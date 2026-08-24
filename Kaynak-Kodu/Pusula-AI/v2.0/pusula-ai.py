import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import time
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = ""


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim)
        )

        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.0, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attn = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ffn = SwiGLU(embed_dim, embed_dim * 2, dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def _attn_forward(self, x):
        return self.attn(self.norm1(x))

    def _ffn_forward(self, x):
        return self.ffn(self.norm2(x))

    def forward(self, x):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            x = x + torch.utils.checkpoint.checkpoint(self._attn_forward, x, use_reentrant=False)
            x = x + torch.utils.checkpoint.checkpoint(self._ffn_forward, x, use_reentrant=False)
        else:
            x = x + self._attn_forward(x)
            x = x + self._ffn_forward(x)
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, pad_idx,
                 dropout=0.0, checkpoint_every=1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, max_seq_len, dropout,
                use_checkpoint=(checkpoint_every > 0 and (i % checkpoint_every == 0))
            )
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embed(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def clean_response(response):
    for etiket in ["Asistan:", "Cevap:", "Kullanıcı:", "İnsan:", "Gpt:", "Sistem:", "Soru:"]:
        if response.startswith(etiket):
            response = response[len(etiket):].strip()
            break

    etiketler = ["İnsan:", "Kullanıcı:", "Asistan:", "Sistem:", "Gpt:", "Cevap:", "Soru:"]
    en_erken_index = None
    for etiket in etiketler:
        index = response.find(etiket)
        if index != -1:
            if en_erken_index is None or index < en_erken_index:
                en_erken_index = index

    if en_erken_index is not None and en_erken_index > 0:
        response = response[:en_erken_index].strip()

    response = re.sub(r'\s+', ' ', response).strip()
    return response


def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.1, top_k=3, repetition_penalty=2.0):
    model.eval()
    device = next(model.parameters()).device
    eos_idx = tokenizer.eos_id()
    pad_idx = model.pad_idx
    max_seq_len = model.max_seq_len

    with torch.no_grad():
        ids = tokenizer.encode(prompt, out_type=int)
        if len(ids) > max_seq_len - 1:
            ids = ids[-(max_seq_len - 1):]
        generated = torch.tensor([ids], dtype=torch.long).to(device)
        generated_tokens = []

        for _ in range(max_new_tokens):
            logits = model(generated)
            next_logits = logits[0, -1, :] / temperature

            for token in set(generated_tokens[-15:]):
                next_logits[token] /= repetition_penalty

            k = min(top_k, next_logits.size(-1))
            top_vals, top_indices = torch.topk(next_logits, k)
            threshold = top_vals[-1]
            next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == eos_idx:
                break
            if next_token.item() == pad_idx:
                continue

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())

            if generated.size(1) >= max_seq_len:
                break

        output_ids = generated[0, len(ids):].tolist()
        response = tokenizer.decode(output_ids).strip()
        response = clean_response(response)

        if len(response) < 3:
            response = "Bu konuda yeterli bilgim yok, lütfen başka bir şey sorun."

        return response


def chat_loop(model, tokenizer):
    print("\n" + "="*60)
    print("Pusula AI ile sohbet (Diyalog Modeli)")
    print("Komutlar: /çıkış, /temizle")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("👤 Misafir: ").strip()
            if not user_input:
                continue

            if user_input.lower() == '/çıkış':
                print("🤖 Pusula AI: Görüşmek üzere! İyi günler.")
                break

            if user_input.lower() == '/temizle':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            prompt = f"{SYSTEM_PROMPT}\n\nİnsan: {user_input}\nAsistan: "

            start_time = time.time()
            response = generate(model, tokenizer, prompt)
            elapsed = time.time() - start_time

            if len(response) > 600:
                response = response[:600] + "..."

            print(f"🤖 Pusula AI: {response}")
            print(f"   (Yanıt süresi: {elapsed:.2f} saniye)\n")

        except KeyboardInterrupt:
            print("\n🤖 Pusula AI: Program sonlandırılıyor.")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")
            print("Lütfen tekrar deneyin.\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Kullanılan cihaz: {device}")

    vocab_size = 16000
    embed_dim = 256
    num_heads = 8
    num_layers = 8
    max_seq_len = 256
    pad_idx = 0
    dropout = 0.1
    checkpoint_every = 1

    model = MiniLLM(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, pad_idx, dropout, checkpoint_every).to(device)

    model_path = os.path.join(BASE_DIR, "pusula_ai_best.pt")
    if not os.path.exists(model_path):
        print("❌ Model dosyası (pusula_ai_best.pt) bulunamadı! Lütfen eğitim.py'yi çalıştırın.")
        return

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    try:
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        print("✅ Model başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return

    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        print("❌ Tokenizer dosyası (tokenizer.model) bulunamadı!")
        return

    try:
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        print("✅ Tokenizer yüklendi.")
    except Exception as e:
        print(f"❌ Tokenizer yüklenirken hata: {e}")
        return

    chat_loop(model, sp)


if __name__ == '__main__':
    main()
