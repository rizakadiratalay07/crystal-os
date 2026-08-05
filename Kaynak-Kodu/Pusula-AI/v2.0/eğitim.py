import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import json
import sentencepiece as spm
import time
import math
import sys
import traceback
import ijson

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

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
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
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ffn = SwiGLU(embed_dim, embed_dim * 4, dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x, mask=None):
        def attn_forward(module, x, mask):
            return module.attn(module.norm1(x), mask)

        def ffn_forward(module, x):
            return module.ffn(module.norm2(x))

        x = x + checkpoint(attn_forward, self, x, mask, use_reentrant=False)
        x = x + checkpoint(ffn_forward, self, x, use_reentrant=False)
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, pad_idx, dropout=0.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight

    def forward(self, input_ids, mask=None):
        B, T = input_ids.shape
        x = self.token_embed(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len, overlap=50):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.overlap = overlap
        self.pad_id = tokenizer.pad_id()
        print("📑 Metinler tokenize ediliyor...")
        self.tokenized = [tokenizer.encode(t, out_type=int) for t in texts]
        print("✅ Tokenizasyon tamamlandı.")
        self.chunk_map = []
        step = max_seq_len - overlap
        chunk_len = max_seq_len + 1
        for i, tokens in enumerate(self.tokenized):
            if len(tokens) < 50:
                continue
            for start in range(0, len(tokens) - chunk_len + 1, step):
                if start + chunk_len <= len(tokens):
                    self.chunk_map.append((i, start))
                else:
                    break
            if len(self.chunk_map) == 0 or self.chunk_map[-1][0] != i:
                self.chunk_map.append((i, 0))
        print(f"✅ Toplam {len(self.chunk_map)} örnek oluşturuldu.")

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        text_idx, start = self.chunk_map[idx]
        tokens = self.tokenized[text_idx]
        chunk = tokens[start:start + self.max_seq_len + 1]
        if len(chunk) < self.max_seq_len + 1:
            chunk = chunk + [self.pad_id] * (self.max_seq_len + 1 - len(chunk))
        else:
            chunk = chunk[:self.max_seq_len + 1]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        target_ids = [t if t != self.pad_id else -100 for t in target_ids]
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long))

def load_wikipedia_texts(json_path, max_articles=None):
    texts = []
    count = 0
    print("📂 JSON dosyası akıllıca okunuyor (ijson)...")
    try:
        with open(json_path, 'rb') as f:
            parser = ijson.parse(f)
            for prefix, event, value in parser:
                if prefix == 'item' and event == 'start_map':
                    item = {}
                elif prefix.startswith('item.') and event == 'string':
                    key = prefix.split('.')[-1]
                    if key in ('baslik', 'metin'):
                        item[key] = value
                elif prefix == 'item' and event == 'end_map':
                    if 'metin' in item and len(item.get('metin', '')) > 100:
                        texts.append(item['metin'])
                        count += 1
                        if max_articles and count >= max_articles:
                            break
                    item = {}
    except ImportError:
        print("⚠️  ijson kurulu değil, eski yöntemle tüm JSON yükleniyor (RAM tüketebilir).")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i, item in enumerate(data):
            if max_articles and i >= max_articles:
                break
            text = item.get('metin', '')
            if text and len(text) > 100:
                texts.append(text)
    print(f"📚 {len(texts)} makale yüklendi.")
    return texts

def train_tokenizer(texts, vocab_size=8000, model_prefix='tokenizer'):
    temp_file = 'bpe_input.txt'
    print("📝 Tokenizer eğitimi için metinler yazılıyor...")
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    print("✅ Dosya oluşturuldu. Eğitim başlıyor...")
    try:
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            pad_id=0, unk_id=1, bos_id=-1, eos_id=2,
            pad_piece='<PAD>', unk_piece='<UNK>', eos_piece='<EOS>',
            character_coverage=0.9995,
            model_type='bpe',
            split_digits=False,
            byte_fallback=False,
        )
        print("✅ Tokenizer eğitimi tamamlandı.")
    except Exception as e:
        print(f"❌ Tokenizer eğitimi sırasında hata: {e}")
        raise
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    print(f"✅ Tokenizer kaydedildi. Kelime sayısı: {vocab_size}")

def get_tokenizer(model_path='tokenizer.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs,
                scaler, scheduler, grad_accum_steps=8):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1)) / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum_steps

        if (batch_idx + 1) % 20 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item()*grad_accum_steps:.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e}")

    epoch_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} tamamlandı | Loss: {epoch_loss:.4f} | Süre: {elapsed:.1f}s")
    return epoch_loss

def main():
    print("🟢 Program başladı.")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Cihaz: {device}")
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        embed_dim = 768
        num_heads = 12
        num_layers = 14
        max_seq_len = 1024
        vocab_size = 16000
        batch_size = 16
        grad_accum_steps = 4
        epochs = 3
        lr = 3e-4
        dropout = 0.1
        max_articles = 40000

        print("📂 Veri yükleniyor...")
        texts = load_wikipedia_texts('wikipedia_tr.json', max_articles)
        if not texts:
            print("❌ Veri bulunamadı! Önce xml_to_json.py çalıştır.")
            return

        if os.path.exists('tokenizer.model'):
            print("🔄 Mevcut tokenizer bulundu, kontrol ediliyor...")
            sp_test = get_tokenizer()
            if sp_test.get_piece_size() == vocab_size:
                print("✅ Tokenizer zaten mevcut, kullanılıyor.")
                sp = sp_test
            else:
                print("ℹ️  Tokenizer vocab boyutu uyuşmuyor, yeniden eğitiliyor.")
                train_tokenizer(texts, vocab_size)
                sp = get_tokenizer()
        else:
            print("🆕 Tokenizer eğitiliyor...")
            train_tokenizer(texts, vocab_size)
            sp = get_tokenizer()

        print(f"🧩 Tokenizer yüklendi, kelime sayısı: {sp.get_piece_size()}")

        print("📊 Veri seti oluşturuluyor...")
        dataset = TextDataset(texts, sp, max_seq_len, overlap=50)
        del texts
        torch.cuda.empty_cache()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        print(f"📦 Toplam batch: {len(dataloader)}")

        pad_idx = sp.pad_id()
        print("🧠 Model oluşturuluyor...")
        model = MiniLLM(
            vocab_size=sp.get_piece_size(),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            dropout=dropout
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Model parametre sayısı: {total_params:,}")

        print("🔍 Model test ediliyor (forward) ...")
        dummy_input = torch.randint(0, sp.get_piece_size(), (batch_size, max_seq_len)).to(device)
        with torch.no_grad():
            out = model(dummy_input)
            print(f"✅ Forward başarılı, çıkış boyutu: {out.shape}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        total_steps = (len(dataloader) // grad_accum_steps) * epochs
        warmup_steps = int(0.1 * total_steps) if total_steps > 0 else 0
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        best_loss = float('inf')
        print(f"\n🚀 Eğitim başlıyor ({epochs} epoch)...")
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            loss = train_epoch(
                model, dataloader, optimizer, criterion, device,
                epoch, epochs, scaler, scheduler, grad_accum_steps
            )

            torch.save(model.state_dict(), f'pusula_ai_epoch{epoch+1}.pt')
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'pusula_ai_best.pt')
                print(f"⭐ En iyi model kaydedildi (loss: {best_loss:.4f})")
            torch.cuda.empty_cache()

        print(f"\n✅ Eğitim tamamlandı. En iyi loss: {best_loss:.4f}")
        print("📁 Modeller: pusula_ai_best.pt, pusula_ai_epoch*.pt")

    except Exception as e:
        print("\n❌ KRİTİK HATA:")
        traceback.print_exc()
        print(f"Hata mesajı: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
