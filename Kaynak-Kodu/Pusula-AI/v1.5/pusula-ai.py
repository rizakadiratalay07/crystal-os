import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

        self.register_buffer('causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1))
        self.register_buffer('positions', torch.arange(max_seq_len).unsqueeze(0))

        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

    def forward(self, x):
        B, T = x.shape
        emb = self.embedding(x) + self.pos_embedding(self.positions[:, :T])
        pad_mask = (x == self.pad_idx)
        mask = self.causal_mask[:T, :T]
        out = self.encoder(src=emb, mask=mask, src_key_padding_mask=pad_mask)
        return self.fc(out)

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_k=40, repetition_penalty=1.2):
    model.eval()
    device = next(model.parameters()).device
    eos_idx = tokenizer.eos_id()
    pad_idx = model.pad_idx
    max_seq_len = model.max_seq_len

    with torch.no_grad():
        ids = tokenizer.encode(prompt)
        
        if len(ids) >= max_seq_len - 50:
            ids = ids[-(max_seq_len - 50):]

        generated = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        prompt_len = generated.size(1)
        generated_tokens = []

        for _ in range(max_new_tokens):
            logits = model(generated)
            next_logits = logits[0, -1, :] / temperature
            
            for token in set(generated_tokens[-10:]):
                next_logits[token] /= repetition_penalty
            
            k = min(top_k, next_logits.size(-1))
            top_vals, top_indices = torch.topk(next_logits, k)
            threshold = top_vals[-1]
            next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            if next_token.item() == eos_idx:
                break
                
            if next_token.item() == pad_idx:
                continue
                
            generated = torch.cat([generated, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            if generated.size(1) >= max_seq_len:
                break
                
        output_ids = generated[0, prompt_len:].tolist()
        response = tokenizer.decode_ids(output_ids)
        
        if len(response.strip()) < 3:
            response = "Anlamadım, lütfen sorunu tekrar eder misiniz?"
            
        return response

def chat_loop(model, tokenizer):
    print("\n" + "="*50)
    print("Pusula AI ile sohbet etmeye hoş geldiniz!")
    print("Komutlar:")
    print("   /çıkış   - Sohbetten çıkar")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("Misafir: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == '/çıkış':
                print("Pusula AI: Görüşmek üzere! İyi günler dilerim.")
                break
            
            prompt = f"Soru: {user_input}\nCevap: "
            
            response = generate(model, tokenizer, prompt)
            
            if len(response) > 500:
                response = response[:500] + "..."
            
            print(f"Pusula AI: {response}")
            
        except KeyboardInterrupt:
            print("\nPusula AI: Program sonlandırılıyor. Görüşmek üzere!")
            break
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            print("Lütfen tekrar deneyin.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    try:
        model = torch.load(os.path.join(BASE_DIR, "pusula_ai.pt"), map_location=device, weights_only=False)
        model.eval()
        print("Model başarıyla yüklendi.")
    except FileNotFoundError:
        print("Model dosyası bulunamadı! Lütfen 'eğitim.py' ile modeli eğitin.")
        return
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {str(e)}")
        return
    
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(BASE_DIR, "tokenizer.model"))
        print("Tokenizer başarıyla yüklendi.")
    except FileNotFoundError:
        print("Tokenizer dosyası bulunamadı!")
        return
    
    chat_loop(model, sp)

if __name__ == '__main__':
    main()
