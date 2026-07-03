import torch
import torch.nn as nn
import torch.optim as optim
import json
import sentencepiece as spm
import os
import time
from datetime import timedelta

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

def load_samples(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for obj in data:
        history = []
        for turn in obj['sohbet']:
            user_msg = turn['misafir']
            assistant_msg = turn['Pusula-AI']
            prompt = ''.join(
                f"Soru: {h['misafir']}\nCevap: {h['Pusula-AI']}\n"
                for h in history
            )
            prompt += f"Soru: {user_msg}\nCevap: "
            samples.append((prompt, assistant_msg))
            history.append({'misafir': user_msg, 'Pusula-AI': assistant_msg})
    return samples

def train_bpe_tokenizer(input_txt, model_prefix='tokenizer', vocab_size=4000):
    spm.SentencePieceTrainer.train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=0, unk_id=1, bos_id=-1, eos_id=2,
        pad_piece='<PAD>', unk_piece='<UNK>', eos_piece='<EOS>',
        character_coverage=0.9995,
        model_type='bpe'
    )

def get_tokenizer(model_path='tokenizer.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def encode_train(prompt, answer, sp, max_len, pad_idx, eos_idx):
    prompt_ids = sp.encode(prompt, out_type=int)
    answer_ids = sp.encode(answer, out_type=int) + [eos_idx]
    if len(answer_ids) > max_len:
        answer_ids = answer_ids[:max_len]
    total_len = len(prompt_ids) + len(answer_ids)
    if total_len > max_len:
        prompt_ids = prompt_ids[-(max_len - len(answer_ids)):]
        if len(prompt_ids) + len(answer_ids) > max_len:
            prompt_ids = []
    input_ids = prompt_ids + answer_ids
    target_ids = [-100] * len(prompt_ids) + answer_ids
    pad_len = max_len - len(input_ids)
    input_ids += [pad_idx] * pad_len
    target_ids += [-100] * pad_len
    return input_ids, target_ids

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)
    optimizer.zero_grad()
    print(f"\nEpoch {epoch+1}/{total_epochs} başlıyor...")
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        inp, tgt = x[:, :-1], y[:, 1:]
        logits = model(inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            progress = (batch_idx + 1) / total_batches * 100
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx+1}/{total_batches} (%{progress:.0f}) | Kayıp: {loss.item():.4f} | Ortalama: {avg_loss:.4f} | Süre: {elapsed:.1f}s")

    epoch_loss = total_loss / total_batches
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} tamamlandı | Kayıp: {epoch_loss:.4f} | Süre: {elapsed:.1f}s")
    return epoch_loss

def main():
    torch.set_num_threads(4) 
    device = torch.device('cpu')
    print(f"Cihaz: {device}")

    for fp in ['bpe_input.txt', 'pusula_ai.pt', 'tokenizer.model', 'tokenizer.vocab']:
        if os.path.exists(fp):
            os.remove(fp)

    embed_dim  = 320
    num_heads  = 4
    num_layers = 5
    max_seq_len = 320
    vocab_size  = 4000
    batch_size  = 24
    epochs      = 30
    lr          = 0.0005

    samples = load_samples('veri_seti/turkce_veri.json')
    print(f"Toplam örnek: {len(samples)}")

    with open('bpe_input.txt', 'w', encoding='utf-8') as f:
        for prompt, answer in samples:
            f.write(prompt + answer + '\n')

    print("Tokenizer eğitiliyor...")
    train_bpe_tokenizer('bpe_input.txt', vocab_size=vocab_size)
    sp = get_tokenizer()
    vocab_size = sp.get_piece_size()
    pad_idx    = sp.pad_id()
    eos_idx    = sp.eos_id()
    print("Tokenizer hazır.")

    print("Veri hazırlanıyor...")
    all_inputs, all_targets = [], []
    for prompt, answer in samples:
        inp, tgt = encode_train(prompt, answer, sp, max_seq_len, pad_idx, eos_idx)
        all_inputs.append(torch.tensor(inp, dtype=torch.long))
        all_targets.append(torch.tensor(tgt, dtype=torch.long))

    dataset = torch.utils.data.TensorDataset(
        torch.stack(all_inputs),
        torch.stack(all_targets)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True
    )
    print(f"Veri hazır. {len(dataloader)} batch/epoch")

    model = MiniLLM(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, pad_idx).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\nEğitim başlıyor. {epochs} epoch, {len(dataloader)} batch/epoch")
    print("=" * 60)

    total_start_time = time.time()
    best_loss = float('inf')
    patience_early_stop = 5
    no_improve_epochs = 0

    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch, epochs)
        
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Kayıp: {loss:.4f} | Öğrenme oranı: {current_lr:.6f}")
        print("-" * 60)

        if loss < best_loss:
            best_loss = loss
            no_improve_epochs = 0
            torch.save(model, 'pusula_ai.pt')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience_early_stop:
                print(f"Erken durdurma: {patience_early_stop} epoch boyunca iyileşme yok.")
                break

    total_training_time = time.time() - total_start_time
    hours, rem = divmod(total_training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    print(f"\nEn iyi kayıp: {best_loss:.4f}")
    print(f"Model 'pusula_ai.pt' olarak kaydedildi.")
    print(f"Toplam eğitim süresi: {time_str} (saat:dakika:saniye)")

if __name__ == '__main__':
    main()
