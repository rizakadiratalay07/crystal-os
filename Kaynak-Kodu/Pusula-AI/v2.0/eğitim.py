import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import sentencepiece as spm
import time
import math
import sys
import traceback
import multiprocessing
import gc
import resource
import json
import pandas as pd
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.multiprocessing.set_sharing_strategy('file_system')


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
        self.ffn = SwiGLU(embed_dim, embed_dim * 2, dropout)  # FFN boyutunu 2x yaparak parametre sayısını azaltıp katman sayısını artırıyoruz
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def _attn_forward(self, x):
        return self.attn(self.norm1(x))

    def _ffn_forward(self, x):
        return self.ffn(self.norm2(x))

    def forward(self, x):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            x = x + checkpoint(self._attn_forward, x, use_reentrant=False)
            x = x + checkpoint(self._ffn_forward, x, use_reentrant=False)
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

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embed(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_id()

        print("📑 Metinler tokenize ediliyor ve diyalog bütünlüğü korunuyor...")
        self.tokenized = []
        skipped = 0
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                print(f"  Tokenize: {i}/{len(texts)}")
            tokens = torch.tensor(tokenizer.encode(text, out_type=int), dtype=torch.int32)
            # Diyalog bütünlüğünü korumak için her metni ayrı örnek olarak alıyoruz
            if len(tokens) == 0:
                skipped += 1
                continue
            # Uzun metinleri baştan kırp (veya son kısmı al)
            if len(tokens) > self.max_seq_len:
                # Son max_seq_len token'ı al (soru-cevap için son kısım daha önemli)
                tokens = tokens[-self.max_seq_len:]
            self.tokenized.append(tokens)
        print(f"✅ Tokenizasyon tamamlandı. Toplam {len(self.tokenized)} örnek, {skipped} atlandı.")

        self.total_tokens = sum(len(tokens) for tokens in self.tokenized)
        print(f"📊 Toplam token sayısı: {self.total_tokens:,}")

        del texts
        gc.collect()

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        tokens = self.tokenized[idx]
        # Hedef için bir sonraki token'ı tahmin et
        input_ids = tokens[:-1]
        target_ids = tokens[1:].clone()

        # Padding uygula (input ve target'ı max_seq_len'e tamamla)
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            pad = torch.full((pad_len,), self.pad_id, dtype=torch.int32)
            input_ids = torch.cat([input_ids, pad])
            target_ids = torch.cat([target_ids, pad])
            target_ids[target_ids == self.pad_id] = -100
        else:
            input_ids = input_ids[:self.max_seq_len]
            target_ids = target_ids[:self.max_seq_len]

        return (input_ids.to(torch.long), target_ids.to(torch.long))


def check_data_sufficiency(total_tokens, total_params):
    print("\n" + "=" * 60)
    print("📊 VERİ SETİ YETERLİLİK ANALİZİ")
    print("=" * 60)
    print(f"Toplam token sayısı: {total_tokens:,}")
    print(f"Model parametre sayısı: {total_params:,}")

    if total_params == 0:
        print("⚠️ Parametre sayısı 0, analiz yapılamadı.")
        return

    ratio = total_tokens / total_params
    print(f"Token/Parametre oranı: {ratio:.2f}")

    if ratio < 5:
        print("🔴 VERİ SETİ YETERSİZ! Model bu kadar az veriyle eğitilemez.")
        print("   Öneri: Veri setini büyütün veya model boyutunu küçültün.")
    elif ratio < 10:
        print("🟡 VERİ SETİ SINIRDA. Eğitim mümkün ancak overfitting riski yüksek.")
        print("   Öneri: Daha fazla veri ekleyin veya model boyutunu azaltın.")
    elif ratio < 50:
        print("🟢 VERİ SETİ YETERLİ. Model eğitimi için uygun miktarda veri var.")
    else:
        print("✅ VERİ SETİ FAZLASIYLA YETERLİ. Model bu veri setiyle rahatlıkla eğitilebilir.")
    print("=" * 60 + "\n")


def load_qa_texts(qa_file_path, max_samples=None):
    print(f"📂 QA veri seti yükleniyor: {qa_file_path}")
    texts = []
    count = 0

    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ QA dosyası bulunamadı: {qa_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON ayrıştırma hatası: {e}")
        return []

    if isinstance(data, dict) and "train" in data:
        samples = data["train"]
        print("   Format: 'train' anahtarı bulundu.")
    elif isinstance(data, list):
        samples = data
        print("   Format: Düz JSON dizisi.")
    else:
        print(f"   ❌ Bilinmeyen JSON formatı: {type(data)}")
        return []

    for sample in samples:
        if max_samples is not None and count >= max_samples:
            break
        try:
            if "input" in sample and "output" in sample:
                inp = sample["input"].strip()
                out = sample["output"].strip()
            elif "question" in sample and "answer" in sample:
                inp = sample["question"].strip()
                out = sample["answer"].strip()
            else:
                inp = sample.get("text") or sample.get("prompt") or ""
                out = sample.get("response") or sample.get("completion") or ""
                inp = inp.strip()
                out = out.strip()

            if inp and out:
                combined = f"İnsan: {inp}\nAsistan: {out}"
                texts.append(combined)
                count += 1
        except Exception:
            continue

    print(f"📚 {len(texts)} QA örneği yüklendi.")
    return texts


def load_parquet_qa_texts(parquet_path, max_samples=None):
    print(f"📂 Parquet QA veri seti yükleniyor: {parquet_path}")
    texts = []
    try:
        df = pd.read_parquet(parquet_path)
        print(f"   Parquet okundu, toplam satır: {len(df)}")

        if 'question' in df.columns and 'answer' in df.columns:
            soru_col, cevap_col = 'question', 'answer'
        elif 'soru' in df.columns and 'cevap' in df.columns:
            soru_col, cevap_col = 'soru', 'cevap'
        else:
            print(f"⚠️ Bilinmeyen sütunlar: {df.columns.tolist()}")
            soru_col, cevap_col = df.columns[0], df.columns[1]
            print(f"   Kullanılacak: '{soru_col}' ve '{cevap_col}'")

        count = 0
        for idx, row in df.iterrows():
            if max_samples is not None and count >= max_samples:
                break
            try:
                soru = str(row[soru_col]).strip()
                cevap = str(row[cevap_col]).strip()
                if soru and cevap and soru != 'nan' and cevap != 'nan':
                    combined = f"İnsan: {soru}\nAsistan: {cevap}"
                    texts.append(combined)
                    count += 1
            except Exception:
                continue

        print(f"📚 {len(texts)} Parquet QA örneği yüklendi.")
        return texts

    except FileNotFoundError:
        print(f"❌ Parquet dosyası bulunamadı: {parquet_path}")
        return []
    except Exception as e:
        print(f"❌ Parquet okuma hatası: {e}")
        return []


def load_chat_texts(chat_file_path, max_samples=None):
    print(f"📂 Chat veri seti yükleniyor: {chat_file_path}")
    texts = []
    count = 0

    try:
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples is not None and count >= max_samples:
                    break
                try:
                    entry = json.loads(line.strip())
                    chat_messages = entry.get("chat", [])
                    if not chat_messages:
                        continue

                    dialog_text = ""
                    for msg in chat_messages:
                        role = msg.get("role", "").strip()
                        content = msg.get("content", "").strip()
                        if not role or not content:
                            continue
                        if role == "user":
                            dialog_text += f"İnsan: {content}\n"
                        elif role == "assistant":
                            dialog_text += f"Asistan: {content}\n"

                    if dialog_text:
                        texts.append(dialog_text.strip())
                        count += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Chat dosyası bulunamadı: {chat_file_path}")
        return []

    print(f"📚 {len(texts)} chat örneği yüklendi.")
    return texts


def load_sharegpt_texts(sharegpt_file_path, max_samples=None):
    print(f"📂 ShareGPT veri seti yükleniyor: {sharegpt_file_path}")
    texts = []
    count = 0

    try:
        with open(sharegpt_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ShareGPT dosyası bulunamadı: {sharegpt_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON ayrıştırma hatası: {e}")
        return []

    if isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
        print("   Format: 'conversations' anahtarı bulundu.")
    elif isinstance(data, list):
        conversations = data
        print("   Format: Düz JSON dizisi.")
    else:
        print(f"   ❌ Bilinmeyen JSON formatı: {type(data)}")
        return []

    for conv in conversations:
        if max_samples is not None and count >= max_samples:
            break

        if isinstance(conv, dict) and "conversations" in conv:
            messages = conv["conversations"]
        elif isinstance(conv, list):
            messages = conv
        else:
            continue

        if not messages:
            continue

        dialog_lines = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            from_field = msg.get("from", "").strip()
            value = msg.get("value", "").strip()
            if not from_field or not value:
                continue

            if from_field == "human":
                role = "İnsan"
            elif from_field == "gpt":
                role = "Asistan"
            else:
                continue

            dialog_lines.append(f"{role}: {value}")

        if dialog_lines:
            full_text = "\n".join(dialog_lines)
            texts.append(full_text)
            count += 1

    print(f"📚 {len(texts)} ShareGPT örneği yüklendi.")
    return texts


def load_daily_dialogues_texts(parquet_path, max_samples=None):
    print(f"📂 Günlük Diyalog veri seti yükleniyor: {parquet_path}")
    texts = []
    try:
        df = pd.read_parquet(parquet_path)
        print(f"   Parquet okundu, toplam satır: {len(df)}")

        count = 0
        for idx, row in df.iterrows():
            if max_samples is not None and count >= max_samples:
                break
            try:
                messages = row.get("messages")
                if messages is None:
                    continue
                if not isinstance(messages, (list, np.ndarray)):
                    continue
                if len(messages) == 0:
                    continue

                dialog_text = ""
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "").strip()
                    content = msg.get("content", "").strip()
                    if not role or not content:
                        continue
                    if role == "user":
                        dialog_text += f"İnsan: {content}\n"
                    elif role == "assistant":
                        dialog_text += f"Asistan: {content}\n"

                if dialog_text:
                    texts.append(dialog_text.strip())
                    count += 1
            except Exception:
                continue

        print(f"📚 {len(texts)} günlük diyalog örneği yüklendi.")
        return texts

    except FileNotFoundError:
        print(f"❌ Günlük diyalog dosyası bulunamadı: {parquet_path}")
        return []
    except Exception as e:
        print(f"❌ Günlük diyalog okuma hatası: {e}")
        return []


def load_ultrachat_texts(parquet_path, max_samples=None):
    print(f"📂 UltraChat veri seti yükleniyor: {parquet_path}")
    texts = []
    try:
        df = pd.read_parquet(parquet_path)
        print(f"   Parquet okundu, toplam satır: {len(df)}")

        count = 0
        for idx, row in df.iterrows():
            if max_samples is not None and count >= max_samples:
                break
            try:
                messages = row.get("messages")
                if messages is None:
                    continue
                if not isinstance(messages, (list, np.ndarray)):
                    continue
                if len(messages) == 0:
                    continue

                dialog_text = ""
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "").strip()
                    content = msg.get("content", "").strip()
                    if not role or not content:
                        continue
                    if role == "user":
                        dialog_text += f"İnsan: {content}\n"
                    elif role == "assistant":
                        dialog_text += f"Asistan: {content}\n"

                if dialog_text:
                    texts.append(dialog_text.strip())
                    count += 1
            except Exception:
                continue

        print(f"📚 {len(texts)} UltraChat örneği yüklendi.")
        return texts

    except FileNotFoundError:
        print(f"❌ UltraChat dosyası bulunamadı: {parquet_path}")
        return []
    except Exception as e:
        print(f"❌ UltraChat okuma hatası: {e}")
        return []


def load_turkish_llm_v10_texts(file_path, max_samples=None):
    print(f"📂 Turkish LLM v10 veri seti yükleniyor: {file_path}")
    texts = []
    count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples is not None and count >= max_samples:
                    break
                try:
                    entry = json.loads(line.strip())
                    prompt = entry.get("prompt", "").strip()
                    completion = entry.get("completion", "").strip()
                    if prompt and completion:
                        combined = f"İnsan: {prompt}\nAsistan: {completion}"
                        texts.append(combined)
                        count += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"❌ Turkish LLM v10 dosyası bulunamadı: {file_path}")
        return []

    print(f"📚 {len(texts)} Turkish LLM v10 örneği yüklendi.")
    return texts


def train_tokenizer(texts, vocab_size=8000, model_prefix='tokenizer', byte_fallback=True):
    if not texts:
        raise ValueError("Tokenizer eğitmek için metin gerekli.")
    temp_file = 'bpe_input.txt'
    print("📝 Tokenizer eğitiliyor...")
    sample_size = min(200000, len(texts))
    with open(temp_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts[:sample_size]):
            if i % 10000 == 0:
                print(f"  Yazıldı: {i}/{sample_size}")
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
            byte_fallback=byte_fallback,
        )
        print("✅ Tokenizer eğitimi tamamlandı.")
    except Exception as e:
        print(f"❌ Tokenizer hatası: {e}")
        raise
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    print(f"✅ Tokenizer kaydedildi. Kelime sayısı: {vocab_size}")


def get_tokenizer(model_path='tokenizer.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.next_input = None
        self.next_target = None
        self._preload()

    def _preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        if self.stream is None:
            self.next_input = self.next_input.to(self.device)
            self.next_target = self.next_target.to(self.device)
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        input_ids = self.next_input
        target_ids = self.next_target
        if input_ids is None:
            return None, None
        if self.stream is not None:
            input_ids.record_stream(torch.cuda.current_stream())
            target_ids.record_stream(torch.cuda.current_stream())
        self._preload()
        return input_ids, target_ids


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs,
                 scaler, scheduler, amp_dtype, use_grad_scaler,
                 grad_accum_steps=1, log_interval=1):
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    prev_log_time = start_time
    prefetcher = CUDAPrefetcher(dataloader, device)
    num_batches = len(dataloader)
    batch_idx = 0
    input_ids, target_ids = prefetcher.next()

    while input_ids is not None:
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1)) / grad_accum_steps

        if use_grad_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_grad_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.detach() * grad_accum_steps

        if (batch_idx + 1) % log_interval == 0:
            current_time = time.time()
            elapsed_since_log = current_time - prev_log_time
            prev_log_time = current_time
            avg_loss = (total_loss / (batch_idx + 1)).item()
            current_loss_value = (loss.detach() * grad_accum_steps).item()
            lr = scheduler.get_last_lr()[0]
            speed = elapsed_since_log / log_interval
            print(f"  Batch {batch_idx+1}/{num_batches} | Loss: {current_loss_value:.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e} | Speed: {speed:.3f}s/batch")

        batch_idx += 1
        input_ids, target_ids = prefetcher.next()

    del prefetcher
    torch.cuda.synchronize()

    epoch_loss = (total_loss / num_batches).item()
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} tamamlandı | Loss: {epoch_loss:.4f} | Süre: {elapsed:.1f}s")
    return epoch_loss


def main():
    multiprocessing.set_start_method('spawn', force=True)
    print("🟢 Program başladı.")

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(65536, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"🔧 Dosya limiti artırıldı: {soft} -> {new_soft}")
    except Exception as e:
        print(f"⚠️ Dosya limiti artırılamadı: {e}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Cihaz: {device}")
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # ============================================================
        #  MODEL KONFİGÜRASYONU
        # ============================================================
        embed_dim = 256
        num_heads = 8
        num_layers = 8
        max_seq_len = 256
        vocab_size = 16000
        batch_size = 100
        grad_accum_steps = 4
        epochs = 20
        lr = 3e-4
        dropout = 0.1
        weight_decay = 0.01
        overlap = 320

        # ----- Veri setleri (hepsi açık) -----
        use_qa_dataset = True
        qa_dataset_path = "turkish-qa-multi-dialog-dataset/data.json"
        qa_max_samples = None

        use_parquet_qa_dataset = True
        parquet_qa_path = "tr_soru_cevap/train-00000-of-00001.parquet"
        parquet_qa_max_samples = None

        use_chat_dataset = True
        chat_dataset_path = "Turkish-Chat_GPT-4O/train.jsonl"
        chat_max_samples = None

        use_sharegpt_dataset = True
        sharegpt_dataset_path = "sharegpt-turkish/dataset.json"
        sharegpt_max_samples = None

        use_daily_dialogues = True
        daily_dialogues_path = "turkish-daily-dialogues-5k/train.parquet"
        daily_max_samples = None

        use_ultrachat_dataset = True
        ultrachat_dataset_path = "UltraChatTR_50k/ultrachat-tr_100k.parquet"
        ultrachat_max_samples = None

        use_turkish_llm_v10_dataset = True
        turkish_llm_v10_path = "Turkish-LLM-v10-Training/train.jsonl"
        turkish_llm_v10_max_samples = None

        log_interval = 50

        amp_dtype = torch.bfloat16
        use_grad_scaler = (amp_dtype == torch.float16)
        use_torch_compile = True
        num_workers = 2
        checkpoint_every = 1
        pct_start = 0.2
        byte_fallback = True
        # ============================================================

        print("📂 Veri yükleniyor...")
        texts = []

        if use_qa_dataset:
            qa_texts = load_qa_texts(qa_dataset_path, qa_max_samples)
            if qa_texts:
                texts.extend(qa_texts)
            else:
                print("⚠️ QA verisi boş veya yüklenemedi.")

        if use_parquet_qa_dataset:
            parquet_texts = load_parquet_qa_texts(parquet_qa_path, parquet_qa_max_samples)
            if parquet_texts:
                texts.extend(parquet_texts)
            else:
                print("⚠️ Parquet QA verisi boş veya yüklenemedi.")

        if use_chat_dataset:
            chat_texts = load_chat_texts(chat_dataset_path, chat_max_samples)
            if chat_texts:
                texts.extend(chat_texts)
            else:
                print("⚠️ Chat verisi boş veya yüklenemedi.")

        if use_sharegpt_dataset:
            sharegpt_texts = load_sharegpt_texts(sharegpt_dataset_path, sharegpt_max_samples)
            if sharegpt_texts:
                texts.extend(sharegpt_texts)
            else:
                print("⚠️ ShareGPT verisi boş veya yüklenemedi.")

        if use_daily_dialogues:
            daily_texts = load_daily_dialogues_texts(daily_dialogues_path, daily_max_samples)
            if daily_texts:
                texts.extend(daily_texts)
            else:
                print("⚠️ Günlük diyalog verisi boş veya yüklenemedi.")

        if use_ultrachat_dataset:
            ultrachat_texts = load_ultrachat_texts(ultrachat_dataset_path, ultrachat_max_samples)
            if ultrachat_texts:
                texts.extend(ultrachat_texts)
            else:
                print("⚠️ UltraChat verisi boş veya yüklenemedi.")

        if use_turkish_llm_v10_dataset:
            turkish_llm_v10_texts = load_turkish_llm_v10_texts(turkish_llm_v10_path, turkish_llm_v10_max_samples)
            if turkish_llm_v10_texts:
                texts.extend(turkish_llm_v10_texts)
            else:
                print("⚠️ Turkish LLM v10 verisi boş veya yüklenemedi.")

        if not texts:
            print("❌ Hiç veri yüklenemedi! Çıkılıyor.")
            return

        print(f"✅ Toplam metin sayısı: {len(texts)}")

        if os.path.exists('tokenizer.model'):
            print("🔄 Tokenizer bulundu, kontrol ediliyor...")
            sp_test = get_tokenizer()
            if sp_test.get_piece_size() == vocab_size:
                print("✅ Tokenizer uygun.")
                sp = sp_test
            else:
                print("ℹ️ Kelime sayısı uyuşmuyor, yeniden eğitiliyor...")
                train_tokenizer(texts, vocab_size, byte_fallback=byte_fallback)
                sp = get_tokenizer()
        else:
            print("🆕 Tokenizer eğitiliyor...")
            train_tokenizer(texts, vocab_size, byte_fallback=byte_fallback)
            sp = get_tokenizer()

        print(f"🧩 Tokenizer yüklendi, kelime sayısı: {sp.get_piece_size()}")

        print("📊 Veri seti oluşturuluyor...")
        dataset = TextDataset(texts, sp, max_seq_len)
        del texts
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(min(4, cpu_count))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )
        print(f"📦 Toplam batch: {len(dataloader)} | Worker: {num_workers}")

        pad_idx = sp.pad_id()
        print(f"🏁 Gradient checkpointing: her {checkpoint_every} blokta")

        print("🧠 Model oluşturuluyor...")
        model = MiniLLM(
            vocab_size=sp.get_piece_size(),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            dropout=dropout,
            checkpoint_every=checkpoint_every,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Model parametre sayısı: {total_params:,}")

        check_data_sufficiency(dataset.total_tokens, total_params)

        print("🔍 İleri besleme testi...")
        dummy_input = torch.randint(0, sp.get_piece_size(), (batch_size, max_seq_len)).to(device)
        with torch.no_grad():
            out = model(dummy_input)
            print(f"✅ Başarılı, çıkış boyutu: {out.shape}")
        del dummy_input, out
        torch.cuda.empty_cache()

        if use_torch_compile:
            try:
                model = torch.compile(model)
                print("✅ torch.compile etkin.")
            except Exception as e:
                print(f"⚠️ torch.compile başarısız: {e}")

        try:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
            print("✅ Fused AdamW etkin.")
        except (TypeError, RuntimeError) as e:
            print(f"⚠️ Fused AdamW kullanılamadı, standart AdamW ile devam: {e}")
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = (len(dataloader) // grad_accum_steps) * epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos',
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

        best_loss = float('inf')
        print(f"\n🚀 Eğitim başlıyor ({epochs} epoch)...")
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            loss = train_epoch(
                model, dataloader, optimizer, criterion, device,
                epoch, epochs, scaler, scheduler, amp_dtype, use_grad_scaler,
                grad_accum_steps,
                log_interval=log_interval
            )

            torch.save(model.state_dict(), f'pusula_ai_epoch{epoch+1}.pt')
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'pusula_ai_best.pt')
                print(f"⭐ En iyi model kaydedildi (loss: {best_loss:.4f})")
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\n✅ Eğitim tamamlandı. En iyi loss: {best_loss:.4f}")
        print("📁 Modeller: pusula_ai_best.pt, pusula_ai_epoch*.pt")

        if 'dataloader' in locals():
            dataloader._iterator = None
            if hasattr(dataloader, '_shutdown_workers'):
                dataloader._shutdown_workers()
            del dataloader

    except Exception as e:
        print("\n❌ KRİTİK HATA:")
        traceback.print_exc()
        print(f"Hata: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
