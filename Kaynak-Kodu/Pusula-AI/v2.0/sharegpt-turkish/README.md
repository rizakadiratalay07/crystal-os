---
license: apache-2.0
task_categories:
- text-generation
language:
- tr
pretty_name: AhiskaAI ShareGPT Turkish
tags:
- turkish
- sharegpt
- slm
- conversational
---

# AhiskaAI ShareGPT Turkish

**AhiskaAI ShareGPT Turkish** is a high-quality, cleaned, and curated conversational dataset optimized for training Small Language Models (SLMs) in Turkish. This dataset is processed specifically for natural language understanding and instruction-following tasks.

---

## 📊 Dataset Details

* **Language:** Turkish (`tr`)
* **Format:** ShareGPT (`.json`)
* **Application:** Suitable for Supervised Fine-Tuning (SFT) and instruction tuning of LLMs/SLMs.

---

## 🛠️ Usage

You can load this dataset directly using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("AhiskaAI/sharegpt-turkish")

# Print the first sample
print(dataset['train'][0])
```
---
## Cleaning & Curation & Translation
To ensure high-quality training runs, this dataset went through a rigorous pipeline:
1. **Machine Translation:** Raw English ShareGPT data was carefully translated into Turkish using Meta's **NLLB-200 (600M parameter)** model, ensuring high-quality cross-lingual alignment.
2. **Structural Validation:** Format inconsistencies and broken JSON inputs were removed.
3. **Quality Filtering:** Low-quality or non-coherent dialogues were purged.
4. **Syntax Verification:** Responses were checked to maintain natural Turkish grammar and conversational context.
