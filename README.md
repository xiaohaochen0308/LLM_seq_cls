# LoRA Fine-Tuning for Qwen2.5-7B-Instruct (Text Classification & Ad Title Ranking)

This project focuses on fine-tuning the Qwen2.5-7B-Instruct model using Low-Rank Adaptation (LoRA) for **two key scenarios**:

1. **Text Classification**: Sentiment analysis, topic classification, etc.  
2. **Text Ranking**: Learning preference between high-performing and low-performing advertisement titles using pairwise ranking.

By integrating PyTorch and the Transformers library, along with pretrained models from the ModelScope platform, this project enables efficient and scalable downstream adaptation of large language models (LLMs).

---

## 🧠 Features

- 🔍 **Two Task Modes**: Choose between classification and ranking with minimal code changes.
- ⚡ **Efficient Training**: LoRA enables fine-tuning large models with minimal resources.
- 🧱 **Modular Structure**: Easy-to-extend code for your own classification or ranking use case.
- ✅ **Compatible with Qwen2.5 series** from ModelScope.

---

## 📦 Setup

```bash
pip install torch transformers peft modelscope
```

---

## 🏁 Quick Start

### Text Classification

```bash
cd Text_Classification/
python lora_cls.py
```

### Text Ranking (Pairwise)

```bash
cd Text_Ranking/
python train.py
```

---

## 📊 Results

### Text Classification (on N24News Abstract)
| Model                        | Accuracy (Acc) |
|-----------------------------|----------------|
| Qwen2.5-7B-Instruct (LoRA)  | **85.26%**     |
| Bert                        | 78.3%          |
| RoBerta                     | 79.7%          |

---

## 📁 Project Structure

```
Qwen_LoRA_Project/
├── README.md                          # Project documentation
├── Text_Classification/              # Text classification module
│   ├── data.py                       # Loads and processes classification data
│   ├── lora_cls.py                   # LoRA fine-tuning and training script
│   ├── train_trans_abs.json          # Training dataset (abstract text)
│   └── test_trans_abs.json           # Test dataset (abstract text)
│
└── Text_Ranking/                     # Ad title ranking model (CTR preference ranking)
    ├── data.py                       # Loads pairwise ad title data
    ├── model.py                      # Defines the TitleScorer model (Qwen + LoRA)
    └── train.py                      # Training and evaluation script for ranking model
```

---

## 🧪 Dataset Format

### For Ranking

```json
{
  "chosen": "<title>高点击标题</title>",
  "rejected": "<title>低点击标题</title>"
}
```

### For Classification

```json
{
  "messages": [{"role": "user", "content": "新闻文本"}],
  "label": 3
}
```

---

## 🛠️ Customization

To train with your own dataset:

- For classification: prepare JSON with `messages` and `label`.
- For ranking: prepare JSONL with `chosen` and `rejected` titles (wrapped in `<title>` tag).

---

## 📜 License

This project is released under the MIT License.

## 🙏 Acknowledgements

- [ModelScope](https://modelscope.cn/)
- [Hugging Face Transformers](https://huggingface.co/)
