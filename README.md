# Qwen2.5-7B-Instruct 文本分类项目

本项目利用 `Qwen2.5-7B-Instruct` 模型进行文本分类任务。通过 `PyTorch` 和 `Transformers` 库，结合 `ModelScope` 平台提供的预训练模型，快速实现高效的文本分类。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 项目简介

本项目旨在提供一个简单易用的文本分类工具，基于 `Qwen2.5-7B-Instruct` 模型，适用于多种文本分类场景，如情感分析、主题分类、意图识别等。通过预训练模型的强大能力，用户可以快速实现高精度的文本分类任务。

## 功能特性

- **支持多种文本分类任务**：情感分析、主题分类、意图识别等。
- **基于强大的预训练模型**：使用 `Qwen2.5-7B-Instruct` 模型，提供高质量的文本分类能力。
- **简单易用**：通过几行代码即可完成模型加载和推理。
- **支持自定义训练**：用户可以根据自己的数据集对模型进行微调。

## 环境准备

在开始之前，请确保已安装以下依赖：

```bash
pip install torch
pip install transformers
pip install modelscope
```

## 模型下载

通过 `ModelScope` 下载 `Qwen2.5-7B-Instruct` 模型：

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct
```

## 快速开始

### 1. 克隆本项目

```bash
git clone https://github.com/your-username/qwen2.5-text-classification.git
cd qwen2.5-text-classification
```

### 2. 加载模型并进行文本分类

在 `main.py` 中，加载模型并对输入文本进行分类：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 输入文本
text = "这是一个示例文本，用于测试分类效果。"

# 分词并转换为模型输入
inputs = tokenizer(text, return_tensors="pt")

# 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 获取预测结果
predicted_class = torch.argmax(logits, dim=1).item()
print(f"预测类别: {predicted_class}")
```

### 3. 运行代码

```bash
python main.py
```

## 项目结构

```
qwen2.5-text-classification/
├── README.md               # 项目说明文档
├── main.py                 # 主程序，加载模型并进行文本分类
├── requirements.txt        # 项目依赖
├── data/                   # 存放训练或测试数据（可选）
└── LICENSE                 # 项目许可证
```

## 自定义训练

如果需要在自己的数据集上微调模型，可以参考以下步骤：

1. 准备数据集，确保数据格式为 `文本` 和 `标签`。
2. 使用 `Transformers` 的 `Trainer` API 进行微调。
3. 保存微调后的模型并加载使用。

示例代码：

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 替换为你的训练数据集
    eval_dataset=eval_dataset,    # 替换为你的验证数据集
)

trainer.train()
```

## 贡献指南

欢迎贡献代码或提出建议

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

- 感谢 [ModelScope](https://www.modelscope.cn/) 提供的 `Qwen2.5-7B-Instruct` 模型。
- 感谢 [Hugging Face](https://huggingface.co/) 提供的 `Transformers` 库。

---

如有任何问题，请提交 Issue 或联系项目维护者。希望本项目对您的文本分类任务有所帮助！ 🚀
