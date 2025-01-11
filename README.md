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
- **基于强大的预训练模型**：使用 `Qwen2.5-7B-Instruct` 模型，提供高质量的文本分类能力。此外，支持不同大小的Qwen2.5.
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

1. 运行代码

```bash
python lora_cls.py
```

## 项目结构

```
Qwen_seq_cls/
├── README.md               
├── lora_cls.py                 
├── requirements.txt        
├── train_trans_abs.json
├── test_trans_abs.json          
└── data.py               
```

## 自定义训练

如果需要在自己的数据集上微调模型，可以参考以下步骤：

1. 准备数据集，确保数据格式为 `文本` 和 `标签`。
2. 参考项目提供数据。


## 贡献指南

欢迎贡献代码或提出建议

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

- 感谢 [ModelScope](https://www.modelscope.cn/) 提供的 `Qwen2.5-7B-Instruct` 模型。
- 感谢 [Hugging Face](https://huggingface.co/) 提供的 `Transformers` 库。

---

如有任何问题，请提交 Issue 或联系项目维护者。希望本项目对您的文本分类任务有所帮助！ 🚀
