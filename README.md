# LoRA Fine-Tuning for Qwen2.5-7B-Instruct Text Classification

This project focuses on fine-tuning the Qwen2.5-7B-Instruct model using Low-Rank Adaptation (LoRA) for text classification tasks. By integrating PyTorch and the Transformers library, along with pretrained models from the ModelScope platform, this approach enables efficient fine-tuning and enhances the performance of large language models (LLMs) in text classification scenarios.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Author
Chen Xiaohao

Email: xiaohaochen@cuc.edu.cn

## Project Overview

This project aims to provide an easy-to-use text classification tool based on the `Qwen2.5-7B-Instruct` model. It is suitable for various text classification scenarios, such as sentiment analysis, topic classification, and intent recognition. With the powerful capabilities of the pretrained model, users can quickly achieve high-accuracy text classification tasks.

## Features

- **Supports various text classification tasks**: Sentiment analysis, topic classification, intent recognition, and more.
- **Powered by a robust pretrained model**: Utilizes the `Qwen2.5-7B-Instruct` model to deliver high-quality text classification. Also supports different sizes of Qwen2.5 models.
- **Easy to use**: Load and infer the model with just a few lines of code.
- **Supports custom training**: Users can fine-tune the model on their datasets.

## Setup

Before starting, ensure the following dependencies are installed:

```bash
pip install torch
pip install transformers
pip install modelscope
pip install peft
```

## Model Download

Download the `Qwen2.5-7B-Instruct` model via `ModelScope`:

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct
```

## Quick Start

1. Run the following script:

```bash
python lora_cls.py
```

## Experimental Results

The experimental results on the **N24News Abstract text classification task** are as follows:

| Model                        | Accuracy (Acc) |
|------------------------------|----------------|
| **Ours (Qwen2.5-7B-Instruct)** | **85.26%**     |
| Bert                         | 78.3%          |
| RoBerta                      | 79.7%          |

The results demonstrate that the `Qwen2.5-7B-Instruct` model performs exceptionally well on text classification tasks, significantly outperforming traditional Bert and RoBerta models.

## Project Structure

```
Qwen_seq_cls/
├── README.md               
├── lora_cls.py                 
├── requirements.txt        
├── train_trans_abs.json
├── test_trans_abs.json          
└── data.py               
```

## Custom Training

To fine-tune the model on your dataset, follow these steps:

1. Prepare your dataset, ensuring it includes `text` and `label` fields.
2. Refer to the project for the provided data format:
```bash
{"messages": [{"role": "user", "content": "His new album features Cardi B, Justin Bieber, Chance the Rapper and countless other stars. But why?"}], "label": 22}
{"messages": [{"role": "user", "content": "An opinionated take on the songwriter's major works, from a delayed debut to a Pulitzer Prize-winning classic."}], "label": 18}
```

## Contribution Guide

Contributions and suggestions are welcome.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Thanks to [ModelScope](https://www.modelscope.cn/) for providing the `Qwen2.5-7B-Instruct` model.
- Thanks to [Hugging Face](https://huggingface.co/) for the `Transformers` library.

---

For any questions, please submit an issue or contact the project maintainers. We hope this project helps with your text classification tasks! 🚀
