from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import json

from torch.utils.data import DataLoader



class TextClassificationDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        """
        初始化数据集
        :param json_file: 存储数据的 JSON 文件路径
        :param tokenizer: 用于文本 Tokenize 的 tokenizer
        :param max_length: 最大文本长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(json_file)

    def load_data(self, json_file):
        """
        从 JSON 文件加载数据
        :param json_file: JSON 文件路径
        :return: 加载的文本和标签列表
        """
        with open(json_file, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中的一条数据
        :param idx: 索引
        :return: 输入的 tokenized 数据和标签
        """
        text = self.data[idx]["messages"][0]["content"]  # 假设内容在 messages -> 0 -> content 中
        label = self.data[idx]["label"]

        # Tokenize 文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # 返回一个包含 tokenized 文本和标签的字典
        return {
            "input_ids": encoding["input_ids"].squeeze(),  # [batch_size, seq_len]
            "attention_mask": encoding["attention_mask"].squeeze(),  # [batch_size, seq_len]
            "labels": torch.tensor(label, dtype=torch.long)  # 标签
        }

# 加载模型和 tokenizer
model_name = "/root/.cache/modelscope/hub/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 假设 `train.json` 存储训练数据
train_dataset = TextClassificationDataset(json_file="train_trans_abs.json", tokenizer=tokenizer)
test_dataset = TextClassificationDataset(json_file="test_trans_abs.json", tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
