from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import json

class TextClassificationDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        """
        Initializes the dataset.
        :param json_file: Path to the JSON file containing the data.
        :param tokenizer: Tokenizer for text tokenization.
        :param max_length: Maximum length of the text sequences.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(json_file)

    def load_data(self, json_file):
        """
        Loads data from a JSON file.
        :param json_file: Path to the JSON file.
        :return: List of text and label entries.
        """
        with open(json_file, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        # Returns the total number of data entries.
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data entry.
        :param idx: Index of the data entry.
        :return: Dictionary containing tokenized input and label.
        """
        text = self.data[idx]["messages"][0]["content"]  # Assumes content is in messages -> 0 -> content
        label = self.data[idx]["label"]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Returns a dictionary containing tokenized input and label
        return {
            "input_ids": encoding["input_ids"].squeeze(),  # [batch_size, seq_len]
            "attention_mask": encoding["attention_mask"].squeeze(),  # [batch_size, seq_len]
            "labels": torch.tensor(label, dtype=torch.long)  # Label
        }

# Load model and tokenizer
model_name = "/root/.cache/modelscope/hub/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assuming `train.json` stores the training data
train_dataset = TextClassificationDataset(json_file="train_trans_abs.json", tokenizer=tokenizer)
test_dataset = TextClassificationDataset(json_file="test_trans_abs.json", tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
