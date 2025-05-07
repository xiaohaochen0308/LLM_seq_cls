from torch.utils.data import Dataset, DataLoader
import torch
import json

class TitlePairDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(json_file)

    def load_data(self, json_file):
        with open(json_file, "r") as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen = item["chosen"]     # 用户偏好的标题
        rejected = item["rejected"] # 不偏好的标题

        chosen_enc = self.tokenizer(
            chosen,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


def get_dataloaders(tokenizer, train_path="train.json", test_path="test.json", batch_size=14):
    train_dataset = TitlePairDataset(json_file=train_path, tokenizer=tokenizer)
    test_dataset = TitlePairDataset(json_file=test_path, tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader