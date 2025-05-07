import torch
import torch.nn as nn
import torch.nn.functional as F

class TitleScorer(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.title_encoder = base_model
        hidden_size = base_model.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1, dtype=torch.bfloat16)
        )

    def forward(self, title_ids, title_mask):
        # Encoder 输出
        output = self.title_encoder(
            input_ids=title_ids,
            attention_mask=title_mask,
            output_hidden_states=True
        )
        last_hidden = output.hidden_states[-1]  # shape: (B, L, D)

        # 使用 mean pooling 作为 title 表征
        title_feat = torch.mean(last_hidden, dim=1)  # shape: (B, D)

        # MLP 输出一个 scalar 得分
        logits = self.mlp(title_feat)  # shape: (B, 1)
        return logits.view(-1)         # shape: (B,)