import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import TitleScorer
from data import get_dataloaders  #  确保返回字段为 chosen/rejected 对
from torch.utils.data import Subset

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, dataloader, device):
    set_seed(42)
    model.eval()
    all_rewards = []
    eval_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            score_chosen = model(chosen_ids, chosen_mask)
            score_rejected = model(rejected_ids, rejected_mask)

            loss = nn.functional.margin_ranking_loss(score_chosen, score_rejected, torch.ones_like(score_chosen))
            eval_loss += loss.item()

            rewards = (score_chosen > score_rejected).float()
            all_rewards.extend(rewards.cpu().numpy())

    accuracy = sum(all_rewards) / len(all_rewards)
    return eval_loss / len(dataloader), accuracy


def train():
    set_seed(42)
    writer = SummaryWriter('logs/')
    model_name = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"

    #  加载 base 模型和 LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = LoraConfig(
        peft_type="LORA",
        task_type="SEQ_CLS",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )
    lora_model = get_peft_model(base_model, lora_config)
    model = TitleScorer(lora_model).to("cuda")

    #  加载数据
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, "train_ctr.jsonl", "test_ctr.jsonl", batch_size=20)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 5
    best_acc = 0.0
    global_step = 0

    val_dataset = Subset(test_dataloader.dataset, range(min(100, len(test_dataloader.dataset))))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_dataloader.batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_rewards = []

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            global_step += 1

            chosen_ids = batch["chosen_input_ids"].cuda()
            chosen_mask = batch["chosen_attention_mask"].cuda()
            rejected_ids = batch["rejected_input_ids"].cuda()
            rejected_mask = batch["rejected_attention_mask"].cuda()

            score_chosen = model(chosen_ids, chosen_mask)
            score_rejected = model(rejected_ids, rejected_mask)

            loss = nn.functional.margin_ranking_loss(score_chosen, score_rejected, torch.ones_like(score_chosen))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            rewards = (score_chosen > score_rejected).float()
            all_rewards.extend(rewards.detach().cpu().numpy())

            if global_step % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                reward_acc = sum(all_rewards) / len(all_rewards)
                writer.add_scalar("Train/Loss", avg_loss, global_step)
                writer.add_scalar("Train/RewardAccuracy", reward_acc, global_step)
                print(f"Step {global_step} | Loss: {avg_loss:.4f} | RewardAcc: {reward_acc:.4f}")

        val_loss, val_acc = evaluate(model, test_dataloader, "cuda")
        writer.add_scalar("Val/RewardAccuracy", val_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        print(f"[Val] Epoch {epoch+1} | Loss: {val_loss:.4f} | RewardAcc: {val_acc:.4f}")

        torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")
        print(f" Saved model at epoch {epoch+1} → model_epoch{epoch+1}.pth")

        # 同时记录最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(" Best model updated")

    writer.close()
    print(f" Best Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train()