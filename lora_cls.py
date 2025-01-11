from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data import train_dataloader, test_dataloader  # 确保导入训练和测试数据加载器
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 新增

class ModelWithLoraClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ModelWithLoraClassifier, self).__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        
        # 多层感知机结构
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes, dtype=torch.bfloat16)
        )

    def forward(self, input_ids, attention_mask=None):
        # 获取基模型的输出
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 获取最后一层的隐藏状态
        
        # 对最后一层的所有 token 进行平均池化
        avg_hidden_state = torch.mean(hidden_states, dim=1)  # 在 token 维度上进行平均池化
        
        # 使用 MLP 进行分类
        logits = self.mlp(avg_hidden_state)
        return logits



# 定义评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            eval_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = eval_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# 定义训练和评估函数
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, writer, save_path="best_model.pth"):
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        # 每个 epoch 开始时打印当前参数数量
        total_params_epoch = sum(p.numel() for p in model.parameters())
        trainable_params_epoch = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nEpoch {epoch+1} - 总参数量: {total_params_epoch}, 可训练参数量: {trainable_params_epoch}")

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # 更新累计损失和准确率
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # 每10个批次打印一次损失和准确率，并记录到TensorBoard
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_samples
                print(f"Batch {batch_idx+1}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                current_batch = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Batch_Loss', avg_loss, current_batch)
                writer.add_scalar('Train/Batch_Accuracy', accuracy, current_batch)

        # 打印每个 epoch 的最终平均损失和准确率，并记录到TensorBoard
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Completed: Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        writer.add_scalar('Train/Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)

        # 评估测试集，并记录到TensorBoard
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)
            print("Best model saved.")

    writer.close()  # 关闭SummaryWriter
    return best_accuracy

# 主程序
if __name__ == "__main__":
    # 创建SummaryWriter实例
    writer = SummaryWriter('logs/')

    # 加载模型和 tokenizer
    model_name = "/root/.cache/modelscope/hub/Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        output_hidden_states=True  # 启用隐藏状态输出
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LoRA 配置
    lora_config = LoraConfig(
        peft_type="LORA",
        task_type="SEQ_CLS",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )

    # 应用 LoRA 到模型
    lora_model = get_peft_model(model, lora_config)

    # 定义分类器模型
    num_classes = 24
    model_with_classifier = ModelWithLoraClassifier(lora_model, num_classes)

    # 定义优化器和损失函数
    optimizer = AdamW(model_with_classifier.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_classifier.to(device)

    # 训练和评估
    num_epochs = 3
    best_accuracy = train_and_evaluate(
        model_with_classifier,
        train_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        device,
        num_epochs,
        writer,
        save_path="best_model.pth"
    )

    print(f"Best Test Accuracy: {best_accuracy:.4f}")