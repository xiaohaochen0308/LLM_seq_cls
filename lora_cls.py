from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data import train_dataloader, test_dataloader  # Ensure the training and testing dataloaders are imported
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # Added

class ModelWithLoraClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ModelWithLoraClassifier, self).__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        
        # Multi-layer Perceptron structure
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
        # Get outputs from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states
        
        # Perform average pooling on the last layer for all tokens
        avg_hidden_state = torch.mean(hidden_states, dim=1)  # Average pooling over the token dimension
        
        # Use MLP for classification
        logits = self.mlp(avg_hidden_state)
        return logits

# Define evaluation function
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

# Define training and evaluation function
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, writer, save_path="best_model.pth"):
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Print the total number of parameters at the start of each epoch
        total_params_epoch = sum(p.numel() for p in model.parameters())
        trainable_params_epoch = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nEpoch {epoch+1} - Total parameters: {total_params_epoch}, Trainable parameters: {trainable_params_epoch}")

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Update cumulative loss and accuracy
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Print loss and accuracy every 10 batches and log them to TensorBoard
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_samples
                print(f"Batch {batch_idx+1}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                current_batch = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Batch_Loss', avg_loss, current_batch)
                writer.add_scalar('Train/Batch_Accuracy', accuracy, current_batch)

        # Print the final average loss and accuracy for each epoch and log them to TensorBoard
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Completed: Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        writer.add_scalar('Train/Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)

        # Evaluate on the test set and log the results to TensorBoard
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)
            print("Best model saved.")

    writer.close()  # Close SummaryWriter
    return best_accuracy

# Main program
if __name__ == "__main__":
    # Create SummaryWriter instance
    writer = SummaryWriter('logs/')

    # Load model and tokenizer
    model_name = "/root/.cache/modelscope/hub/Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        output_hidden_states=True  # Enable output of hidden states
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LoRA configuration
    lora_config = LoraConfig(
        peft_type="LORA",
        task_type="SEQ_CLS",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)

    # Define classifier model
    num_classes = 24
    model_with_classifier = ModelWithLoraClassifier(lora_model, num_classes)

    # Define optimizer and loss function
    optimizer = AdamW(model_with_classifier.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_classifier.to(device)

    # Train and evaluate
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
