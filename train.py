import time
from tqdm import tqdm
from config import *
from sklearn.metrics import accuracy_score as accuracy
from label_generation import id2label_dict, label2id_dict
from VQADataset import VQADataset

from transformers import ViltForQuestionAnswering
import torch
from torch.utils.data import DataLoader, random_split
from transformers import ViltProcessor
import wandb


wandb.init(project="VQA", settings=wandb.Settings(start_method='fork'))
wandb.run.name= model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 num_labels=len(id2label_dict),
                                                 id2label=id2label_dict,
                                                 label2id=label2id_dict)
model.to(device)

# Load the processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# Create the dataset
dataset = VQADataset(df=df,
                     processor=processor, tokenizer=tokenizer)

# Split the dataset
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = (dataset_size - train_size) // 2
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], 
                                                        generator=torch.Generator().manual_seed(SEED))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Set the optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Set the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Initialize the total loss for this epoch
    total_loss = 0
    num_data = 0
    num_batch = len(train_dataloader)
    t1 = time.time()
    train_preds = []
    train_labels = []
    # Iterate over the batches in the train_dataloader
    for idx, batch in enumerate(train_dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}
        # Clear out the gradients from the previous batch
        optimizer.zero_grad()
        # Forward pass
        outputs = model(**batch)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        _, actual_labels = torch.max(batch['labels'], dim=1)
        train_preds.extend(predicted_labels.cpu().tolist())
        train_labels.extend(actual_labels.cpu().tolist())
        # Compute the loss
        loss = loss_fn(outputs.logits, batch['labels'])
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
        total_loss += loss.item()
        average_loss = total_loss / (idx+1)
        elapsed = int(time.time() - t1)
        eta = int(elapsed / (idx+1) * (num_batch-(idx+1)))
        print(f"Train Epoch: {epoch+1} Progress: [{idx+1}/{num_batch}] Running Loss: {average_loss:.4f} Time: {elapsed}s ETA: {eta} s", end="\r")
        wandb.log({"Running Train Loss": average_loss, "Epoch": epoch})
    # Calculate the average loss for this epoch
    average_loss = total_loss / len(train_dataloader)

    model.eval()
            
    train_acc = accuracy(train_labels, train_preds)  # Use 'micro' or 'weighted' as needed
    wandb.log({'Train Accuracy': train_acc, "Epoch":epoch})
    # Calculate Accuracy on the validation dataset
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            _, actual_labels = torch.max(batch['labels'], dim=1)
            val_preds.extend(predicted_labels.cpu().tolist())
            val_labels.extend(actual_labels.cpu().tolist())
    val_acc = accuracy(val_labels, val_preds, )  # Use 'micro' or 'weighted' as needed
    wandb.log({'Validation Accuracy': val_acc, "Epoch":epoch})


# Calculate Accuracy on the test dataset
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        _, actual_labels = torch.max(batch['labels'], dim=1)
        test_preds.extend(predicted_labels.cpu().tolist())
        test_labels.extend(actual_labels.cpu().tolist())
test_acc = accuracy(test_labels, test_preds)  # Use 'micro' or 'weighted' as needed
wandb.log({'Test Accuracy': test_acc})

# Print metrics for this epoch
print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")
print(f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Save the model
torch.save(model.state_dict(), "saved_model.pth")