import time
from tqdm import tqdm
from config import *
from label_generation import id2label_dict, label2id_dict
from VQADataset import VQADataset

import torch
from torch.utils.data import DataLoader, random_split
from transformers import ViltProcessor
import wandb
import torch.nn as nn
import copy
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViltForQuestionAnswering

wandb.init(project="VQA", settings=wandb.Settings(start_method='fork'))
wandb.run.name= model_name 

# Initialize the best validation accuracy
best_val_acc = 0.0
best_model_state_dict = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 num_labels=len(id2label_dict),
                                                 id2label=id2label_dict,
                                                 label2id=label2id_dict)

# Move the model to the desired device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# Load the processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# Create the dataset
dataset = VQADataset(df=df, processor=processor, tokenizer=tokenizer)

# Split the dataset
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = (dataset_size - train_size) // 2
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], 
                                                        generator=torch.Generator().manual_seed(42))

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
    num_batch = len(train_dataloader)
    t1 = time.time()

    # Iterate over the batches in the train_dataloader
    for idx, batch in enumerate(train_dataloader):
        # Move the batch to the device
        batch = {key: value.to(device) for key, value in batch.items()}

        # Clear out the gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**batch)

        # Compute the loss
        loss = loss_fn(outputs.logits, batch['labels'])

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Accumulate the total loss for this epoch
        total_loss += loss.item()
        average_loss = total_loss / (idx+1)

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (idx+1) * (num_batch-(idx+1)))
        print(f"Epoch: {epoch+1} Progress: [{idx+1}/{num_batch}] Running Loss: {average_loss:.4f} Time: {elapsed}s ETA: {eta} s", end="\r")
        wandb.log({"Running Train Loss": average_loss, "Epoch": epoch})

    # Calculate the average loss for this epoch
    average_loss = total_loss / len(train_dataloader)
    # Calculate training accuracy
    train_labels = []
    train_preds = []

    with torch.no_grad():
        for batch in train_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            
            # Extract the true labels (convert one-hot to class indices)
            batch_labels = torch.argmax(batch['labels'], dim=1).cpu().numpy()
            train_labels.extend(batch_labels)
            
            # Extract the predicted labels
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            train_preds.extend(predicted_labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_acc:.4f}, Average Loss: {average_loss:.4f}")

    # Calculate validation accuracy
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            
            # Extract the true labels (convert one-hot to class indices)
            batch_labels = torch.argmax(batch['labels'], dim=1).cpu().numpy()
            val_labels.extend(batch_labels)
            
            # Extract the predicted labels
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            val_preds.extend(predicted_labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(val_labels, val_preds)

    # Plot confusion matrix and log to WandB
    classes = list(id2label_dict.values())
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Validation Set")
    
    # Save the plot to a file
    plt.savefig("confusion_matrix.png")
    
    # Log the confusion matrix plot to WandB
    wandb.log({"Confusion Matrix": wandb.Image("confusion_matrix.png")})

    # Close the plot to avoid displaying it in the console
    plt.close()

# Calculate test accuracy after the training loop
test_labels = []
test_preds = []

with torch.no_grad():
    for batch in test_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        
        # Extract the true labels (convert one-hot to class indices)
        batch_labels = torch.argmax(batch['labels'], dim=1).cpu().numpy()
        test_labels.extend(batch_labels)
        
        # Extract the predicted labels
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        test_preds.extend(predicted_labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
wandb.log({'Test Accuracy': test_acc})

if val_acc > best_val_acc:
    best_val_acc = val_acc
    best_model_state_dict = copy.deepcopy(model.state_dict())  # Save the best model

# Save the best model
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, "best_model.pth")

# Print metrics after the training loop
print(f"Test Accuracy: {test_acc:.4f}")
