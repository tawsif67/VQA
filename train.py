from sklearn.metrics import accuracy_score
from config import *
from label_generation import id2label_dict, label2id_dict
from VQADataset import VQADataset

from transformers import ViltForQuestionAnswering
import torch
from torch.utils.data import DataLoader, random_split
from transformers import ViltProcessor



# Assuming you have already defined your 'df', 'processor', 'tokenizer', 'VQADataset', and other variables

# Load the model
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


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the model
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 num_labels=len(id2label_dict),
                                                 id2label=id2label_dict,
                                                 label2id=label2id_dict)
model.to(device)

# Set the optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Set the loss function
loss_fn = torch.nn.CrossEntropyLoss()
scaler = GradScaler()  # Initialize the GradScaler
# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Initialize the total loss for this epoch
    total_loss = 0

    # Iterate over the batches in the train_dataloader
    for batch in train_dataloader:
        # Move the batch to the device
        batch = {key: value.to(device) for key, value in batch.items()}

        # Clear out the gradients from the previous batch
        optimizer.zero_grad()

        With autocast():
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch['labels'])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        average_loss = total_loss / (train_dataloader_with_progress.n + 1)
        train_dataloader_with_progress.set_postfix({"Avg Loss": average_loss:.4f})

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

    # Calculate the average loss for this epoch
    average_loss = total_loss / len(train_dataloader)

    # Calculate accuracy on the train dataset
    model.eval()
    train_preds = []
    train_labels = []
    with torch.no_grad():
        for batch in train_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            train_preds.extend(predicted_labels.cpu().tolist())
            train_labels.extend(batch['labels'].cpu().tolist())
    train_accuracy = accuracy_score(train_labels, train_preds)

    # Calculate accuracy on the validation dataset
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            val_preds.extend(predicted_labels.cpu().tolist())
            val_labels.extend(batch['labels'].cpu().tolist())
    val_accuracy = accuracy_score(val_labels, val_preds)

    # Calculate accuracy on the test dataset
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            test_preds.extend(predicted_labels.cpu().tolist())
            test_labels.extend(batch['labels'].cpu().tolist())
    test_accuracy = accuracy_score(test_labels, test_preds)

    # Print metrics for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "saved_model.pth")