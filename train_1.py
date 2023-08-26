import time
import torch
from torch.optim.lr_scheduler import CyclicLR
import wandb

# Initialize the cyclic learning rate scheduler
base_lr = 0.001  # Set your desired base learning rate
max_lr = 0.01   # Set your desired maximum learning rate
step_size = len(train_dataloader) * 2  # Choose the step size for cycling
scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size)

num_epochs = 10  # Set the number of epochs

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    total_loss = 0
    num_data = 0
    num_batch = len(train_dataloader)
    t1 = time.time()

    for idx, batch in enumerate(train_dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()

        # Update cyclic learning rate
        scheduler.step()

        total_loss += loss.item()
        average_loss = total_loss / (idx+1)
        elapsed = int(time.time() - t1)
        eta = int(elapsed / (idx+1) * (num_batch - (idx + 1)))

        print(f"Epoch: {epoch + 1} Progress: [{idx + 1}/{num_batch}] Running Loss: {average_loss:.4f} Time: {elapsed}s ETA: {eta} s", end="\r")

        wandb.log({"Running Train Loss": average_loss, "Epoch": epoch})

    average_loss = total_loss / len(train_dataloader)

    # Validation and test evaluations
    for dataset_name, dataloader in [("Validation", val_dataloader), ("Test", test_dataloader)]:
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                _, predicted_labels = torch.max(outputs.logits, dim=1)
                preds.extend(predicted_labels.cpu().tolist())
                labels.extend(batch['labels'].cpu().tolist())
        accuracy_score = accuracy(labels, preds)  # Use 'micro' or 'weighted' as needed
        wandb.log({f'{dataset_name} Accuracy': accuracy_score, "Epoch": epoch})

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")
