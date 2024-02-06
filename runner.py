from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import datetime
from tqdm import tqdm


def run(hparams, model, name, criterion, optimizer, val_loader, train_loader, device):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    readable_run_name = f"{name}_{current_time}"
    writer = SummaryWriter(log_dir=f"runs/{readable_run_name}")

    writer.add_hparams(hparams, {})

    # function for model validation
    def validation(global_step):
        # Validation step
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total = 0
        with torch.no_grad():
            # Iterate over the validation data and update the average loss
            # but only use 10% of the batches choosen randomly
            number_of_batches = int(len(val_loader) * 0.1)
            for i, (inputs, labels) in enumerate(val_loader):
                if i > number_of_batches:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
            val_loss /= number_of_batches
            val_accuracy = val_corrects.float() / total
            writer.add_scalars(
                "loss", {"validation loss": val_loss}, global_step=global_step
            )
            writer.add_scalar(
                "validation accuracy", val_accuracy, global_step=global_step
            )

            return val_loss, val_accuracy
            # print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    # Train the model
    num_epochs = hparams["epochs"]  # Adjust this value as needed
    val_loss, val_acc = 0, 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (inputs, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}")
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                global_step = epoch * len(train_loader) + i
                if i % 10 == 9:  # Log every 10 mini-batches
                    writer.add_scalars(
                        "loss",
                        {"training loss": running_loss / 10},
                        global_step=global_step,
                    )
                    running_loss = 0.0

                # Validation every 50 mini-batches
                if i % 50 == 49:
                    val_loss, val_acc = validation(global_step)

                tepoch.set_postfix(
                    {"loss": loss.item(), "val_loss": val_loss, "val_acc": val_acc}
                )

    writer.close()
