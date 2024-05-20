import torch
from tqdm import tqdm


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(data_loader, leave=True)

    for input_ids, labels in loop:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids)
        logits = outputs.logits
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()

    return total_loss / len(data_loader)


def test_accuracy(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            _, predicted = torch.max(logits, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy
