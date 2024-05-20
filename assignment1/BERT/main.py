import torch
import json
import logging
import argparse
import json
import datetime

from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataLoader import load_jsonl, prepare_data_for_model, split_data, NLIDataset
from training import train_epoch, evaluate, test_accuracy
from transformers import BertTokenizer, BertForSequenceClassification


if __name__ == '__main__':
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()

    # Extract arguments
    batch_size = args.bs
    learning_rate = args.lr
    num_epochs = 5
    max_length = 512

    # check available device
    cuda_able = torch.cuda.is_available()
    mps_able = (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    device = 'cuda:0' if cuda_able else 'mps' if mps_able else 'cpu'
    # Load and prepare data
    filename = '../dataset/mismatched.jsonl'
    data = load_jsonl(filename)
    inputs, labels = prepare_data_for_model(data)

    # Spilt data into train, val, and test sets
    inputs_train, labels_train, inputs_val, labels_val, inputs_test, labels_test = split_data(
        inputs, labels, train_size=0.8
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    model.to(device)

    train_dataset = NLIDataset(inputs_train, labels_train, tokenizer, max_length)
    val_dataset = NLIDataset(inputs_val, labels_val, tokenizer, max_length)
    test_dataset = NLIDataset(inputs_test, labels_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_state = None
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")

        # Check best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1}")
            logging.info(f"New best model found at epoch {epoch + 1}")

    model.load_state_dict(best_model_state)

    test_acc = test_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.3f}")
    logging.info(f"Test Accuracy: {test_acc:.3f}")
    # Prepare results data
    results = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'best_val_loss': best_val_loss,
        'test_accuracy': test_acc
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = f'results/results_{timestamp}.json'

    # Write results to JSON file
    with open(results_filename, 'w') as f:
        json.dump(results, f)
