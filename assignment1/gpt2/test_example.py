import torch
import random

from transformers import GPT2Tokenizer
from torch.optim import AdamW
from dataLoader import load_jsonl, prepare_data_for_model, split_data, NLIDataset
from transformers.adapters import AutoAdapterModel, AdapterConfig
from training import train_epoch, evaluate
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Extract arguments
    batch_size = 32
    learning_rate = 1e-05
    non_linearity = "swish"
    reduction_factor = 16
    num_epochs = 5
    max_length = 128

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

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = AutoAdapterModel.from_pretrained('gpt2')
    # Load adapter config
    config = AdapterConfig.load("houlsby", non_linearity=non_linearity, reduction_factor=reduction_factor)
    # Load adapter
    adapter_name = "multinli"
    model.load_adapter("nli/multinli@ukp", config=config, load_as=adapter_name)
    model.set_active_adapters(adapter_name)
    tokenizer.pad_token = tokenizer.eos_token
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

        # Check best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1}")

    model.load_state_dict(best_model_state)

    model.eval()
    index = 73
    with torch.no_grad():
        sentences = inputs_test[index:index + 1]
        label = labels_test[index:index + 1]
        print(f"Sentence1: {sentences[0][0]}\nSentence2: {sentences[0][1]}\nGround-Truth:\nRelationShip: {label[0]}")

        test_dataset = NLIDataset(sentences, labels, tokenizer, max_length)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for input_, label in test_loader:
            input_ = input_.to(device)
            label = label.to(device)
            output = model(input_)
            logit = output.logits
            _, predicted = torch.max(logit, dim=1)

            for label_str in test_dataset.label_map:
                if test_dataset.label_map[label_str] == predicted[0]:
                    print(f"Predicted:\nRelationShip: {label_str}")
                    break
