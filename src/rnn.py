import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function to save results
def save_results(results, filename="results/metrics/rnn_results.json"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

# RNN model definition
class ProteinRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProteinRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Training and evaluation function
def train_and_evaluate_rnn(train_loader, test_loader, input_size, output_size, device):
    # Initialize model, loss, and optimizer
    model = ProteinRNN(input_size=input_size, hidden_size=2048, output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Adjust number of epochs as needed
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")

    # Evaluation loop
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU
            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())  # Move back to CPU for metrics
            all_targets.extend(y_batch.cpu().tolist())

    # Metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, output_dict=True)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)

    # Save results
    results = {
        "model": "RNN",
        "accuracy": accuracy,
        "classification_report": report
    }
    save_results(results)

    return model


if __name__ == "__main__":
    # Define amino acids for one-hot encoding
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    # Function to one-hot encode sequences
    def one_hot_encode(sequence, max_length=100):
        encoding = np.zeros((max_length, len(AMINO_ACIDS)))
        for i, aa in enumerate(sequence[:max_length]):
            if aa in AA_TO_INDEX:
                encoding[i, AA_TO_INDEX[aa]] = 1
        return encoding  # No need to flatten; keep (max_length, len(AMINO_ACIDS))

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = pd.read_csv("data/processed/sequences.csv")

    # Encode sequences into numerical format
    print("Encoding sequences...")
    data["encoded_sequence"] = data["sequence"].apply(lambda seq: one_hot_encode(seq))
    X = np.stack(data["encoded_sequence"].values)  # Shape: (num_samples, max_length, num_features)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["class"])

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=676, stratify=y
    )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders for batching
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

    # Train and evaluate the RNN model
    input_size = X_train.shape[2]  # Number of features (20 amino acids)
    output_size = len(np.unique(y))  # Number of classes
    train_and_evaluate_rnn(train_loader, test_loader, input_size, output_size, device)
