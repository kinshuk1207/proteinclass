import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Define amino acids for one-hot encoding
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Function to one-hot encode sequences
def one_hot_encode(sequence, max_length=100):
    encoding = np.zeros((max_length, len(AMINO_ACIDS)))
    for i, aa in enumerate(sequence[:max_length]):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1
    return encoding.flatten()  # Flatten for compatibility with ML models

# Load data
data = pd.read_csv("data/processed/sequences.csv")

# Encode sequences into numerical format
print("Encoding sequences...")
data["encoded_sequence"] = data["sequence"].apply(lambda seq: one_hot_encode(seq))

# Split into features and labels
X = np.stack(data["encoded_sequence"].values)  # Features (one-hot encoded sequences)
y = data["class"].values  # Labels (structural classes)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression model
print("Training the model...")
model = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="ovr")
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

