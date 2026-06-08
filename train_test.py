import os
import numpy as np
from sklearn.model_selection import train_test_split

# models
from src.gmm import train_gmm_models
from src.hmm import train_hmm_models

# prediction functions
from src.gmm import predict_digit_gmm
from src.hmm import predict_digit_hmm

import matplotlib.pyplot as plt
import seaborn as sns


# Load Data
from src.audio_feature_extraction import load_data

data_path = "recordings"   # change this if dataset is elsewhere
train_data, test_data = load_data(data_path)


# Train / Validation Split
train_split, val_split = train_test_split(train_data, test_size=0.2, random_state=42)


# Train Models
gmm_models = train_gmm_models(train_split)
hmm_models = train_hmm_models(train_split)

# Evaluation Function
def evaluate(data, models, predict_fn):
    correct = 0

    for mfcc, digit in data:
        pred = predict_fn(mfcc, models)
        if pred == digit:
            correct += 1

    return correct / len(data) if len(data) > 0 else 0


# Confusion Matrix
def confusion_matrix(data, models, predict_fn):
    cm = np.zeros((10, 10), dtype=int)

    for mfcc, digit in data:
        pred =int(predict_fn(mfcc, models))
        cm[digit][pred] += 1

    return cm


# Per-digit Accuracy
def per_digit_accuracy(cm):
    acc = []

    for i in range(10):
        correct = cm[i][i]
        total = np.sum(cm[i])
        if total ==0:
            acc.append(0)
        else:
            acc.append(correct / total)

    return acc


# Validation Accuracy
gmm_val = evaluate(val_split, gmm_models, predict_digit_gmm)
hmm_val = evaluate(val_split, hmm_models, predict_digit_hmm)

print("\nValidation Accuracy")
print("GMM:", gmm_val)
print("HMM:", hmm_val)


# Test Accuracy
gmm_test = evaluate(test_data, gmm_models, predict_digit_gmm)
hmm_test = evaluate(test_data, hmm_models, predict_digit_hmm)

print("\nTest Accuracy")
print("GMM:", gmm_test)
print("HMM:", hmm_test)


# Confusion Matrix
gmm_cm = confusion_matrix(test_data, gmm_models, predict_digit_gmm)
hmm_cm = confusion_matrix(test_data, hmm_models, predict_digit_hmm)

print("\nGMM Confusion Matrix:\n", gmm_cm)
print("\nHMM Confusion Matrix:\n", hmm_cm)


# Per-digit Accuracy
gmm_digit = per_digit_accuracy(gmm_cm)
hmm_digit = per_digit_accuracy(hmm_cm)

print("\nPer-digit Accuracy")
for i in range(10):
    print(f"Digit {i}: GMM={gmm_digit[i]:.2f}, HMM={hmm_digit[i]:.2f}")



os.makedirs("results", exist_ok=True)

# 1. Confusion Matrix plot
def plot_confusion(cm, title, filename):
    plt.figure(figsize=(9,7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        cbar=True
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

plot_confusion(gmm_cm, "GMM Confusion Matrix", "gmm_confusion.png")
plot_confusion(hmm_cm, "HMM Confusion Matrix", "hmm_confusion.png")


# 2. Accuracy Comparison Plot
models = ["GMM", "HMM"]
val_scores = [gmm_val, hmm_val]
test_scores = [gmm_test, hmm_test]

x = np.arange(len(models))

plt.figure(figsize=(6,4))

bars1 = plt.bar(x - 0.2, val_scores, width=0.4, label="Validation")
bars2 = plt.bar(x + 0.2, test_scores, width=0.4, label="Test")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Validation vs Test Accuracy")
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha='center')

plt.savefig("results/accuracy_comparison.png")
plt.close()


# 3. Per-digit Accuracy Plot

digits = np.arange(10)

plt.figure(figsize=(8,5))

plt.plot(digits, gmm_digit, marker='o', linewidth=2, label="GMM")
plt.plot(digits, hmm_digit, marker='s', linewidth=2, label="HMM")

for i in range(10):
    plt.text(digits[i], gmm_digit[i] + 0.03, f"{gmm_digit[i]:.2f}", ha='center', fontsize=9)
    plt.text(digits[i], hmm_digit[i] - 0.07, f"{hmm_digit[i]:.2f}", ha='center', fontsize=9)

plt.xlabel("Digit (0–9)")
plt.ylabel("Accuracy")
plt.title("Per-Digit Accuracy Comparison")
plt.xticks(digits)
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig("results/per_digit_line.png")
plt.close()


# DONE
print("\nAll final plots saved in 'results' folder ")