import argparse
import json
import random
import sys
from collections import defaultdict
import numpy as np

# Try importing your specific DeepLog implementation
try:
    from deeplog import DeepLog
except ImportError:
    print("Error: Could not import 'DeepLog'. Make sure deeplog.py is in the same directory.")
    sys.exit(1)

# --- Configuration ---
# You can tweak these or add them as command line args if needed
CONFIG = {
    "train_ratio": 0.8,  # 80% of normal data for training
    "window_size": 10,  # Sequence context window
    "hidden_size": 64,  # LSTM hidden dimension
    "num_layers": 2,  # Number of LSTM layers
    "batch_size": 32,
    "epochs": 5
}


def load_and_preprocess(filepath):
    """
    Reads the JSON log file and converts text templates into integer sequences
    grouped by Block ID.
    """
    print(f"[*] Loading data from {filepath}...")

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{filepath}'. Check format.")
        sys.exit(1)

    logs = data.get("logs", [])
    if not logs:
        print("Error: JSON file contains no 'logs' field.")
        sys.exit(1)

    print(f"[*] Processing {len(logs)} logs...")

    # Mappings
    vocab = {}  # template_id -> integer index
    sessions = defaultdict(list)  # block_id -> list of integer indices
    block_labels = {}  # block_id -> 0 (Normal) or 1 (Anomaly)

    for log in logs:
        t_id = log.get("template_id", "unknown")
        blk_id = log.get("block_id", "unknown")
        is_anom = log.get("is_anomaly", 0)

        # 1. Build Vocab (Map string IDs to Ints)
        if t_id not in vocab:
            vocab[t_id] = len(vocab)

        # 2. Build Sequence
        sessions[blk_id].append(vocab[t_id])

        # 3. Label Block
        # If any log in a block is an anomaly, the whole block is treated as anomalous
        if blk_id not in block_labels:
            block_labels[blk_id] = 0

        if is_anom == 1:
            block_labels[blk_id] = 1

    print(f"[*] Found {len(sessions)} unique blocks (sessions).")
    print(f"[*] Vocab Size: {len(vocab)} unique templates.")

    return sessions, block_labels, vocab


def split_dataset(sessions, block_labels):
    """
    Splits data into:
    1. Train Set (Normal blocks only)
    2. Test Set Normal (Remaining Normal blocks)
    3. Test Set Anomaly (All Anomaly blocks)
    """
    normal_seqs = []
    anomaly_seqs = []

    for blk_id, seq in sessions.items():
        if block_labels[blk_id] == 1:
            anomaly_seqs.append(seq)
        else:
            normal_seqs.append(seq)

    # Randomize normal sequences before splitting
    random.shuffle(normal_seqs)

    # Split Normal Data
    split_idx = int(len(normal_seqs) * CONFIG["train_ratio"])

    train_data = normal_seqs[:split_idx]
    test_normal = normal_seqs[split_idx:]
    test_anomaly = anomaly_seqs

    print(f"[*] Data Split Summary:")
    print(f"    - Training Sequences (Normal): {len(train_data)}")
    print(f"    - Testing Sequences (Normal):  {len(test_normal)}")
    print(f"    - Testing Sequences (Anomaly): {len(test_anomaly)}")

    return train_data, test_normal, test_anomaly


def evaluate_model(model, sequences, label_name):
    """
    Runs the model on a set of sequences and counts anomalies detected.
    """
    detected_count = 0
    total = len(sequences)

    if total == 0:
        return 0.0

    print(f"[*] Evaluating {label_name} set ({total} sequences)...")

    for seq in sequences:
        # NOTE: Adjust 'detect' to match your DeepLog API (e.g., .predict, .evaluate)
        # We assume .detect() returns True if anomaly, False if normal
        if hasattr(model, 'detect'):
            is_anomaly = model.detect(seq)
        elif hasattr(model, 'predict'):
            # Some implementations return a probability or class, assume predict handles logic
            pred = model.predict(seq)
            is_anomaly = True if pred == 1 else False  # Adjust logic based on your library
        else:
            print("Error: Your DeepLog class does not have a detect() or predict() method.")
            sys.exit(1)

        if is_anomaly:
            detected_count += 1

    return detected_count


def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run DeepLog Anomaly Detection on JSON logs.")
    parser.add_argument("filename", help="Path to the JSON log file (e.g., log.json)")
    args = parser.parse_args()

    # 2. Prepare Data
    sessions, labels, vocab = load_and_preprocess(args.filename)
    train_seqs, test_normal_seqs, test_anomaly_seqs = split_dataset(sessions, labels)

    # 3. Initialize DeepLog
    # We pass the vocab size so the model knows the embedding dimension
    input_size = len(vocab)

    print(f"[*] Initializing DeepLog model with input_size={input_size}...")

    model = DeepLog(
        input_size=input_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=input_size
    )

    # 4. Train
    print("[*] Starting Training Phase...")
    # NOTE: Adjust .fit arguments based on your DeepLog library's definition
    model.fit(
        train_seqs,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"]
    )
    print("[*] Training Complete.")

    # 5. Evaluation
    print("\n--- Evaluation Results ---")

    # Check False Positives (Normal data detected as Anomaly)
    fp_count = evaluate_model(model, test_normal_seqs, "Normal Test")
    fp_rate = fp_count / len(test_normal_seqs) if len(test_normal_seqs) > 0 else 0

    # Check True Positives (Anomaly data detected as Anomaly)
    tp_count = evaluate_model(model, test_anomaly_seqs, "Anomaly Test")
    recall = tp_count / len(test_anomaly_seqs) if len(test_anomaly_seqs) > 0 else 0

    # Precision & F1 (Requires us to combine counts)
    # Precision = TP / (TP + FP)
    precision = 0
    if (tp_count + fp_count) > 0:
        precision = tp_count / (tp_count + fp_count)

    f1_score = 0
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print("-" * 30)
    print(f"False Positives: {fp_count} / {len(test_normal_seqs)}")
    print(f"True Positives:  {tp_count} / {len(test_anomaly_seqs)}")
    print("-" * 30)
    print(f"False Positive Rate: {fp_rate:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1_score:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()