import argparse
import sys
import os
import csv
import pickle
import time
import numpy as np
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from detection.detect import avg_block_time

try:
    import ijson
except ImportError:
    print("Error: pip install ijson")
    sys.exit(1)

# --- Configuration ---
CONFIG = {
    "limit": 3_000_000,  # 1M records is easy for Isolation Forest
    "contamination": 0.05,  # Estimate: ~3% of data is anomalous (Tune this!)
    "cache_file": "iforest_features.pkl"
}


def load_labels(csv_path):
    print(f"[*] Loading Labels from {csv_path}...")
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 2: continue
            # 1 = Anomaly, 0 = Normal
            labels[row[0]] = 1 if row[1].strip() == "Anomaly" else 0
    return labels


def extract_features(json_path, label_path):
    if os.path.exists(CONFIG["cache_file"]):
        print(f"[*] Loading cached features from {CONFIG['cache_file']}...")
        with open(CONFIG["cache_file"], 'rb') as f:
            return pickle.load(f)

    labels_map = load_labels(label_path)
    print(f"[*] Streaming logs from {json_path}...")

    # 1. First Pass: Identify all unique templates (Vocabulary)
    #    and group logs by Block ID
    sessions = defaultdict(lambda: defaultdict(int))
    unique_templates = set()

    with open(json_path, 'rb') as f:
        parser = ijson.items(f, 'logs.item')
        for i, log in enumerate(parser):
            if i >= CONFIG["limit"]: break
            if i % 100000 == 0: print(f"    Processed {i} logs...", end='\r')

            bid = log.get("block_id")
            tid = log.get("template_id")

            # Only process if we have a label for this block
            if bid and bid in labels_map:
                sessions[bid][tid] += 1
                unique_templates.add(tid)

    print(f"\n[*] Vectorizing {len(sessions)} blocks...")

    # 2. Convert to Matrix (Blocks x Templates)
    # Sort templates to ensure consistent column order
    sorted_templates = sorted(list(unique_templates))
    template_to_idx = {t: i for i, t in enumerate(sorted_templates)}

    n_samples = len(sessions)
    n_features = len(sorted_templates)

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    block_ids = list(sessions.keys())

    for row_idx, bid in enumerate(block_ids):
        # Fill the row with counts
        for tid, count in sessions[bid].items():
            col_idx = template_to_idx[tid]
            X[row_idx, col_idx] = count

        y[row_idx] = labels_map[bid]

    print(f"[*] Feature Matrix Shape: {X.shape}")

    with open(CONFIG["cache_file"], 'wb') as f:
        pickle.dump((X, y), f)

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("csv_file")
    args = parser.parse_args()

    # 1. Get Data
    X, y_true = extract_features(args.json_file, args.csv_file)

    # 2. Split (Standard Machine Learning Split)
    # We train on a mix of Normal+Anomaly (unsupervised) OR just Normal
    # Isolation Forest works best if we train on "Mostly Normal" data.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

    print(f"[*] Training Isolation Forest on {len(X_train)} blocks...")

    # 3. Model
    # contamination: The expected proportion of outliers in the dataset
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=CONFIG["contamination"],
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    iso_forest.fit(X_train)

    # 4. Predict
    print("[*] Predicting...")
    # Returns -1 for anomaly, 1 for normal
    pred_start = time.time()

    preds_raw = iso_forest.predict(X_test)

    pred_end = time.time()
    inference_time = pred_end - pred_start

    # Calculate latency
    num_samples = X_test.shape[0]
    avg_block_time_per_block = (inference_time / num_samples) * 1000

    # Convert to our format: 1=Anomaly, 0=Normal
    y_pred = [1 if x == -1 else 0 for x in preds_raw]

    print("\n" + "=" * 40)
    print("PERFORMANCE METRICS")
    print("=" * 40)
    print(f"Total Inference Time: {i}")
    # 5. Evaluate
    print("\n" + "=" * 40)
    print("      ISOLATION FOREST RESULTS")
    print("=" * 40)
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    print("=" * 40)

    import joblib
    print("[*] Saving the Champion Model...")
    joblib.dump(iso_forest, "drain_hdfs_model.pkl")
    print("[*] Model saved to drain_hdfs_model.pkl")


if __name__ == "__main__":
    main()