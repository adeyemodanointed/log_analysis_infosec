import json
import argparse
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from deeplog import DeepLog
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="DeepLog anomaly detection on standardized logs")
parser.add_argument("logfile", type=str, help="Path to the standardized JSON log file")
args = parser.parse_args()
log_file_path = args.logfile

# -----------------------------
# 2. Load the log file
# -----------------------------
print("Importing log file...")
with open(log_file_path, "r") as f:
    data = json.load(f)
logs = data["logs"]
print(f"Loaded {len(logs)} log entries from {log_file_path}")

# -----------------------------
# 3. Group logs by block_id
# -----------------------------
sessions = {}
labels = {}
for log in logs:
    block = log["block_id"]
    if block not in sessions:
        sessions[block] = []
        labels[block] = []
    sessions[block].append(log["template_id"])
    # Replace None with 0 (assume unknown = normal)
    labels[block].append(0 if log["is_anomaly"] is None else log["is_anomaly"])

# -----------------------------
# 4. Prepare sequences and labels
# -----------------------------
sequences = [sessions[block] for block in sessions]
sequence_labels = [labels[block] for block in labels]

# -----------------------------
# 5. Encode template IDs to integers
# -----------------------------
all_templates = [t for seq in sequences for t in seq]
le = LabelEncoder()
le.fit(all_templates)
sequences_encoded = [le.transform(seq).tolist() for seq in sequences]

# -----------------------------
# 6. Split normal sequences into train/test
# -----------------------------
normal_sequences = [seq for seq, lbls in zip(sequences_encoded, sequence_labels) if all(l == 0 for l in lbls)]
normal_labels = [lbls for lbls in sequence_labels if all(l == 0 for l in lbls)]

train_sequences, test_normal_sequences, _, test_normal_labels = train_test_split(
    normal_sequences, normal_labels, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Prepare full test set (normal + anomalous)
# -----------------------------
anomalous_sequences = [seq for seq, lbls in zip(sequences_encoded, sequence_labels) if any(l == 1 for l in lbls)]
anomalous_labels = [lbls for lbls in sequence_labels if any(l == 1 for l in lbls)]

test_sequences = test_normal_sequences + anomalous_sequences
test_labels = test_normal_labels + anomalous_labels

# -----------------------------
# 8. Prepare training X and y (predict last template) with context window
# -----------------------------
context_len = 50  # last 50 templates per sequence


def prepare_XY_context(sequences, context_len):
    X_all = []
    y_all = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        seq_trunc = seq[-context_len:]
        X_all.append(torch.tensor(seq_trunc[:-1], dtype=torch.long))
        y_all.append(torch.tensor(seq_trunc[-1], dtype=torch.long))
    return X_all, y_all


X_train_list, y_train_list = prepare_XY_context(train_sequences, context_len)
X_train = pad_sequence(X_train_list, batch_first=True)
y_train = torch.stack(y_train_list)

# -----------------------------
# 9. Initialize DeepLog
# -----------------------------
num_templates = len(le.classes_)
model = DeepLog(
    input_size=num_templates,
    hidden_size=256,
    output_size=num_templates,
    num_layers=2
)

# -----------------------------
# 10. Train DeepLog
# -----------------------------
batch_size = 64  # smaller batch for speed
print("Training DeepLog on normal sequences...")
start_train = time.time()
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epoch=15,
    learning_rate=0.001
)
end_train = time.time()
print(f"Training completed in {end_train - start_train:.2f} seconds.")


# -----------------------------
# 11. Prepare test sequences as tensors
# -----------------------------
def prepare_test_tensors(sequences, context_len):
    tensors = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        seq_trunc = seq[-context_len:]
        tensors.append(torch.tensor(seq_trunc, dtype=torch.long))
    return tensors


test_sequences_tensors = prepare_test_tensors(test_sequences, context_len)

# -----------------------------
# 12. Predict anomalies with progress logging
# -----------------------------
print("Predicting anomalies and measuring inference time per block...")
predictions = []
block_times = []

for i, seq_tensor in enumerate(test_sequences_tensors, 1):
    start_block = time.time()

    seq_tensor_batch = seq_tensor.unsqueeze(0)  # add batch dim
    pred_seq, _ = model.predict(seq_tensor_batch, k=1)
    end_block = time.time()

    # Ensure integer predictions
    pred_seq = pred_seq.squeeze().tolist()
    if isinstance(pred_seq, int):
        pred_seq = [pred_seq]

    actual_last = seq_tensor[-1].item()
    anomaly_flag = 1 if pred_seq[0] != actual_last else 0
    predictions.append([int(anomaly_flag)] * len(seq_tensor))

    block_times.append(end_block - start_block)

    if i % 1000 == 0 or i == len(test_sequences_tensors):
        print(f"Processed {i}/{len(test_sequences_tensors)} blocks ({i / len(test_sequences_tensors) * 100:.2f}%)")

# -----------------------------
# 13. Flatten for evaluation and ensure integers
# -----------------------------
y_true = []
y_pred = []

for lbls, pred in zip(test_labels, predictions):
    lbls_clean = [0 if l is None else int(l) for l in lbls[1:]]
    pred_clean = [int(p) for p in pred[1:len(lbls_clean) + 1]]
    y_true.extend(lbls_clean)
    y_pred.extend(pred_clean)

y_true = [0 if v == 0 else 1 for v in y_true]
y_pred = [0 if v == 0 else 1 for v in y_pred]

# -----------------------------
# 14. Compute metrics
# -----------------------------
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# -----------------------------
# 15. Inference time statistics
# -----------------------------
avg_block_time = np.mean(block_times)
max_block_time = np.max(block_times)
min_block_time = np.min(block_times)

print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print(f"Inference time per block - Avg: {avg_block_time:.4f}s, Min: {min_block_time:.4f}s, Max: {max_block_time:.4f}s")
