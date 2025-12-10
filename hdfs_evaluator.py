"""
HDFS Anomaly Detection Accuracy Evaluator
Evaluates Drain3 vs Librelog on HDFS dataset with block-level labels
"""

import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
from pathlib import Path


class HDFSAnomalyEvaluator:
    """Evaluate anomaly detection on HDFS logs with block-level labels"""

    def __init__(self, anomaly_labels_file):
        """
        Initialize with HDFS block-level labels

        Args:
            anomaly_labels_file: Path to anomaly_label.csv
        """
        print(f"📂 Loading HDFS anomaly labels from: {anomaly_labels_file}")

        self.labels_df = pd.read_csv(anomaly_labels_file)

        # Convert to binary labels
        self.labels_df['is_anomaly'] = self.labels_df['Label'].apply(
            lambda x: 1 if str(x).lower().strip() == 'anomaly' else 0
        )

        # Create lookup dict
        self.block_labels = dict(zip(
            self.labels_df['BlockId'].astype(str),
            self.labels_df['is_anomaly']
        ))

        total = len(self.block_labels)
        anomalous = sum(self.block_labels.values())

        print(f"✅ Loaded {total} block labels")
        print(f"   - Normal blocks: {total - anomalous} ({(total - anomalous) / total * 100:.1f}%)")
        print(f"   - Anomalous blocks: {anomalous} ({anomalous / total * 100:.1f}%)")

    def load_parsed_results(self, parsed_file, method_name):
        """
        Load Drain3 or Librelog parsed results

        Args:
            parsed_file: Path to parsed output JSON
            method_name: "drain3" or "librelog"
        """
        print(f"\n📂 Loading {method_name} parsed results...")

        with open(parsed_file, 'r') as f:
            data = json.load(f)

        # Extract log_mapping with block IDs
        if 'log_mapping' in data:
            logs = data['log_mapping']
        elif 'logs' in data:
            logs = data['logs']
        else:
            raise ValueError(f"Cannot find log entries in {parsed_file}")

        print(f"✅ Loaded {len(logs)} parsed logs")

        return logs, data

    def evaluate_method(self, parsed_logs, detected_anomaly_templates, method_name):
        """
        Evaluate a single method against ground truth

        Args:
            parsed_logs: List of parsed log entries with block_id and cluster_id
            detected_anomaly_templates: Set of template/cluster IDs marked as anomalous
            method_name: "drain3" or "librelog"
        """
        print(f"\n🔍 Evaluating {method_name} against ground truth...")

        # Map each log to its prediction
        y_true = []
        y_pred = []
        processed_blocks = set()

        for log in parsed_logs:
            block_id = log.get('block_id')
            if not block_id or block_id in processed_blocks:
                continue

            # Get ground truth for this block
            true_label = self.block_labels.get(block_id)
            if true_label is None:
                continue  # Skip blocks not in ground truth

            # Determine prediction: if any log in this block matches anomalous template
            cluster_id = str(log.get('cluster_id') or log.get('template_id') or log.get('EventId'))
            predicted = 1 if cluster_id in detected_anomaly_templates else 0

            y_true.append(true_label)
            y_pred.append(predicted)
            processed_blocks.add(block_id)

        if len(y_true) == 0:
            print("❌ Error: No matching blocks found between parsed logs and ground truth")
            return None

        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            'method': method_name,
            'total_blocks': len(y_true),
            'true_anomalous_blocks': int(y_true.sum()),
            'predicted_anomalous_blocks': int(y_pred.sum()),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

        print(f"\n✅ {method_name.upper()} RESULTS:")
        print(f"   Blocks evaluated: {metrics['total_blocks']}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   TP: {tp:4d}  FP: {fp:4d}")
        print(f"   FN: {fn:4d}  TN: {tn:4d}")

        return metrics

    def compare_methods(self, drain_metrics, librelog_metrics, output_file='output/hdfs_comparison.json'):
        """Compare Drain3 vs Librelog"""
        print("\n" + "=" * 70)
        print("📊 COMPARATIVE ANALYSIS: DRAIN3 vs LIBRELOG on HDFS")
        print("=" * 70)

        comparison = {
            'dataset': 'HDFS',
            'drain3': drain_metrics,
            'librelog': librelog_metrics,
            'winner': {}
        }

        # Compare each metric
        print(f"\n{'Metric':<15} {'Drain3':<12} {'Librelog':<12} {'Winner':<12} {'Difference'}")
        print("-" * 70)

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            d_val = drain_metrics[metric]
            l_val = librelog_metrics[metric]

            if d_val > l_val:
                winner = 'drain3'
                diff = d_val - l_val
            elif l_val > d_val:
                winner = 'librelog'
                diff = l_val - d_val
            else:
                winner = 'tie'
                diff = 0

            comparison['winner'][metric] = {
                'method': winner,
                'difference': diff,
                'improvement_pct': (diff / max(d_val, l_val, 0.001) * 100)
            }

            winner_str = winner.upper() if winner != 'tie' else 'TIE'
            improvement = f"+{diff:.4f}" if winner != 'tie' else "-"

            print(f"{metric.capitalize():<15} {d_val:<12.4f} {l_val:<12.4f} {winner_str:<12} {improvement}")

        # Overall winner
        drain_wins = sum(1 for m in ['accuracy', 'precision', 'recall', 'f1']
                         if comparison['winner'][m]['method'] == 'drain3')
        librelog_wins = sum(1 for m in ['accuracy', 'precision', 'recall', 'f1']
                            if comparison['winner'][m]['method'] == 'librelog')

        print("\n" + "=" * 70)
        if drain_wins > librelog_wins:
            print(f"🏆 OVERALL WINNER: DRAIN3 ({drain_wins}/4 metrics)")
        elif librelog_wins > drain_wins:
            print(f"🏆 OVERALL WINNER: LIBRELOG ({librelog_wins}/4 metrics)")
        else:
            print(f"🤝 RESULT: TIE ({drain_wins}-{librelog_wins})")
        print("=" * 70)

        # Save report
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\n💾 Detailed comparison saved to: {output_file}")

        return comparison


def main():
    """Main evaluation workflow"""
    print("""
    ╔══════════════════════════════════════════════════╗
    ║   HDFS ANOMALY DETECTION EVALUATOR               ║
    ║   Drain3 vs Librelog Accuracy Comparison         ║
    ╚══════════════════════════════════════════════════╝
    """)

    try:
        # Load ground truth
        evaluator = HDFSAnomalyEvaluator('data/HDFS_v1/preprocessed/anomaly_label.csv')

        # Load parsed results
        drain_logs, drain_data = evaluator.load_parsed_results(
            'output/HDFS_v1_drain3_standardized.json',
            'drain3'
        )

        librelog_logs, librelog_data = evaluator.load_parsed_results(
            'output/HDFS_v1_librelog_standardized.json',
            'librelog'
        )

        # Load anomaly detection results from web tool
        print("\n📂 Loading anomaly detection results...")
        with open('output/anomaly-detection-results.json', 'r') as f:
            detection_results = json.load(f)

        # Extract detected anomaly template IDs
        drain_anomalies = set()
        for anomaly in detection_results.get('topAnomalies', {}).get('drain3', []):
            template_id = str(anomaly.get('EventId') or anomaly.get('template_id') or anomaly.get('cluster_id'))
            drain_anomalies.add(template_id)

        librelog_anomalies = set()
        for anomaly in detection_results.get('topAnomalies', {}).get('librelog', []):
            template_id = str(anomaly.get('EventId') or anomaly.get('template_id') or anomaly.get('cluster_id'))
            librelog_anomalies.add(template_id)

        print(f"   Drain3: {len(drain_anomalies)} anomalous templates")
        print(f"   Librelog: {len(librelog_anomalies)} anomalous templates")

        # Evaluate both methods
        drain_metrics = evaluator.evaluate_method(
            drain_logs,
            drain_anomalies,
            'drain3'
        )

        librelog_metrics = evaluator.evaluate_method(
            librelog_logs,
            librelog_anomalies,
            'librelog'
        )

        # Compare
        if drain_metrics and librelog_metrics:
            comparison = evaluator.compare_methods(
                drain_metrics,
                librelog_metrics,
                'output/hdfs_comparison.json'
            )

            print("\n✅ Evaluation complete!")
            print("\n📝 Results ready for your paper:")
            print("   - Precision: Accuracy of detected anomalies")
            print("   - Recall: Coverage of actual anomalies")
            print("   - F1 Score: Overall detection quality")
            print("   - Use these metrics to show which method preserves anomaly information better!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Required files:")
        print("   1. data/anomaly_label.csv (HDFS labels)")
        print("   2. output/HDFS_v1_drain3_standardized.json (Drain3 parsed output)")
        print("   3. output/HDFS_v1_librelog_standardized.json (Librelog parsed output)")
        print("   4. output/anomaly-detection-results.json (Detection results)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()