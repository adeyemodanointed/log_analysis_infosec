"""
Complete Librelog Parser Template
- Parses logs with Librelog
- Outputs standardized format directly
- Calculates performance metrics
- Optional accuracy evaluation

NOTE: Replace the actual Librelog parsing logic with your implementation
"""

import re
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path


class CompleteLibrelogParser:
    """All-in-one Librelog parser with metrics and evaluation"""

    def __init__(self, ground_truth_file=None):
        """
        Initialize parser

        Args:
            ground_truth_file: Optional ground truth for accuracy
        """
        print("🚀 Initializing Complete Librelog Parser...")

        # Initialize your Librelog parser here
        # self.librelog_parser = YourLibrelogParser()

        # Statistics
        self.total_logs = 0
        self.templates = {}
        self.log_mapping = []
        self.start_time = None
        self.end_time = None

        # Ground truth (optional)
        self.ground_truth = None
        self.has_ground_truth = False
        if ground_truth_file:
            self.load_ground_truth(ground_truth_file)

        print("✅ Parser initialized successfully!")

    def load_ground_truth(self, ground_truth_file):
        """Load ground truth labels"""
        print(f"📂 Loading ground truth from: {ground_truth_file}")

        try:
            if ground_truth_file.endswith('.csv'):
                self.ground_truth = pd.read_csv(ground_truth_file)
            elif ground_truth_file.endswith('.json'):
                with open(ground_truth_file, 'r') as f:
                    data = json.load(f)
                    self.ground_truth = pd.DataFrame(data if isinstance(data, list) else data.get('logs', []))

            # Find label column
            label_col = None
            for col in ['Label', 'label', 'is_anomaly', 'Anomaly']:
                if col in self.ground_truth.columns:
                    label_col = col
                    break

            if label_col:
                self.ground_truth['is_anomaly'] = self.ground_truth[label_col].apply(
                    lambda x: 1 if str(x).lower().strip() in ['anomaly', '1', 'true', 'alert'] else 0
                )
                self.has_ground_truth = True

                total = len(self.ground_truth)
                anomalies = self.ground_truth['is_anomaly'].sum()
                print(f"✅ Loaded {total} labels ({anomalies} anomalies, {total - anomalies} normal)")
            else:
                print(f"⚠️  Warning: No label column found")

        except Exception as e:
            print(f"⚠️  Warning: Could not load ground truth: {e}")

    def extract_block_id(self, log_line):
        """Extract HDFS block ID if present"""
        match = re.search(r'blk_-?\d+', log_line)
        return match.group(0) if match else None

    def parse_log(self, log_line, line_id):
        """
        Parse a single log with Librelog

        REPLACE THIS with your actual Librelog parsing logic!
        """
        # Extract block ID
        block_id = self.extract_block_id(log_line)

        # TODO: Replace with your Librelog parsing
        # Example structure (replace with actual Librelog output):
        # result = self.librelog_parser.parse(log_line)
        # template_id = result['template_id']
        # template = result['template']

        # PLACEHOLDER - Replace with actual Librelog logic
        template_id = f"group_{line_id % 10}"  # This is just a placeholder!
        template = log_line  # Replace with actual template extraction

        # Update statistics
        self.total_logs += 1

        if template_id not in self.templates:
            self.templates[template_id] = {
                "template": template,
                "count": 0,
                "log_ids": [],
                "block_ids": set()
            }

        self.templates[template_id]["count"] += 1
        self.templates[template_id]["log_ids"].append(line_id)

        if block_id:
            self.templates[template_id]["block_ids"].add(block_id)

        # Get ground truth label if available
        is_anomaly = None
        if self.has_ground_truth and line_id <= len(self.ground_truth):
            is_anomaly = self.ground_truth.iloc[line_id - 1]['is_anomaly']

        # Store mapping
        self.log_mapping.append({
            "log_id": line_id,
            "template_id": template_id,
            "template": template,
            "original_log": log_line,
            "block_id": block_id,
            "cluster_size": None,  # Will be filled later
            "is_anomaly": int(is_anomaly) if is_anomaly is not None else None
        })

        return {
            "line_id": line_id,
            "template_id": template_id,
            "template": template,
            "block_id": block_id
        }

    def parse_file(self, input_file):
        """Parse log file with timing"""
        print(f"\n📂 Parsing logs from: {input_file}")
        print("⏱️  Starting timer...\n")

        self.start_time = time.time()
        results = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    result = self.parse_log(line, line_id=i)
                    results.append(result)

                    # Progress indicator
                    if i % 1000 == 0:
                        elapsed = time.time() - self.start_time
                        rate = i / elapsed
                        print(f"  Processed {i:,} logs... ({rate:.0f} logs/sec)")

            self.end_time = time.time()

            # Update cluster sizes
            for log in self.log_mapping:
                template_id = log['template_id']
                log['cluster_size'] = self.templates[template_id]['count']

            print(f"\n✅ Parsing completed!")
            print(f"   Total logs: {self.total_logs:,}")
            print(f"   Time taken: {self.end_time - self.start_time:.2f} seconds")
            print(f"   Speed: {self.total_logs / (self.end_time - self.start_time):.0f} logs/sec")

        except FileNotFoundError:
            print(f"❌ Error: File not found - {input_file}")
            return []
        except Exception as e:
            print(f"❌ Error parsing file: {e}")
            import traceback
            traceback.print_exc()
            return []

        return results

    def calculate_metrics(self):
        """Calculate parsing and compression metrics"""
        parsing_time = self.end_time - self.start_time if self.end_time else 0

        metrics = {
            "parsing_performance": {
                "total_logs": self.total_logs,
                "unique_templates": len(self.templates),
                "parsing_time_seconds": round(parsing_time, 3),
                "logs_per_second": round(self.total_logs / parsing_time, 2) if parsing_time > 0 else 0,
                "avg_time_per_log_ms": round((parsing_time * 1000) / self.total_logs, 4) if self.total_logs > 0 else 0
            },
            "compression_metrics": {
                "compression_ratio": round(self.total_logs / len(self.templates), 2) if len(self.templates) > 0 else 0,
                "template_coverage": round((len(self.templates) / self.total_logs) * 100,
                                           2) if self.total_logs > 0 else 0
            }
        }

        return metrics

    def print_statistics(self):
        """Print detailed statistics"""
        print("\n" + "=" * 70)
        print("📊 PARSING STATISTICS")
        print("=" * 70)

        metrics = self.calculate_metrics()

        print("\n🚀 Performance:")
        print(f"   Total logs processed: {metrics['parsing_performance']['total_logs']:,}")
        print(f"   Unique templates: {metrics['parsing_performance']['unique_templates']:,}")
        print(f"   Parsing time: {metrics['parsing_performance']['parsing_time_seconds']:.2f} seconds")
        print(f"   Speed: {metrics['parsing_performance']['logs_per_second']:.0f} logs/sec")
        print(f"   Avg per log: {metrics['parsing_performance']['avg_time_per_log_ms']:.4f} ms")

        print("\n📦 Compression:")
        print(f"   Compression ratio: {metrics['compression_metrics']['compression_ratio']:.2f}x")
        print(f"   Template coverage: {metrics['compression_metrics']['template_coverage']:.2f}%")

        # Top templates
        sorted_templates = sorted(
            self.templates.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )

        print("\n🔝 Top 10 Most Frequent Templates:")
        print("-" * 70)
        for i, (template_id, data) in enumerate(sorted_templates[:10], 1):
            print(f"\n{i}. Template ID: {template_id}")
            print(f"   Pattern: {data['template'][:100]}...")  # Truncate if long
            print(f"   Count: {data['count']:,} ({data['count'] / self.total_logs * 100:.1f}%)")

    def export_standardized(self, output_file):
        """Export in standardized format"""
        print(f"\n💾 Exporting standardized output to: {output_file}")

        metrics = self.calculate_metrics()

        standardized_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": "librelog",
                "total_logs": self.total_logs,
                "unique_templates": len(self.templates),
                "parsing_time_seconds": metrics['parsing_performance']['parsing_time_seconds'],
                "logs_per_second": metrics['parsing_performance']['logs_per_second'],
                "compression_ratio": metrics['compression_metrics']['compression_ratio'],
                "has_ground_truth": self.has_ground_truth
            },
            "logs": self.log_mapping,
            "metrics": metrics
        }

        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(standardized_data, f, indent=2)
            print("✅ Standardized output exported successfully!")
            return output_file
        except Exception as e:
            print(f"❌ Error exporting: {e}")
            return None


def main():
    """Main function"""
    print("""
    ╔════════════════════════════════════════════════╗
    ║   COMPLETE LIBRELOG PARSER                     ║
    ║   Parsing + Metrics + Standardized Output      ║
    ╚════════════════════════════════════════════════╝
    """)

    # Configuration
    DATASET = 'HDFS_v1'  # Change this: 'HDFS_2k', 'Apache', 'BGL', etc.
    FILENAME = 'HDFS'
    INPUT_FILE = f'data/{DATASET}/{FILENAME}.log'
    GROUND_TRUTH = f'data/{DATASET}/preprocessed/anomaly_label.csv'  # Optional, set to None if not available
    OUTPUT_DIR = 'output'

    print(f"📋 Configuration:")
    print(f"   Dataset: {DATASET}")
    print(f"   Input: {INPUT_FILE}")
    print(f"   Ground truth: {GROUND_TRUTH if GROUND_TRUTH else 'None'}")
    print()

    # Initialize parser
    parser = CompleteLibrelogParser(ground_truth_file=GROUND_TRUTH)

    # Parse logs
    results = parser.parse_file(INPUT_FILE)

    if not results:
        print("❌ Parsing failed. Exiting.")
        return

    # Print statistics
    parser.print_statistics()

    # Export standardized format
    standardized_file = f'{OUTPUT_DIR}/{DATASET}_librelog_standardized.json'
    parser.export_standardized(standardized_file)

    print("\n" + "=" * 70)
    print("🎉 PARSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print("\n📁 Output File:")
    print(f"   {standardized_file}")

    print("\n💡 Ready for comparison with Drain3!")

    print("\n📊 Quick Stats:")
    metrics = parser.calculate_metrics()
    print(f"   ⚡ Speed: {metrics['parsing_performance']['logs_per_second']:.0f} logs/sec")
    print(f"   📦 Compression: {metrics['compression_metrics']['compression_ratio']:.1f}x")
    print(f"   🎯 Templates: {metrics['parsing_performance']['unique_templates']}")


if __name__ == "__main__":
    main()