"""
Complete Drain Log Parser
- Parses logs with Drain3
- Outputs standardized format directly
- Calculates performance metrics
- Optional accuracy evaluation with ground truth
"""

import re
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class CompleteDrainParser:
    """All-in-one Drain parser with metrics and evaluation"""

    def __init__(self, config_file="drain_config.ini", ground_truth_file=None):
        """
        Initialize parser

        Args:
            config_file: Drain3 configuration
            ground_truth_file: Optional ground truth for accuracy (CSV or JSON)
        """
        print("🚀 Initializing Complete Drain Log Parser...")

        # Load Drain3 configuration
        config = TemplateMinerConfig()
        config.load(config_file)
        self.template_miner = TemplateMiner(config=config)

        # Statistics
        self.total_logs = 0
        self.clusters = {}
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
        """Load ground truth labels for accuracy evaluation"""
        print(f"📂 Loading ground truth from: {ground_truth_file}")

        try:
            if ground_truth_file.endswith(".csv"):
                self.ground_truth = pd.read_csv(ground_truth_file)
            elif ground_truth_file.endswith(".json"):
                with open(ground_truth_file, "r") as f:
                    data = json.load(f)
                    self.ground_truth = pd.DataFrame(
                        data if isinstance(data, list) else data.get("logs", [])
                    )

            # Find label column
            label_col = None
            for col in ["Label", "label", "is_anomaly", "Anomaly"]:
                if col in self.ground_truth.columns:
                    label_col = col
                    break

            if label_col:
                # Convert to binary
                self.ground_truth["is_anomaly"] = self.ground_truth[label_col].apply(
                    lambda x: (
                        1
                        if str(x).lower().strip() in ["anomaly", "1", "true", "alert"]
                        else 0
                    )
                )
                self.has_ground_truth = True

                total = len(self.ground_truth)
                anomalies = self.ground_truth["is_anomaly"].sum()
                print(
                    f"✅ Loaded {total} labels ({anomalies} anomalies, {total - anomalies} normal)"
                )
            else:
                print(f"⚠️  Warning: No label column found in ground truth")

        except Exception as e:
            print(f"⚠️  Warning: Could not load ground truth: {e}")

    def extract_block_id(self, log_line):
        """Extract HDFS block ID if present"""
        match = re.search(r"blk_-?\d+", log_line)
        return match.group(0) if match else None

    def preprocess_log(self, log_line):
        """Preprocess log message"""
        # Remove timestamp
        log_line = re.sub(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", "", log_line)
        log_line = re.sub(r"^\d{6}\s+\d{6}", "", log_line)  # HDFS format

        # Remove process ID
        log_line = re.sub(r"^\s*\d+\s+", "", log_line)

        # Remove log level
        log_line = re.sub(
            r"\b(INFO|ERROR|WARN|DEBUG|TRACE|notice|error|info)\b",
            "",
            log_line,
            flags=re.IGNORECASE,
        )

        # Replace IP addresses
        log_line = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>", log_line)

        # Replace ports
        log_line = re.sub(r":(\d{4,5})\b", ":<PORT>", log_line)

        # Replace block IDs
        log_line = re.sub(r"blk_-?\d+", "<BLOCK>", log_line)

        # Replace large numbers
        log_line = re.sub(r"\b\d{5,}\b", "<NUM>", log_line)

        # Clean up spaces
        log_line = " ".join(log_line.split())

        return log_line.strip()

    def parse_log(self, log_line, line_id):
        """Parse a single log line"""
        # Extract block ID before preprocessing
        block_id = self.extract_block_id(log_line)

        # Preprocess
        cleaned_log = self.preprocess_log(log_line)

        # Parse with Drain
        result = self.template_miner.add_log_message(cleaned_log)

        # Update statistics
        self.total_logs += 1
        cluster_id = result["cluster_id"]

        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = {
                "template": result["template_mined"],
                "count": 0,
                "log_ids": [],
                "block_ids": set(),
            }

        self.clusters[cluster_id]["count"] += 1
        self.clusters[cluster_id]["log_ids"].append(line_id)

        if block_id:
            self.clusters[cluster_id]["block_ids"].add(block_id)

        # Get ground truth label if available
        is_anomaly = None
        if self.has_ground_truth and line_id <= len(self.ground_truth):
            is_anomaly = self.ground_truth.iloc[line_id - 1]["is_anomaly"]

        # Store mapping
        self.log_mapping.append(
            {
                "log_id": line_id,
                "template_id": str(cluster_id),
                "template": result["template_mined"],
                "original_log": log_line,
                "block_id": block_id,
                "cluster_size": None,  # Will be filled later
                "is_anomaly": int(is_anomaly) if is_anomaly is not None else None,
            }
        )

        return {
            "line_id": line_id,
            "cluster_id": cluster_id,
            "template": result["template_mined"],
            "block_id": block_id,
        }

    def parse_file(self, input_file):
        """Parse log file with timing"""
        print(f"\n📂 Parsing logs from: {input_file}")
        print("⏱️  Starting timer...\n")

        self.start_time = time.time()
        results = []

        try:
            with open(input_file, "r", encoding="utf-8") as f:
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

            # Update cluster sizes in log_mapping
            for log in self.log_mapping:
                cluster_id = int(log["template_id"])
                log["cluster_size"] = self.clusters[cluster_id]["count"]

            print(f"\n✅ Parsing completed!")
            print(f"   Total logs: {self.total_logs:,}")
            print(f"   Time taken: {self.end_time - self.start_time:.2f} seconds")
            print(
                f"   Speed: {self.total_logs / (self.end_time - self.start_time):.0f} logs/sec"
            )

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
                "unique_templates": len(self.clusters),
                "parsing_time_seconds": round(parsing_time, 3),
                "logs_per_second": (
                    round(self.total_logs / parsing_time, 2) if parsing_time > 0 else 0
                ),
                "avg_time_per_log_ms": (
                    round((parsing_time * 1000) / self.total_logs, 4)
                    if self.total_logs > 0
                    else 0
                ),
            },
            "compression_metrics": {
                "compression_ratio": (
                    round(self.total_logs / len(self.clusters), 2)
                    if len(self.clusters) > 0
                    else 0
                ),
                "template_coverage": (
                    round((len(self.clusters) / self.total_logs) * 100, 2)
                    if self.total_logs > 0
                    else 0
                ),
            },
        }

        return metrics

    def print_statistics(self):
        """Print detailed statistics"""
        print("\n" + "=" * 70)
        print("PARSING STATISTICS")
        print("=" * 70)

        metrics = self.calculate_metrics()

        print("\n🚀 Performance:")
        print(
            f"   Total logs processed: {metrics['parsing_performance']['total_logs']:,}"
        )
        print(
            f"   Unique templates: {metrics['parsing_performance']['unique_templates']:,}"
        )
        print(
            f"   Parsing time: {metrics['parsing_performance']['parsing_time_seconds']:.2f} seconds"
        )
        print(
            f"   Speed: {metrics['parsing_performance']['logs_per_second']:.0f} logs/sec"
        )
        print(
            f"   Avg per log: {metrics['parsing_performance']['avg_time_per_log_ms']:.4f} ms"
        )

        print("\n📦 Compression:")
        print(
            f"   Compression ratio: {metrics['compression_metrics']['compression_ratio']:.2f}x"
        )
        print(
            f"   Template coverage: {metrics['compression_metrics']['template_coverage']:.2f}%"
        )

        # Top templates
        sorted_clusters = sorted(
            self.clusters.items(), key=lambda x: x[1]["count"], reverse=True
        )

        print("\n🔝 Top 10 Most Frequent Templates:")
        print("-" * 70)
        for i, (cluster_id, data) in enumerate(sorted_clusters[:10], 1):
            print(f"\n{i}. Template ID: {cluster_id}")
            print(f"   Pattern: {data['template']}")
            print(
                f"   Count: {data['count']:,} ({data['count'] / self.total_logs * 100:.1f}%)"
            )

    def export_standardized(self, output_file):
        """Export in standardized format"""
        print(f"\n💾 Exporting standardized output to: {output_file}")

        metrics = self.calculate_metrics()

        standardized_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": "drain3",
                "total_logs": self.total_logs,
                "unique_templates": len(self.clusters),
                "parsing_time_seconds": metrics["parsing_performance"][
                    "parsing_time_seconds"
                ],
                "logs_per_second": metrics["parsing_performance"]["logs_per_second"],
                "compression_ratio": metrics["compression_metrics"][
                    "compression_ratio"
                ],
                "has_ground_truth": self.has_ground_truth,
            },
            "logs": self.log_mapping,
            "metrics": metrics,
        }

        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(standardized_data, f, indent=2)
            print("✅ Standardized output exported successfully!")
            return output_file
        except Exception as e:
            print(f"❌ Error exporting: {e}")
            return None

    def export_legacy_format(self, output_file):
        """Export in original cluster format (for compatibility)"""
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_logs": self.total_logs,
                "total_clusters": len(self.clusters),
            },
            "clusters": {},
        }

        for cluster_id, data in self.clusters.items():
            export_data["clusters"][str(cluster_id)] = {
                "template": data["template"],
                "count": data["count"],
                "log_ids": data["log_ids"],
            }

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Legacy format also saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save legacy format: {e}")


def main():
    """Main function"""
    print(
        """
    ╔════════════════════════════════════════════════╗
    ║   COMPLETE DRAIN LOG PARSER                    ║
    ║   Parsing + Metrics + Standardized Output      ║
    ╚════════════════════════════════════════════════╝
    """
    )

    # Configuration
    DATASET = "HDFS_v1"
    FILENAME = "HDFS"
    INPUT_FILE = f"data/{DATASET}/{FILENAME}.log"
    GROUND_TRUTH = f"data/{DATASET}/preprocessed/anomaly_label.csv"  # Optional, set to None if not available
    OUTPUT_DIR = "output"

    print(f"📋 Configuration:")
    print(f"   Dataset: {DATASET}")
    print(f"   Input: {INPUT_FILE}")
    print(
        f"   Ground truth: {GROUND_TRUTH if GROUND_TRUTH else 'None (no accuracy evaluation)'}"
    )
    print()

    # Initialize parser
    parser = CompleteDrainParser(
        config_file="drain_config.ini", ground_truth_file=GROUND_TRUTH
    )

    # Parse logs
    results = parser.parse_file(INPUT_FILE)

    if not results:
        print("❌ Parsing failed. Exiting.")
        return

    # Print statistics
    parser.print_statistics()

    # Export standardized format
    standardized_file = f"{OUTPUT_DIR}/{DATASET}_drain3_standardized.json"
    parser.export_standardized(standardized_file)

    # Also export legacy format
    legacy_file = f"{OUTPUT_DIR}/{DATASET}_drain3_legacy.json"
    parser.export_legacy_format(legacy_file)

    print("\n" + "=" * 70)
    print("🎉 PARSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Summary
    print("\n📁 Output Files:")
    print(f"   1. Standardized: {standardized_file}")
    print(f"   2. Legacy format: {legacy_file}")

    print("\n💡 Next Steps:")
    print("   1. Run Librelog parser on the same file")
    print("   2. Upload both standardized outputs to anomaly detection tool")
    print("   3. Export results and evaluate accuracy!")

    print("\n📊 Quick Stats:")
    metrics = parser.calculate_metrics()
    print(
        f"   ⚡ Speed: {metrics['parsing_performance']['logs_per_second']:.0f} logs/sec"
    )
    print(
        f"   📦 Compression: {metrics['compression_metrics']['compression_ratio']:.1f}x"
    )
    print(f"   🎯 Templates: {metrics['parsing_performance']['unique_templates']}")


if __name__ == "__main__":
    main()
