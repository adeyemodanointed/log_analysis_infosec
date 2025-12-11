"""
Complete Librelog Parser with Standardized Output & Metrics
Integrates your existing LibreLog implementation
"""

import os
import sys
import json
import time
import re
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict

# Add LibreLog parser directory to path
sys.path.append('LibreLog/parser')

try:
    from LibreLog.parser.grouping import LogParser
    from LibreLog.parser.llama_parser import LlamaParser
    from LibreLog.parser.regex_manager import RegexManager

    LIBRELOG_AVAILABLE = True
except ImportError as e:
    print("⚠️  LibreLog modules not found. Make sure LibreLog/ is in current directory.", e)
    LIBRELOG_AVAILABLE = False


@dataclass
class ParsedLog:
    """Structured parsed log result"""
    log_id: int
    original_log: str
    template: str
    template_id: str
    variables: Dict
    confidence: float
    parsing_time: float


class CompleteLibreLogParser:
    """All-in-one LibreLog parser with metrics and standardized output"""

    def __init__(self,
                 model_path="models/Meta-Llama-3-8B-Instruct",
                 device="auto",
                 max_new_tokens=256,
                 temperature=0.1,
                 ground_truth_file=None):
        """
        Initialize parser

        Args:
            model_path: Path to Llama-3 model
            device: 'cpu', 'cuda', or 'auto'
            ground_truth_file: Optional ground truth for accuracy
        """

        print("🚀 Initializing Complete LibreLog Parser...")
        print("=" * 70)

        self.model_path = model_path
        self.device = self._get_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Initialize LibreLog components
        self.grouping = None
        self.llm_parser = None
        self.regex_manager = None

        # Statistics
        self.total_logs = 0
        self.templates = {}
        self.log_mapping = []
        self.start_time = None
        self.end_time = None
        self.llm_calls = 0
        self.groups_created = 0

        # Ground truth
        self.ground_truth = None
        self.has_ground_truth = False
        if ground_truth_file:
            self.load_ground_truth(ground_truth_file)

        # Initialize components
        if LIBRELOG_AVAILABLE:
            self._initialize_components()
        else:
            print("❌ LibreLog not available. Install requirements first.")

        print("✅ Parser initialized successfully!\n")

    def _get_device(self, device_choice):
        """Determine device for inference"""
        if device_choice == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                print("✅ Apple Silicon GPU (MPS) available")
            else:
                device = "cpu"
                print("ℹ️  Using CPU (consider GPU for faster processing)")
        else:
            device = device_choice

        return device

    def _initialize_components(self):
        """Initialize LibreLog components"""

        print("[1/3] Initializing Log Grouping...")
        try:
            self.grouping = LogParser(
                depth=4,
                st=0.5
            )
            print("✅ Log Grouping initialized")
        except Exception as e:
            print(f"❌ Grouping failed: {e}")

        print("\n[2/3] Loading Llama-3-8B Model...")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")
        print("⏱️  This may take 2-5 minutes...\n")

        try:
            self.llm_parser = LlamaParser(
                model_path=self.model_path,
                device_map=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            print("✅ Llama-3 Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print(f"Tip: Ensure model is at: {self.model_path}")

        print("\n[3/3] Initializing Regex Manager...")
        try:
            self.regex_manager = RegexManager()
            print("✅ Regex Manager initialized")
        except Exception as e:
            print(f"❌ Regex Manager failed: {e}")

        print()

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
                print(f"✅ Loaded {total} labels ({anomalies} anomalies)")
            else:
                print(f"⚠️  Warning: No label column found")

        except Exception as e:
            print(f"⚠️  Could not load ground truth: {e}")

    def extract_block_id(self, log_line):
        """Extract HDFS block ID if present"""
        match = re.search(r'blk_-?\d+', log_line)
        return match.group(0) if match else None

    def parse_log(self, log_line, line_id):
        """Parse a single log with LibreLog"""

        start_time = time.time()
        block_id = self.extract_block_id(log_line)

        if not self.grouping or not self.llm_parser:
            # Fallback if components not initialized
            return self._fallback_parse(log_line, line_id, block_id, start_time)

        try:
            # Step 1: Add to grouping tree
            group_id = self.grouping.add_log(log_line)
            template_id = f"group_{group_id}"

            # Step 2: Get or generate template
            if group_id in self.templates:
                template = self.templates[group_id]['template']
                confidence = 0.95
            else:
                # Generate new template using LLM
                group = self.grouping.get_group(group_id)

                if len(group.logs) >= 3:
                    # Use LLM for template extraction
                    template = self.llm_parser.extract_template(group.logs)
                    self.llm_calls += 1
                    self.groups_created += 1
                    confidence = 0.85
                else:
                    # Not enough samples, use log as template
                    template = log_line
                    confidence = 0.70

                # Store template
                self.templates[group_id] = {
                    'template': template,
                    'count': 0,
                    'log_ids': []
                }

            # Update template stats
            self.templates[group_id]['count'] += 1
            self.templates[group_id]['log_ids'].append(line_id)

            # Step 3: Extract variables
            variables = self.regex_manager.extract_variables(log_line, template) if self.regex_manager else {}

            # Get ground truth if available
            is_anomaly = None
            if self.has_ground_truth and line_id <= len(self.ground_truth):
                is_anomaly = self.ground_truth.iloc[line_id - 1]['is_anomaly']

            parsing_time = time.time() - start_time
            self.total_logs += 1

            # Store mapping
            self.log_mapping.append({
                "log_id": line_id,
                "template_id": template_id,
                "template": template,
                "original_log": log_line,
                "block_id": block_id,
                "cluster_size": None,  # Will be filled later
                "confidence": confidence,
                "variables": variables,
                "is_anomaly": int(is_anomaly) if is_anomaly is not None else None
            })

            return {
                "line_id": line_id,
                "template_id": template_id,
                "template": template,
                "block_id": block_id
            }

        except Exception as e:
            print(f"⚠️  Error parsing log {line_id}: {e}")
            return self._fallback_parse(log_line, line_id, block_id, start_time)

    def _fallback_parse(self, log_line, line_id, block_id, start_time):
        """Fallback parsing when LLM unavailable"""
        template_id = "group_fallback"

        if template_id not in self.templates:
            self.templates[template_id] = {'template': log_line, 'count': 0, 'log_ids': []}

        self.templates[template_id]['count'] += 1
        self.templates[template_id]['log_ids'].append(line_id)

        self.log_mapping.append({
            "log_id": line_id,
            "template_id": template_id,
            "template": log_line,
            "original_log": log_line,
            "block_id": block_id,
            "cluster_size": 1,
            "confidence": 0.0,
            "variables": {},
            "is_anomaly": None
        })

        return {"line_id": line_id, "template_id": template_id, "template": log_line, "block_id": block_id}

    def parse_file(self, input_file):
        """Parse log file with timing"""
        print(f"\n📂 Parsing logs from: {input_file}")
        print("⏱️  Starting timer...\n")

        self.start_time = time.time()
        results = []

        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                logs = [line.strip() for line in f if line.strip()]

            print(f"Total logs to process: {len(logs):,}\n")

            for i, log_line in enumerate(logs, 1):
                result = self.parse_log(log_line, line_id=i)
                results.append(result)

                if i % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = i / elapsed
                    print(f"  Processed {i:,}/{len(logs):,} logs... ({rate:.0f} logs/sec)")

            self.end_time = time.time()

            # Update cluster sizes
            for log in self.log_mapping:
                template_id = log['template_id']
                group_id = int(template_id.split('_')[1]) if '_' in template_id else 0
                if group_id in self.templates:
                    log['cluster_size'] = self.templates[group_id]['count']

            print(f"\n✅ Parsing completed!")
            print(f"   Total logs: {self.total_logs:,}")
            print(f"   Time taken: {self.end_time - self.start_time:.2f} seconds")
            print(f"   Speed: {self.total_logs / (self.end_time - self.start_time):.0f} logs/sec")
            print(f"   LLM calls: {self.llm_calls}")

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
                "avg_time_per_log_ms": round((parsing_time * 1000) / self.total_logs, 4) if self.total_logs > 0 else 0,
                "llm_calls": self.llm_calls,
                "groups_created": self.groups_created
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
        print(f"   LLM calls made: {metrics['parsing_performance']['llm_calls']}")

        print("\n📦 Compression:")
        print(f"   Compression ratio: {metrics['compression_metrics']['compression_ratio']:.2f}x")
        print(f"   Template coverage: {metrics['compression_metrics']['template_coverage']:.2f}%")

        # Top templates
        sorted_templates = sorted(
            self.templates.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        print("\n🔝 Top 10 Most Frequent Templates:")
        print("-" * 70)
        for i, (group_id, data) in enumerate(sorted_templates[:10], 1):
            print(f"\n{i}. Template ID: group_{group_id}")
            print(f"   Pattern: {data['template'][:80]}...")
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
                "llm_calls": self.llm_calls,
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
    DATASET = 'HDFS_v1'
    FILENAME= "HDFS"
    INPUT_FILE = f"data/{DATASET}/{FILENAME}.log"
    GROUND_TRUTH = f"data/{DATASET}/preprocessed/anomaly_label.csv"
    OUTPUT_DIR = 'output'
    MODEL_PATH = 'models/Meta-Llama-3-8B-Instruct'

    print(f"📋 Configuration:")
    print(f"   Dataset: {DATASET}")
    print(f"   Input: {INPUT_FILE}")
    print(f"   Ground truth: {GROUND_TRUTH if os.path.exists(GROUND_TRUTH) else 'None'}")
    print(f"   Model: {MODEL_PATH}")
    print()

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("Please download Llama-3-8B-Instruct model first.")
        print("See README for instructions.")
        return

    # Initialize parser
    parser = CompleteLibreLogParser(
        model_path=MODEL_PATH,
        device="auto",
        ground_truth_file=GROUND_TRUTH if os.path.exists(GROUND_TRUTH) else None
    )

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
    print(f"   🤖 LLM calls: {metrics['parsing_performance']['llm_calls']}")


if __name__ == "__main__":
    main()