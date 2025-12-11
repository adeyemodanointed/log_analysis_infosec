#!/usr/bin/env python3
"""
LibreLog-Compatible Parser
Produces Drain3-compatible standardized output:
- identical metadata fields
- identical logs[] schema
- identical metrics structure
- template_id = "group_<id>"
- NO variables in standardized output
"""

import re
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path


# ============================================================
#  LIBRELOG INTERNAL-LIKE COMPONENTS
# ============================================================

class RegexPreprocessor:
    """LibreLog-style text normalizer"""
    def __init__(self):
        self.patterns = [
            (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
            (re.compile(r":\d{2,5}\b"), ":<PORT>"),
            (re.compile(r"blk_-?\d+"), "<BLOCK>"),
            (re.compile(r"\b\d+\b"), "<NUM>"),
        ]

    def normalize(self, text):
        for p, r in self.patterns:
            text = p.sub(r, text)
        return " ".join(text.split())


class LlamaTemplateGenerator:
    """
    LibreLog-style template generation.
    Real LibreLog uses Llama 3.1 70B; here we mimic behavior:
    - Collapse dynamic values into <*>
    - Preserve structure
    - Respect placeholders created by RegexPreprocessor
    """
    def generate_template(self, normalized_text):
        tokens = normalized_text.split()
        result = []
        for t in tokens:
            if t in ["<IP>", "<PORT>", "<NUM>", "<BLOCK>"]:
                result.append(t)
            elif re.search(r"\d", t):
                result.append("<*>")
            else:
                result.append(t)
        return " ".join(result)


class TemplateMatcher:
    """Maps templates to stable group IDs"""
    def __init__(self):
        self.template_to_id = {}
        self.id_to_template = {}
        self.next_id = 1

    def match_or_create(self, template):
        if template in self.template_to_id:
            return self.template_to_id[template]

        cid = self.next_id
        self.next_id += 1

        self.template_to_id[template] = cid
        self.id_to_template[cid] = template

        return cid


class ClusterManager:
    """Tracks cluster statistics"""
    def __init__(self):
        self.clusters = {}

    def add(self, cid, template, log_id, block_id):
        if cid not in self.clusters:
            self.clusters[cid] = {
                "template": template,
                "count": 0,
                "log_ids": [],
                "block_ids": set(),
            }

        self.clusters[cid]["count"] += 1
        self.clusters[cid]["log_ids"].append(log_id)

        if block_id:
            self.clusters[cid]["block_ids"].add(block_id)


# ============================================================
#  MAIN PARSER
# ============================================================

class CompleteLibreLogParser:

    def __init__(self, ground_truth_file=None):
        print("🚀 Initializing LibreLog Parser...")

        self.preprocessor = RegexPreprocessor()
        self.template_gen = LlamaTemplateGenerator()
        self.matcher = TemplateMatcher()
        self.clusters = ClusterManager()

        self.total_logs = 0
        self.log_mapping = []

        self.start_time = None
        self.end_time = None

        # ground truth
        self.ground_truth = None
        self.has_ground_truth = False
        if ground_truth_file:
            self.load_gt(ground_truth_file)

        print("✅ LibreLog Parser ready.\n")

    # --------------------------------------------------------

    def load_gt(self, file):
        print(f"📂 Loading ground truth: {file}")

        try:
            if file.endswith(".csv"):
                self.ground_truth = pd.read_csv(file)
            elif file.endswith(".json"):
                with open(file, "r") as f:
                    raw = json.load(f)
                self.ground_truth = pd.DataFrame(raw)

            if "Label" in self.ground_truth.columns:
                self.ground_truth["is_anomaly"] = self.ground_truth["Label"].apply(
                    lambda x: 1 if str(x).lower().strip() in ["1", "true", "anomaly"] else 0
                )
                self.has_ground_truth = True

            print("   ✔ Ground truth loaded.\n")

        except Exception as e:
            print(f"⚠️ Error loading ground truth: {e}\n")

    # --------------------------------------------------------

    def extract_block_id(self, line):
        m = re.search(r"blk_-?\d+", line)
        return m.group(0) if m else None

    # --------------------------------------------------------

    def parse_log(self, original, line_id):
        block_id = self.extract_block_id(original)
        normalized = self.preprocessor.normalize(original)
        template = self.template_gen.generate_template(normalized)
        cid = self.matcher.match_or_create(template)

        self.clusters.add(cid, template, line_id, block_id)
        self.total_logs += 1

        # ground truth lookup
        gt = None
        if self.has_ground_truth and line_id <= len(self.ground_truth):
            gt = int(self.ground_truth.iloc[line_id - 1]["is_anomaly"])

        # store for standardized output
        self.log_mapping.append({
            "log_id": line_id,
            "template_id": f"group_{cid}",
            "template": template,
            "original_log": original,
            "block_id": block_id,
            "cluster_size": None,  # filled after entire file parsed
            "is_anomaly": gt
        })

    # --------------------------------------------------------

    def parse_file(self, input_file):
        print(f"📄 Parsing logs from: {input_file}")

        self.start_time = time.time()

        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    self.parse_log(line, i)

                if i % 200000 == 0:
                    elapsed = time.time() - self.start_time
                    print(f"   ⏳ {i:,} logs processed ({i/elapsed:.0f} logs/sec)")

        self.end_time = time.time()

        # Fill cluster sizes
        for entry in self.log_mapping:
            cid = int(entry["template_id"].replace("group_", ""))
            entry["cluster_size"] = self.clusters.clusters[cid]["count"]

        print(f"\n✔ Completed parsing {self.total_logs:,} logs.\n")

    # --------------------------------------------------------

    def compute_metrics(self):
        t = self.end_time - self.start_time
        u = len(self.clusters.clusters)

        return {
            "parsing_performance": {
                "total_logs": self.total_logs,
                "unique_templates": u,
                "parsing_time_seconds": round(t, 3),
                "logs_per_second": round(self.total_logs / t, 2),
                "avg_time_per_log_ms": round((t * 1000) / self.total_logs, 4)
            },
            "compression_metrics": {
                "compression_ratio": round(self.total_logs / u, 2),
                "template_coverage": round((u / self.total_logs) * 100, 2)
            }
        }

    # --------------------------------------------------------

    def export_standardized(self, output_path):
        metrics = self.compute_metrics()

        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "method": "librelog",
                "total_logs": self.total_logs,
                "unique_templates": len(self.clusters.clusters),
                "parsing_time_seconds": metrics["parsing_performance"]["parsing_time_seconds"],
                "logs_per_second": metrics["parsing_performance"]["logs_per_second"],
                "compression_ratio": metrics["compression_metrics"]["compression_ratio"],
                "has_ground_truth": self.has_ground_truth
            },
            "logs": self.log_mapping,
            "metrics": metrics
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"💾 Standardized LibreLog output written: {output_path}")
        return output_path


# ============================================================
#  MAIN EXECUTION (RUN AS SCRIPT)
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║   LIBRELOG PARSER (DRAIN3-COMPATIBLE OUTPUT)         ║
╚══════════════════════════════════════════════════════╝
""")

    DATASET = "HDFS_v1"
    INPUT_FILE = f"data/{DATASET}/HDFS.log"
    GROUND_TRUTH = f"data/{DATASET}/preprocessed/anomaly_label.csv"
    OUTPUT_FILE = f"output/{DATASET}_librelog_standardized.json"

    print(f"📋 Dataset: {DATASET}")
    print(f"📄 Input file: {INPUT_FILE}")
    print(f"🔍 Ground truth: {GROUND_TRUTH}")
    print()

    parser = CompleteLibreLogParser(ground_truth_file=GROUND_TRUTH)
    parser.parse_file(INPUT_FILE)
    parser.export_standardized(OUTPUT_FILE)

    print("\n🎉 Done!\n")


if __name__ == "__main__":
    main()
