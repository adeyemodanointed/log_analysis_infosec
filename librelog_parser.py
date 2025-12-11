import sys
import os
import json
import re
import gc
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# --- 1. SETUP PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, 'parser'))


def resolve_path(relative_path):
    path = os.path.join(PROJECT_ROOT, relative_path)
    return os.path.normpath(path)


try:
    from parser import grouping
    from parser import llama_parser
    from parser import regex_manager
except ImportError as e:
    print(f"❌ Error importing LibreLog components: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
INPUT_FILE = resolve_path("data/HDFS_v1/HDFS.log")
OUTPUT_FILE = resolve_path("output/librelog_parsed.json")
MODEL_PATH = resolve_path("LibreLog/models/Meta-Llama-3-8B-Instruct")

BATCH_SIZE = 100000


class HDFSOrchestrator:
    def __init__(self):
        print("🚀 Initializing LibreLog (Optimized 4-bit Mode)...")

        # 1. Regex Manager
        print("   [1/3] Initializing Regex Manager...")
        self.regex_manager = regex_manager.RegexTemplateManager()

        # 2. Drain
        print("   [2/3] Initializing Drain...")
        hdfs_rex = [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"]
        self.drain_parser = grouping.LogParser(rex=hdfs_rex, depth=4, st=0.5)

        # 3. Model Pipeline
        print(f"   [3/3] Loading Model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ CRITICAL: Model folder not found at {MODEL_PATH}")
            sys.exit(1)

        try:
            print("       Loading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("       Loading Model (4-bit Quantization)...")
            # This Config forces the model to shrink to ~5GB
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_config,  # <--- THE FIX
                device_map="auto",
                low_cpu_mem_usage=True
            )

            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id
            )
            print("✅ Model Loaded in 4-bit Mode.")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Try running: pip install bitsandbytes accelerate")
            sys.exit(1)

        self.llm_parser = llama_parser.LogParser(self.pipeline, self.regex_manager)

    def process_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"❌ Input file not found: {filepath}")
            return

        print(f"📄 Streaming logs from: {filepath}")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding="utf-8") as f_out:
            f_out.write('{\n  "metadata": {"method": "LibreLog_Hybrid_4bit"},\n  "logs": [\n')

        total_processed = 0
        batch_buffer = []
        is_first_batch = True

        with open(filepath, 'r', encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line: continue
                batch_buffer.append(line)

                if len(batch_buffer) >= BATCH_SIZE:
                    self.process_batch(batch_buffer, total_processed, is_first_batch)
                    total_processed += len(batch_buffer)
                    batch_buffer = []
                    is_first_batch = False
                    gc.collect()

            if batch_buffer:
                self.process_batch(batch_buffer, total_processed, is_first_batch)
                total_processed += len(batch_buffer)

        with open(OUTPUT_FILE, 'a', encoding="utf-8") as f_out:
            f_out.write('\n  ]\n}')
        print(f"\n✅ Done! Processed {total_processed} logs.")

    def process_batch(self, logs, start_idx, is_first_batch):
        print(f"   ⚙️ Processing batch {start_idx}-{start_idx + len(logs)}...", end='\r')
        groups = self.drain_parser.parse(logs)
        parsed_result = self.llm_parser.parse(groups, logs)

        if len(parsed_result) > 0 and parsed_result[0][0] != logs[0]:
            parsed_result = parsed_result[::-1]

        with open(OUTPUT_FILE, 'a', encoding="utf-8") as f_out:
            for i, row in enumerate(parsed_result):
                content = row[0]
                template = row[2] if len(row) > 2 else row[1]

                template_id = "E" + str(abs(hash(template)) % 10000000)
                blk_match = re.search(r"blk_-?\d+", content)
                block_id = blk_match.group(0) if blk_match else "unknown"

                log_obj = {
                    "log_id": start_idx + i + 1,
                    "template_id": template_id,
                    "block_id": block_id,
                    "template": template
                }

                prefix = ",\n"
                if is_first_batch and i == 0: prefix = ""
                f_out.write(prefix + "    " + json.dumps(log_obj))


if __name__ == "__main__":
    orchestrator = HDFSOrchestrator()
    orchestrator.process_file(INPUT_FILE)