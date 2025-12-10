"""
librelog_parser_v1.py
Complete implementation of LibreLog for multi-source log parsing
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Add LibreLog parser directory to path
sys.path.append('LibreLog/parser')


try:
    from Librelog.parser.grouping import LogGrouping
    from Librelog.parser.llama_parser import LlamaParser
    from Librelog.parser.regex_manager import RegexManager
    LIBRELOG_AVAILABLE = True
except ImportError:
    print("⚠️  LibreLog modules not found. Make sure you're in the correct directory.")
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

class LibreLogParser:
    """
    Complete LibreLog parser implementation
    Handles multiple log formats
    """
    
    def __init__(self, 
                 model_path="models/Meta-Llama-3-8B-Instruct",
                 device="auto",
                 max_new_tokens=256,
                 temperature=0.1):
        """
        Initialize LibreLog parser
        
        Args:
            model_path: Path to Llama-3 model
            device: Device for inference ('cpu', 'cuda', or 'auto')
            max_new_tokens: Maximum tokens for LLM generation
            temperature: Temperature for LLM sampling
        """
        
        print("="*70)
        print("Initializing LibreLog Parser")
        print("="*70)
        
        self.model_path = model_path
        self.device = self._get_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Initialize components
        self.grouping = None
        self.llm_parser = None
        self.regex_manager = None
        
        # Statistics
        self.stats = {
            "total_logs": 0,
            "groups_created": 0,
            "llm_calls": 0,
            "total_time": 0.0,
            "templates": {}
        }
        
        # Initialize LibreLog components
        if LIBRELOG_AVAILABLE:
            self._initialize_components()
        else:
            print("❌ LibreLog not available")
    
    def _get_device(self, device_choice):
        """Determine device for inference"""
        if device_choice == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("ℹ️  Using CPU (consider GPU for faster processing)")
        else:
            device = device_choice
        
        return device
    
    def _initialize_components(self):
        """Initialize LibreLog components"""
        
        print("\n[1/3] Initializing Log Grouping...")
        try:
            self.grouping = LogGrouping(
                depth=4,  # Fixed-depth tree
                similarity_threshold=0.5
            )
            print("✅ Log Grouping initialized")
        except Exception as e:
            print(f"❌ Grouping initialization failed: {e}")
        
        print("\n[2/3] Loading Llama-3-8B Model...")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")
        print("This may take 2-5 minutes...")
        
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
            print("Tip: Make sure model is downloaded to:", self.model_path)
        
        print("\n[3/3] Initializing Regex Manager...")
        try:
            self.regex_manager = RegexManager()
            print("✅ Regex Manager initialized")
        except Exception as e:
            print(f"❌ Regex Manager initialization failed: {e}")
        
        print("\n" + "="*70)
        print("LibreLog Parser Ready!")
        print("="*70 + "\n")
    
    def parse_log(self, log: str, log_id: int = 0) -> ParsedLog:
        """
        Parse a single log message
        
        Args:
            log: Raw log message
            log_id: Unique log identifier
            
        Returns:
            ParsedLog object with template and metadata
        """
        
        start_time = time.time()
        
        if not self.grouping or not self.llm_parser:
            return ParsedLog(
                log_id=log_id,
                original_log=log,
                template=log,
                template_id="error",
                variables={},
                confidence=0.0,
                parsing_time=0.0
            )
        
        try:
            # Step 1: Add to grouping tree
            group_id = self.grouping.add_log(log)
            
            # Step 2: Get or generate template
            if group_id in self.stats["templates"]:
                # Template already exists
                template = self.stats["templates"][group_id]
                confidence = 0.95
            else:
                # Generate new template using LLM
                group = self.grouping.get_group(group_id)
                
                if len(group.logs) >= 3:
                    # Use LLM for template extraction
                    template = self.llm_parser.extract_template(group.logs)
                    self.stats["llm_calls"] += 1
                    self.stats["groups_created"] += 1
                else:
                    # Not enough samples, use first log as template
                    template = log
                
                # Store template
                self.stats["templates"][group_id] = template
                confidence = 0.85
            
            # Step 3: Extract variables using regex
            variables = self.regex_manager.extract_variables(log, template)
            
            # Update statistics
            self.stats["total_logs"] += 1
            parsing_time = time.time() - start_time
            self.stats["total_time"] += parsing_time
            
            return ParsedLog(
                log_id=log_id,
                original_log=log,
                template=template,
                template_id=f"group_{group_id}",
                variables=variables,
                confidence=confidence,
                parsing_time=parsing_time
            )
            
        except Exception as e:
            print(f"Error parsing log: {e}")
            return ParsedLog(
                log_id=log_id,
                original_log=log,
                template=log,
                template_id="error",
                variables={},
                confidence=0.0,
                parsing_time=time.time() - start_time
            )
    
    def parse_file(self, file_path: str, max_logs: int = None) -> List[ParsedLog]:
        """
        Parse logs from file
        
        Args:
            file_path: Path to log file
            max_logs: Maximum number of logs to parse (None = all)
            
        Returns:
            List of ParsedLog objects
        """
        
        print(f"\n{'='*70}")
        print(f"Parsing: {file_path}")
        print(f"{'='*70}")
        
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                logs = [line.strip() for line in f if line.strip()]
            
            if max_logs:
                logs = logs[:max_logs]
            
            print(f"Total logs to parse: {len(logs)}")
            print()
            
            for i, log in enumerate(logs, 1):
                result = self.parse_log(log, log_id=i)
                results.append(result)
                
                if i % 100 == 0:
                    print(f"Progress: {i}/{len(logs)} logs parsed")
            
            print(f"\n✅ Completed parsing {len(results)} logs")
            print(f"Total time: {self.stats['total_time']:.2f}s")
            print(f"Average time: {self.stats['total_time']/len(results)*1000:.2f}ms per log")
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return results
    
    def export_results(self, results: List[ParsedLog], output_file: str):
        """Export parsing results to JSON"""
        
        data = {
            "metadata": {
                "total_logs": len(results),
                "unique_templates": len(set(r.template for r in results)),
                "avg_parsing_time": sum(r.parsing_time for r in results) / len(results),
                "llm_calls": self.stats["llm_calls"]
            },
            "results": [asdict(r) for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results exported to: {output_file}")
    
    def print_summary(self, results: List[ParsedLog]):
        """Print parsing summary"""
        
        print(f"\n{'='*70}")
        print("PARSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Logs Parsed:    {len(results)}")
        print(f"Unique Templates:     {len(set(r.template for r in results))}")
        print(f"LLM Calls Made:       {self.stats['llm_calls']}")
        print(f"Groups Created:       {self.stats['groups_created']}")
        print(f"Avg Parsing Time:     {sum(r.parsing_time for r in results)/len(results)*1000:.2f}ms")
        print(f"Total Time:           {self.stats['total_time']:.2f}s")
        print()
        
        # Show template examples
        print("Sample Templates:")
        print("-" * 70)
        unique_templates = list(set(r.template for r in results))[:5]
        for i, template in enumerate(unique_templates, 1):
            print(f"{i}. {template[:65]}...")
        print(f"{'='*70}\n")


def main():
    """Main execution function"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     LibreLog Parser - Complete Implementation                    ║
    ║     Multi-Source Log Parsing with Llama-3-8B                     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize parser
    parser = LibreLogParser(
        model_path="models/Meta-Llama-3-8B-Instruct",
        device="auto"  # Use GPU if available
    )
    
    # Parse sample logs
    #log_file = "data/sample_logs/custom_logs.txt"
    log_file = "data/HDFS.log"
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("Please create the file with sample logs")
        return
    
    # Parse logs
    results = parser.parse_file(log_file)  # Parse first 50 logs
    
    # Print summary
    parser.print_summary(results)
    
    # Export results
    os.makedirs("output", exist_ok=True)
    parser.export_results(results, "output/HDFSlog_results.json")
    
    # Show sample results
    print("\nSample Parsed Logs:")
    print("="*70)
    for result in results[:3]:
        print(f"\nOriginal: {result.original_log[:60]}...")
        print(f"Template: {result.template[:60]}...")
        print(f"Variables: {result.variables}")
        print(f"Confidence: {result.confidence:.2f}")
    
    print("\n✅ Parsing completed successfully!")
    print(f"Check output/librelog_results.json for full results\n")


if __name__ == "__main__":
    main()