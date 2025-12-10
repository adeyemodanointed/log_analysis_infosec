"""
Drain Log Parser - Simple Implementation
Author: Your Name
Date: 2024-11-22
"""

import re
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import json
from datetime import datetime

class DrainLogParser:
    """Simple log parser using Drain3 algorithm"""
    
    def __init__(self, config_file='drain_config.ini'):
        """
        Initialize the Drain parser
        
        Args:
            config_file: Path to configuration file
        """
        print("🚀 Initializing Drain Log Parser...")
        
        # Load configuration
        config = TemplateMinerConfig()
        config.load(config_file)
        
        # Initialize template miner
        self.template_miner = TemplateMiner(config=config)
        
        # Statistics
        self.total_logs = 0
        self.clusters = {}
        
        print("✅ Parser initialized successfully!")
    
    def preprocess_log(self, log_line):
        """
        Preprocess log message by removing common patterns
        
        Args:
            log_line: Raw log line
            
        Returns:
            Cleaned log content
        """
        # Remove timestamp (assuming format: YYYY-MM-DD HH:MM:SS)
        log_line = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', log_line)
        
        # Remove log level (INFO, ERROR, WARN, DEBUG)
        log_line = re.sub(r'\b(INFO|ERROR|WARN|DEBUG|TRACE)\b', '', log_line)
        
        # Remove IP addresses (will be replaced with * in template)
        log_line = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', log_line)
        
        # Remove block IDs (HDFS specific)
        log_line = re.sub(r'blk_\d+', '<BLOCK_ID>', log_line)
        
        # Remove numbers that look like sizes/IDs
        log_line = re.sub(r'\b\d{5,}\b', '<NUM>', log_line)
        
        # Clean up extra spaces
        log_line = ' '.join(log_line.split())
        
        return log_line.strip()
    
    def parse_log(self, log_line):
        """
        Parse a single log line
        
        Args:
            log_line: Raw log line
            
        Returns:
            Parsing result dictionary
        """
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
                "log_ids": []
            }
        
        self.clusters[cluster_id]["count"] += 1
        self.clusters[cluster_id]["log_ids"].append(self.total_logs)
        
        return {
            "log_id": self.total_logs,
            "original": log_line,
            "cleaned": cleaned_log,
            "cluster_id": cluster_id,
            "template": result["template_mined"],
            "change_type": result["change_type"]
        }
    
    def parse_file(self, input_file):
        """
        Parse an entire log file
        
        Args:
            input_file: Path to input log file
            
        Returns:
            List of parsing results
        """
        print(f"\n📂 Reading logs from: {input_file}")
        
        results = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    result = self.parse_log(line)
                    results.append(result)
                    
                    # Progress indicator
                    if i % 1000 == 0:
                        print(f"  Processed {i} logs...")
            
            print(f"✅ Completed! Processed {self.total_logs} logs")
            
        except FileNotFoundError:
            print(f"❌ Error: File not found - {input_file}")
            return []
        except Exception as e:
            print(f"❌ Error parsing file: {e}")
            return []
        
        return results
    
    def print_statistics(self):
        """Print parsing statistics"""
        print("\n" + "="*60)
        print("📊 PARSING STATISTICS")
        print("="*60)
        print(f"Total logs processed: {self.total_logs}")
        print(f"Total clusters found: {len(self.clusters)}")
        print(f"Average logs per cluster: {self.total_logs / len(self.clusters):.2f}")
        print()
        
        # Sort clusters by count
        sorted_clusters = sorted(
            self.clusters.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        print("Top 10 Most Frequent Log Templates:")
        print("-" * 60)
        for i, (cluster_id, data) in enumerate(sorted_clusters[:10], 1):
            print(f"\n{i}. Cluster ID: {cluster_id}")
            print(f"   Template: {data['template']}")
            print(f"   Occurrences: {data['count']}")
            print(f"   Log IDs: {data['log_ids'][:5]}..." if len(data['log_ids']) > 5 else f"   Log IDs: {data['log_ids']}")
    
    def export_results(self, output_file='output/results.json'):
        """
        Export parsing results to JSON file
        
        Args:
            output_file: Path to output file
        """
        print(f"\n💾 Exporting results to: {output_file}")
        
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_logs": self.total_logs,
                "total_clusters": len(self.clusters)
            },
            "clusters": {}
        }
        
        for cluster_id, data in self.clusters.items():
            export_data["clusters"][str(cluster_id)] = {
                "template": data["template"],
                "count": data["count"],
                "log_ids": data["log_ids"]
            }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            print("✅ Results exported successfully!")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
    
    def get_template_for_log(self, log_line):
        """
        Get template for a specific log line
        
        Args:
            log_line: Raw log line
            
        Returns:
            Template string
        """
        cleaned = self.preprocess_log(log_line)
        result = self.template_miner.match(cleaned)
        
        if result:
            return result.get_template()
        else:
            return "No matching template found"


def main():
    """Main function"""
    print("""
    ╔════════════════════════════════════════╗
    ║   DRAIN LOG PARSER - IMPLEMENTATION    ║
    ║   Fast & Accurate Log Template Mining  ║
    ╚════════════════════════════════════════╝
    """)
    
    # Initialize parser
    parser = DrainLogParser(config_file='drain_config.ini')
    
    # Parse log file
    # results = parser.parse_file('data/sample_logs.txt')
    filename = 'Mac'
    results = parser.parse_file(f'data/{filename}.log')
    
    # Print statistics
    parser.print_statistics()
    
    # Export results
    parser.export_results(f'output/{filename}.json')
    
    print("\n" + "="*60)
    print("🎉 Parsing completed successfully!")
    print("="*60)
    
    # Example: Test single log
    print("\n\n📝 Testing single log parsing:")
    test_log = "2024-11-22 10:15:45 INFO Received block blk_9999 of size 12345678 from /10.251.99.99"
    template = parser.get_template_for_log(test_log)
    print(f"Input:    {test_log}")
    print(f"Template: {template}")


if __name__ == "__main__":
    main()