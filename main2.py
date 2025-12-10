"""
Log Format Standardizer
Converts Drain3 and Librelog outputs to a unified format for fair comparison
"""

import json
from datetime import datetime
from pathlib import Path


class LogFormatConverter:
    """Convert different log parser outputs to standardized format"""

    @staticmethod
    def convert_drain_to_standard(drain_file, output_file):
        """
        Convert Drain3 cluster format to standardized log-level format

        Args:
            drain_file: Path to Drain3 output JSON
            output_file: Path to save standardized output
        """
        print(f"📂 Converting Drain3 output: {drain_file}")

        with open(drain_file, 'r', encoding='utf-8') as f:
            drain_data = json.load(f)

        # Extract metadata
        metadata = drain_data.get('metadata', {})
        clusters = drain_data.get('clusters', {})

        # Convert to log-level format
        logs = []
        log_counter = 0

        for cluster_id, cluster_data in clusters.items():
            template = cluster_data['template']
            count = cluster_data['count']
            log_ids = cluster_data.get('log_ids', [])

            # Create a log entry for each occurrence
            for i, original_log_id in enumerate(log_ids):
                log_counter += 1
                logs.append({
                    "log_id": log_counter,
                    "original_log_id": original_log_id,
                    "template_id": cluster_id,
                    "template": template,
                    "cluster_size": count,
                    "original_log": None  # Not available in cluster format
                })

        # Create standardized output
        standardized = {
            "metadata": {
                "timestamp": metadata.get('timestamp', datetime.now().isoformat()),
                "total_logs": len(logs),
                "unique_templates": len(clusters),
                "method": "drain3",
                "original_total_logs": metadata.get('total_logs', len(logs)),
                "original_total_clusters": metadata.get('total_clusters', len(clusters))
            },
            "logs": logs
        }

        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(standardized, f, indent=2)

        print(f"✅ Converted {len(logs)} logs with {len(clusters)} unique templates")
        print(f"💾 Saved to: {output_file}")
        return standardized

    @staticmethod
    def convert_librelog_to_standard(librelog_file, output_file):
        """
        Convert Librelog output to standardized format

        Args:
            librelog_file: Path to Librelog output JSON
            output_file: Path to save standardized output
        """
        print(f"📂 Converting Librelog output: {librelog_file}")

        with open(librelog_file, 'r', encoding='utf-8') as f:
            librelog_data = json.load(f)

        # Extract metadata
        metadata = librelog_data.get('metadata', {})
        results = librelog_data.get('results', [])

        # Convert to standardized format
        logs = []
        template_counts = {}

        for result in results:
            template_id = result.get('template_id', f"group_{result['log_id']}")
            template = result.get('template', '')

            # Count template occurrences
            if template_id not in template_counts:
                template_counts[template_id] = 0
            template_counts[template_id] += 1

            logs.append({
                "log_id": result['log_id'],
                "original_log_id": result['log_id'],
                "template_id": template_id,
                "template": template,
                "original_log": result.get('original_log'),
                "variables": result.get('variables'),
                "confidence": result.get('confidence'),
                "parsing_time": result.get('parsing_time')
            })

        # Add cluster size to each log
        for log in logs:
            log['cluster_size'] = template_counts[log['template_id']]

        # Create standardized output
        standardized = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_logs": len(logs),
                "unique_templates": len(template_counts),
                "method": "librelog",
                "avg_parsing_time": metadata.get('avg_parsing_time'),
                "llm_calls": metadata.get('llm_calls')
            },
            "logs": logs
        }

        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(standardized, f, indent=2)

        print(f"✅ Converted {len(logs)} logs with {len(template_counts)} unique templates")
        print(f"💾 Saved to: {output_file}")
        return standardized

    @staticmethod
    def batch_convert(drain_file, librelog_file, output_dir='output/standardized'):
        """
        Convert both files at once

        Args:
            drain_file: Path to Drain3 output
            librelog_file: Path to Librelog output
            output_dir: Directory to save standardized outputs
        """
        print("\n" + "=" * 60)
        print("🔄 BATCH CONVERSION - Standardizing Log Formats")
        print("=" * 60 + "\n")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Convert Drain3
        drain_output = Path(output_dir) / 'drain3_standardized.json'
        drain_data = LogFormatConverter.convert_drain_to_standard(
            drain_file, drain_output
        )

        print()

        # Convert Librelog
        librelog_output = Path(output_dir) / 'librelog_standardized.json'
        librelog_data = LogFormatConverter.convert_librelog_to_standard(
            librelog_file, librelog_output
        )

        # Print comparison
        print("\n" + "=" * 60)
        print("📊 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<30} {'Drain3':<15} {'Librelog':<15}")
        print("-" * 60)
        print(
            f"{'Total Logs':<30} {drain_data['metadata']['total_logs']:<15} {librelog_data['metadata']['total_logs']:<15}")
        print(
            f"{'Unique Templates':<30} {drain_data['metadata']['unique_templates']:<15} {librelog_data['metadata']['unique_templates']:<15}")
        print(f"{'Method':<30} {drain_data['metadata']['method']:<15} {librelog_data['metadata']['method']:<15}")
        print("=" * 60)

        print("\n✅ Standardization complete! Ready for anomaly detection.")
        print(f"📁 Files saved in: {output_dir}/")

        return drain_output, librelog_output


def main():
    """Example usage"""
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   LOG FORMAT STANDARDIZER                     ║
    ║   Unified format for fair comparison          ║
    ╚═══════════════════════════════════════════════╝
    """)

    # Example: Convert both files
    drain_file = 'output/Apache.json'  # Your Drain3 output
    librelog_file = 'output/Apachelog_results.json'  # Your Librelog output

    try:
        drain_std, librelog_std = LogFormatConverter.batch_convert(
            drain_file,
            librelog_file,
            output_dir='output/standardized'
        )

        print("\n🎯 Next Steps:")
        print("1. Upload both standardized files to the anomaly detection tool")
        print("2. Run the comparison")
        print("3. Export results for your report")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Usage:")
        print("  1. Update the file paths in main()")
        print("  2. Ensure both Drain3 and Librelog outputs exist")
        print("  3. Run the script again")


if __name__ == "__main__":
    main()