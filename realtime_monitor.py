import time
from main import DrainLogParser

def monitor_logs(log_file='data/sample_logs.txt'):
    """Monitor logs in real-time (simulation)"""
    
    parser = DrainLogParser()
    
    print("🔍 Starting real-time log monitoring...")
    print("Press Ctrl+C to stop\n")
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                result = parser.parse_log(line)
                
                print(f"[{result['log_id']}] Template: {result['template'][:60]}...")
                print(f"     Change: {result['change_type']}")
                
                # Simulate real-time delay
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")
        parser.print_statistics()

if __name__ == "__main__":
    monitor_logs()