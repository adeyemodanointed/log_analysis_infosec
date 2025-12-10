import json
import matplotlib.pyplot as plt

def visualize_clusters(json_file='output/results.json'):
    """Visualize cluster distribution"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract cluster counts
    clusters = data['clusters']
    counts = [cluster['count'] for cluster in clusters.values()]
    templates = [cluster['template'][:50] + '...' if len(cluster['template']) > 50 
                 else cluster['template'] for cluster in clusters.values()]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(counts)), counts, color='steelblue')
    plt.yticks(range(len(counts)), templates, fontsize=8)
    plt.xlabel('Number of Logs')
    plt.title('Log Template Distribution')
    plt.tight_layout()
    plt.savefig('output/cluster_distribution.png', dpi=300)
    plt.show()
    
    print("📊 Visualization saved to: output/cluster_distribution.png")

if __name__ == "__main__":
    visualize_clusters()