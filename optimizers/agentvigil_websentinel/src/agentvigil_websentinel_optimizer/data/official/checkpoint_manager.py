"""
Utility script for managing fuzzer checkpoints.

Provides commands to:
- Inspect checkpoint information
- Delete checkpoints
- Export/import checkpoints
"""

import argparse
import json
from pathlib import Path


def inspect_checkpoint(checkpoint_path: Path):
    """Display information about a checkpoint file."""
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Checkpoint Information: {checkpoint_path}")
    print(f"{'='*60}")
    print(f"Current Iteration: {data['current_iteration']}")
    print(f"Total Nodes: {len(data['nodes'])}")
    print(f"Exploration Factor: {data['exploration_factor']}")
    
    # Coverage statistics
    coverage_bitmap = data['coverage_bitmap']
    total_tasks = len(coverage_bitmap)
    covered_tasks = sum(1 for v in coverage_bitmap.values() if v > 0)
    coverage_pct = (covered_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    print(f"\nCoverage:")
    print(f"  Tasks Covered: {covered_tasks}/{total_tasks} ({coverage_pct:.1f}%)")
    print(f"  Coverage Score: {sum(coverage_bitmap.values())}")
    
    # Results statistics
    if data['results_list']:
        all_scores = [score for results in data['results_list'] for score in results]
        print(f"\nResults:")
        print(f"  Completed Iterations: {len(data['results_list'])}")
        print(f"  Total Scores: {len(all_scores)}")
        if all_scores:
            print(f"  Best Score: {max(all_scores):.4f}")
            print(f"  Average Score: {sum(all_scores)/len(all_scores):.4f}")
    
    # Node statistics
    nodes = data['nodes']
    total_visits = sum(node['visits'] for node in nodes)
    avg_visits = total_visits / len(nodes) if nodes else 0
    best_node = max(nodes, key=lambda n: n['seed']['performance']) if nodes else None
    
    print(f"\nNode Statistics:")
    print(f"  Total Visits: {total_visits}")
    print(f"  Average Visits per Node: {avg_visits:.2f}")
    if best_node:
        print(f"  Best Node Performance: {best_node['seed']['performance']:.4f}")
        print(f"  Best Node ID: {best_node['seed']['id']}")
    
    print(f"\n{'='*60}\n")


def delete_checkpoint(checkpoint_path: Path, force: bool = False):
    """Delete a checkpoint file."""
    if not checkpoint_path.exists():
        print(f"Checkpoint file not found: {checkpoint_path}")
        return
    
    if not force:
        response = input(f"Are you sure you want to delete {checkpoint_path}? (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled.")
            return
    
    checkpoint_path.unlink()
    print(f"Deleted checkpoint: {checkpoint_path}")


def export_seeds(checkpoint_path: Path, output_path: Path):
    """Export all seeds from a checkpoint to a separate JSON file."""
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    
    seeds = [node['seed'] for node in data['nodes']]
    
    # Sort by performance
    seeds.sort(key=lambda s: s['performance'], reverse=True)
    
    with open(output_path, "w") as f:
        json.dump(seeds, f, indent=2)
    
    print(f"Exported {len(seeds)} seeds to {output_path}")
    print(f"Top 3 seeds by performance:")
    for i, seed in enumerate(seeds[:3], 1):
        print(f"  {i}. {seed['id']}: {seed['performance']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Manage fuzzer checkpoints")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a checkpoint file")
    inspect_parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a checkpoint file")
    delete_parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    delete_parser.add_argument("--force", "-f", action="store_true", 
                               help="Skip confirmation prompt")
    
    # Export seeds command
    export_parser = subparsers.add_parser("export", 
                                          help="Export seeds from checkpoint")
    export_parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    export_parser.add_argument("output", type=Path, help="Output path for seeds JSON")
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        inspect_checkpoint(args.checkpoint)
    elif args.command == "delete":
        delete_checkpoint(args.checkpoint, args.force)
    elif args.command == "export":
        export_seeds(args.checkpoint, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
