# Fuzzer Save and Resume Feature

This document explains how to use the save and resume functionality for the MCTSFuzzer.

## Features

- **Automatic Checkpointing**: Save fuzzer state at configurable intervals
- **Resume from Checkpoint**: Continue fuzzing from where you left off
- **Preserve Complete State**: Includes MCTS tree, coverage bitmap, and iteration progress
- **Graceful Interruption Handling**: Save checkpoint on Ctrl+C or errors

## Quick Start

### Starting a New Fuzzing Session with Checkpointing

```python
from fuzzer import MCTSFuzzer, MutatorConfig

fuzzer = MCTSFuzzer(
    mutator_config=MutatorConfig(helper_model="gpt-4o-mini"),
    seeds=initial_seeds,
    out_dir="tmp/logs",
    target_tasks=task_ids,
    target_agent=agent,
    checkpoint_interval=5,  # Save checkpoint every 5 iterations
)

# Run fuzzing
await fuzzer.fuzz_loop(num_loops=100)
```

### Resuming from a Checkpoint

```python
from fuzzer import MCTSFuzzer, MutatorConfig

# Load fuzzer from checkpoint
fuzzer = MCTSFuzzer.from_checkpoint(
    checkpoint_path="tmp/logs/checkpoint.json",
    mutator_config=MutatorConfig(helper_model="gpt-4o-mini"),
    out_dir="tmp/logs",
    target_tasks=task_ids,
    target_agent=agent,
    checkpoint_interval=5,
)

# Continue fuzzing (will resume from saved iteration)
await fuzzer.fuzz_loop(num_loops=100)
```

### Automatic Resume Logic

See `run_with_resume.py` for a complete example that automatically resumes from checkpoint if one exists:

```bash
python run_with_resume.py
```

This script will:
1. Check if a checkpoint exists in the output directory
2. Resume from checkpoint if found, otherwise start fresh
3. Save checkpoints periodically during execution
4. Save checkpoint on interruption (Ctrl+C) or errors

## Checkpoint File Format

The checkpoint is saved as a JSON file containing:

```json
{
  "current_iteration": 10,
  "coverage_bitmap": {"task1": 1, "task2": 0, ...},
  "nodes": [
    {
      "seed": {
        "id": "seed_xyz",
        "text": "...",
        "score": 0.75,
        "performance": 0.5,
        "results": [1, 0, 1, ...]
      },
      "visits": 3,
      "total_reward": 2.25,
      "parent_indices": [0, 1]
    },
    ...
  ],
  "results_list": [[0.5, 0.6], [0.7, 0.8], ...],
  "coverage_list": [0.1, 0.15, 0.2, ...],
  "exploration_factor": 1.41
}
```

## Checkpoint Management

Use the `checkpoint_manager.py` utility to manage checkpoints:

### Inspect Checkpoint Information

```bash
python checkpoint_manager.py inspect tmp/logs/checkpoint.json
```

Output:
```
============================================================
Checkpoint Information: tmp/logs/checkpoint.json
============================================================
Current Iteration: 10
Total Nodes: 45
Exploration Factor: 1.41

Coverage:
  Tasks Covered: 15/20 (75.0%)
  Coverage Score: 18

Results:
  Completed Iterations: 10
  Total Scores: 30
  Best Score: 0.8500
  Average Score: 0.6234

Node Statistics:
  Total Visits: 135
  Average Visits per Node: 3.00
  Best Node Performance: 0.8200
  Best Node ID: seed_abc123
============================================================
```

### Export Seeds from Checkpoint

```bash
python checkpoint_manager.py export tmp/logs/checkpoint.json seeds_exported.json
```

### Delete Checkpoint

```bash
python checkpoint_manager.py delete tmp/logs/checkpoint.json
```

Or skip confirmation:
```bash
python checkpoint_manager.py delete tmp/logs/checkpoint.json --force
```

## Configuration Options

### MCTSFuzzer Parameters

- `checkpoint_interval` (int): Number of iterations between checkpoints (default: 1)
  - Set to 1 to save after every iteration (safest, but slower)
  - Set to higher values (e.g., 5, 10) for better performance

### Example Configurations

**Maximum Safety** (save every iteration):
```python
fuzzer = MCTSFuzzer(
    checkpoint_interval=1,
    ...
)
```

**Balanced** (save every 5 iterations):
```python
fuzzer = MCTSFuzzer(
    checkpoint_interval=5,
    ...
)
```

**Performance Priority** (save every 10 iterations):
```python
fuzzer = MCTSFuzzer(
    checkpoint_interval=10,
    ...
)
```

## Manual Checkpoint Operations

### Save Checkpoint Manually

```python
# Save to default location (out_dir/checkpoint.json)
fuzzer.save_checkpoint()

# Save to custom location
fuzzer.save_checkpoint("custom_checkpoint.json")
```

### Check Current State

```python
# Get current iteration
current_iter = fuzzer.current_iteration

# Get current coverage
coverage = fuzzer._get_coverage()

# Get number of nodes
num_nodes = len(fuzzer.mcts_tree.nodes)
```

## Best Practices

1. **Set Appropriate Checkpoint Interval**: Balance between safety and performance
   - Long-running fuzzing (100+ iterations): checkpoint every 5-10 iterations
   - Quick experiments: checkpoint every 1-2 iterations

2. **Handle Interruptions Gracefully**: The example script shows how to catch KeyboardInterrupt and save checkpoint

3. **Backup Important Checkpoints**: Copy checkpoint files before making significant changes

4. **Monitor Disk Space**: Checkpoint files can be large (1-10 MB depending on node count)

5. **Use Checkpoint Manager**: Regularly inspect checkpoints to monitor progress

## Troubleshooting

### Checkpoint File Corrupted

If a checkpoint file is corrupted, you may need to start fresh or use an earlier checkpoint if you made backups.

### Resume Not Working as Expected

Make sure:
- The checkpoint file exists and is valid JSON
- You're passing the same `target_tasks` when resuming
- The output directory matches the one used when saving

### High Memory Usage

Large MCTS trees can consume significant memory. Consider:
- Limiting the number of nodes retained
- Using a higher checkpoint interval
- Pruning low-performing nodes periodically

## Advanced Usage

### Custom Checkpoint Location

```python
# Save with custom name including timestamp
import time
checkpoint_name = f"checkpoint_{int(time.time())}.json"
fuzzer.save_checkpoint(fuzzer.out_dir / checkpoint_name)
```

### Multiple Checkpoints

Keep multiple checkpoints for recovery:

```python
# Save numbered checkpoints
for i in range(num_loops):
    if i % checkpoint_interval == 0:
        fuzzer.save_checkpoint(f"checkpoint_{i:04d}.json")
```

### Checkpoint Rotation

Implement checkpoint rotation to save disk space:

```python
def save_checkpoint_with_rotation(fuzzer, max_checkpoints=5):
    """Save checkpoint and keep only the latest N checkpoints."""
    checkpoints = sorted(fuzzer.out_dir.glob("checkpoint_*.json"))
    
    # Delete old checkpoints
    while len(checkpoints) >= max_checkpoints:
        oldest = checkpoints.pop(0)
        oldest.unlink()
    
    # Save new checkpoint
    checkpoint_name = f"checkpoint_{fuzzer.current_iteration:04d}.json"
    fuzzer.save_checkpoint(fuzzer.out_dir / checkpoint_name)
```
