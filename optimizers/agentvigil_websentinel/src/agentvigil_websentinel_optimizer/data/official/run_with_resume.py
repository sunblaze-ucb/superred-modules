"""
Example script showing how to use save and resume functionality with MCTSFuzzer.

This script demonstrates:
1. Starting a new fuzzing session with checkpointing enabled
2. Resuming from a checkpoint if one exists
3. Configuring checkpoint interval
"""

import asyncio
import json
import logging
from pathlib import Path
from agent import SimulatedWebAgent
from fuzzer import MCTSFuzzer, MutatorConfig
from new_seeds import new_seeds

out_dir = Path("tmp/logs")
out_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
logging.getLogger("fuzzer").setLevel(logging.DEBUG)
logging.getLogger("mutate").setLevel(logging.DEBUG)
logging.getLogger("agent").setLevel(logging.DEBUG)

logger.info(f"Output directory: {out_dir}")

# Load task data
logger.info("Loading attack data")
with open(
    "/home/zhunwang/projects/websentinel/agentvigil/adaptive_attack_data.json"
) as f:
    tasks = json.loads(f.read())

target_task_ids = [task["id"] for task in tasks]
logger.info(f"Loaded {len(tasks)} tasks")

# Configuration
mutator_config = MutatorConfig(helper_model="gpt-4o-mini-2024-07-18")
num_loops = 100  # Total number of fuzzing iterations
checkpoint_interval = 5  # Save checkpoint every 5 iterations
checkpoint_path = out_dir / "checkpoint.json"

# Initialize agent
logger.info("Initializing agent")
agent = SimulatedWebAgent(model="gpt-4o-2024-05-13", apply_detection=False)

# Check if checkpoint exists to resume
if checkpoint_path.exists():
    logger.info(f"Found existing checkpoint at {checkpoint_path}")
    logger.info("Resuming from checkpoint...")
    
    fuzzer = MCTSFuzzer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        mutator_config=mutator_config,
        out_dir=out_dir,
        target_tasks=target_task_ids,
        target_agent=agent,
        checkpoint_interval=checkpoint_interval,
    )
    logger.info(
        f"Resumed from iteration {fuzzer.current_iteration} with {len(fuzzer.mcts_tree.nodes)} nodes"
    )
else:
    logger.info("No checkpoint found, starting fresh fuzzing session")
    
    fuzzer = MCTSFuzzer(
        mutator_config=mutator_config,
        seeds=new_seeds,
        out_dir=out_dir,
        target_tasks=target_task_ids,
        target_agent=agent,
        checkpoint_interval=checkpoint_interval,
    )
    logger.info(
        f"Starting fuzzing with {len(new_seeds)} seeds and {len(target_task_ids)} target tasks"
    )

# Run fuzzing
try:
    asyncio.run(fuzzer.fuzz_loop(num_loops))
    logger.info("Fuzzing completed successfully")
except KeyboardInterrupt:
    logger.info("Fuzzing interrupted by user")
    logger.info("Saving checkpoint before exit...")
    fuzzer.save_checkpoint()
    logger.info("Checkpoint saved. You can resume later using this checkpoint.")
except Exception as e:
    logger.error(f"Fuzzing failed with error: {e}", exc_info=True)
    logger.info("Saving checkpoint before exit...")
    fuzzer.save_checkpoint()
    raise
