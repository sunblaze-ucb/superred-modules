import asyncio
import json
import logging
from pathlib import Path
from agent import SimulatedWebAgent
from fuzzer import MCTSFuzzer, MutatorConfig
from new_seeds import new_seeds

out_dir = Path("tmp/logs.20260126.subset.websentinel")
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

logger.info("Loading attack data")
with open(
    "/home/zhunwang/projects/websentinel/agentvigil/adaptive_attack_data.json"
) as f:
    tasks = json.loads(f.read())

target_task_ids = [task["id"] for task in tasks]

logger.info(f"Loaded {len(tasks)} tasks")

logger.info("Initializing agent and fuzzer")
agent = SimulatedWebAgent(model="gpt-4o-2024-05-13", apply_detection=True, batch_size=10)
fuzzer = MCTSFuzzer(
    mutator_config=MutatorConfig(
        helper_model="gpt-4o-2024-05-13",
    ),
    seeds=new_seeds,
    out_dir=out_dir,
    target_tasks=target_task_ids,
    target_agent=agent,
    subset_ratio=1,
    population_size=10,
)
logger.info(
    f"Starting fuzzing with {len(new_seeds)} seeds and {len(target_task_ids)} target tasks"
)

try:
    asyncio.run(fuzzer.fuzz_loop(20))
    logger.info("Fuzzing completed successfully")
except Exception as e:
    logger.error(f"Fuzzing failed with error: {e}", exc_info=True)
    raise
