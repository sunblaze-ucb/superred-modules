import asyncio
import hashlib
import json
import logging
import math
import random
import time
import uuid
from os import PathLike
from pathlib import Path
from typing import Protocol

import numpy as np
from mutate import MutationMethod, Mutator
from pydantic import BaseModel, Field
from pydantic_core import to_json

logger = logging.getLogger(__name__)


def gen_id():
    return str(uuid.uuid4())


class MutatorConfig(BaseModel):
    helper_model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 3


class Seed(BaseModel):
    id: str = Field(default_factory=lambda: "seed_" + gen_id())
    text: str
    mutation_method: MutationMethod | None = None
    mutation_seed: str | list[str] | None = None
    score: float = 0.0
    performance: float = 0.0
    results: dict[str, int] = Field(default_factory=dict)


class FuzzOneLoopReturn(BaseModel):
    error: str = ""
    seed: Seed | None = None
    results: list[float] = Field(default_factory=list)
    coverage: float = 0.0


class MonteCarloTree:
    def __init__(self, seeds: list[Seed], exploration_factor: float = 1.41):
        """
        Initialize the Monte Carlo Tree with seeds.

        Args:
            seeds (list[Seed]): List of seed objects to explore.
            exploration_factor (float): Controls exploration vs exploitation trade-off.
        """

        self.nodes = {MonteCarloNode(seed) for seed in seeds}
        self.exploration_factor = exploration_factor

    def update(self, seed_id: str, reward: float):
        """
        Update the statistics for the selected seed.

        Args:        #for us to init a bitmap
        [score_seed(seed) for seed in seeds]
            seed_id (str): The ID of the seed to update.
            reward (float): The reward received for this seed.
        """
        for node in self.nodes:
            if node.seed.id == seed_id:
                node.update(reward)
                break

    def select_node(self, n: int = 1):
        """
        Explicit mechanism to select a node or nodes to visit.

        Args:
            n (int): Number of nodes to select (default is 1).

        Returns:
            MonteCarloNode | list[MonteCarloNode]: The selected Monte Carlo node(s).
        """
        total_visits = sum(node.visits for node in self.nodes)
        if n == 1:
            selected_node = max(
                self.nodes,
                key=lambda node: node.average_reward
                + self.exploration_factor
                * math.sqrt(math.log(total_visits + 1) / (node.visits + 1e-5)),
            )
            return [selected_node]
        elif n == 2:
            selected_nodes = sorted(
                self.nodes,
                key=lambda node: node.average_reward
                + self.exploration_factor
                * math.sqrt(math.log(total_visits + 1) / (node.visits + 1e-5)),
                reverse=True,
            )[:2]
            return selected_nodes


class MonteCarloNode:
    def __init__(self, seed: Seed, parents: list["MonteCarloNode"] = None):
        """
        Initialize a Monte Carlo Node for a specific seed.

        Args:
            seed (Seed): The seed associated with this node.
            parent (MonteCarloNode, optional): The parent node in the tree. Defaults to None.
        """
        self.seed = seed
        self.visits = 0
        self.total_reward = 0
        self.parents = parents  # Track the parent node

    @property
    def average_reward(self):
        """Compute the average reward for this node."""
        return self.total_reward / self.visits if self.visits > 0 else 0

    def update(self, reward: float):
        """
        Update the node with a new reward.

        Args:
            reward (float): The reward to incorporate.
        """
        self.visits += 1
        self.total_reward += reward


class TargetAgentProtocol(Protocol):
    async def run(self, task_list: list[str], injection: str) -> list[int]:
        """
        Run the target agent with the given task list and injection.
        Args:
            task_list (list[str]): List of tasks to perform.
            injection (str): The injection string.
        Returns:
            list[int]: List of results.
        """
        ...


class MCTSFuzzer:
    def __init__(
        self,
        mutator_config: MutatorConfig,
        seeds: list[Seed],
        out_dir: PathLike,
        target_agent: TargetAgentProtocol,
        target_tasks: list[str],
        name="fuzz",
        population_size: int = 3,
        checkpoint_interval: int = 1,
        subset_ratio: float = 1.0,
    ):
        self.name = name
        logger.info(
            f"Initializing MCTSFuzzer with {len(seeds)} seeds and {len(target_tasks)} target tasks"
        )
        self.mutator = Mutator(
            helper_model=mutator_config.helper_model,
            api_key=mutator_config.api_key,
            base_url=mutator_config.base_url,
            max_retries=mutator_config.max_retries,
        )
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory set to: {self.out_dir}")
        self.mcts_tree = MonteCarloTree(seeds)
        self.target_agent = target_agent
        self.target_tasks = target_tasks
        self.checkpoint_interval = checkpoint_interval
        self.current_iteration = 0
        self.results_list = []
        self.coverage_list = []
        self.subset_ratio = subset_ratio
        self.population_size = population_size

        self._init_coverage(target_tasks)

    def _init_coverage(self, keys: list[str]):
        self.coverage_bitmap = {key: 0 for key in keys}

    def _get_coverage(self) -> float:
        return sum(self.coverage_bitmap.values()) / len(self.coverage_bitmap)

    def save_checkpoint(self, checkpoint_path: PathLike | None = None):
        """Save the current fuzzer state to a checkpoint file.

        Args:
            checkpoint_path: Path to save checkpoint. If None, uses default path in out_dir.
        """
        if checkpoint_path is None:
            checkpoint_path = self.out_dir / "checkpoint.json"
        else:
            checkpoint_path = Path(checkpoint_path)

        # Build node map for parent relationships
        node_map = {id(node): idx for idx, node in enumerate(self.mcts_tree.nodes)}

        # Serialize nodes with parent indices
        nodes_data = []
        for node in self.mcts_tree.nodes:
            parent_indices = (
                [node_map[id(parent)] for parent in node.parents]
                if node.parents is not None
                else None
            )
            nodes_data.append(
                {
                    "seed": node.seed.model_dump(),
                    "visits": node.visits,
                    "total_reward": node.total_reward,
                    "parent_indices": parent_indices,
                }
            )

        checkpoint_data = {
            "current_iteration": self.current_iteration,
            "coverage_bitmap": self.coverage_bitmap,
            "nodes": nodes_data,
            "results_list": self.results_list,
            "coverage_list": self.coverage_list,
            "exploration_factor": self.mcts_tree.exploration_factor,
            "population_size": self.population_size,
            "subset_ratio": self.subset_ratio,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: PathLike,
        mutator_config: MutatorConfig,
        out_dir: PathLike,
        target_agent: TargetAgentProtocol,
        target_tasks: list[str],
        name="fuzz",
        checkpoint_interval: int = 1,
    ) -> "MCTSFuzzer":
        """Create a fuzzer instance from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.
            mutator_config: Mutator configuration.
            out_dir: Output directory.
            target_agent: Target agent instance.
            target_tasks: List of target task IDs.
            name: Name of the fuzzer.
            checkpoint_interval: How often to save checkpoints.

        Returns:
            MCTSFuzzer: Restored fuzzer instance.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from iteration {checkpoint_data['current_iteration']}")

        # Recreate seeds from checkpoint
        nodes_list = []
        for node_data in checkpoint_data["nodes"]:
            seed = Seed(**node_data["seed"])
            node = MonteCarloNode(seed=seed)
            node.visits = node_data["visits"]
            node.total_reward = node_data["total_reward"]
            nodes_list.append((node, node_data["parent_indices"]))

        # Reconstruct parent relationships
        for node, parent_indices in nodes_list:
            if parent_indices is not None:
                node.parents = [nodes_list[idx][0] for idx in parent_indices]

        # Extract just the nodes (without parent_indices)
        nodes = [node for node, _ in nodes_list]

        # Create fuzzer instance with initial seeds (will be overridden)
        initial_seeds = [node.seed for node in nodes]
        fuzzer = cls(
            mutator_config=mutator_config,
            seeds=initial_seeds,
            out_dir=out_dir,
            target_agent=target_agent,
            target_tasks=target_tasks,
            name=name,
            checkpoint_interval=checkpoint_interval,
            population_size=checkpoint_data.get("population_size", 3),
            subset_ratio=checkpoint_data.get("subset_ratio", 1.0),
        )

        # Override with restored state
        fuzzer.mcts_tree.nodes = set(nodes)
        fuzzer.mcts_tree.exploration_factor = checkpoint_data["exploration_factor"]
        fuzzer.coverage_bitmap = checkpoint_data["coverage_bitmap"]
        fuzzer.current_iteration = checkpoint_data["current_iteration"]
        fuzzer.results_list = checkpoint_data["results_list"]
        fuzzer.coverage_list = checkpoint_data["coverage_list"]

        logger.info(
            f"Loaded {len(nodes)} nodes, coverage: {fuzzer._get_coverage():.4f}"
        )

        return fuzzer

    def _update_coverage(self, task_results: dict[str, int]) -> float:
        """
        Update the coverage information based on the given node.

        Args:
            node (MonteCarloNode): The node containing the seed to evaluate.

        Returns:
            float: The updated coverage value.
        """
        new_coverage = 0
        for key, result in task_results.items():
            if result > self.coverage_bitmap.get(key, 0):
                self.coverage_bitmap[key] = result
                new_coverage += 1
        total_keys = len(self.coverage_bitmap)
        return new_coverage / total_keys if total_keys > 0 else 0.0

    def backpropagate(self, node: MonteCarloNode, reward: float):
        """
        Recursively backpropagate the reward up the tree, updating all parent nodes.

        Args:
            node (MonteCarloNode): The current node being updated.
            reward (float): The reward to propagate up the tree.
        """
        node.update(reward)  # Update the current node
        if node.parents is None:  # Check if there are no parents
            return
        for parent in node.parents:
            self.backpropagate(parent, reward)  # Recursively propagate to parents

    async def score_seed(
        self, node: MonteCarloNode, coverage_weight: float = 1.0
    ) -> float:
        """
        Score a seed based on its performance

        Args:
            node (MonteCarloNode): The node containing the seed to score.
            coverage_weight (float): Weight for coverage in the scoring.

        Returns:
            float: The computed score for the seed.
        """
        logger.debug(f"Scoring seed {node.seed.id}")
        # sample a subset of target tasks for efficiency
        if self.subset_ratio < 1.0:
            subset_size = max(1, int(len(self.target_tasks) * self.subset_ratio))
            sampled_tasks = random.sample(self.target_tasks, subset_size)
        else:
            sampled_tasks = self.target_tasks
        results = await self.target_agent.run(sampled_tasks, node.seed.text)
        task_results = {task: result for task, result in zip(sampled_tasks, results)}
        coverage = self._update_coverage(task_results)
        performance = sum(results) / len(results) if results else 0.0
        score = performance + coverage_weight * coverage
        for task, result in task_results.items():
            node.seed.results[task] = result
        node.seed.score = score
        node.seed.performance = performance
        logger.debug(
            f"Seed {node.seed.id} scored: performance={performance:.4f}, coverage={coverage:.4f}, total_score={score:.4f}"
        )
        return score

    async def mutate(
        self, method: str, nodes: list[MonteCarloNode] | None = None
    ) -> MonteCarloNode | None:
        try:
            if method == "crossover":
                # Ensure two nodes are provided for crossover
                if not nodes or len(nodes) != 2:
                    logger.warning("Crossover requires exactly two nodes")
                    return None

                node1, node2 = nodes

                # Use Mutator's crossover_mutate
                mutated_text = await self.mutator.mutate(
                    [node1.seed.text, node2.seed.text], method
                )
                if mutated_text is None:
                    logger.warning(
                        f"Failed to mutate seeds {node1.seed.id} and {node2.seed.id} with method {method}"
                    )
                    return None

                mutation_id = f"{node1.seed.id}_{node2.seed.id}_{method[:8]}"
                new_seed = Seed(
                    id=mutation_id,
                    text=mutated_text,
                    mutation_method=MutationMethod(method),
                    mutation_seed=[node1.seed.id, node2.seed.id],
                )
            else:
                # Single-seed mutation
                if not nodes or len(nodes) != 1:
                    logger.warning("Simple mutation requires exactly one node")
                    return None

                node = nodes[0]

                # Use Mutator's simple_mutate
                mutated_text = await self.mutator.mutate(node.seed.text, method)
                if mutated_text is None:
                    logger.warning(
                        f"Failed to mutate seed {node.seed.id} with method {method}"
                    )
                    return None

                id_hash = hashlib.md5(f"{node.seed.id}".encode()).hexdigest()[:6]
                mutation_id = id_hash + method
                new_seed = Seed(
                    id=mutation_id,
                    text=mutated_text,
                    mutation_method=MutationMethod(method),
                    mutation_seed=node.seed.id,
                )

            # Score
            try:
                result = await self.score_seed(MonteCarloNode(seed=new_seed))
            except Exception as e:
                result = None

            # if failure to score
            if result is None:
                # kill seed
                return None

            # Create a MonteCarloNode from the seed
            new_node = MonteCarloNode(seed=new_seed, parents=nodes)

            # Update the tree with the new node
            self.mcts_tree.nodes.add(new_node)

            # Backpropagation: Update scores for the selected nodes
            self.backpropagate(new_node, new_seed.score)

            return new_node
        except Exception as e:
            logger.error(f"Error during mutation with method {method}: {e}")
            return None

    async def fuzz_one_loop(self, run_id: str | None = None) -> FuzzOneLoopReturn:
        run_id = f"run_{gen_id()}"
        logger.info(f"Starting fuzz loop {run_id}")
        loop_nodes = []

        population_size = self.population_size  # Consider making this configurable

        coverage_before = self._get_coverage()
        logger.debug(f"Coverage before loop: {coverage_before:.4f}")

        for _ in range(population_size):
            method = random.choice([method.value for method in MutationMethod])
            new_node = None

            # Retry mechanism for mutations
            retries = 3

            for _ in range(retries):
                if method == "crossover":
                    new_node = await self.mutate(
                        method, nodes=self.mcts_tree.select_node(n=2)
                    )
                else:
                    new_node = await self.mutate(
                        method, nodes=self.mcts_tree.select_node(n=1)
                    )

                if new_node is not None:
                    break

            if new_node is None:
                logger.warning(
                    f"Failed to mutate a seed with method {method} after {retries} attempts"
                )
                continue

            # Add the new node to the tree
            loop_nodes.append(new_node)

        coverage_after = self._get_coverage()
        logger.info(
            f"Loop {run_id} completed. Coverage: {coverage_before:.4f} -> {coverage_after:.4f}"
        )

        if loop_nodes:
            logger.debug(f"Generated {len(loop_nodes)} new nodes")
            return FuzzOneLoopReturn(
                seed=loop_nodes[0].seed,
                results=[node.seed.score for node in loop_nodes],
                coverage=coverage_after,
            )
        logger.warning("No seeds generated in this loop")
        return FuzzOneLoopReturn(error="No seeds generated in this loop")

    async def fuzz_loop(self, num_loops: int = 1):
        logger.info(f"Starting fuzz loop with {num_loops} iterations")

        # Check if resuming from checkpoint
        start_idx = self.current_iteration
        if start_idx > 0:
            logger.info(f"Resuming from iteration {start_idx}")
        else:
            # score seeds for bitmap, but don't prioritize for coverage yet
            logger.info(f"Scoring initial {len(self.mcts_tree.nodes)} seeds")
            for idx, node in enumerate(self.mcts_tree.nodes):
                try:
                    await self.score_seed(node, coverage_weight=0)
                    logger.debug(
                        f"Scored initial seed {idx + 1}/{len(self.mcts_tree.nodes)}: {node.seed.score:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to score initial seed {idx + 1}: {e}")
                    continue

            with open(self.out_dir / "initial_seeds.json", "wb") as f:
                f.write(to_json([node.seed for node in self.mcts_tree.nodes], indent=2))

        for idx in range(start_idx, num_loops):
            logger.info(f"=== Fuzzing iteration {idx + 1}/{num_loops} ===")
            ret = await self.fuzz_one_loop(f"step_{idx}")
            if ret.error:
                logger.info(f"Fuzzing error: {ret.error}")
            else:
                self.results_list.append(ret.results)
                self.coverage_list.append(ret.coverage)
                logger.info(
                    f"Iteration {idx + 1} results: avg_score={sum(ret.results) / len(ret.results):.4f}, coverage={ret.coverage:.4f}"
                )

            self.current_iteration = idx + 1

            await asyncio.sleep(1)

            # Save fuzz log
            with open(self.out_dir / "fuzz_log.json", "wb") as f:
                f.write(
                    to_json(
                        {
                            "results": self.results_list,
                            "coverage": self.coverage_list,
                        },
                        indent=2,
                    )
                )

            # Save all seeds
            with open(self.out_dir / "seeds_after_fuzz.json", "wb") as f:
                f.write(to_json([node.seed for node in self.mcts_tree.nodes], indent=2))

            # Save checkpoint periodically
            if (idx + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint()
                logger.info(f"Checkpoint saved at iteration {idx + 1}")

        seeds = [node.seed for node in self.mcts_tree.nodes]

        seeds.sort(key=lambda seed: seed.performance, reverse=True)
        logger.info(
            f"Fuzzing completed. Total seeds: {len(seeds)}, Best performance: {seeds[0].performance:.4f}"
        )

        # Save final checkpoint
        self.save_checkpoint()
        logger.info("Final checkpoint saved")

        return [seed for seed in seeds], self.coverage_list
