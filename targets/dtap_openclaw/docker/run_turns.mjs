#!/usr/bin/env node
// In-container entrypoint for one OpenClaw DTAP episode.
//
// Reads /state/task.json (written by the host driver: build_task_json) and runs
// the OpenClaw CLI once per turn, exactly as upstream DTAP does on the host:
//   openclaw --profile <profile> agent --local --message <turn>
//            --thinking <level> --session-id <session_id>
// with OPENCLAW_TRAJECTORY=1 so OpenClaw writes the per-session trajectory JSONL
// under OPENCLAW_TRAJECTORY_DIR (the bound /state/traces). The host then reads
// that trace back via dtap_openclaw_target.trajectory.convert.
//
// HOME is /state, so the per-profile config at /state/.openclaw-<profile>/openclaw.json
// (written by the host) is the config OpenClaw loads for `--profile <profile>`.

import { spawnSync } from "node:child_process";
import { readFileSync, mkdirSync } from "node:fs";

const STATE = process.env.HOME || "/state";
const taskPath = `${STATE}/task.json`;

let task;
try {
  task = JSON.parse(readFileSync(taskPath, "utf-8"));
} catch (err) {
  console.error(`[run_turns] cannot read ${taskPath}: ${err}`);
  process.exit(2);
}

const turns = Array.isArray(task.turns) && task.turns.length ? task.turns : [""];
const sessionId = task.session_id || "dtap-session";
const profile = task.profile || "dtap";
const thinking = task.thinking || "off";
const traceDir = task.trace_dir || `${STATE}/traces`;

mkdirSync(traceDir, { recursive: true });

const env = {
  ...process.env,
  HOME: STATE,
  OPENCLAW_TRAJECTORY: "1",
  OPENCLAW_TRAJECTORY_DIR: traceDir,
};

// Mirror upstream OpenClawAgent.run(): each turn is independent. A failed turn is
// logged and SWALLOWED (no break) -- upstream's _run_openclaw_cli returns
// success:False on a non-zero exit or timeout and run() keeps looping, then always
// generates the trajectory. So run every turn and let the episode complete; the
// host then extracts whatever trace was written and re-queries env state (an attack
// that mutated state and then failed a turn is still judged).
let failures = 0;
for (const turn of turns) {
  const args = [
    "--profile", profile,
    "agent",
    "--local",
    "--message", String(turn),
    "--thinking", thinking,
    "--session-id", sessionId,
  ];
  const result = spawnSync("openclaw", args, { env, stdio: "inherit" });
  if (result.error) {
    console.error(`[run_turns] failed to spawn openclaw: ${result.error} (continuing)`);
    failures += 1;
    continue;
  }
  if (typeof result.status === "number" && result.status !== 0) {
    console.error(`[run_turns] openclaw exited ${result.status} on a turn (continuing)`);
    failures += 1;
    continue;
  }
}

if (failures) {
  console.error(`[run_turns] episode completed with ${failures} failed turn(s)`);
}
// Exit 0: per-turn failures are non-fatal (mirrors upstream). Only a catastrophic
// setup failure -- an unreadable task.json (handled above) -- exits non-zero.
process.exit(0);
