#!/usr/bin/env node
/**
 * Wrapper for seclaw agent that respects maxToolIterations from config.
 * Usage: node seclaw-agent-wrapper.js "message" [session_id]
 */
const { loadConfig, getProvider, getApiBase, getProviderName } = require("/opt/seclaw/dist/config/loader");
const { getWorkspacePath } = require("/opt/seclaw/dist/config/schema");
const { LiteLLMProvider } = require("/opt/seclaw/dist/providers/litellm_provider");
const { MessageBus } = require("/opt/seclaw/dist/bus/queue");
const { AgentLoop } = require("/opt/seclaw/dist/agent/loop");

async function main() {
  const message = process.argv[2];
  const sessionId = process.argv[3] || "cli:default";
  if (!message) {
    console.error("Usage: node seclaw-agent-wrapper.js <message> [session_id]");
    process.exit(1);
  }

  const config = loadConfig();
  const workspacePath = getWorkspacePath(config);
  const p = getProvider(config);
  const provider = new LiteLLMProvider({
    apiKey: p?.apiKey,
    apiBase: getApiBase(config),
    defaultModel: config.agents.defaults.model,
    extraHeaders: p?.extraHeaders ?? undefined,
    providerName: getProviderName(config),
  });

  const bus = new MessageBus();
  const agentLoop = new AgentLoop({
    bus,
    provider,
    workspace: workspacePath,
    braveApiKey: config.tools.web?.search?.apiKey || undefined,
    execConfig: config.tools.exec,
    restrictToWorkspace: config.tools.restrictToWorkspace,
    securityConfig: config.security,
    maxIterations: config.agents.defaults.maxToolIterations || 100,
  });

  const response = await agentLoop.processDirect(message, sessionId);
  console.log(response ?? "");
  process.exit(0);
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
