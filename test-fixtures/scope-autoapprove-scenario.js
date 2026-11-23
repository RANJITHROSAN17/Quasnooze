/**
 * test-fixtures/scope-autoapprove-scenario.js — one scoped agent turn with
 * PINCH_AUTO_APPROVE=true, reported as JSON on stdout.
 *
 * Spawned by test/mcp-router.test.js, which owns the fake model server and
 * passes its URL in the environment. Deliberately outside test/: `node --test`
 * collects every .js under a directory called test, and this is a scenario a
 * test drives, not a test.
 *
 * The point: scoped MCP mode (issue #156) requires originating-session
 * approval *even when* the owner set PINCH_AUTO_APPROVE. config is frozen at
 * import from the environment, so the proof runs in a process that has the
 * flag set and reports what the approval gate actually did.
 */
process.env.LLM_PROVIDER = "ollama";
process.env.PINCH_GUARD = "off";
process.env.PINCH_AUTO_APPROVE = "true";

const { createMcpTaskScope } = await import("../src/mcp/router.js");
const { toQuasnoozeTool } = await import("../src/mcp.js");
const { runAgent, answerApprovalFor } = await import("../src/agent.js");
const { registerTool } = await import("../src/tools.js");
const { config } = await import("../src/config.js");

const calls = [];
const client = {
  name: "scase",
  trusted: true,
  tools: [
    {
      name: "read",
      description: "read a file",
      inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
    },
  ],
  async callTool(tool, args) {
    calls.push({ tool, args });
    return { ok: true, output: "read ok" };
  },
};
const tool = toQuasnoozeTool(client, client.tools[0]);
registerTool(tool);

const registry = new Map([["scase", client]]);
const scope = createMcpTaskScope({ sessionId: "origin-aa", allow: ["scase:read"], registry });

let approvals = 0;
const result = await runAgent({
  sessionId: "worker-aa",
  text: "handle the inbound work item now",
  channel: "web",
  mcpScope: scope,
  onEvent: (event) => {
    if (event.type === "approval") {
      approvals += 1;
      answerApprovalFor({ sessionId: "origin-aa", channel: "web" }, true);
    }
  },
});

console.log(
  JSON.stringify({
    autoApprove: config.autoApprove,
    approvals,
    calls,
    ok: result.trace.some((t) => t.tool === "mcp__scase__read" && t.ok),
  }),
);
