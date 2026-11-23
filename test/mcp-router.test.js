/**
 * test/mcp-router.test.js — the task-scoped MCP view (issue #156).
 *
 * The acceptance criteria are about what a fabricated tool call cannot do, so
 * most of these run against deterministic fake registry clients with call
 * spies (zero calls is the assertion), plus a fake model that emits exactly
 * the tool call under test. One test rides the real stdio wire against
 * test-fixtures/fake-mcp-server.js to prove the registry path works end to end.
 *
 * The three criteria:
 *   1. a filtered or spoofed server/tool cannot be invoked even by fabricated
 *      model output; names remain collision-safe;
 *   2. a legacy trusted server still needs approval in scoped mode and is
 *      refused unattended; denial causes zero calls;
 *   3. MCP crash, oversized result and malformed schema produce bounded
 *      errors; no server is spawned solely by model instruction.
 */
import test, { after } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURE = join(ROOT, "test-fixtures", "fake-mcp-server.js");

process.env.PINCH_DATA = mkdtempSync(join(tmpdir(), "pinch-scope-data-"));
process.env.PINCH_WORKSPACE = mkdtempSync(join(tmpdir(), "pinch-scope-ws-"));
process.env.LLM_PROVIDER = "ollama";
process.env.OLLAMA_MODEL = "fake";
process.env.PINCH_GUARD = "off";
process.env.PINCH_AUTO_APPROVE = "false";

/* ────────────────────────── a fake model ──────────────────────────────── */
// It emits whatever tool call the current test queued, then says "done" —
// that is "fabricated model output" under the tester's control.

const model = { plan: { toolCall: null }, requests: [] };
const modelServer = createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", () => {
    let parsed = {};
    try {
      parsed = JSON.parse(body || "{}");
    } catch {}
    model.requests.push(parsed);
    const hasToolResult = (parsed.messages || []).some((m) => m.role === "tool");
    const message =
      model.plan.toolCall && !hasToolResult
        ? {
            content: "",
            tool_calls: [{ function: { name: model.plan.toolCall.name, arguments: model.plan.toolCall.args } }],
          }
        : { content: "done" };
    const payload = JSON.stringify({ message, done: true });
    if (parsed.stream) {
      res.writeHead(200, { "content-type": "application/x-ndjson" });
      res.write(payload + "\n");
      res.end();
    } else {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(payload);
    }
  });
});
await new Promise((r) => modelServer.listen(0, "127.0.0.1", r));
process.env.OLLAMA_URL = `http://127.0.0.1:${modelServer.address().port}`;
after(async () => {
  modelServer.closeAllConnections();
  await new Promise((r) => modelServer.close(r));
});

const {
  createMcpTaskScope,
  normalizeScopeAllow,
  isMcpTaskScope,
  isWellFormedToolSchema,
  McpScopeError,
  MCP_SCOPE_LIMITS,
} = await import("../src/mcp/router.js");
const { McpClient, toQuasnoozeTool, mcpToolName, parseMcpToolName, mcpClients, MCP_RESULT_CHAR_LIMIT } = await import("../src/mcp.js");
const { runAgent, answerApprovalFor, pendingApprovalsFor } = await import("../src/agent.js");
const { registerTool, TOOL_MAP, DANGEROUS } = await import("../src/tools.js");

const quiet = { info() {}, warn() {} };

/* ────────────────────────── fake registry ─────────────────────────────── */

function def(name, { properties = {}, required = [], description = `${name} tool`, inputSchema } = {}) {
  return {
    name,
    description,
    inputSchema: inputSchema !== undefined ? inputSchema : { type: "object", properties, ...(required.length ? { required } : {}) },
  };
}

/** A registry client with a call spy. `calls.length` is the "zero calls" proof. */
function makeClient(name, tools, { trusted = false, onCall } = {}) {
  const calls = [];
  return {
    name,
    trusted,
    tools,
    calls,
    async callTool(tool, args) {
      calls.push({ tool, args });
      if (onCall) return onCall(tool, args);
      return { ok: true, output: `${tool} ok` };
    },
  };
}

/** Register a fake client's tools the way startMcpTools does; returns cleanup. */
function register(client) {
  const names = [];
  for (const definition of client.tools) {
    const tool = toQuasnoozeTool(client, definition);
    registerTool(tool);
    names.push(tool.name);
  }
  return () => {
    for (const name of names) {
      TOOL_MAP.delete(name);
      DANGEROUS.delete(name);
    }
  };
}

function oneToolCall(name, args = {}) {
  model.plan.toolCall = { name, args };
}

function offers(request) {
  return (request?.tools || []).map((t) => t.function?.name);
}

/* ────────────────── names remain collision-safe (criterion 1) ─────────── */

test("names: the mcp__server__tool namespace cannot collide", () => {
  assert.equal(mcpToolName("files", "read_file"), "mcp__files__read_file");
  assert.deepEqual(parseMcpToolName("mcp__files__read_file"), { server: "files", tool: "read_file" });
  assert.deepEqual(parseMcpToolName("mcp__my_server__my_tool"), { server: "my_server", tool: "my_tool" });

  // "__" in either part would make one name parse two ways.
  assert.throws(() => mcpToolName("a__b", "c"), /without "__"/);
  assert.throws(() => mcpToolName("a", "b__c"), /without "__"/);
  assert.throws(() => mcpToolName("a b", "c"), /must be/);
  assert.throws(() => mcpToolName("", "c"), /must be/);
  assert.equal(parseMcpToolName("mcp__a__b__c"), null, "an ambiguous name must not parse");
  assert.equal(parseMcpToolName("mcp__a__b "), null);
  assert.equal(parseMcpToolName("shell"), null);
  assert.deepEqual(parseMcpToolName("mcp__a__b"), { server: "a", tool: "b" });
});

/* ──────────── scope creation rejects bad names (criterion 1) ──────────── */

test("scope: creation rejects unknown names, collisions and unsafe names", () => {
  const registry = new Map([["files", makeClient("files", [def("read")])]]);
  const cases = [
    [{ sessionId: "s", allow: ["ghost:tool"], registry }, /not in the running MCP registry/, "unknown-server"],
    [{ sessionId: "s", allow: ["files:write"], registry }, /advertises no tool/, "unknown-tool"],
    [{ sessionId: "s", allow: ["files:read", "files:read"], registry }, /twice/, "name-collision"],
    [{ sessionId: "s", allow: ["a__b:c"], registry }, /without "__"/, "unsafe-name"],
    [{ sessionId: "s", allow: ["a:b__c"], registry }, /without "__"/, "unsafe-name"],
    [{ sessionId: "s", allow: ["files:read:extra"], registry }, /without "__"/, "unsafe-name"],
    [{ sessionId: "", allow: [], registry }, /session/, "bad-origin"],
    [{ sessionId: "s", allow: "files:read", registry }, /array/, "bad-allowlist"],
    [{ sessionId: "s", allow: [42], registry }, /must be/, "bad-allowlist"],
  ];
  for (const [opts, pattern, code] of cases) {
    assert.throws(
      () => createMcpTaskScope(opts),
      (err) =>
        err instanceof McpScopeError &&
        err.code === code &&
        pattern.test(err.message) &&
        err.message.length <= 400,
      `expected ${JSON.stringify(opts.allow)} to be rejected as ${code}`,
    );
  }
  // A map key that does not match the client's own name is an alias, and an
  // alias could dispatch under a name it never owned.
  const aliased = new Map([["files", makeClient("other", [def("read")])]]);
  assert.throws(() => createMcpTaskScope({ sessionId: "s", allow: ["files:read"], registry: aliased }), McpScopeError);
});

test("scope: object and list allowlist forms normalise to the same pairs", () => {
  const entries = normalizeScopeAllow([{ server: "s", tools: ["b", "a"] }, "t:c"]);
  assert.deepEqual(entries.map((e) => e.name), ["mcp__s__a", "mcp__s__b", "mcp__t__c"]);
  assert.ok(Object.isFrozen(entries));
});

/* ─────────────── a worker cannot widen its scope (criterion 1) ────────── */

test("scope: a worker attempt to expand the scope is refused", async () => {
  const allow = ["files:read"];
  const client = makeClient("files", [def("read"), def("write")]);
  const registry = new Map([["files", client]]);
  const scope = createMcpTaskScope({ sessionId: "s", allow, registry });
  assert.ok(isMcpTaskScope(scope));
  assert.ok(Object.isFrozen(scope));

  assert.throws(() => scope.expand(["files:write"]), (err) => err instanceof McpScopeError && err.code === "scope-fixed");
  assert.throws(() => scope.add(["files:write"]), (err) => err.code === "scope-fixed");

  // Mutating the owner's input after the fact must not widen anything either.
  allow.push("files:write");
  assert.equal(scope.has("mcp__files__write"), false);
  const result = await scope.dispatch({ name: "mcp__files__write", args: {} });
  assert.equal(result.ok, false);
  assert.equal(client.calls.length, 0);

  // And the scope's own lists are frozen data.
  assert.throws(() => scope.entries.push({}), TypeError);
});

/* ───────── filtered and spoofed names never dispatch (criterion 1) ───── */

test("scope: offered schemas are filtered to the owner-approved names", () => {
  const registry = new Map([
    ["files", makeClient("files", [def("read"), def("write")])],
    ["gh", makeClient("gh", [def("pr")])],
  ]);
  const scope = createMcpTaskScope({ sessionId: "s", allow: ["files:read"], registry });
  const filtered = scope.filterSchemas([
    { type: "function", function: { name: "shell", parameters: {} } },
    { type: "function", function: { name: "mcp__files__read", parameters: {} } },
    { type: "function", function: { name: "mcp__files__write", parameters: {} } },
    { type: "function", function: { name: "mcp__gh__pr", parameters: {} } },
    { type: "function", function: { name: "mcp__ghost__run", parameters: {} } },
  ]);
  assert.deepEqual(filtered.map((s) => s.function.name), ["shell", "mcp__files__read"]);
});

test("scope: a filtered or spoofed name is refused at dispatch with zero server calls", async () => {
  const client = makeClient("files", [def("read"), def("write")]);
  const other = makeClient("gh", [def("pr")]);
  const registry = new Map([["files", client], ["gh", other]]);
  const scope = createMcpTaskScope({ sessionId: "s", allow: ["files:read"], registry });

  for (const name of [
    "mcp__files__write", // registered but filtered out of this task
    "mcp__gh__pr", // a whole other server
    "mcp__files__read ", // whitespace spoof
    "mcp__files__READ", // case spoof
    "mcp__ghost__run", // fabricated server
    "shell", // a built-in is not an MCP call and not in this allowlist
    "mcp__",
    "",
  ]) {
    const result = await scope.dispatch({ name, args: {} });
    assert.equal(result.ok, false, `"${name}" must be refused`);
    assert.match(result.output, /^Refused:/);
    assert.ok(result.output.length <= MCP_SCOPE_LIMITS.maxErrorChars, "refusals are bounded");
  }
  assert.equal(client.calls.length, 0, "a refused dispatch must not reach the server");
  assert.equal(other.calls.length, 0);
  // No server is spawned or added by any of it (criterion 3).
  assert.deepEqual([...registry.keys()], ["files", "gh"]);
});

test("scope: an approved call dispatches to the exact server and tool", async () => {
  const client = makeClient("files", [def("read", { properties: { path: { type: "string" } }, required: ["path"] })]);
  const registry = new Map([["files", client]]);
  const scope = createMcpTaskScope({ sessionId: "s", allow: ["files:read"], registry });

  const ok = await scope.dispatch({ name: "mcp__files__read", args: { path: "notes.txt" } });
  assert.equal(ok.ok, true);
  assert.deepEqual(client.calls, [{ tool: "read", args: { path: "notes.txt" } }]);
  assert.match(ok.output, /<untrusted-source tool="mcp__files__read">/);

  // The same argument contract as every other tool, enforced inside dispatch.
  const missing = await scope.dispatch({ name: "mcp__files__read", args: {} });
  assert.equal(missing.ok, false);
  assert.match(missing.output, /Missing required/);
  const unknown = await scope.dispatch({ name: "mcp__files__read", args: { path: "x", bogus: 1 } });
  assert.equal(unknown.ok, false);
  assert.match(unknown.output, /Unknown argument/);
  assert.equal(client.calls.length, 1, "validation refusals do not reach the server");
});

/* ────────────── scoped mode is stricter than legacy (criterion 2) ─────── */

test("scope: scoped tools are dangerous even for a legacy-trusted server", async () => {
  const client = makeClient("files", [def("read")], { trusted: true });
  const legacy = toQuasnoozeTool(client, client.tools[0]);
  assert.equal(legacy.dangerous, false, "legacy trusted opt-out must stay compatible");

  const registry = new Map([["files", client]]);
  const scope = createMcpTaskScope({ sessionId: "s", allow: ["files:read"], registry });
  const scoped = scope.toolFor("mcp__files__read");
  assert.equal(scoped.dangerous, true, "scoped mode marks every MCP call dangerous");
  assert.deepEqual(scoped.mcp, { server: "files", tool: "read", scoped: true });

  // Unattended scoped calls are refused outright, with zero calls.
  const refused = await scope.dispatch({ name: "mcp__files__read", args: { path: "x" }, autonomy: "unattended" });
  assert.equal(refused.ok, false);
  assert.match(refused.output, /unattended/);
  assert.equal(client.calls.length, 0);
});

/* ────────────── bounded failures, no spawning (criterion 3) ───────────── */

test("scope: a crashed server is a bounded error, not a throw", async () => {
  const client = makeClient("flaky", [def("poke")], {
    onCall: () => {
      throw new Error(`server "flaky" exited (SIGKILL): ${"x".repeat(5000)}`);
    },
  });
  const registry = new Map([["flaky", client]]);
  const scope = createMcpTaskScope({ sessionId: "s", allow: ["flaky:poke"], registry });
  const result = await scope.dispatch({ name: "mcp__flaky__poke", args: {} });
  assert.equal(result.ok, false);
  assert.match(result.output, /exited/);
  assert.match(result.output, /<untrusted-source/);
  assert.ok(result.output.length < 600, `crash errors are bounded, got ${result.output.length}`);
});

test("scope: an oversized result is capped into the context", async () => {
  const client = makeClient("chatty", [def("spew")], { onCall: () => ({ ok: true, output: "z".repeat(100_000) }) });
  const registry = new Map([["chatty", client]]);
  const scope = createMcpTaskScope({ sessionId: "s", allow: ["chatty:spew"], registry });
  const result = await scope.dispatch({ name: "mcp__chatty__spew", args: {} });
  assert.equal(result.ok, true);
  assert.match(result.output, /truncated/);
  assert.match(result.output, /88000 characters not shown/);
  assert.ok(result.output.length < MCP_RESULT_CHAR_LIMIT + 300);
});

test("scope: a malformed schema is a bounded error, is not offered, and never dispatched", async () => {
  assert.equal(isWellFormedToolSchema(undefined), true, "no schema is defaulted, not malformed");
  for (const bad of ["not-a-schema", [1], { type: "string" }, { properties: "x" }, { required: "y" }, { required: [1] }]) {
    assert.equal(isWellFormedToolSchema(bad), false, `${JSON.stringify(bad)} must count as malformed`);
  }
  assert.equal(isWellFormedToolSchema({ type: "object", properties: { a: { type: "string" } }, required: ["a"] }), true);

  const client = makeClient("weird", [
    { name: "bad", description: "b", inputSchema: "not-a-schema" },
    def("good"),
  ]);
  const registry = new Map([["weird", client]]);
  const scope = createMcpTaskScope({ sessionId: "s", allow: ["weird:bad", "weird:good"], registry });

  const offered = scope
    .filterSchemas([{ function: { name: "mcp__weird__bad" } }, { function: { name: "mcp__weird__good" } }])
    .map((s) => s.function.name);
  assert.deepEqual(offered, ["mcp__weird__good"]);

  const result = await scope.dispatch({ name: "mcp__weird__bad", args: {} });
  assert.equal(result.ok, false);
  assert.match(result.output, /malformed schema/);
  assert.ok(result.output.length <= MCP_SCOPE_LIMITS.maxErrorChars);
  assert.equal(client.calls.length, 0);

  const tool = scope.toolFor("mcp__weird__bad");
  assert.equal(tool.schemaMalformed, true);
  assert.equal(tool.dangerous, true);
});

test("scope: dispatch rides the real stdio wire", async () => {
  const client = new McpClient({ name: "fake", command: process.execPath, args: [FIXTURE], log: quiet });
  await client.start();
  try {
    let calls = 0;
    const original = client.callTool.bind(client);
    client.callTool = async (tool, args) => {
      calls += 1;
      return original(tool, args);
    };
    const registry = new Map([["fake", client]]);
    const scope = createMcpTaskScope({ sessionId: "wire-1", allow: ["fake:add"], registry });

    const result = await scope.dispatch({ name: "mcp__fake__add", args: { a: 2, b: 40 } });
    assert.equal(result.ok, true);
    assert.match(result.output, /42/);
    assert.match(result.output, /<untrusted-source tool="mcp__fake__add">/);
    assert.equal(calls, 1);

    const refused = await scope.dispatch({ name: "mcp__fake__echo_env" });
    assert.equal(refused.ok, false);
    assert.match(refused.output, /outside this task's approved MCP scope/);
    assert.equal(calls, 1, "a filtered tool must not reach the real server");
  } finally {
    await client.close();
  }
});

/* ─────────── the agent loop under a scope (fabricated output) ─────────── */

test("agent: a fabricated out-of-scope call is refused and the server sees nothing", async () => {
  const client = makeClient("salpha", [def("read"), def("write")], { trusted: true });
  const cleanup = register(client);
  try {
    const registry = new Map([["salpha", client]]);
    const scope = createMcpTaskScope({ sessionId: "origin-a", allow: ["salpha:read"], registry });
    oneToolCall("mcp__salpha__write", { path: "secrets.txt" }); // registered, but not in this task
    const start = model.requests.length;
    const result = await runAgent({
      sessionId: "worker-a",
      text: "handle the inbound work item now",
      channel: "web",
      mcpScope: scope,
    });
    const requests = model.requests.slice(start);

    assert.ok(result.trace.some((t) => t.tool === "mcp__salpha__write" && t.reason === "out-of-scope"));
    assert.equal(client.calls.length, 0, "a filtered tool must not be invoked even by fabricated model output");

    // It was never offered in the first place...
    const offered = offers(requests[0]);
    assert.ok(offered.includes("mcp__salpha__read"));
    assert.ok(!offered.includes("mcp__salpha__write"));
    assert.ok(offered.includes("shell") && offered.includes("read_file"), "built-ins stay offered");
    // ...and the refusal is what the model was told.
    const feedback = requests.at(-1).messages.filter((m) => m.role === "tool").map((m) => m.content).join("\n");
    assert.match(feedback, /outside this task's approved MCP scope/);
    assert.ok(!feedback.includes("Available:"), "an unknown call must not leak the global tool list");
  } finally {
    cleanup();
  }
});

test("agent: a spoofed server name is refused with zero calls and nothing spawned", async () => {
  const client = makeClient("sbeta", [def("read")]);
  const cleanup = register(client);
  try {
    const registry = new Map([["sbeta", client]]);
    const scope = createMcpTaskScope({ sessionId: "origin-b", allow: ["sbeta:read"], registry });
    oneToolCall("mcp__ghost__run", { cmd: "whatever" });
    const result = await runAgent({
      sessionId: "worker-b",
      text: "handle the inbound work item now",
      channel: "web",
      mcpScope: scope,
    });
    assert.ok(result.trace.some((t) => t.tool === "mcp__ghost__run" && t.reason === "out-of-scope"));
    assert.equal(client.calls.length, 0);
    assert.deepEqual([...registry.keys()], ["sbeta"], "no server is spawned or added by a fabricated name");
  } finally {
    cleanup();
  }
});

test("agent: a scoped call needs the originating session's approval; denial causes zero calls", async () => {
  const client = makeClient("sgamma", [def("read", { properties: { path: { type: "string" } }, required: ["path"] })], {
    trusted: true, // legacy-trusted: in scoped mode that opt-out is ignored
  });
  const cleanup = register(client);
  try {
    assert.ok(!DANGEROUS.has("mcp__sgamma__read"), "the legacy opt-out registered it as non-dangerous");
    const registry = new Map([["sgamma", client]]);
    const scope = createMcpTaskScope({ sessionId: "origin-9", allow: ["sgamma:read"], registry });
    oneToolCall("mcp__sgamma__read", { path: "notes.txt" });

    let approvals = 0;
    const result = await runAgent({
      sessionId: "worker-7",
      text: "handle the inbound work item now",
      channel: "web",
      mcpScope: scope,
      onEvent: (event) => {
        if (event.type !== "approval") return;
        approvals += 1;
        // The prompt belongs to the session that owns the scope, not the worker.
        assert.ok(pendingApprovalsFor("origin-9").some((p) => p.id === event.id));
        assert.equal(pendingApprovalsFor("worker-7").length, 0);
        // A yes in the worker's conversation must not reach it...
        assert.equal(answerApprovalFor({ sessionId: "worker-7", channel: "web" }, true), false);
        // ...and the originating session's answer is the one that counts.
        assert.equal(answerApprovalFor({ sessionId: "origin-9", channel: "web" }, false), true);
      },
    });

    assert.equal(approvals, 1, "a legacy-trusted server still needs approval in scoped mode");
    assert.equal(client.calls.length, 0, "denial causes zero calls");
    assert.ok(result.trace.some((t) => t.tool === "mcp__sgamma__read" && t.denied));
  } finally {
    cleanup();
  }
});

test("agent: an approved scoped call runs exactly once", async () => {
  const client = makeClient("sdelta", [def("read", { properties: { path: { type: "string" } }, required: ["path"] })], {
    trusted: true,
  });
  const cleanup = register(client);
  try {
    const registry = new Map([["sdelta", client]]);
    const scope = createMcpTaskScope({ sessionId: "origin-1", allow: ["sdelta:read"], registry });
    oneToolCall("mcp__sdelta__read", { path: "notes.txt" });

    const result = await runAgent({
      sessionId: "worker-1",
      text: "handle the inbound work item now",
      channel: "web",
      mcpScope: scope,
      onEvent: (event) => {
        if (event.type === "approval") answerApprovalFor({ sessionId: "origin-1", channel: "web" }, true);
      },
    });

    assert.deepEqual(client.calls, [{ tool: "read", args: { path: "notes.txt" } }]);
    const trace = result.trace.find((t) => t.tool === "mcp__sdelta__read");
    assert.ok(trace.ok);
    assert.match(String(trace.output), /<untrusted-source/);
  } finally {
    cleanup();
  }
});

test("agent: scoped calls ask for approval even with PINCH_AUTO_APPROVE on", async () => {
  // config is frozen from the environment at import, so the proof runs in a
  // child process that has the flag set (test-fixtures/scope-autoapprove-scenario.js).
  // Its turn talks to this file's fake model, which serves the queued call.
  oneToolCall("mcp__scase__read", { path: "notes.txt" });
  try {
    const { stdout } = await execFileAsync(
      process.execPath,
      [join(ROOT, "test-fixtures", "scope-autoapprove-scenario.js")],
      { env: { ...process.env, PINCH_AUTO_APPROVE: "true" } },
    );
    const report = JSON.parse(stdout.trim().split("\n").at(-1));
    assert.equal(report.autoApprove, true, "the scenario must actually run with the flag on");
    assert.equal(report.approvals, 1, "scoped mode requires originating-session approval, full stop");
    assert.deepEqual(report.calls, [{ tool: "read", args: { path: "notes.txt" } }]);
    assert.equal(report.ok, true);
  } finally {
    model.plan.toolCall = null;
  }
});

test("agent: unattended turns refuse scoped calls even for a legacy-trusted server", async () => {
  const client = makeClient("szeta", [def("read")], { trusted: true });
  const cleanup = register(client);
  try {
    const registry = new Map([["szeta", client]]);
    const scope = createMcpTaskScope({ sessionId: "origin-3", allow: ["szeta:read"], registry });
    oneToolCall("mcp__szeta__read", { path: "notes.txt" });

    const result = await runAgent({
      sessionId: "sched-1",
      text: "handle the inbound work item now",
      channel: "web",
      autonomy: "unattended",
      mcpScope: scope,
    });
    assert.equal(client.calls.length, 0, "an unattended scoped call must never run");
    const trace = result.trace.find((t) => t.tool === "mcp__szeta__read");
    assert.ok(trace.denied);
    assert.equal(trace.reason, "unattended");
  } finally {
    cleanup();
  }
});

test("agent: a malformed scoped schema is refused before any approval", async () => {
  const client = makeClient("seta", [{ name: "bad", description: "b", inputSchema: "not-a-schema" }]);
  const cleanup = register(client);
  try {
    const registry = new Map([["seta", client]]);
    const scope = createMcpTaskScope({ sessionId: "origin-4", allow: ["seta:bad"], registry });
    oneToolCall("mcp__seta__bad", {});

    let approvals = 0;
    const result = await runAgent({
      sessionId: "worker-4",
      text: "handle the inbound work item now",
      channel: "web",
      mcpScope: scope,
      onEvent: (event) => {
        if (event.type === "approval") {
          approvals += 1;
          answerApprovalFor({ sessionId: "origin-4", channel: "web" }, true);
        }
      },
    });
    assert.equal(approvals, 0, "there is nothing an approval could authorise");
    assert.equal(client.calls.length, 0);
    assert.ok(result.trace.some((t) => t.tool === "mcp__seta__bad" && t.reason === "malformed-schema"));
  } finally {
    cleanup();
  }
});

/* ──────────────────── legacy mode stays compatible ────────────────────── */

test("agent: without a scope, a legacy-trusted MCP tool runs with no prompt", async () => {
  const client = makeClient("stheta", [def("read")], { trusted: true });
  const cleanup = register(client);
  try {
    oneToolCall("mcp__stheta__read", { path: "notes.txt" });
    let approvals = 0;
    const result = await runAgent({
      sessionId: "legacy-1",
      text: "handle the inbound work item now",
      channel: "web",
      onEvent: (event) => {
        if (event.type === "approval") {
          approvals += 1;
          answerApprovalFor({ sessionId: "legacy-1", channel: "web" }, true);
        }
      },
    });
    assert.equal(approvals, 0, "legacy mode must not suddenly ask");
    assert.equal(client.calls.length, 1);
    assert.ok(result.trace.some((t) => t.tool === "mcp__stheta__read" && t.ok));
  } finally {
    cleanup();
  }
});

test("agent: runAgent refuses a mcpScope that is not a task scope", async () => {
  await assert.rejects(
    () => runAgent({ sessionId: "bad-scope", text: "hello there", channel: "web", mcpScope: { has() { return true; } } }),
    /createMcpTaskScope/,
  );
});

/* ──────────── the default registry is read-only for scopes ────────────── */

test("scope: the live registry is never written to", () => {
  const before = mcpClients().size;
  assert.throws(() => createMcpTaskScope({ sessionId: "s", allow: ["nobody:nothing"] }), McpScopeError);
  assert.equal(mcpClients().size, before);
});
