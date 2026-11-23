/**
 * mcp/router.js — a task-scoped view over the stdio MCP registry (issue #156).
 *
 * The registry in `../mcp.js` is global and namespaced: every configured
 * server's tools are registered once at boot, and an owner-trusted server
 * (`PINCH_MCP_TRUSTED`) skips the danger gate. A worker or any orchestration
 * task must not inherit that surface, and a router must not assume every
 * server is gated — so this module adds an optional per-task view that a
 * caller keeps for one task and throws away.
 *
 * The rules that matter, and why:
 *
 *  1. **The scope is fixed at creation.** It matches owner-approved server and
 *     tool names against the servers *already running* in the registry. It
 *     never starts, installs or configures anything, so no model instruction
 *     can widen the scope or spawn a server — a fabricated name is a bounded
 *     refusal with zero server calls, not a lookup that boots something.
 *  2. **Scoped mode is stricter than legacy, never looser.** Every scoped call
 *     is marked dangerous regardless of the legacy trusted opt-out, the
 *     approval belongs to the session that owns the scope (see `sessionBound`
 *     in `../agent.js`), and unattended calls are refused outright. Unscoped
 *     legacy behaviour is untouched.
 *  3. **Failure is bounded.** Unknown, spoofed or colliding names, a crashed
 *     server, an oversized result and a malformed schema all produce short
 *     `{ ok: false, output }` errors — never a throw at dispatch, never a
 *     hang, never server-controlled text unbounded into the context.
 *
 * No remote transport, no auto-install, no shared credentials: dispatch rides
 * the same stdio clients the registry already started, with the same scrubbed
 * environment and the same untrusted-output label.
 */
import {
  MCP_RESULT_CHAR_LIMIT,
  isSafeMcpName,
  labelUntrustedMcp,
  mcpClients,
  mcpToolName,
} from "../mcp.js";

/** A scope refusal with a machine-readable code and a bounded message. */
export class McpScopeError extends Error {
  constructor(code, message) {
    super(String(message).slice(0, 400));
    this.name = "McpScopeError";
    this.code = code;
  }
}

export const MCP_SCOPE_LIMITS = Object.freeze({
  /** A task scope is a small allowlist, not a second registry. */
  maxEntries: 32,
  /** Result text one scoped call may carry into the context. */
  maxResultChars: MCP_RESULT_CHAR_LIMIT,
  /** A refusal or failure is a sentence, not a core dump. */
  maxErrorChars: 400,
  maxSessionChars: 200,
});

/* ─────────────────────────── schema sanity ────────────────────────────── */

/**
 * A tool input schema must be an object schema to be offered or dispatched.
 * `undefined`/`null` is "no schema given" and is defaulted by the caller; a
 * string, an array, a non-object `type`, or a `properties`/`required` of the
 * wrong shape is malformed and yields a bounded error, not a crash.
 */
export function isWellFormedToolSchema(schema) {
  if (schema === undefined || schema === null) return true;
  if (typeof schema !== "object" || Array.isArray(schema)) return false;
  if (schema.type !== undefined && schema.type !== "object") return false;
  if (
    schema.properties !== undefined &&
    (typeof schema.properties !== "object" || schema.properties === null || Array.isArray(schema.properties))
  ) {
    return false;
  }
  if (
    schema.required !== undefined &&
    (!Array.isArray(schema.required) || schema.required.some((key) => typeof key !== "string"))
  ) {
    return false;
  }
  return true;
}

/** The same contract as `validateArgs` in agent.js, enforced inside dispatch. */
function scopedArgError(schema, args) {
  const required = Array.isArray(schema.required) ? schema.required : [];
  const allowed = Object.keys(schema.properties || {});
  const given = Object.keys(args);
  const missing = required.filter((key) => args[key] === undefined || args[key] === null);
  if (missing.length) {
    return `Missing required argument${missing.length === 1 ? "" : "s"}: ${missing.join(", ")}.`;
  }
  const unknown = given.filter((key) => !allowed.includes(key));
  if (allowed.length && unknown.length) {
    return `Unknown argument${unknown.length === 1 ? "" : "s"}: ${unknown.join(", ")}. Allowed: ${allowed.join(", ")}.`;
  }
  return null;
}

/* ────────────────────────── allowlist parsing ─────────────────────────── */

const clip = (value, max = 120) => {
  const text = String(value ?? "");
  return text.length > max ? `${text.slice(0, max)}…` : text;
};

/**
 * Normalise the owner-approved allowlist into `{ server, tool, name }` entries.
 *
 * Accepts `"server:tool"` strings, `{ server, tool }` and
 * `{ server, tools: [...] }` objects. Unknown shapes, unsafe names and
 * collisions (the same namespaced name twice) are rejected loudly — a scope
 * with one ambiguous entry is worse than no scope at all.
 */
export function normalizeScopeAllow(allow) {
  if (!Array.isArray(allow)) {
    throw new McpScopeError("bad-allowlist", "an MCP task scope needs an array of owner-approved server/tool names");
  }
  const raw = [];
  for (const item of allow) {
    if (typeof item === "string") {
      const sep = item.indexOf(":");
      if (sep === -1) {
        throw new McpScopeError("bad-allowlist", `scope entry ${clip(item)} must look like "server:tool"`);
      }
      raw.push({ server: item.slice(0, sep).trim(), tool: item.slice(sep + 1).trim() });
    } else if (item && typeof item === "object" && !Array.isArray(item)) {
      const tools = Array.isArray(item.tools) ? item.tools : [item.tool];
      for (const tool of tools) raw.push({ server: item.server, tool });
    } else {
      throw new McpScopeError("bad-allowlist", `scope entry ${clip(item)} must be a "server:tool" string or a { server, tool } object`);
    }
  }
  if (raw.length > MCP_SCOPE_LIMITS.maxEntries) {
    throw new McpScopeError("bad-allowlist", `an MCP task scope names at most ${MCP_SCOPE_LIMITS.maxEntries} tools (got ${raw.length})`);
  }

  const seen = new Set();
  const entries = [];
  for (const { server, tool } of raw) {
    const s = String(server ?? "").trim();
    const t = String(tool ?? "").trim();
    if (!isSafeMcpName(s)) {
      throw new McpScopeError("unsafe-name", `scope server name ${clip(s)} must be 1-64 letters, digits, dash or underscore, without "__"`);
    }
    if (!isSafeMcpName(t)) {
      throw new McpScopeError("unsafe-name", `scope tool name ${clip(t)} must be 1-64 letters, digits, dash or underscore, without "__"`);
    }
    // mcpToolName is unambiguous for safe names, so equal names mean equal pairs.
    const name = mcpToolName(s, t);
    if (seen.has(name)) {
      throw new McpScopeError("name-collision", `scope names "${name}" twice; server/tool pairs must be unique`);
    }
    seen.add(name);
    entries.push(Object.freeze({ server: s, tool: t, name }));
  }
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return Object.freeze(entries);
}

/* ──────────────────────────── the task scope ──────────────────────────── */

const SCOPES = new WeakSet();

export function isMcpTaskScope(value) {
  return SCOPES.has(value);
}

/**
 * Build the task-scoped view. Throws `McpScopeError` on anything ambiguous:
 * an unknown or duplicate name is rejected *here*, at owner-approved scope
 * creation, and never falls through to a lookup that could start a server.
 *
 * @param {object} opts
 * @param {string} opts.sessionId the session that owns this scope — scoped
 *   approvals are bound to it and answered there.
 * @param {Array} opts.allow owner-approved `server`/`tool` names (see above).
 * @param {Map|object} [opts.registry] live clients; defaults to the stdio
 *   registry. Passed explicitly in tests. Read only — never written, never
 *   grown.
 */
export function createMcpTaskScope({ sessionId, allow, registry = mcpClients() } = {}) {
  const origin = typeof sessionId === "string" ? sessionId.trim() : "";
  if (!origin || origin.length > MCP_SCOPE_LIMITS.maxSessionChars) {
    throw new McpScopeError("bad-origin", "an MCP task scope needs the id of the session that owns it");
  }
  const entries = normalizeScopeAllow(allow);

  const resolveClient = (server) => {
    if (registry instanceof Map) return registry.get(server);
    if (registry && typeof registry === "object") return registry[server];
    return undefined;
  };

  const byName = new Map();
  for (const entry of entries) {
    const client = resolveClient(entry.server);
    if (!client || typeof client.callTool !== "function" || client.name !== entry.server) {
      throw new McpScopeError(
        "unknown-server",
        `server "${entry.server}" is not in the running MCP registry; a task scope can only name servers already started, never start one`,
      );
    }
    const listed = Array.isArray(client.tools) ? client.tools : [];
    const matches = listed.filter((definition) => definition && definition.name === entry.tool);
    if (matches.length === 0) {
      throw new McpScopeError("unknown-tool", `server "${entry.server}" advertises no tool "${entry.tool}"`);
    }
    if (matches.length > 1) {
      throw new McpScopeError("name-collision", `server "${entry.server}" advertises "${entry.tool}" more than once`);
    }
    const definition = matches[0];
    const schema = definition.inputSchema === undefined ? { type: "object", properties: {} } : definition.inputSchema;
    byName.set(
      entry.name,
      Object.freeze({
        ...entry,
        client,
        definition,
        schema,
        schemaOk: isWellFormedToolSchema(schema),
      }),
    );
  }

  const names = Object.freeze([...byName.keys()]);
  const frozenEntries = Object.freeze(names.map((name) => { const e = byName.get(name); return Object.freeze({ server: e.server, tool: e.tool, name }); }));

  const refuse = (text) => ({
    ok: false,
    output: String(text).slice(0, MCP_SCOPE_LIMITS.maxErrorChars),
  });

  /** Exact-string membership. A filtered or spoofed name is simply not here. */
  function has(name) {
    return byName.has(String(name));
  }

  /**
   * Filter offered schemas: built-ins pass through, MCP entries pass only when
   * the owner approved the name AND the server's schema is well formed. A
   * malformed-schema tool is not offered; a direct call gets the bounded error
   * from `dispatch`.
   */
  function filterSchemas(schemas) {
    return (Array.isArray(schemas) ? schemas : []).filter((schema) => {
      const name = schema?.function?.name ?? schema?.name;
      if (typeof name !== "string" || !name.startsWith("mcp__")) return true;
      const entry = byName.get(name);
      return Boolean(entry && entry.schemaOk);
    });
  }

  /**
   * The scoped view of one approved tool. Always `dangerous: true` — scoped
   * mode marks every MCP call dangerous regardless of any legacy trusted
   * opt-out — and `run` routes through `dispatch`, so the allowlist is checked
   * again at the door even if this object reached a caller by another path.
   */
  function toolFor(name) {
    const entry = byName.get(String(name));
    if (!entry) return null;
    return {
      name: entry.name,
      description: `[${entry.server} MCP server, task-scoped] ${entry.definition.description || entry.tool}`,
      parameters: entry.schemaOk ? entry.schema : { type: "object", properties: {} },
      schemaMalformed: !entry.schemaOk,
      dangerous: true,
      mcp: { server: entry.server, tool: entry.tool, scoped: true },
      async run(args) {
        return dispatch({ name: entry.name, args });
      },
    };
  }

  /**
   * The single door to the server. Enforces the allowlist again here — at
   * dispatch — so fabricated model output cannot invoke a filtered or spoofed
   * name even if an offered-schemas filter were bypassed. Every refusal is
   * bounded and ends with zero server calls.
   */
  async function dispatch({ name, args = {}, autonomy = "interactive" } = {}) {
    const key = String(name ?? "");
    const entry = byName.get(key);
    if (!entry) {
      return refuse(`Refused: ${clip(key)} is outside this task's approved MCP scope. No server was called.`);
    }
    if (autonomy === "unattended") {
      return refuse(`Refused: ${key} needs a human's approval and nobody is here; a scoped MCP call is never unattended. No server was called.`);
    }
    if (!entry.schemaOk) {
      return refuse(`Refused: server "${entry.server}" advertises a malformed schema for "${entry.tool}"; the call was not sent.`);
    }
    if (!args || typeof args !== "object" || Array.isArray(args)) {
      return refuse(`Refused: ${key} takes an object of arguments. No server was called.`);
    }
    const invalid = scopedArgError(entry.schema, args);
    if (invalid) return refuse(`Refused: ${invalid} No server was called.`);

    try {
      const result = await entry.client.callTool(entry.tool, args);
      const raw = String(result?.output ?? "");
      const kept = raw.slice(0, MCP_SCOPE_LIMITS.maxResultChars);
      const body =
        kept.length < raw.length
          ? `${kept}\n\n[output truncated — ${raw.length - kept.length} characters not shown]`
          : kept;
      return { ok: Boolean(result?.ok), output: labelUntrustedMcp(entry.name, body) };
    } catch (err) {
      // A crashed or hung-over server is a bounded error carrying only its own
      // words, wrapped like any other server text.
      const why = clip(err?.message || err, MCP_SCOPE_LIMITS.maxErrorChars);
      return { ok: false, output: labelUntrustedMcp(entry.name, `the call failed: ${why}`) };
    }
  }

  /**
   * A worker cannot widen its scope. The allowlist is frozen data; this is the
   * loud, named refusal for anything that tries.
   */
  function expand() {
    throw new McpScopeError("scope-fixed", "a task scope is fixed when it is created; a worker cannot add servers or tools to it");
  }

  const scope = {
    kind: "mcp-task-scope",
    sessionId: origin,
    names,
    entries: frozenEntries,
    has,
    filterSchemas,
    toolFor,
    dispatch,
    expand,
    /** Alias of `expand` — every route to widening says the same thing. */
    add: expand,
  };
  Object.freeze(scope);
  SCOPES.add(scope);
  return scope;
}
