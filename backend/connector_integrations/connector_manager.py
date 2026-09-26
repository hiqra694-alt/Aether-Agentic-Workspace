"""
Connector Manager.

Owns connections to zero or more *external* MCP servers over the HTTP/SSE
transport, discovers their tool schemas, and merges those schemas alongside
Aether's own native tool registry (api.chat.tools.ALL_TOOLS) into one
list shaped for the LLM -- optionally filtered down to a per-request set of
enabled connectors (Phase 3).

Static connectors only (e.g. a generic MCP_SERVER_URLS entry such as Brave
Search): connected once at app startup, held in the MCPClientManager
singleton's `_connections` for the process's lifetime, and shared read-only
across every request.

The Google Managed MCP (Workspace/Gmail) and GitHub MCP connectors that
used to also live here -- each opened as a short-lived, per-user session
via create_workspace_session/create_github_session/create_gmail_mcp_session
-- have been removed: Google Workspace is being rebuilt as native Python
tools instead (see connector_integrations/intent_router.py, repurposed to
route those), and GitHub is dropped entirely. The frontend's
google_gmail/google_calendar/google_drive toggles, and the
backend/api/connectors OAuth-token endpoints and user_oauth_tokens table
they rely on, are unaffected by this removal and are reused as-is by
whatever replaces this wiring.

Strictly additive: nothing in this module mutates the existing Groq
provider, RAG pipeline, native tools, or Supabase wiring. Each external
server is identified by a stable `connector_id` (e.g. "brave_search")
rather than its raw URL, so the frontend can toggle connectors on/off by a
human-readable id and a server's tools can be traced back to the connector
that offers them.
"""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

from core.config import get_settings

logger = logging.getLogger(__name__)

# Ceiling on how long any single MCP server's connect()+initialize()+
# list_tools() handshake may take, applied via asyncio.wait_for everywhere
# a connection is opened. Without this, a server that accepts the TCP/SSE
# connection but never completes (or never responds to) the MCP handshake
# -- e.g. a misconfigured URL pointing at a port nothing is actually
# serving MCP on -- hangs that await forever: at startup that means the
# whole FastAPI app never finishes starting, and mid-request it means the
# user's HTTP request never completes either. A plain connection refused/
# DNS failure already fails fast on its own; this constant exists for the
# slower, hung-not-failed case those don't cover.
CONNECT_TIMEOUT_SECONDS = 10.0


@dataclass
class MCPServerSpec:
    """One external MCP server to connect to: a stable id the frontend/
    config can refer to it by, its SSE endpoint, and any headers the
    connection needs."""
    connector_id: str
    url: str
    headers: Optional[dict[str, str]] = field(default=None)


async def _call_tool_on_session(session: ClientSession, tool_name: str, arguments: dict) -> str:
    """
    Shared execution + CallToolResult-parsing logic used by every connected,
    singleton-registered connector's session. Returns the result as a
    string ready to drop straight into a `role: "tool"` message, the same
    way api.chat.tools.execute_tool's return value already is.

    Never raises: a JSON-RPC error or transport failure degrades to a JSON
    error string instead of blowing up the calling turn, matching how
    every native tool in api.chat.tools already degrades on failure.
    """
    try:
        result = await session.call_tool(tool_name, arguments)
    except Exception:
        logger.exception(f"MCP: call_tool('{tool_name}') failed.")
        return json.dumps({"error": f"MCP tool '{tool_name}' execution failed."})

    text_parts = [block.text for block in result.content if isinstance(block, types.TextContent)]
    result_text = "\n".join(text_parts) if text_parts else ""

    # NOTE: the SDK exposes the wire-level `isError` field under that same
    # camelCase name as the model attribute -- there is no snake_case
    # `is_error` alias, same as Tool.inputSchema below.
    if getattr(result, "isError", False):
        logger.warning(f"MCP: tool '{tool_name}' returned an error result: {result_text}")
        return json.dumps({"error": result_text or f"MCP tool '{tool_name}' returned an error."})

    if not result_text:
        return json.dumps({"result": f"MCP tool '{tool_name}' executed but returned no text content."})

    return result_text


class MCPServerConnection:
    """A single live session to one external MCP server, plus its cached
    tool schemas. `connect`/`close` must be awaited from the same asyncio
    task (true of a FastAPI lifespan context manager, which runs its
    startup and shutdown halves in one task -- and equally true of a
    single chat request's async generator) -- the underlying transport
    uses anyio task groups that aren't safe to enter in one task and exit
    in another."""

    def __init__(
        self,
        connector_id: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
    ):
        self.connector_id = connector_id
        self.url = url
        self.headers = headers
        self.session: Optional[ClientSession] = None
        self.tools: list[types.Tool] = []
        self._exit_stack = contextlib.AsyncExitStack()

    async def connect(self) -> None:
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client(self.url, headers=self.headers)
        )
        self.session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools = result.tools

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self.session = None

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self.session is None:
            logger.warning(f"MCP: call_tool('{tool_name}') on connector '{self.connector_id}' with no active session.")
            return json.dumps({"error": f"MCP connector '{self.connector_id}' has no active session."})
        return await _call_tool_on_session(self.session, tool_name, arguments)


class MCPClientManager:
    """
    Accepts a list of MCPServerSpec (external MCP servers over HTTP/SSE,
    e.g. a generic MCP_SERVER_URLS entry such as Brave Search) and manages
    connecting to all of them, caching their advertised tools per
    connector_id, and shutting every session down cleanly. A single server
    being unreachable is logged and skipped rather than raised -- one bad
    MCP endpoint must never prevent the FastAPI app itself from starting.
    """

    def __init__(self, servers: Optional[list[MCPServerSpec]] = None):
        self.server_specs: list[MCPServerSpec] = list(servers or [])
        self._connections: dict[str, MCPServerConnection] = {}  # keyed by connector_id

    async def initialize(self) -> None:
        for spec in self.server_specs:
            connection = MCPServerConnection(spec.connector_id, spec.url, spec.headers)
            try:
                await asyncio.wait_for(connection.connect(), timeout=CONNECT_TIMEOUT_SECONDS)
            except Exception:
                logger.exception(f"MCP: failed to connect connector '{spec.connector_id}' at '{spec.url}'; skipping.")
                await connection.close()
                continue
            self._connections[spec.connector_id] = connection
            logger.info(
                f"MCP: connected connector '{spec.connector_id}' ({spec.url}) -- "
                f"discovered {len(connection.tools)} tool(s)."
            )

    async def shutdown(self) -> None:
        for connection in self._connections.values():
            try:
                await connection.close()
            except Exception:
                logger.exception(f"MCP: error while closing connector '{connection.connector_id}'.")
        self._connections.clear()

    def list_connector_ids(self) -> list[str]:
        """Every connector_id currently connected."""
        return list(self._connections.keys())

    def list_cached_tools(self, connector_ids: Optional[Iterable[str]] = None) -> list[types.Tool]:
        """Every MCP tool discovered across connected servers, flattened
        into one list. When `connector_ids` is given, only tools from
        connectors in that set are included -- None (the default) means
        every connected connector, preserving the pre-filtering behavior.
        Read-only against the cache populated by initialize() -- never
        opens a connection itself."""
        allowed = set(connector_ids) if connector_ids is not None else None
        tools: list[types.Tool] = []
        for connector_id, connection in self._connections.items():
            if allowed is not None and connector_id not in allowed:
                continue
            tools.extend(connection.tools)
        return tools

    def get_connection_for_tool(self, tool_name: str) -> Optional[MCPServerConnection]:
        """The MCPServerConnection that owns `tool_name`, for the
        execution layer to route a call to the right server. None if no
        connected server currently advertises that tool. Deliberately
        ignores connector filtering -- a tool that was actually offered to
        the model this turn (i.e. passed the enabled_connectors filter
        when the schema list was built) must always be executable, so this
        looks across every connected connector, not just an enabled
        subset."""
        for connection in self._connections.values():
            if any(t.name == tool_name for t in connection.tools):
                return connection
        return None

    def get_session_for_tool(self, tool_name: str) -> Optional[ClientSession]:
        """The ClientSession that owns `tool_name` -- see
        get_connection_for_tool. Kept as a thin convenience wrapper for
        anything that only needs the raw session."""
        connection = self.get_connection_for_tool(tool_name)
        return connection.session if connection else None

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Executes `tool_name` on whichever connected, singleton-registered
        MCP server advertised it."""
        connection = self.get_connection_for_tool(tool_name)
        if connection is None:
            logger.warning(f"MCP: call_tool('{tool_name}') requested but no connected server advertises it.")
            return json.dumps({"error": f"No connected MCP server advertises tool '{tool_name}'."})
        return await connection.call_tool(tool_name, arguments)


def _build_default_server_specs() -> list[MCPServerSpec]:
    """
    Reads Settings to build the list of STATIC MCP servers to connect to
    at startup: `mcp_server_urls` (generic, unnamed URLs) each get an
    auto-assigned connector_id ("mcp_server_1", "mcp_server_2", ...).
    """
    settings = get_settings()
    specs: list[MCPServerSpec] = []

    for i, url in enumerate(settings.mcp_server_url_list, start=1):
        specs.append(MCPServerSpec(connector_id=f"mcp_server_{i}", url=url))

    return specs


# Single shared instance for the whole app -- main.py's lifespan calls
# initialize()/shutdown() on exactly this object, and api.chat.services
# imports this same instance to merge schemas and route tool calls.
# Defined here (rather than in main.py) so importing it never risks a
# circular import through main -> api.chat.routers -> api.chat.services.
mcp_manager = MCPClientManager(servers=_build_default_server_specs())


def mcp_tool_to_native_schema(tool: types.Tool) -> dict[str, Any]:
    """Converts an MCP `Tool` (name/description/inputSchema -- the SDK
    exposes the wire-level field under that same camelCase name as the
    model attribute, with no snake_case alias) into this app's native tool
    schema shape -- {"type": "function", "function": {...}} -- matching
    api.chat.tools.ALL_TOOLS, so both can sit in the same list handed to
    the LLM."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def get_merged_tool_schemas(
    native_tools: list[dict[str, Any]],
    manager: MCPClientManager,
    enabled_connectors: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Merges Aether's hardcoded native tool schemas with the cached
    schemas of tools discovered from connected MCP servers into one unified
    list, in the same shape the LLM provider expects.

    Native tools are always included -- `enabled_connectors` only ever
    filters which *MCP* connectors' tools are offered this turn, never the
    native registry. When `enabled_connectors` is None (the default), every
    connected connector's tools are included, exactly matching pre-Phase-3
    behavior -- so an old caller, or a request that never sends the field,
    is unaffected.

    A native tool name always wins on collision -- native tools are this
    app's own audited implementations, so an external server advertising
    e.g. "calculator" never shadows them.
    """
    native_names = {t["function"]["name"] for t in native_tools}
    merged = list(native_tools)
    for tool in manager.list_cached_tools(enabled_connectors):
        if tool.name in native_names:
            logger.warning(
                f"MCP: external tool '{tool.name}' collides with a native tool name; keeping the native one."
            )
            continue
        merged.append(mcp_tool_to_native_schema(tool))
    return merged
