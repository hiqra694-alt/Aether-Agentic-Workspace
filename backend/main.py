import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat.routers import router as chat_router
from api.chat.tools import ALL_TOOLS
from api.connectors.routers import router as connectors_router
from api.connectors.google_oauth import router as google_oauth_router
from api.documents.routers import router as documents_router
from api.memory.routers import router as memory_router
from api.tasks.routers import router as tasks_router
from core.scheduler import run_scheduler_loop
from connector_integrations.connector_manager import mcp_manager, get_merged_tool_schemas
from canvas.router import router as canvas_router
from voice.router import router as voice_router

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MCP Phase 1 (Foundation): connect to any configured external MCP
    # servers and cache their tool schemas. Purely additive -- does not
    # touch Groq routing, Supabase, or any existing native tool. If no
    # server URLs are configured yet, this is a no-op.
    await mcp_manager.initialize()

    # Startup diagnostic: log the schema list as of connect-time. As of
    # Phase 2, ChatService.stream_chat computes this same merge fresh on
    # every turn (see api/chat/services.py) and passes it to the provider --
    # this log is just a one-time snapshot for verifying what connected.
    merged_schemas = get_merged_tool_schemas(ALL_TOOLS, mcp_manager)
    logger.info(
        "MCP: merged tool schema list at startup (native + MCP):\n"
        f"{json.dumps(merged_schemas, indent=2)}"
    )

    # Phase 5: background task/reminder scheduler. Runs as a plain
    # asyncio.Task alongside request handling (never blocking it) and is
    # stopped via an asyncio.Event rather than task.cancel(), so a poll that's
    # mid-flight when shutdown begins finishes cleanly instead of being cut
    # off partway through a DB call.
    scheduler_stop_event = asyncio.Event()
    scheduler_task = asyncio.create_task(run_scheduler_loop(scheduler_stop_event))

    yield

    scheduler_stop_event.set()
    await scheduler_task

    await mcp_manager.shutdown()


app = FastAPI(title="Aether API", lifespan=lifespan)


@app.get("/", tags=["health"])
async def health():
    return {"status": "ok", "service": "Aether API"}


@app.get("/api/health", tags=["health"])
async def api_health():
    return {"status": "ok", "service": "Aether API"}


# Restrict CORS to production domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aether-agentic-workspace.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat_router)
app.include_router(connectors_router)
# Unified Google Workspace OAuth (Phase 2) -- native Gmail/Calendar/Drive
# tools' auth flow, see api/connectors/google_oauth.py. Separate router from
# connectors_router (same /api/connectors prefix family) since it owns its
# own /google/authorize + /google/callback routes.
app.include_router(google_oauth_router)
app.include_router(documents_router)
app.include_router(memory_router)
app.include_router(tasks_router)
# Canvas feature (Phase 1) -- fully isolated Google Drive OAuth + export
# routes, see canvas/router.py. Never touches connector_integrations/.
app.include_router(canvas_router)
# Voice agent (Phase 1) -- fully isolated LiveKit token-issuance route, see
# voice/router.py. Never touches api/chat/routers.py, the RAG pipeline, or
# any MCP connector.
app.include_router(voice_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
