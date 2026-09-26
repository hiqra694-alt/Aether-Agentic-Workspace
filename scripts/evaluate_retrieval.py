"""
Standalone retrieval-only evaluation harness.

Loads scripts/eval_dataset.json and, for each query, calls the app's real
`get_relevant_context` (backend/api/documents/services.py) against the live
Supabase + Voyage AI backend. The LLM is never invoked - this measures the
retriever in isolation.

Run from anywhere with:
    python scripts/evaluate_retrieval.py

Requires backend/.env to define SUPABASE_URL, VOYAGE_API_KEY, and
SUPABASE_SERVICE_ROLE_KEY. The service-role key is required (not the anon
key) because get_relevant_context's `match_document_chunks` RPC is scoped by
an explicit filter_user_id argument rather than the caller's JWT, and the
document_chunks table itself denies SELECT to unauthenticated/anon roles.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPTS_DIR.parent / "backend"
EVAL_DATASET_PATH = SCRIPTS_DIR / "eval_dataset.json"

# eval_dataset.json was built from the real document_chunks rows already
# uploaded under this Supabase user; override with EVAL_USER_ID to evaluate
# against a different account's documents.
DEFAULT_EVAL_USER_ID = "de162a33-ebe8-4a7b-b508-f6f22b6d2a40"

# get_relevant_context is session-scoped (migration 003); override with
# EVAL_SESSION_ID if eval_dataset.json's chunks were uploaded under a
# different session.
DEFAULT_EVAL_SESSION_ID = "00000000-0000-4000-8000-000000000001"

TOP_K = 5  # fetch 5 once; Hit Rate@3 is derived by slicing the same ranked list


def _bootstrap_backend_imports():
    """
    Adds backend/ to sys.path (mirroring backend/tests/*.py's own path setup)
    so this script can import api.documents.services exactly as the running
    app does, and loads backend/.env so core.config.get_settings() and the
    service-role key are both available.
    """
    sys.path.insert(0, str(BACKEND_DIR))
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")


_bootstrap_backend_imports()

from supabase import create_client  # noqa: E402
from core.config import get_settings  # noqa: E402
from api.documents import services  # noqa: E402
from api.documents.services import get_relevant_context  # noqa: E402


def _build_supabase_client():
    settings = get_settings()
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not settings.supabase_url or not service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY are missing from "
            "backend/.env. Add SUPABASE_SERVICE_ROLE_KEY (Supabase project "
            "settings -> API -> service_role key) to run this script."
        )
    return create_client(settings.supabase_url, service_role_key)


def _is_hit(entry: dict, retrieved_chunks: list) -> bool:
    return any(
        chunk.document_name == entry["expected_document"]
        and entry["expected_chunk_text"] in chunk.chunk_text
        for chunk in retrieved_chunks
    )


def _reciprocal_rank(entry: dict, retrieved_chunks: list) -> float:
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if (
            chunk.document_name == entry["expected_document"]
            and entry["expected_chunk_text"] in chunk.chunk_text
        ):
            return 1.0 / rank
    return 0.0


def _preflight_check_reranker() -> None:
    """
    Calls services._get_reranker() and scores one trivial pair before any
    eval config runs. _hybrid_rerank's own reranker call is wrapped in a
    try/except that silently falls back to the RRF-fused order on any
    failure (see its docstring), so a reranker that's silently broken
    (e.g. no network to fetch its ONNX weights on first run) would make
    the "Vector + BM25 + rerank" row silently equal the "Vector + BM25
    (RRF)" row instead of failing loudly.
    """
    try:
        reranker = services._get_reranker()
        list(reranker.rerank("preflight check", ["trivial pair"]))
    except Exception as exc:
        print(f"FATAL: reranker failed to load/score during preflight check: {exc}", file=sys.stderr)
        sys.exit(1)


def _raising_reranker() -> None:
    raise RuntimeError("reranker disabled for 'Vector + BM25 (RRF)' eval configuration")


def _make_memoizing_embedder(original_embed_with_retry):
    """
    Wraps services._embed_with_retry with a cache keyed on
    (tuple(texts), input_type). The three configurations differ only in
    post-retrieval ranking, not in what's embedded, so without this the
    same 28 query embeddings get re-requested from Voyage AI once per
    configuration (84 calls instead of 28) -- and re-embedding also risks
    each configuration ranking a subtly different candidate pool if Voyage
    ever returns a non-identical embedding for a repeat call. On a cache
    miss the original function is called, so its existing retry/backoff
    behavior is unchanged.
    """
    cache = {}
    stats = {"calls": 0, "cache_hits": 0}

    def _memoizing_embed_with_retry(texts, input_type, *args, **kwargs):
        stats["calls"] += 1
        key = (tuple(texts), input_type)
        if key in cache:
            stats["cache_hits"] += 1
            return cache[key]
        result = original_embed_with_retry(texts, input_type, *args, **kwargs)
        cache[key] = result
        return result

    return _memoizing_embed_with_retry, stats


async def _run_config(eval_set: list, supabase, user_id: str, session_id: str) -> tuple:
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []

    for entry in eval_set:
        chunks = await get_relevant_context(supabase, entry["query"], user_id, session_id, top_k=TOP_K)

        hit3 = _is_hit(entry, chunks[:3])
        hit5 = _is_hit(entry, chunks[:5])
        rr = _reciprocal_rank(entry, chunks[:5])

        hits_at_3 += int(hit3)
        hits_at_5 += int(hit5)
        reciprocal_ranks.append(rr)

    n = len(eval_set)
    hit_rate_at_3 = hits_at_3 / n if n else 0.0
    hit_rate_at_5 = hits_at_5 / n if n else 0.0
    mrr_at_5 = sum(reciprocal_ranks) / n if n else 0.0
    return hit_rate_at_3, hit_rate_at_5, mrr_at_5


async def evaluate() -> None:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    user_id = os.environ.get("EVAL_USER_ID", DEFAULT_EVAL_USER_ID)
    session_id = os.environ.get("EVAL_SESSION_ID", DEFAULT_EVAL_SESSION_ID)
    supabase = _build_supabase_client()

    _preflight_check_reranker()

    results = []

    original_embed_with_retry = services._embed_with_retry
    memoizing_embed_with_retry, embed_stats = _make_memoizing_embedder(original_embed_with_retry)
    services._embed_with_retry = memoizing_embed_with_retry
    try:
        # "Vector only": the RPC already returns rows ordered by vector
        # similarity, so slicing the top_k off the untouched rows measures
        # vector search alone.
        original_hybrid_rerank = services._hybrid_rerank
        services._hybrid_rerank = lambda query, rows, top_k: rows[:top_k]
        try:
            results.append(("Vector only", await _run_config(eval_set, supabase, user_id, session_id)))
        finally:
            services._hybrid_rerank = original_hybrid_rerank

        # "Vector + BM25 (RRF)": _hybrid_rerank's own except clause already
        # falls back to the RRF-fused order when the reranker raises, so
        # this measures production's own fallback path rather than a
        # separate reimplementation of it.
        original_get_reranker = services._get_reranker
        services._get_reranker = _raising_reranker
        try:
            results.append(("Vector + BM25 (RRF)", await _run_config(eval_set, supabase, user_id, session_id)))
        finally:
            services._get_reranker = original_get_reranker

        # "Vector + BM25 + rerank": no replacement, full production path.
        results.append(("Vector + BM25 + rerank", await _run_config(eval_set, supabase, user_id, session_id)))
    finally:
        services._embed_with_retry = original_embed_with_retry

    n = len(eval_set)
    print(f"| Configuration | Hit Rate@3 | Hit Rate@5 | MRR@5 |")
    print(f"|---|---|---|---|")
    for name, (hit_rate_at_3, hit_rate_at_5, mrr_at_5) in results:
        print(f"| {name} | {hit_rate_at_3:.3f} | {hit_rate_at_5:.3f} | {mrr_at_5:.3f} |")
    print()
    print(f"Query count: {n}, TOP_K: {TOP_K}")
    print(
        f"Embedding calls: {embed_stats['calls']} requested, "
        f"{embed_stats['cache_hits']} served from cache, "
        f"{embed_stats['calls'] - embed_stats['cache_hits']} sent to Voyage AI"
    )


if __name__ == "__main__":
    asyncio.run(evaluate())
