"""
Temporary diagnostic: for the queries that scored 0.000 RR@5 in
scripts/evaluate_retrieval.py, determine whether the miss is a RECALL
FAILURE (the expected chunk never surfaces in a large raw vector candidate
pool, so no amount of local reranking could have found it) or a RERANKER
FAILURE (the expected chunk is present in the vector pool but ranked below
what the hybrid rerank stage in api.documents.services.get_relevant_context
returned).

Bypasses get_relevant_context entirely and calls the match_document_chunks
RPC directly with a large top_k, so BM25/RRF/cross-encoder reranking never
runs here — this isolates the vector search stage in isolation.

Run from anywhere with:
    python scripts/diagnose_misses.py

Requires the same backend/.env as evaluate_retrieval.py (SUPABASE_URL,
VOYAGE_API_KEY, SUPABASE_SERVICE_ROLE_KEY).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPTS_DIR.parent / "backend"
EVAL_DATASET_PATH = SCRIPTS_DIR / "eval_dataset.json"

DEFAULT_EVAL_USER_ID = "de162a33-ebe8-4a7b-b508-f6f22b6d2a40"
DEFAULT_EVAL_SESSION_ID = "00000000-0000-4000-8000-000000000001"

# The 9 queries that scored 0.000 RR@5 in the last evaluate_retrieval.py run.
FAILING_IDS = {
    "eval_008", "eval_010", "eval_019", "eval_020",
    "eval_021", "eval_022", "eval_023", "eval_024", "eval_028",
}

TOP_25 = 25
TOP_100 = 100


def _bootstrap_backend_imports():
    sys.path.insert(0, str(BACKEND_DIR))
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")


_bootstrap_backend_imports()

from supabase import create_client  # noqa: E402
from core.config import get_settings  # noqa: E402
from api.documents.services import _embed_with_retry, DEFAULT_MATCH_THRESHOLD  # noqa: E402


def _build_supabase_client():
    settings = get_settings()
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not settings.supabase_url or not service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY are missing from "
            "backend/.env."
        )
    return create_client(settings.supabase_url, service_role_key)


def _raw_vector_search(supabase, query: str, user_id: str, session_id: str, top_k: int) -> list[dict]:
    """
    Calls match_document_chunks directly (no BM25/RRF/reranking), mirroring
    exactly what get_relevant_context used to do before the hybrid rerank
    stage was added.
    """
    query_embedding = _embed_with_retry([query], input_type="query")[0]
    res = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": DEFAULT_MATCH_THRESHOLD,
            "match_count": top_k,
            "filter_user_id": user_id,
            "filter_session_id": session_id,
            "filter_document_name": None,
        },
    ).execute()
    return res.data or []


def _find_rank(entry: dict, rows: list[dict]) -> int | None:
    """1-indexed rank of the expected chunk in rows, or None if absent."""
    for rank, row in enumerate(rows, start=1):
        if (
            row["document_name"] == entry["expected_document"]
            and entry["expected_chunk_text"] in row["chunk_text"]
        ):
            return rank
    return None


async def diagnose() -> None:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    failing_entries = [e for e in eval_set if e["id"] in FAILING_IDS]
    missing_ids = FAILING_IDS - {e["id"] for e in failing_entries}
    if missing_ids:
        print(f"WARNING: ids not found in eval_dataset.json: {sorted(missing_ids)}")

    user_id = os.environ.get("EVAL_USER_ID", DEFAULT_EVAL_USER_ID)
    session_id = os.environ.get("EVAL_SESSION_ID", DEFAULT_EVAL_SESSION_ID)
    supabase = _build_supabase_client()

    rows_out = []
    for entry in failing_entries:
        rows_100 = _raw_vector_search(supabase, entry["query"], user_id, session_id, TOP_100)
        rank = _find_rank(entry, rows_100)

        in_top_25 = "Yes" if (rank is not None and rank <= TOP_25) else "No"
        if rank is None:
            in_top_100 = "No"
            diagnosis = "RECALL FAILURE"
        else:
            in_top_100 = f"Rank {rank}"
            diagnosis = "RECALL FAILURE" if rank > TOP_25 else "RERANKER FAILURE"

        rows_out.append((entry["id"], in_top_25, in_top_100, diagnosis))

    col_id, col_25, col_100, col_diag = 10, 14, 14, 18
    print(f"{'Query ID':<{col_id}} {'Top 25?':<{col_25}} {'Top 100 (rank)':<{col_100}} {'Diagnosis':<{col_diag}}")
    print("-" * (col_id + col_25 + col_100 + col_diag))
    for qid, in25, in100, diag in rows_out:
        print(f"{qid:<{col_id}} {in25:<{col_25}} {in100:<{col_100}} {diag:<{col_diag}}")

    print()
    recall_failures = sum(1 for *_ , d in rows_out if d == "RECALL FAILURE")
    reranker_failures = len(rows_out) - recall_failures
    print(f"Recall failures (not in top {TOP_100}, or beyond top {TOP_25} with no rerank recourse): {recall_failures}")
    print(f"Reranker failures (in top {TOP_25} but hybrid rerank still dropped them): {reranker_failures}")


if __name__ == "__main__":
    asyncio.run(diagnose())
