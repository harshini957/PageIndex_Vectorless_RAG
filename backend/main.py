"""
Contract IQ — Vectorless RAG backend powered by PageIndex
FastAPI server that handles contract upload, indexing, and Q&A
"""

import os
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile

from dotenv import load_dotenv
from pageindex import PageIndexClient

# ── Load environment variables from .env ─────────────────────────────────────
# load_dotenv() reads the .env file in the project root and injects every
# key=value pair into os.environ before anything else runs.
# If a variable is already set in the shell environment, it is NOT overwritten
# (override=False is the default) — so production env vars always win.
load_dotenv()

# ── Validate required keys at startup so errors are obvious immediately ───────
PAGEINDEX_API_KEY = os.environ.get("PAGEINDEX_API_KEY", "")
if not PAGEINDEX_API_KEY:
    raise RuntimeError(
        "PAGEINDEX_API_KEY is not set.\n"
        "Add it to your .env file:  PAGEINDEX_API_KEY=pi-your-key-here"
    )

# ── Init ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Contract IQ", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

# In-memory store: doc_id → metadata (replace with DB in production)
documents: dict[str, dict] = {}

# ── Models ────────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    doc_id: str
    question: str
    conversation_history: Optional[list[dict]] = []

class CompareRequest(BaseModel):
    doc_ids: list[str]
    question: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_contract(file: UploadFile = File(...)):
    """
    Upload a PDF contract.
    PageIndex processes it into a hierarchical ToC tree — no chunking, no embeddings.
    Returns doc_id immediately; processing happens async.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Write upload to temp file (PageIndex SDK takes a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pi_client.submit_document(tmp_path)
        doc_id = result["doc_id"]

        documents[doc_id] = {
            "doc_id": doc_id,
            "filename": file.filename,
            "status": "processing",
            "uploaded_at": time.time(),
            "tree": None,
        }
        return {"doc_id": doc_id, "filename": file.filename, "status": "processing"}
    finally:
        os.unlink(tmp_path)


@app.get("/document/{doc_id}/status")
async def get_document_status(doc_id: str):
    """
    Poll processing status and retrieve the ToC tree once ready.
    The tree is a JSON hierarchy — section titles, node_ids, page numbers, summaries.
    No vector index is built — the tree IS the index.
    """
    result = pi_client.get_tree(doc_id, node_summary=True)
    status = result.get("status", "processing")

    if status == "completed" and doc_id in documents:
        documents[doc_id]["status"] = "completed"
        documents[doc_id]["tree"] = result.get("result", [])

    return {
        "doc_id": doc_id,
        "status": status,
        "filename": documents.get(doc_id, {}).get("filename"),
        "tree_node_count": len(result.get("result", [])) if status == "completed" else None,
    }


@app.get("/document/{doc_id}/tree")
async def get_document_tree(doc_id: str):
    """
    Return the full hierarchical tree structure PageIndex generated.
    This is a human-readable, LLM-navigable JSON tree — not a vector space.
    Each node has: node_id, title, page_index, text, summary, sub-nodes.
    """
    result = pi_client.get_tree(doc_id, node_summary=True)
    if result.get("status") != "completed":
        raise HTTPException(202, "Document still processing")
    return {"doc_id": doc_id, "tree": result.get("result", [])}


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """
    Ask a question about a contract using PageIndex reasoning-based RAG.

    Under the hood PageIndex:
    1. Loads the ToC tree into the LLM context window (in-context index)
    2. LLM reasons: which node(s) to navigate to?
    3. Fetches raw content by node_id
    4. Iterates if context insufficient
    5. Returns answer with node citations (fully traceable)

    No vector similarity. No embeddings. Pure structural reasoning.
    """
    # Build messages — PageIndex chat API is multi-turn aware
    messages = list(req.conversation_history) + [
        {"role": "user", "content": req.question}
    ]

    response = pi_client.chat_completions(
        messages=messages,
        doc_id=req.doc_id,
    )

    answer = response["choices"][0]["message"]["content"]
    return {
        "answer": answer,
        "doc_id": req.doc_id,
        "question": req.question,
    }


@app.post("/ask/stream")
async def ask_question_stream(req: QuestionRequest):
    """Streaming version — tokens appear as they're generated."""
    messages = list(req.conversation_history) + [
        {"role": "user", "content": req.question}
    ]

    def generate():
        for chunk in pi_client.chat_completions(
            messages=messages,
            doc_id=req.doc_id,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/compare")
async def compare_contracts(req: CompareRequest):
    """
    Cross-document reasoning — compare two contracts side by side.
    PageIndex handles multi-doc retrieval natively.
    """
    if len(req.doc_ids) < 2:
        raise HTTPException(400, "Provide at least 2 doc_ids to compare")

    response = pi_client.chat_completions(
        messages=[{"role": "user", "content": req.question}],
        doc_id=req.doc_ids,
    )
    return {
        "answer": response["choices"][0]["message"]["content"],
        "compared_docs": req.doc_ids,
    }


@app.get("/documents")
async def list_documents():
    """List all uploaded contracts."""
    return {"documents": list(documents.values())}


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete contract and its PageIndex tree."""
    pi_client.delete_document(doc_id)
    documents.pop(doc_id, None)
    return {"deleted": doc_id}