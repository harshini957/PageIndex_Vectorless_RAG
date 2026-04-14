# PageIndex — Technical Deep Dive

> Vectorless, reasoning-based RAG for long structured documents.
> No embeddings. No vector database. No chunking. Pure structural reasoning.

---

## Table of Contents

1. [What is PageIndex?](#1-what-is-pageindex)
2. [Why Not Vector RAG?](#2-why-not-vector-rag)
3. [How PageIndex Works — The Full Pipeline](#3-how-pageindex-works--the-full-pipeline)
4. [How the ToC Tree Is Generated](#4-how-the-toc-tree-is-generated)
5. [The Tree Data Structure](#5-the-tree-data-structure)
6. [The Reasoning Retrieval Loop](#6-the-reasoning-retrieval-loop)
7. [In-Context Index — The Key Innovation](#7-in-context-index--the-key-innovation)
8. [Two Integration Modes](#8-two-integration-modes)
9. [The Chat API (Mode 1)](#9-the-chat-api-mode-1)
10. [Self-Managed Pipeline (Mode 2)](#10-self-managed-pipeline-mode-2)
11. [Pricing and Credits](#11-pricing-and-credits)
12. [Quick Start](#12-quick-start)

---

## 1. What is PageIndex?

PageIndex is a **reasoning-based RAG framework** built by VectifyAI (September 2025). Instead of the standard approach of chunking documents and searching a vector index for semantically similar passages, PageIndex:

1. Transforms a PDF into a **hierarchical tree** (like an intelligent Table of Contents)
2. Uses an **LLM to reason** over that tree to decide which sections are relevant
3. Fetches only the **exact raw content** of those sections
4. Generates a **traceable, cited answer**

The core insight: **retrieval is a navigation problem, not a similarity problem.** When a human expert answers a question from a 200-page document, they don't compute cosine similarity — they read the table of contents, decide which chapter to open, find the right section, and read it. PageIndex teaches LLMs to do the same.

**Benchmark result:** PageIndex-powered systems achieved **98.7% accuracy on FinanceBench**, the hardest financial document QA benchmark — significantly outperforming all vector RAG approaches.

---

## 2. Why Not Vector RAG?

Standard vector RAG has five structural failure modes for long, structured documents:

| Failure Mode | What Happens | Example |
|---|---|---|
| **Query–knowledge mismatch** | Queries express intent; embeddings match surface form | "What did they owe?" vs. "total liabilities: $4.2M" |
| **Similarity ≠ relevance** | Many passages share semantics but differ in relevance | All EGFR drug trials look similar; only one matches |
| **Hard chunking** | Fixed-size chunks sever sentences, tables, clauses | Indemnification clause split across two chunks |
| **No chat history** | Each query is independent; context is lost | "What about liabilities?" after asking about assets |
| **Cross-references missed** | "See Appendix G" has no semantic link to Appendix G | Footnote reference to a table on page 84 |

PageIndex addresses all five by replacing similarity search with structural reasoning.

---

## 3. How PageIndex Works — The Full Pipeline

```
                    Your PDF
                       │
                       ▼
         ┌─────────────────────────┐
         │   PageIndex OCR +       │
         │   Structure Parser      │  ← Detects headings, sections,
         │                         │    hierarchy from the document
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │   JSON ToC Tree         │  ← Hierarchical index stored
         │   (the index)           │    as a JSON file, not vectors
         └────────────┬────────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
     ┌─────────────┐   ┌──────────────┐
     │  Mode 1:    │   │  Mode 2:     │
     │  Chat API   │   │  Self-managed│
     │  (PageIndex │   │  (your LLM + │
     │   LLM)      │   │   PageIndex  │
     └──────┬──────┘   │   retrieval) │
            │          └──────┬───────┘
            └────────┬────────┘
                     ▼
          ┌────────────────────┐
          │  Answer + Citations │
          │  (node_id trace)   │
          └────────────────────┘
```

---

## 4. How the ToC Tree Is Generated

This is the most technically interesting part. PageIndex does **not** use fixed-size chunking. Instead it performs structural document analysis.

### Step 1 — OCR and layout analysis

PageIndex reads the raw PDF and extracts:
- Text content per page
- Font sizes and weights (to identify headings vs. body text)
- Visual layout cues (indentation, whitespace, numbering patterns)
- Existing structural markers (section numbers like `1.`, `1.1`, `A.`, etc.)

### Step 2 — Hierarchy detection

The parser identifies the **heading hierarchy** of the document:

```
Document root
├── Level 1 heading  (e.g. "1. Definitions"  — large bold font)
│   ├── Level 2 heading  (e.g. "1.1 Confidential Information")
│   └── Level 2 heading  (e.g. "1.2 Receiving Party")
├── Level 1 heading  (e.g. "2. Obligations")
│   └── Level 2 heading  (e.g. "2.1 Non-Disclosure")
└── ...
```

This is fundamentally different from chunking at 512 tokens. A section stays together as a **logical unit**, regardless of its length.

### Step 3 — Node creation

Each detected section becomes a **node** with:
- A unique `node_id`
- The section `title`
- The `page_index` where it starts
- The raw `text` content of that section
- An LLM-generated `summary` (used during tree search, not retrieval)
- An array of `nodes` (child sections, recursively)

### Step 4 — Summary generation

For each node, PageIndex generates a short summary that describes what the section is about. This summary is what the LLM reads during tree search — it's compact enough to fit the entire tree in a context window, but informative enough to guide navigation.

---

## 5. The Tree Data Structure

The tree is a **recursive JSON structure**. This is the exact schema PageIndex returns from `get_tree()`:

```json
[
  {
    "node_id": "0001",
    "title": "Agreement Overview",
    "page_index": 1,
    "text": "This Master Services Agreement (the Agreement) is entered into...",
    "summary": "Defines the scope and parties of the MSA between Acme Corp and Beta Ltd.",
    "nodes": []
  },
  {
    "node_id": "0002",
    "title": "Definitions",
    "page_index": 3,
    "text": "For the purposes of this Agreement, the following terms shall have...",
    "summary": "Defines key terms: Confidential Information, Intellectual Property, Services.",
    "nodes": [
      {
        "node_id": "0003",
        "title": "Confidential Information",
        "page_index": 3,
        "text": "Confidential Information means any non-public information disclosed...",
        "summary": "Defines what constitutes Confidential Information and exclusions.",
        "nodes": []
      },
      {
        "node_id": "0004",
        "title": "Intellectual Property",
        "page_index": 4,
        "text": "Intellectual Property means all patents, trademarks, copyrights...",
        "summary": "Defines IP ownership and license grant scope.",
        "nodes": []
      }
    ]
  },
  {
    "node_id": "0005",
    "title": "Obligations of the Parties",
    "page_index": 6,
    "text": "Each party shall fulfil the following obligations...",
    "summary": "Covers non-disclosure duties, data handling, and compliance requirements.",
    "nodes": [
      {
        "node_id": "0006",
        "title": "Non-Disclosure Obligations",
        "page_index": 6,
        "text": "The Receiving Party shall not disclose Confidential Information...",
        "summary": "30-day written notice required; carve-outs for legal compulsion.",
        "nodes": []
      }
    ]
  }
]
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `node_id` | `string` | Unique identifier — used to fetch raw content |
| `title` | `string` | Section heading as it appears in the document |
| `page_index` | `integer` | Page number where this section starts (1-indexed) |
| `text` | `string` | Full raw text content of this section |
| `summary` | `string` | LLM-generated summary (shown during tree search) |
| `nodes` | `array` | Child nodes — same structure, recursively nested |

### Key properties of the tree

- **Lossless**: every word from the original document is preserved in some node's `text` field
- **Navigable**: the tree can be serialised to JSON and loaded directly into an LLM context window
- **Addressable**: any section can be fetched by `node_id` without re-reading the full document
- **Hierarchical**: parent nodes contain semantic context; child nodes contain specific detail

---

## 6. The Reasoning Retrieval Loop

This is the mechanism that replaces vector similarity search.

```
User query arrives
        │
        ▼
┌────────────────────────────────────────────────────┐
│  Load ToC tree (summaries only) into context window│
│  — this is the "in-context index"                  │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│  LLM reads tree and reasons:                       │
│  "Indemnification is in Section 9, node_0024"      │
│                                                    │
│  Returns JSON: { "node_list": ["0024", "0025"] }   │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│  Fetch raw text by node_id                         │
│  — exact section content, not a similarity result  │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│  Is the context sufficient to answer?              │
│  YES → generate answer with citations              │
│  NO  → return to tree, fetch adjacent nodes        │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
              Answer + node_id citations
```

### The tree search prompt (what the LLM actually sees)

```
You are given a question and a tree structure of a document.
Find all node IDs that likely contain the answer to the question.

Question: What is the indemnification clause?

Document tree:
[
  { "node_id": "0001", "title": "Parties", "summary": "Names the contracting parties." },
  { "node_id": "0009", "title": "Indemnification", "summary": "Covers mutual indemnification,
     hold harmless provisions, and liability caps under Section 9." },
  { "node_id": "0010", "title": "Limitation of Liability", "summary": "Caps total liability
     at 12 months of fees paid." },
  ...
]

Reply ONLY as JSON:
{
  "thinking": "The question asks about indemnification. Node 0009 is explicitly titled
               Indemnification and its summary directly matches.",
  "node_list": ["0009"]
}
```

The LLM never sees the full document text during this step — only the compact summaries. This keeps the tree search fast and focused.

---

## 7. In-Context Index — The Key Innovation

The term **in-context index** describes where the ToC tree lives during retrieval.

In vector RAG, the index is an **external database** (Pinecone, Weaviate, ChromaDB). The LLM cannot read it directly — a separate ANN search step queries it and returns results. The LLM only sees the top-k chunks.

In PageIndex, the index is the **JSON tree** loaded directly into the LLM's context window. The LLM can:
- Read all section titles and summaries at once
- Reason about which sections are relevant based on document structure
- Follow cross-references: "Section 9 refers to Appendix B — navigate to Appendix B node"
- Use conversation history: "The user already asked about assets — now they're asking about liabilities, same financial section"

This is why PageIndex calls it "in-context" — the index lives in the same context as the LLM's reasoning, not in an external service.

### Why this beats vector search on structured documents

| Dimension | Vector RAG | PageIndex |
|---|---|---|
| Retrieval unit | Fixed-size chunk (512–1024 tokens) | Logical section (variable size) |
| Retrieval mechanism | Cosine/dot-product similarity | LLM reasoning over structure |
| Index location | External vector DB | LLM context window |
| Cross-reference handling | Not supported | LLM follows references by node navigation |
| Chat history awareness | Not supported | LLM uses prior turns when searching tree |
| Explainability | Similarity score (opaque) | node_id + reasoning trace (transparent) |
| Exact number retrieval | Poor (numbers smeared by embeddings) | Excellent (exact text in node) |

---

## 8. Two Integration Modes

PageIndex offers two ways to integrate — the right choice depends on whether you want PageIndex to handle the LLM or bring your own.

```
Mode 1: Chat API                Mode 2: Self-Managed
─────────────────────           ────────────────────────────
Your app                        Your app
    │                               │
    ▼                               ▼
chat_completions()              get_tree(doc_id)
    │                               │
    ▼                               ▼
PageIndex runs:                 YOUR LLM: tree search
  - tree loading                    │
  - LLM tree search                 ▼
  - node fetching               fetch nodes by node_id
  - LLM answer gen                  │
    │                               ▼
    ▼                           YOUR LLM: answer generation
  Answer                            │
                                    ▼
                                  Answer

You pay: PageIndex credits      You pay: PageIndex indexing
                                         + your LLM API
```

---

## 9. The Chat API (Mode 1)

The simplest path. PageIndex provides the LLM — you just send messages and a `doc_id`.

```python
from pageindex import PageIndexClient

pi = PageIndexClient(api_key="pi-your-key")

response = pi.chat_completions(
    messages=[
        {"role": "user", "content": "What is the indemnification clause?"}
    ],
    doc_id="pi-abc123",
    enable_citations=True,   # adds page refs like <doc=contract.pdf;page=9>
    temperature=0,
)

print(response["choices"][0]["message"]["content"])
```

### Multi-turn conversation

Pass the full conversation history in `messages` — PageIndex uses prior turns when navigating the tree:

```python
messages = [
    {"role": "user",      "content": "What are the payment terms?"},
    {"role": "assistant", "content": "Payment is due net-30 per Section 4.2."},
    {"role": "user",      "content": "What happens if payment is late?"},  # new question
]

response = pi.chat_completions(messages=messages, doc_id=doc_id)
```

### Streaming

```python
for chunk in pi.chat_completions(messages=messages, doc_id=doc_id, stream=True):
    print(chunk, end='', flush=True)
```

### Observing the retrieval steps (stream_metadata)

```python
for chunk in pi.chat_completions(
    messages=messages, doc_id=doc_id,
    stream=True, stream_metadata=True
):
    meta = chunk.get("block_metadata", {})
    if meta.get("type") == "mcp_tool_use_start":
        print(f"\n[Searching: {meta.get('tool_name')}]")
    elif meta.get("type") == "mcp_tool_result_start":
        print(f"\n[Retrieved content]")

    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    if content:
        print(content, end='', flush=True)
```

---

## 10. Self-Managed Pipeline (Mode 2)

You wire PageIndex as a retrieval tool only — your own LLM does the reasoning. This gives you full control over which model is used, the exact prompt, and the cost structure.

```python
import json
import asyncio
import openai  # or anthropic, or google.generativeai
from pageindex import PageIndexClient
import pageindex.utils as utils

pi = PageIndexClient(api_key="pi-your-key")
llm = openai.AsyncOpenAI(api_key="sk-your-openai-key")

async def ask(doc_id: str, query: str) -> str:

    # Step 1 — get the tree from PageIndex (no LLM involved yet)
    tree = pi.get_tree(doc_id, node_summary=True)["result"]

    # Step 2 — strip full text; only give LLM the compact summaries for tree search
    tree_compact = utils.remove_fields(tree.copy(), fields=["text"])

    search_prompt = f"""
You have the structure of a document. Find node IDs likely to answer the question.

Question: {query}

Tree:
{json.dumps(tree_compact, indent=2)}

Reply ONLY as JSON:
{{
  "thinking": "<your reasoning>",
  "node_list": ["node_id_1", "node_id_2"]
}}
"""

    result = await llm.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": search_prompt}],
        temperature=0,
    )
    node_ids = json.loads(result.choices[0].message.content)["node_list"]

    # Step 3 — fetch exact content from selected nodes
    node_map = utils.create_node_mapping(tree)
    context = "\n\n".join(
        f"[{node_map[nid]['title']} | p.{node_map[nid]['page_index']}]\n{node_map[nid]['text']}"
        for nid in node_ids if nid in node_map
    )

    # Step 4 — generate answer with your LLM using the retrieved content
    answer_result = await llm.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}],
        temperature=0,
    )
    return answer_result.choices[0].message.content

answer = asyncio.run(ask("pi-abc123", "What is the governing law?"))
print(answer)
```

### Swapping LLMs

The LLM call in Steps 2 and 4 is a plain async function — swap the provider with one line:

```python
# Anthropic Claude
import anthropic
client = anthropic.Anthropic(api_key="...")
resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               messages=[{"role": "user", "content": prompt}])
return resp.content[0].text

# Ollama (local, no API key)
import httpx
async with httpx.AsyncClient() as c:
    resp = await c.post("http://localhost:11434/api/generate",
                        json={"model": "llama3", "prompt": prompt, "stream": False})
return resp.json()["response"]
```

---

## 11. Pricing and Credits

PageIndex uses a credit system. Credits are consumed for two operations:

| Operation | Cost |
|---|---|
| Page indexing | 1 credit per page (one-time, on upload) |
| Chat API query | 1 credit per query |

### Plans

| Plan | Price | Monthly credits | Max active pages |
|---|---|---|---|
| Free trial | $0 | 200 | 200 |
| Standard | $30/mo | 1,000 | 10,000 |
| Pro | $50/mo | 2,000 | 50,000 |
| Max | $100/mo | 6,000 | 500,000 |

- **Mode 1 (Chat API)**: you pay both indexing credits and query credits to PageIndex
- **Mode 2 (self-managed)**: you only pay indexing credits to PageIndex; LLM calls go to your OpenAI/Anthropic/Google bill

Get your API key at: `https://dash.pageindex.ai/api-keys`

---

## 12. Quick Start

```bash
# Install
pip install pageindex

# Set your key
export PAGEINDEX_API_KEY="pi-your-key-here"
```

```python
import os, time
from pageindex import PageIndexClient

pi = PageIndexClient(api_key=os.environ["PAGEINDEX_API_KEY"])

# 1. Upload and index a PDF
doc = pi.submit_document("./contract.pdf")
doc_id = doc["doc_id"]

# 2. Wait for processing
while pi.get_tree(doc_id).get("status") != "completed":
    time.sleep(4)

# 3. Inspect the tree
tree = pi.get_tree(doc_id, node_summary=True)["result"]
for node in tree:
    print(f"[{node['node_id']}] p.{node['page_index']}  {node['title']}")

# 4. Ask a question
response = pi.chat_completions(
    messages=[{"role": "user", "content": "What is the termination clause?"}],
    doc_id=doc_id,
)
print(response["choices"][0]["message"]["content"])
```

---

## References

- [PageIndex GitHub](https://github.com/VectifyAI/PageIndex)
- [Official documentation](https://docs.pageindex.ai)
- [Technical blog post](https://pageindex.ai/blog/pageindex-intro)
- [FinanceBench benchmark results](https://pageindex.ai/blog/pageindex-intro)
- [Cookbook — Vectorless RAG](https://docs.pageindex.ai/cookbook/vectorless-rag-pageindex)
- [Cookbook — Agentic Vectorless RAG](https://docs.pageindex.ai/cookbook/agentic-vectorless-rag-pageindex)