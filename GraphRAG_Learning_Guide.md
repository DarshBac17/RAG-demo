# Graph-RAG Travel Chatbot — Learning Guide

*A complete walkthrough of RAG, Graph-RAG, vector databases, and LLM integration, built hands-on in Google Colab with the free Groq API.*

---

## 1. What This Project Is

A small, request-response travel chatbot (no UI — just `input()`/`print()` in a Colab notebook) that:

1. Generates day-by-day itineraries for a place the user asks about
2. Answers factual questions about places
3. Compares travel options with real cost/duration numbers

It's built specifically as a **teaching example** — every moving part is visible and hand-wired, nothing is hidden behind a framework, so each concept below can be traced directly back to a specific cell in the notebook.

---

## 2. Core Concepts

### RAG (Retrieval-Augmented Generation)

Instead of asking an LLM to answer purely from what it memorized during training, RAG retrieves relevant information *right before* generation and hands it to the model as context. This reduces hallucination and lets the bot answer using facts it was never trained on (like our dummy travel data).

### Graph RAG

Plain RAG retrieves by **similarity search only** — embed the query, find the nearest vectors. Graph RAG adds a second retrieval mechanism: a **knowledge graph** of typed relationships (`city → HAS_ATTRACTION → attraction`, `city → TRANSPORT_TO → transport_option`, `city → HAS_CHUNK → wikipedia_chunk`). After similarity search finds "seed" nodes, the system **walks the graph** one hop outward from each seed, pulling in connected nodes that might not be textually similar to the query at all.

Example: asking about "Paris" finds the `paris` node by similarity, and the graph walk then pulls in the Eiffel Tower, the Louvre, and flight options — none of which necessarily share vocabulary with the word "Paris" alone.

### Chunking

Long text (a full Wikipedia article) can't be embedded as one meaningful vector — it's too large, and a single vector for a huge document ends up representing a vague "average of everything." **Chunking** splits long text into smaller, individually-embeddable pieces (in this project: ~80-word windows fetched from Wikipedia's intro section, each linked back to its city node in the graph).

### Embeddings & Vector Database

An **embedding** is a fixed-length list of numbers (384 of them, here) representing a piece of text's meaning — texts with similar meaning produce vectors that are mathematically close together. A **vector database** (ChromaDB) stores these vectors and can quickly find the closest ones to a new query vector — this is what "semantic search" actually means under the hood.

---

## 3. End-to-End Architecture

```
User query
   │
   ▼
[1] Embed query ─────────────► sentence-transformers (all-MiniLM-L6-v2)
   │
   ▼
[2] Vector similarity search ─► ChromaDB → "seed" nodes
   │
   ▼
[3] Graph expansion (1-hop) ──► networkx → related nodes (attractions, transport, chunks)
   │                              (capped at max_context_nodes to avoid hub-node explosion)
   ▼
[4] Build context string ─────► seed + expanded nodes' text, joined
   │
   ▼
[5] Pick a prompt template ───► itinerary / place-info / budget-comparison
   │
   ▼
[6] Call the LLM ─────────────► Groq API (openai/gpt-oss-20b)
   │
   ▼
Response to user
```

---

## 4. Data Layer

Four kinds of hand-written records, each with a short `description` string:

- **`city`** — a destination (Paris, Tokyo, Goa, Delhi)
- **`attraction`** — a place to visit inside a city, with entry cost and time needed
- **`transport`** — a way to travel between two cities, with **exact** `cost_usd` and `duration_hours`
- **`category`** — a tag like `landmark`, `culture`, `nature`

**Why the numbers stay hand-written instead of scraped:** an LLM asked to compare transport costs needs *exact, structured* numbers — a Wikipedia article won't reliably contain "Delhi to Goa train costs $35," and even if it did, prose is the wrong format for a cost-comparison table. Structured facts and long-form narrative text are different kinds of data, solved by different mechanisms — see the "chunking doesn't replace structured data" note in section 9.

---

## 5. The Knowledge Graph

Built with `networkx.DiGraph()` — a **directed** graph, since relationships are inherently one-way ("Paris has the Eiffel Tower" makes sense; the reverse doesn't).

| Edge | Meaning |
|---|---|
| `city → HAS_ATTRACTION → attraction` | this city contains this attraction |
| `attraction → HAS_CATEGORY → category` | this attraction is tagged as this type |
| `city → TRANSPORT_TO → transport → TRANSPORT_TO → city` | this route connects these two cities |
| `city → HAS_CHUNK → wikipedia_chunk` | this piece of real-world text is about this city |

`G.successors(node)` = what this node points *to*. `G.predecessors(node)` = what points *to* this node. The retrieval function checks **both directions**, because depending on which node the vector search hits, the useful related information could be either "downstream" (a city → its attractions) or "upstream" (an attraction → its city).

---

## 6. Real Content + Chunking

Rather than relying only on hand-written city blurbs, the notebook fetches real Wikipedia content:

```python
params = {
    "action": "query", "prop": "extracts",
    "explaintext": True,
    "exintro": True,       # lead section only — NOT the whole article
    "titles": title, "format": "json",
}
headers = {"User-Agent": "GraphRAGTravelChatbot/1.0 (educational Colab notebook)"}
```

Two important lessons learned here (see section 9 for the full story):
- **Wikipedia requires a `User-Agent` header** — requests without one get a 403, regardless of query correctness.
- **`exintro: True` matters a lot** — without it, a full Wikipedia article (tens of thousands of words) gets chunked into 200+ pieces, and a hub node with 200+ graph edges can blow up the retrieval context and crash generation with an out-of-memory error.

Each chunk becomes its own graph node (`type="chunk"`), connected to its city with a `HAS_CHUNK` edge — so vector search can find a specific fact buried in real prose, and the graph walk can still connect it back to the right city.

---

## 7. Embeddings & Vector Store

```python
embed_model = SentenceTransformer("all-MiniLM-L6-v2")   # free, local, 384-dim vectors
collection = chroma_client.get_or_create_collection(name="travel_nodes")

documents  = [...]   # text to embed, one per graph node
metadatas  = [...]   # extra structured fields (type, name)
embeddings = embed_model.encode(documents).tolist()      # matrix: N documents × 384 dims

collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
```

`encode(documents)` returns an **N × 384 matrix** — N rows (one per document), 384 columns (the fixed embedding size for this model). `.tolist()` just converts it from a NumPy array to plain Python lists, which is the format Chroma's API expects. Each *row* of that matrix is one node's vector, and it's stored tied to that node's id, text, and metadata.

**Important operational lesson:** the vector store and the graph must stay in sync. If you rebuild `G` (e.g., re-running the graph or chunking cells) without also rebuilding the vector store, Chroma keeps stale entries from the old graph — leading to a `KeyError` when retrieval finds a seed node that no longer exists in `G`. The fix used here: reset the Chroma collection every time this cell runs, and defensively skip any id not currently in `G`.

---

## 8. Graph-RAG Retrieval Function

```python
def graph_rag_retrieve(query, top_k=3, hops=1, max_context_nodes=15, verbose=False):
    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    seed_ids = [nid for nid in results["ids"][0] if nid in G]      # drop stale ids

    expanded_ids = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(hops):
        next_frontier = set()
        for nid in frontier:
            if nid in G:
                next_frontier.update(G.successors(nid))
                next_frontier.update(G.predecessors(nid))
        expanded_ids.update(next_frontier)
        frontier = next_frontier

    # Cap total context size — prevents a high-degree hub node (e.g. a city with
    # hundreds of chunk neighbors) from blowing up the prompt size
    if len(expanded_ids) > max_context_nodes:
        non_seed = expanded_ids - set(seed_ids)
        prioritized = sorted(non_seed, key=lambda nid: 0 if G.nodes[nid]["type"] != "chunk" else 1)
        keep = max(max_context_nodes - len(seed_ids), 0)
        expanded_ids = set(seed_ids) | set(prioritized[:keep])

    context_lines = [
        f"{'[SEED]' if nid in seed_ids else '[GRAPH-LINKED]'} "
        f"({G.nodes[nid]['type']}) {G.nodes[nid].get('name','')}: {G.nodes[nid].get('description','')}"
        for nid in expanded_ids if nid in G
    ]
    return "\n".join(context_lines)
```

**What each safeguard is for:**
- `if nid in G` (twice) — stale vector-store entries never crash the function
- `max_context_nodes` cap — bounds the prompt size regardless of how many neighbors any single node has, prioritizing compact structured facts (attractions, transport) over generic text chunks when trimming

---

## 9. Groq API Integration (the LLM layer)

### Why Groq

Groq is a hardware company (not to be confused with xAI's *Grok*) that runs open-weight models on custom inference chips. It offers a genuinely free API tier, is very fast, and — critically for this project — supports much larger output limits than a small model run locally on a free Colab GPU, without any risk of local out-of-memory errors.

### Setup

```python
from openai import OpenAI  # Groq's API is OpenAI-compatible

groq_client = OpenAI(api_key="YOUR_FREE_GROQ_API_KEY", base_url="https://api.groq.com/openai/v1")

def call_llm(system_prompt, user_prompt, temperature=0.4, max_new_tokens=1200):
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content
```

### Choosing a model — a real lesson in API integration

Guessing a model name (`llama-3.3-70b-versatile`) failed with a 404 — model catalogs change over time and by account. The reliable approach:

```python
models = groq_client.models.list()
for m in models.data:
    print(m.id)
```

This lists every model *your specific key* can access. From that list, models had to be sorted by actual purpose:

| Type of model | Examples from the list | Usable for chat? |
|---|---|---|
| Speech-to-text | `whisper-large-v3`, `whisper-large-v3-turbo` | ❌ |
| Text-to-speech | `canopylabs/orpheus-v1-english` | ❌ |
| Safety/classifier models | `meta-llama/llama-prompt-guard-2-*`, `openai/gpt-oss-safeguard-20b` | ❌ |
| General chat/instruct | **`openai/gpt-oss-20b`**, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `allam-2-7b` | ✅ |

**Lesson:** an API's model list often mixes chat models with audio, safety, and specialty models under one endpoint — always check what a model is *for*, not just whether it's listed.

### `max_new_tokens` — a subtle bug

A 7-day itinerary request returned only 4 days, cut off mid-sentence. The cause: `max_new_tokens=250` was a hard ceiling on output length, left over from an earlier fix for a different problem (local GPU memory pressure). The model wasn't choosing to stop — it ran out of budget. Fix: raise the limit generously (`1200`), and optionally scale it per-intent:

```python
max_tokens = 1200 if intent == "itinerary" else 300   # itineraries need much more room
```

---

## 10. Prompts

```python
SYSTEM_PROMPT = (
    "You are a helpful travel assistant. Answer ONLY using the CONTEXT provided below. "
    "If the context does not contain enough information, say so honestly instead of making facts up. "
    "Be concise, practical, and use bullet points where helpful."
)
```

This is the **grounding instruction** — it tells the model to rely on retrieved context rather than its own training data, directly reducing hallucination. Three task-specific templates (itinerary / place-info / budget) shape the output format for each use case, with the retrieved context and the user's query injected via `.format(context=..., query=...)`.

---

## 11. Orchestration

```python
def route_intent(query):
    q = query.lower()
    if any(w in q for w in ["itinerary", "plan", "day", "days", "schedule"]):
        return "itinerary"
    if any(w in q for w in ["cost", "budget", "cheap", "price", "flight", "train", "bus"]):
        return "budget"
    return "place_info"

def travel_chatbot(query, top_k=4, hops=1, verbose=False):
    intent = route_intent(query)
    context = graph_rag_retrieve(query, top_k=top_k, hops=hops, verbose=verbose)
    prompt = TEMPLATES[intent].format(context=context, query=query)
    answer = call_llm(SYSTEM_PROMPT, prompt)
    return answer
```

A simple keyword-based router picks which prompt template to use — no extra LLM call needed for classification, which keeps the demo simple and free of additional API cost/latency.

---

## 12. Debugging Log — Real Issues Hit, and Why They Happened

This project surfaced several genuinely common RAG/LLM engineering problems — documented here because *understanding the failure* teaches as much as the working code does.

| Symptom | Root cause | Fix |
|---|---|---|
| Wikipedia fetch returns 403 Forbidden | No `User-Agent` header — Wikimedia's API rejects anonymous-looking requests | Add a descriptive `User-Agent` header to the request |
| `KeyError: 'tokyo_chunk_162'` in retrieval | Vector store (Chroma) held stale node ids from a previous run after the graph was rebuilt without re-syncing | Reset the Chroma collection whenever the graph is rebuilt; defensively skip ids not in `G` |
| `CUDA out of memory` trying to allocate 45+ GiB | A city node had 200+ chunk edges (full Wikipedia article, not just the intro); the 1-hop graph walk pulled in *all* of them into one prompt — a "hub node explosion" | Fetch only the article intro (`exintro: True`); cap total expanded context with `max_context_nodes` |
| `Model not found` (xAI Grok, then Groq) | Guessed model name strings that didn't match what the account actually had access to | Call the provider's `models.list()` endpoint and use an exact id from the real response |
| 7-day itinerary request returns only ~4 days, cut off mid-sentence | `max_new_tokens` was too low for the length of content requested | Raise the token limit, scale it per request type |
| `PermissionDeniedError` — "team doesn't have credits" | xAI's developer API is pay-per-token; the account had $0 balance | Either add billing credits, or switch to a free-tier provider (Groq) |

**The throughline:** almost none of these were "the RAG logic is wrong" — they were **operational/integration issues** (auth headers, sync state, resource limits, exact API contracts) that show up in almost any real-world LLM application, RAG or not. Recognizing which category a bug falls into (data sync vs. API contract vs. resource limit vs. logic error) is itself a core skill.

---

## 13. Recap Table

| Concept | Where it lives in the notebook |
|---|---|
| Data configuration | Hand-written dicts (structured facts) + real Wikipedia extracts (narrative text), unified as graph nodes |
| Vector DB role | ChromaDB stores embeddings, enables semantic (meaning-based) search instead of keyword match |
| What makes it "Graph" RAG | `networkx` edges let retrieval expand beyond similarity search into explicitly related nodes |
| Chunking | Splits long text into embeddable pieces; doesn't replace structured numeric data, which stays hand-written |
| LLM | Groq API, OpenAI-compatible SDK, `openai/gpt-oss-20b` |
| Role of prompts | System prompt grounds the model in retrieved context; task templates shape output format |
| Orchestration | `route_intent()` → `graph_rag_retrieve()` → prompt template → `call_llm()` |

---

## 14. Ideas to Extend

- Swap the keyword-based router for an LLM-based intent classifier
- Try `hops=2` and observe how context grows — and starts pulling in less-relevant nodes (a direct, hands-on look at the hub-node tradeoff from section 9)
- Persist Chroma to disk (`chromadb.PersistentClient`) instead of in-memory, so data survives across runs
- Replace the hand-written attraction data with a real structured source (e.g. Wikivoyage sections, a Places API) — keep the transport/cost table hand-written or API-sourced, since that's volatile pricing data no encyclopedia will have
- Compare this hand-built version against the companion **LangChain notebook**, which reimplements the same pipeline using `HuggingFaceEmbeddings`, `Chroma`, a custom `BaseRetriever`, and LCEL chains — useful for seeing what a framework abstracts away versus what stays custom either way
