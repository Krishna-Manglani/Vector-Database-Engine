"""

Dependencies:
    pip install flask requests

Optional (for RAG):
    Install Ollama from https://ollama.com
    ollama pull nomic-embed-text
    ollama pull llama3.2

Run:
    python vectordb_server.py
    → http://localhost:8080
"""

from __future__ import annotations

import math
import heapq
import random
import threading
import time
from collections import defaultdict
from typing import Callable, Optional
from flask import Flask, request, jsonify, send_file, abort
import requests
import os

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # demo vectors (16D categorical)

# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    def __init__(self, id: int, metadata: str, category: str, emb: list[float]):
        self.id       = id
        self.metadata = metadata
        self.category = category
        self.emb      = emb

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a)
    nb  = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))

def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

DistFn = Callable[[list[float], list[float]], float]

def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":    return cosine
    if metric == "manhattan": return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        pairs = [(dist(q, v.emb), v.id) for v in self.items]
        pairs.sort()
        return pairs[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item  = item
        self.left:  Optional[KDNode] = None
        self.right: Optional[KDNode] = None

class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left  = self._insert(node.left,  v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: list[float], k: int, depth: int,
             dist: DistFn, heap: list):
        """heap is a max-heap stored as [(-dist, id), ...]"""
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item.id))

        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer,  q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist, heap)
        return sorted((-d, id_) for d, id_ in heap)

    def rebuild(self, items: list[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    def __init__(self, item: VectorItem, max_lyr: int):
        self.item    = item
        self.max_lyr = max_lyr
        # nbrs[layer] = list of neighbour IDs
        self.nbrs: list[list[int]] = [[] for _ in range(max_lyr + 1)]

class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.M        = m
        self.M0       = 2 * m
        self.ef_build = ef_build
        self.mL       = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt  = -1
        self.G: dict[int, HNSWNode] = {}
        self._rng = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int,
                      lyr: int, dist: DistFn) -> list[tuple[float, int]]:
        vis: set[int] = {ep}
        # min-heap for candidates; max-heap for found (stored as negatives)
        d0 = dist(q, self.G[ep].item.emb)
        cands = [(d0, ep)]
        found = [(-d0, ep)]  # max-heap

        while cands:
            cd, cid = heapq.heappop(cands)
            worst = -found[0][0]
            if len(found) >= ef and cd > worst:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id_) for d, id_ in found]
        result.sort()
        return result

    def _select_nbrs(self, cands: list[tuple[float, int]], max_m: int) -> list[int]:
        return [id_ for _, id_ in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        id_  = item.id
        lvl  = self._rand_level()
        self.G[id_] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt  = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            node = self.G.get(ep)
            if node and lc < len(node.nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            maxM = self.M0 if lc == 0 else self.M
            sel  = self._select_nbrs(W, maxM)
            self.G[id_].nbrs[lc] = sel

            for nid in sel:
                nd = self.G.get(nid)
                if nd is None:
                    continue
                while len(nd.nbrs) <= lc:
                    nd.nbrs.append([])
                nd.nbrs[lc].append(id_)
                if len(nd.nbrs[lc]) > maxM:
                    ds = [(dist(nd.item.emb, self.G[c].item.emb), c)
                          for c in nd.nbrs[lc] if c in self.G]
                    ds.sort()
                    nd.nbrs[lc] = [c for _, c in ds[:maxM]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id_

    def knn(self, q: list[float], k: int, ef: int,
            dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            node = self.G.get(ep)
            if node and lc < len(node.nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id: int):
        if id not in self.G:
            return
        for node in self.G.values():
            for layer in node.nbrs:
                if id in layer:
                    layer.remove(id)
        if self.entry_pt == id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id:
                    self.entry_pt = nid
                    break
        del self.G[id]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for id_, nd in self.G.items():
            nodes.append({
                "id":       id_,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr":   nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer":       self.top_layer,
            "nodeCount":      len(self.G),
            "nodesPerLayer":  nodes_per_layer,
            "edgesPerLayer":  edges_per_layer,
            "nodes":          nodes,
            "edges":          edges,
        }

    def __len__(self):
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims   = dims
        self._store: dict[int, VectorItem] = {}
        self._bf    = BruteForce()
        self._kdt   = KDTree(dims)
        self._hnsw  = HNSW(16, 200)
        self._lock  = threading.Lock()
        self._next  = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self._lock:
            v = VectorItem(self._next, meta, cat, emb)
            self._next += 1
            self._store[v.id] = v
            self._bf.insert(v)
            self._kdt.insert(v)
            self._hnsw.insert(v, dist)
            return v.id

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self._store:
                return False
            del self._store[id]
            self._bf.remove(id)
            self._hnsw.remove(id)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: list[float], k: int,
               metric: str, algo: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0  = time.perf_counter()
            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self._store:
                    v = self._store[id_]
                    hits.append({
                        "id":        v.id,
                        "metadata":  v.metadata,
                        "category":  v.category,
                        "embedding": v.emb,
                        "distance":  round(d, 6),
                    })
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)

            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": time_fn(lambda: self._bf.knn(q, k, dfn)),
                "kdtreeUs":     time_fn(lambda: self._kdt.knn(q, k, dfn)),
                "hnswUs":       time_fn(lambda: self._hnsw.knn(q, k, 50, dfn)),
                "itemCount":    len(self._store),
            }

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def __len__(self):
        with self._lock:
            return len(self._store)

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base        = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        try:
            r = requests.post(
                f"{self.base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    step   = chunk_words - overlap_words
    chunks = []
    i      = 0
    while i < len(words):
        end   = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

class DocItem:
    def __init__(self, id: int, title: str, text: str, emb: list[float]):
        self.id    = id
        self.title = title
        self.text  = text
        self.emb   = emb

class DocumentDB:
    def __init__(self):
        self._store: dict[int, DocItem] = {}
        self._hnsw  = HNSW(16, 200)
        self._bf    = BruteForce()
        self._lock  = threading.Lock()
        self._next  = 1
        self._dims  = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(self._next, title, text, emb)
            self._next += 1
            self._store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int,
               max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, 50, cosine)
            return [(d, self._store[id_])
                    for d, id_ in raw
                    if id_ in self._store and d <= max_dist]

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self._store:
                return False
            del self._store[id]
            self._hnsw.remove(id)
            self._bf.remove(id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def get_dims(self) -> int:
        return self._dims

    def __len__(self):
        with self._lock:
            return len(self._store)

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    data = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in data:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  FLASK APP
# =====================================================================

app    = Flask(__name__)
db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)

# ── CORS ─────────────────────────────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options(_path=""):
    return "", 204

# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────────

@app.route("/search")
def search():
    v_str = request.args.get("v", "")
    q     = [float(x) for x in v_str.split(",") if x.strip()]
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k      = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo   = request.args.get("algo",   "hnsw")
    return jsonify(db.search(q, k, metric, algo))

@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(force=True, silent=True) or {}
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": id_})

@app.route("/delete/<int:id_>", methods=["DELETE"])
def delete(id_):
    ok = db.remove(id_)
    return jsonify({"ok": ok})

@app.route("/items")
def items():
    return jsonify([
        {"id": v.id, "metadata": v.metadata,
         "category": v.category, "embedding": v.emb}
        for v in db.all()
    ])

@app.route("/benchmark")
def benchmark():
    v_str = request.args.get("v", "")
    q     = [float(x) for x in v_str.split(",") if x.strip()]
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k      = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(q, k, metric))

@app.route("/hnsw-info")
def hnsw_info():
    return jsonify(db.hnsw_info())

# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body  = request.get_json(force=True, silent=True) or {}
    title = body.get("title", "")
    text  = body.get("text",  "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids    = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            }), 503
        chunk_title = (f"{title} [{i+1}/{len(chunks)}]"
                       if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})

@app.route("/doc/delete/<int:id_>", methods=["DELETE"])
def doc_delete(id_):
    ok = doc_db.remove(id_)
    return jsonify({"ok": ok})

@app.route("/doc/list")
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        result.append({
            "id":      d.id,
            "title":   d.title,
            "preview": preview,
            "words":   len(d.text.split()),
        })
    return jsonify(result)

@app.route("/doc/search", methods=["POST"])
def doc_search():
    body     = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    hits = doc_db.search(q_emb, k)
    return jsonify({
        "contexts": [
            {"id": doc.id, "title": doc.title, "distance": round(d, 4)}
            for d, doc in hits
        ]
    })

@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body     = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    # Step 1: embed question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    # Step 2: retrieve top-k chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx_parts = [
        f"[{i+1}] {doc.title}:\n{doc.text}\n"
        for i, (_, doc) in enumerate(hits)
    ]
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        "Context:\n" + "\n".join(ctx_parts) + "\n"
        f"Question: {question}\n\nAnswer:"
    )

    # Step 4: generate
    answer = ollama.generate(prompt)

    # Step 5: return
    return jsonify({
        "answer":   answer,
        "model":    ollama.gen_model,
        "contexts": [
            {"id": doc.id, "title": doc.title,
             "text": doc.text, "distance": round(d, 4)}
            for d, doc in hits
        ],
        "docCount": len(doc_db),
    })

# ── STATUS / STATS ────────────────────────────────────────────────────

@app.route("/status")
def status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "docCount":        len(doc_db),
        "docDims":         doc_db.get_dims(),
        "demoDims":        DIMS,
        "demoCount":       len(db),
    })

@app.route("/stats")
def stats():
    return jsonify({
        "count":      len(db),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"],
    })

# ── SERVE index.html ─────────────────────────────────────────────────

@app.route("/")
def index():
    if os.path.exists("index.html"):
        return send_file("index.html")
    return "VectorDB Engine running. Place index.html alongside this file.", 200

# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    app.run(host="0.0.0.0", port=8080, threaded=True)