#!/usr/bin/env python3
"""
narrative_builder.py

Usage:
    python narrative_builder.py --dataset path/to/news.json --topic "AI regulation" [--top_n 200] [--output out.json]

Requirements:
    pip install sentence-transformers faiss-cpu scikit-learn hdbscan networkx nltk tqdm python-dateutil

Notes:
 - Dataset must be a JSON list of article objects. Each article should ideally contain:
    - date (string)
    - headline or title (string)
    - url (string)
    - summary or text (string)
    - source_rating (numeric)
 - The script filters articles with source_rating > 8 (strict).
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from dateutil import parser as dateparser
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)

# ----------------------------
# Configurable parameters
# ----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"  # change to a larger model if you have resources
EMBED_BATCH = 256
EMBEDDINGS_FILE = "news_embeddings.npy"
META_FILE = "news_metadata.json"
INDEX_FILE = "news_faiss.index"  # not used explicitly in this script but reserved if you add FAISS persistence
TOP_N_DEFAULT = 200
SIM_BUILD_THRESHOLD = 0.72
SIM_ADD_THRESHOLD = 0.55
SIM_CONTRADICT_THRESHOLD = 0.78
MAX_SUMMARY_SENTENCES = 8
MAX_CLUSTER_COUNT = 12
MIN_CLUSTER_COUNT = 2
# ----------------------------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    text = p.read_text(encoding='utf-8', errors='ignore')
    # Try to parse as JSON
    try:
        data = json.loads(text)
    except Exception:
        # Try NDJSON: one JSON object per line
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        if items:
            return items
        raise

    # If the JSON is an object with an 'items' key, use that list
    if isinstance(data, dict):
        if 'items' in data and isinstance(data['items'], list):
            return data['items']
        # try common keys
        for k in ('data','articles','results'):
            if k in data and isinstance(data[k], list):
                return data[k]
        # otherwise, cannot interpret
        raise ValueError("JSON root is an object but does not contain a top-level list of articles (keys tried: items,data,articles,results)")

    if not isinstance(data, list):
        raise ValueError("Expected dataset JSON to be a list of article objects.")
    return data

def filter_by_rating(data: List[Dict[str, Any]], min_rating: float = 8.0000001) -> List[Dict[str, Any]]:
    out = []
    for a in data:
        try:
            r = a.get("source_rating", None)
            if r is None:
                continue
            if float(r) > 8.0:
                out.append(a)
        except Exception:
            continue
    return out

def article_text_for_embedding(a: Dict[str, Any]) -> str:
    # Compose the text we will embed for each article: headline + summary/text + url (optional)
    h = a.get('headline') or a.get('title') or ''
    s = a.get('summary') or a.get('text') or a.get('content') or ''
    combined = (h + ". " + s).strip()
    if not combined:
        combined = h or s or a.get('url','')
    return combined

def ensure_embeddings(model: SentenceTransformer, data: List[Dict[str, Any]], dataset_path: str = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    If embeddings file exists and matches data length, load; otherwise compute and save.
    Returns embeddings (n x d) and metadata list (article_index -> meta).
    """
    # Build metadata list (we use stable ordering of `data`)
    metadata = []
    for i, a in enumerate(data):
        metadata.append({
            "idx": i,
            "headline": a.get('headline') or a.get('title'),
            "date": a.get('date'),
            "url": a.get('url')
        })

    # build dataset-specific cache filenames to avoid collisions
    stem = Path(dataset_path).stem if dataset_path else "news"
    safe_model = EMBED_MODEL.replace('/', '_')
    embeddings_path = f"{stem}_embeddings_{safe_model}.npy"
    meta_path = f"{stem}_metadata.json"

    # If embeddings file exists and length matches, load it
    if Path(embeddings_path).exists():
        try:
            emb = np.load(embeddings_path)
            if emb.shape[0] == len(data):
                return emb, metadata
            else:
                # mismatch -> compute new embeddings
                pass
        except Exception:
            pass

    # compute embeddings in batches
    texts = [article_text_for_embedding(a) for a in data]
    embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Computing embeddings"):
        batch = texts[i:i+EMBED_BATCH]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    np.save(embeddings_path, embeddings)
    Path(meta_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return embeddings, metadata

def find_relevant_articles(topic: str, data: List[Dict[str, Any]], embeddings: np.ndarray, model: SentenceTransformer, top_n: int) -> List[Dict[str, Any]]:
    q_emb = model.encode([topic], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]  # shape (n_articles,)
    top_idx = np.argsort(-sims)[:top_n]
    selected = []
    for idx in top_idx:
        art = data[idx].copy()
        art['_sim_to_topic'] = float(sims[idx])
        # Keep the embedding as a numpy array in-memory (avoid converting to large Python lists)
        art['_embedding'] = embeddings[idx]
        selected.append(art)
    return selected

def summarize_narrative(selected: List[Dict[str, Any]], max_sentences: int = MAX_SUMMARY_SENTENCES) -> str:
    """
    Simple extractive summary: choose top articles by similarity and produce short sentences.
    For higher-quality summaries, integrate a summarization model (not included here).
    """
    if not selected:
        return ""
    # sort by sim score, then by date
    def safe_date(a):
        try:
            return dateparser.parse(a.get('date')) if a.get('date') else None
        except Exception:
            return None
    sorted_sel = sorted(selected, key=lambda a: (-a.get('_sim_to_topic',0.0), safe_date(a) or ""))
    sentences = []
    for i, a in enumerate(sorted_sel[:max_sentences]):
        headline = a.get('headline') or a.get('title') or ""
        date = a.get('date') or ""
        snippet = (a.get('summary') or a.get('text') or "")[:200]
        if headline:
            sentences.append(f"{headline} ({date})")
        elif snippet:
            sentences.append(snippet)
    # Form 5-10 sentence paragraph by joining with space.
    # Optionally, we can prepend a lead sentence.
    lead = f"A narrative on the topic based on {len(selected)} relevant, highly-rated articles."
    body = " ".join(sentences)
    # Limit to roughly 5-10 sentences: if body has more, truncate.
    return f"{lead} {body}"

def build_timeline(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timeline = []
    for a in selected:
        dt = None
        try:
            if a.get('date'):
                dt = dateparser.parse(a.get('date'))
        except Exception:
            dt = None
        timeline.append((dt, a))
    # sort by date ascending (oldest first). If date missing, place at end.
    timeline_sorted = sorted(timeline, key=lambda x: (x[0] is None, x[0]))
    out = []
    for dt, a in timeline_sorted:
        out.append({
            "date": dt.isoformat() if dt else None,
            "headline": a.get('headline') or a.get('title'),
            "url": a.get('url'),
            "why_it_matters": (a.get('summary') or a.get('text') or "")[:280]
        })
    return out

def cluster_articles(selected: List[Dict[str, Any]], suggested_k: int = None) -> List[Dict[str, Any]]:
    """
    Cluster using KMeans. Determine k based on number of articles if not provided.
    Returns list of clusters each with cluster_id and member articles.
    """
    if not selected:
        return []
    embeddings = np.array([np.array(a['_embedding']) for a in selected])
    n = len(selected)
    # choose cluster count if not provided
    if suggested_k is None:
        k = max(MIN_CLUSTER_COUNT, min(MAX_CLUSTER_COUNT, n // 10))
        k = min(k, n)  # cannot have more clusters than items
    else:
        k = max(1, min(suggested_k, n))
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(embeddings)
    clusters = {}
    for lbl, art in zip(labels, selected):
        clusters.setdefault(int(lbl), []).append({
            "headline": art.get('headline') or art.get('title'),
            "url": art.get('url'),
            "date": art.get('date'),
            "score": art.get('_sim_to_topic')
        })
    out = [{"cluster_id": cid, "articles": arts} for cid, arts in clusters.items()]
    return out

# Basic rule-based sentiment approximation (very lightweight)
_POS = {"good","positive","supports","yes","increase","rise","win","successful","benefit","beneficial","approve","approved"}
_NEG = {"bad","negative","opposes","against","no","decline","drop","lose","loss","harm","reject","rejected","accuse","accuses"}

def simple_sentiment_label(text: str) -> str:
    if not text:
        return "neutral"
    tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
    pos_score = sum(1 for t in tokens if t in _POS)
    neg_score = sum(1 for t in tokens if t in _NEG)
    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"

def build_graph(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a directed graph where nodes are article indices (0..n-1) and edges carry relation types:
    - builds_on: later article very similar to earlier one (SI >= SIM_BUILD_THRESHOLD)
    - adds_context: later article moderately similar (SI >= SIM_ADD_THRESHOLD)
    - contradicts: high similarity + opposite sentiment (heuristic)
    - escalates: similarity + later date with increased 'intensity' (heuristic)
    """
    G = nx.DiGraph()
    n = len(selected)
    if n == 0:
        return {"nodes": [], "edges": []}
    # add nodes
    for i, a in enumerate(selected):
        G.add_node(i, headline=a.get('headline') or a.get('title'), url=a.get('url'), date=a.get('date'))

    # embeddings and similarities
    embs = np.array([np.array(a['_embedding']) for a in selected])
    sims = cosine_similarity(embs)

    # prepare dates and sentiment
    dates = []
    for a in selected:
        try:
            d = dateparser.parse(a.get('date')) if a.get('date') else None
        except Exception:
            d = None
        dates.append(d)
    sentiments = [simple_sentiment_label(a.get('summary') or a.get('text') or "") for a in selected]

    # add edges by pairwise comparison
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim_ij = float(sims[i, j])
            di = dates[i]
            dj = dates[j]
            # consider edge direction from earlier to later (if both dates exist)
            later = False
            if di and dj and dj > di:
                later = True
            # builds_on: high similarity and j is later
            if sim_ij >= SIM_BUILD_THRESHOLD and later:
                G.add_edge(i, j, relation="builds_on", score=sim_ij)
            # adds_context: moderate similarity and j is later
            elif sim_ij >= SIM_ADD_THRESHOLD and later:
                G.add_edge(i, j, relation="adds_context", score=sim_ij)
            # contradicts: high similarity + opposite sentiment
            if sim_ij >= SIM_CONTRADICT_THRESHOLD:
                s_i = sentiments[i]
                s_j = sentiments[j]
                if s_i != "neutral" and s_j != "neutral" and s_i != s_j:
                    G.add_edge(i, j, relation="contradicts", score=sim_ij)
            # escalate heuristic: j later and j text appears more "intense" (simple proxy: length or keywords)
            if later:
                len_i = len(selected[i].get('summary') or selected[i].get('text') or "")
                len_j = len(selected[j].get('summary') or selected[j].get('text') or "")
                if sim_ij >= SIM_ADD_THRESHOLD and len_j > len_i * 1.25:
                    # treat as escalate candidate
                    G.add_edge(i, j, relation="escalates", score=sim_ij)

    # convert to serializable structure
    nodes = []
    for nid, d in G.nodes(data=True):
        node = {"id": int(nid)}
        node.update({k: v for k, v in d.items()})
        nodes.append(node)
    edges = []
    for u, v, d in G.edges(data=True):
        edge = {"source": int(u), "target": int(v)}
        edge.update({k: v for k, v in d.items()})
        edges.append(edge)
    return {"nodes": nodes, "edges": edges}

def build_output(narrative_summary: str, timeline: List[Dict[str, Any]], clusters: List[Dict[str, Any]], graph: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "narrative_summary": narrative_summary,
        "timeline": timeline,
        "clusters": clusters,
        "graph": graph
    }

def main():
    parser = argparse.ArgumentParser(description="Narrative builder from news JSON")
    parser.add_argument("--dataset", required=False, default=None, help="Path to news JSON dataset (list of article dicts). If omitted, script will attempt to auto-detect a JSON dataset in the current directory.")
    parser.add_argument("--topic", required=True, help="Topic to build narrative for")
    parser.add_argument("--top_n", type=int, default=TOP_N_DEFAULT, help="Number of top relevant articles to consider")
    parser.add_argument("--output", required=False, help="Optional path to save output JSON (otherwise print to stdout)")
    args = parser.parse_args()
    # Auto-detect dataset if user didn't pass one
    def auto_detect_dataset() -> str:
        # preference order
        candidates = ["news.json", "dataset.json", "news_84mb.json", "news_84mb_data.json"]
        cwd = Path('.').resolve()
        for c in candidates:
            p = cwd / c
            if p.exists():
                return str(p)
        # fallback: if there's exactly one .json in cwd, use it
        json_files = list(cwd.glob('*.json'))
        if len(json_files) == 1:
            return str(json_files[0])
        # otherwise, no automatic choice
        return None

    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = auto_detect_dataset()
        if dataset_path:
            print(f"Auto-detected dataset: {dataset_path}")
        else:
            parser.error("No dataset specified and none could be auto-detected in the current directory. Please pass --dataset path/to/news.json")

    print("Loading dataset...")
    data = load_dataset(dataset_path)
    print(f"Total articles in dataset: {len(data)}")

    print("Filtering by source_rating > 8...")
    data = filter_by_rating(data, min_rating=8.0000001)
    print(f"Articles after rating filter: {len(data)}")
    if len(data) == 0:
        out = build_output("", [], [], {"nodes": [], "edges": []})
        if args.output:
            Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Wrote output to {args.output}")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Preparing (or loading) embeddings...")
    embeddings, metadata = ensure_embeddings(model, data, dataset_path=dataset_path)

    # limit top_n to dataset size
    top_n = min(args.top_n, len(data))
    print(f"Finding top {top_n} relevant articles for topic: {args.topic!r} ...")
    selected = find_relevant_articles(args.topic, data, embeddings, model, top_n=top_n)
    print(f"Selected {len(selected)} relevant articles.")

    print("Building narrative summary...")
    narrative_summary = summarize_narrative(selected, max_sentences=MAX_SUMMARY_SENTENCES)

    print("Building timeline...")
    timeline = build_timeline(selected)

    print("Clustering articles into themes...")
    clusters = cluster_articles(selected)

    print("Building narrative graph (relations)...")
    graph = build_graph(selected)

    output = build_output(narrative_summary, timeline, clusters, graph)
    json_out = json.dumps(output, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(json_out, encoding='utf-8')
        print(f"Saved narrative JSON to: {args.output}")
    else:
        print(json_out)

if __name__ == "__main__":
    main()
