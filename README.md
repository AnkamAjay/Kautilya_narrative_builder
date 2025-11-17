<<<<<<< HEAD
# Narrative Builder
# Kautilya_narrative_builder

This repository contains `narrative_builder.py`, a script that builds structured narratives from a JSON news dataset.

Usage examples:

```
python narrative_builder.py --dataset path/to/news.json --topic "Jubilee Hills elections"
python narrative_builder.py --topic "AI regulation"
python narrative_builder.py --topic "Israel-Iran conflict" --output narrative_ai.json
```

If `--dataset` is omitted, the script will try to auto-detect a `.json` file in the current directory (prefers `news.json`, `dataset.json`, `news_84mb.json`).

Output:
- JSON with keys: `narrative_summary`, `timeline`, `clusters`, `graph`.

Performance tips:
- Install `faiss-cpu` for faster nearest-neighbor search.
- Increase available RAM or use a smaller embedding model to reduce memory.
- The script caches embeddings to `<dataset>_embeddings_<model>.npy` to avoid recomputing.

Requirements:
- See `requirements.txt`.

Notes:
- Dataset must be a JSON array of article objects (or an object with an `items` array) containing at least `date`/`published_at`, `headline`/`title`, `url`, `summary`/`text`/`story`, and `source_rating`.
- The script filters articles with `source_rating > 8` by default.
