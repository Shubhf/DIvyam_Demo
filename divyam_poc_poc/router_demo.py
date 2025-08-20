import argparse, json, time, hashlib, sqlite3, statistics, random, re
from pathlib import Path
from typing import Optional, Tuple, List

BASE = Path(__file__).resolve().parent
CACHE_DB = BASE / "cache.sqlite3"
DEFAULT_WORKLOAD = BASE / "workload.jsonl"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Cost model ($ per 1k tokens) — tweak if you want different ratios
COST = {"small": 0.05, "big": 2.50}

def tokenize(prompt: str) -> int:
    # crude token proxy
    return max(1, int(len(prompt.split()) * 1.3))

# ---------- Cache ----------
def init_cache():
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, tokens INT)")
    con.commit(); con.close()

def clear_cache():
    if CACHE_DB.exists():
        CACHE_DB.unlink()

def cache_get(key: str) -> Optional[Tuple[str,int]]:
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute("SELECT response, tokens FROM cache WHERE key = ?", (key,))
    row = cur.fetchone()
    con.close()
    return (row[0], row[1]) if row else None

def cache_set(key: str, response: str, tokens: int):
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO cache (key, response, tokens) VALUES (?, ?, ?)", (key, response, tokens))
    con.commit(); con.close()

def norm_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())

def cache_key(prompt: str) -> str:
    return hashlib.sha256(norm_prompt(prompt).encode()).hexdigest()

# ---------- Models (simulated) ----------
def small_model(prompt: str):
    """
    Swap this with your quantized local model call (e.g., llama.cpp / GGUF / bitsandbytes).
    Keep the same return shape: (response_text, token_count, latency_seconds).
    """
    tokens = tokenize(prompt) + 30
    start = time.time()
    time.sleep(random.uniform(0.08, 0.12))  # simulate faster local model
    return f"[SMALL] {prompt[:60]} ...", tokens, time.time() - start

def big_model(prompt: str):
    """
    Swap this with your production LLM API call. Keep the same return shape.
    """
    tokens = tokenize(prompt) + 80
    start = time.time()
    time.sleep(random.uniform(0.35, 0.45))  # simulate slower API model
    return f"[BIG] {prompt[:60]} ...", tokens, time.time() - start

# ---------- Router ----------
def router_score(prompt: str) -> float:
    # Probability that prompt is "easy" (route to small model)
    p = prompt.lower(); length = len(p.split())
    easy = ["what is","2+2","define","capital of","json","convert","short"]
    hard = ["summarize","explain","compare","trade-off","causal","diagnose"]
    score = 0.6 if length < 10 else 0.4
    if any(k in p for k in easy): score += 0.2
    if any(k in p for k in hard): score -= 0.2
    return max(0.0, min(1.0, score))

def should_route_small(prompt: str, threshold: float = 0.55) -> bool:
    return router_score(prompt) >= threshold

# ---------- IO ----------
def load_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSONL at line {idx}: {e.msg}. Line: {s[:120]}")
    if not items:
        raise ValueError("Workload is empty after filtering. Add lines or check file.")
    return items

# ---------- Runner ----------
def run(mode: str, workload_path: Path):
    init_cache()
    items = load_jsonl(workload_path)

    rows, latencies = [], []
    total_cost = cache_hits = small_ct = big_ct = 0

    for it in items:
        pid, prompt = it["id"], it["prompt"]
        key = cache_key(prompt)

        # Treat "full" like "routing+cache"
        if mode in ("routing+cache", "full"):
            hit = cache_get(key)
            if hit:
                response, tokens = hit
                cache_hits += 1
                latency = 0.005
                used = "cache"
                cost = 0.0
            else:
                if should_route_small(prompt):
                    response, tokens, latency = small_model(prompt); used = "small"; small_ct += 1
                    cost = (tokens/1000) * COST["small"]
                else:
                    response, tokens, latency = big_model(prompt); used = "big"; big_ct += 1
                    cost = (tokens/1000) * COST["big"]
                cache_set(key, response, tokens)

        else:
            if mode == "baseline":
                response, tokens, latency = big_model(prompt); used = "big"; big_ct += 1
                cost = (tokens/1000) * COST["big"]
            elif mode == "routing":
                if should_route_small(prompt):
                    response, tokens, latency = small_model(prompt); used = "small"; small_ct += 1
                    cost = (tokens/1000) * COST["small"]
                else:
                    response, tokens, latency = big_model(prompt); used = "big"; big_ct += 1
                    cost = (tokens/1000) * COST["big"]
            else:
                raise ValueError("Unknown mode")

        total_cost += cost
        latencies.append(latency)
        rows.append({
            "id": pid, "mode": mode, "route": used,
            "prompt_tokens": tokens, "latency_ms": round(latency*1000, 1),
            "cost_usd": round(cost, 6), "cache_hit": used == "cache"
        })

    # Robust percentiles when n is small
    lat_sorted = sorted(latencies); n = len(lat_sorted)
    p50 = round(lat_sorted[n//2] * 1000, 1)
    p95_idx = max(0, min(n-1, int(round(0.95*(n-1)))))
    p95 = round(lat_sorted[p95_idx] * 1000, 1)

    out = {
        "mode": mode,
        "total_cost_usd": round(total_cost, 4),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "cache_hits": cache_hits,
        "small_routed": small_ct,
        "big_routed": big_ct,
        "n": len(rows)
    }

    stamp = int(time.time())
    metrics_path = RESULTS_DIR / f"metrics_{mode}_{stamp}.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nArtifacts:\n  {metrics_path}\n  cache_db: {CACHE_DB}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline","routing","routing+cache","full"])
    ap.add_argument("--workload", default=str(DEFAULT_WORKLOAD), help="Path to a JSONL workload file")
    ap.add_argument("--clear-cache", action="store_true", help="Clear SQLite cache before run")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for repeatable latency sims")
    args = ap.parse_args()

    if args.clear_cache:
        clear_cache()
    random.seed(args.seed)

    run(args.mode, Path(args.workload))
