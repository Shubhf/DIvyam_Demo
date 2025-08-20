# Routing + Caching Proof of Concept

This is a minimal demo showing how a routing + caching layer can help reduce **latency** and **cost** when deciding between a small (cheap, fast) model and a large (expensive, slower) model.

---

## ⚡ What This Demo Does
- **Baseline:** Always calls the "big" model.  
- **Routing + Cache:**  
  - Uses a router to decide whether a query should go to the small or big model.  
  - Caches results so repeated queries are served instantly with no cost.

Currently, both models are **stubbed locally** (no external API calls). The idea is to show the workflow — you can later plug in an actual model (quantized small model or API-based large model).

---

## 🚀 How to Run
Clone the repo and run any of the modes:

```bash
# baseline (big model only)
python router_demo.py --mode baseline

# routing only
python router_demo.py --mode routing

# routing + cache
python router_demo.py --mode routing+cache

# full (routing + cache with small model placeholder)
python router_demo.py --mode full

```
📂 Outputs

Each run generates:

Metrics JSON in results/

Cache DB (cache.sqlite3)

---

Example output:
```

{
  "mode": "routing+cache",
  "total_cost_usd": 0.0,
  "p50_latency_ms": 5.0,
  "p95_latency_ms": 5.0,
  "cache_hits": 1,
  "small_routed": 0,
  "big_routed": 0,
  "n": 1
}
```
---
### 🔧 Next Steps

- Plug in a quantized small model into small_model() in router_demo.py.

- Connect a real large model API into big_model().

- Run comparative experiments on real workloads.
