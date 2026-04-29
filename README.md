# VecSRCH

<p align="center">
  <strong>Vector search for 28,000+ Olympiad math problems</strong>
</p>

<p align="center">
  <a href="https://jeonqxrhsgptutufrwpo.supabase.co/functions/v1/search">
    <img src="https://img.shields.io/badge/Live%20Demo-View%20Site-blue?style=for-the-badge" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Dataset-28K%2B%20Problems-green?style=for-the-badge" alt="Dataset Size">
  <img src="https://img.shields.io/badge/Embeddings-Qwen3--4B-purple?style=for-the-badge" alt="Embeddings">
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#setup">Setup</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#license">License</a>
</p>

---

## Overview

VecSRCH is a semantic search engine for competitive math. It lets you find problems from the [MathNet dataset](https://huggingface.co/datasets/ShadenA/MathNet) using vector embeddings instead of just keyword matching.

**What it does**

- Search by concept ("geometry triangle circumcircle") not just exact words
- Edge-deployed for fast queries anywhere
- Shows diagrams when problems have them
- Dark mode because obviously
- Filter by competition, country, whatever

---

## Architecture

```
Browser (GitHub Pages) 
    ↓
Edge Function (Supabase/Deno)
    ↓
OpenRouter (embeddings) + Postgres (pgvector)
```

| Thing | What we used |
|-------|--------------|
| Frontend | Static HTML/CSS/JS on GitHub Pages |
| API | Supabase Edge Functions |
| Database | PostgreSQL + pgvector |
| Embeddings (live) | OpenRouter - `qwen/qwen3-embedding-4b` |
| Embeddings (build) | LM Studio locally |
| Dataset | [ShadenA/MathNet](https://huggingface.co/datasets/ShadenA/MathNet) (~28K problems) |

---

## Setup

### What you need

- Python 3.12+ (3.14 breaks stuff)
- Supabase account
- OpenRouter API key
- LM Studio for building the database

### 1. Install stuff

```bash
pip install -r requirements.txt
```

### 2. Build the database

Start LM Studio with `text-embedding-qwen3-embedding-4b` loaded, then:

```bash
python build_db.py
```

It'll ask for your Supabase `service_role` key. Copy it from Supabase dashboard.

### 3. Set Edge Function secrets

In Supabase Dashboard > Project Settings > Edge Functions > Secrets, add:

| Secret | Value |
|--------|-------|
| `OPENROUTER_API_KEY` | Your OpenRouter key |
| `SUPABASE_URL` | Your project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Your service role key |

---

## Deployment

### GitHub Pages

1. Fork this repo
2. Settings > Pages
3. Source: `main` branch, root folder
4. Site goes live at `https://<username>.github.io/VecSRCH`

### Custom domain

Add a `CNAME` file with your domain if you want.

---

## API

### POST `/functions/v1/search`

Search problems by semantic similarity.

**Request**

```json
{
  "query": "geometry triangle circumcircle",
  "count": 10
}
```

**Response**

```json
{
  "results": [
    {
      "id": "mathnet_1234",
      "problem_markdown": "...",
      "similarity": 0.87,
      "country": "USA",
      "competition": "IMO",
      "topics_flat": ["geometry", "circles"],
      "has_images": true,
      "num_images": 2
    }
  ]
}
```

**Rate limit:** 20 req/min per IP (hard limit in Edge Function).

---

## Dataset

[MathNet](https://huggingface.co/datasets/ShadenA/MathNet) has:

- 28,385 competition math problems
- 4,000+ with figures/diagrams
- Various sources (IMO, AIME, USAMO, etc.)
- Multiple languages

---

## License

MIT. See [LICENSE](./LICENSE).

---

<p align="center">
  Built by <a href="https://github.com/Alek-Nedelchev">Aleksandar Nedelchev</a> & <a href="https://github.com/Richard-Zzzzz">Richard Zhang</a>
</p>
