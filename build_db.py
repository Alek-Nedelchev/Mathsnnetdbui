"""Download MathNet from HuggingFace, embed via LM Studio, upload to Supabase."""

import json
import sys
import time
import requests
from datasets import load_dataset
from supabase import create_client

SUPABASE_URL = "https://reonrsbfjbzhebjrfeia.supabase.co"
EXISTING_IDS_PAGE = 1000
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBED_MODEL = "text-embedding-qwen3-embedding-4b"
EMBED_DIMENSIONS = 2560
BATCH_SIZE = 50
EMBED_DELAY = 0.05
MAX_EMBED_RETRIES = 3
RETRY_BACKOFF = 2.0


def check_lm_studio():
    try:
        resp = requests.get(LM_STUDIO_URL.replace("/embeddings", "/models"), timeout=5)
        resp.raise_for_status()
        return True
    except requests.ConnectionError:
        print("ERROR: Cannot connect to LM Studio at http://127.0.0.1:1234")
        print("Make sure LM Studio is running with the embedding model loaded.")
        return False
    except Exception as e:
        print(f"ERROR: LM Studio check failed: {e}")
        return False


def get_embedding(text: str) -> list[float]:
    for attempt in range(MAX_EMBED_RETRIES):
        try:
            resp = requests.post(
                LM_STUDIO_URL,
                json={"model": EMBED_MODEL, "input": text},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            if not embedding or len(embedding) != EMBED_DIMENSIONS:
                raise ValueError(f"Unexpected embedding dimension: {len(embedding)}, expected {EMBED_DIMENSIONS}")
            return embedding
        except requests.ConnectionError:
            if attempt < MAX_EMBED_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                print(f"  Connection lost, retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
        except requests.HTTPError as e:
            if resp.status_code >= 500 and attempt < MAX_EMBED_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                print(f"  Server error {resp.status_code}, retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Invalid embedding response: {e}")


def get_existing_ids(supabase) -> set:
    existing = set()
    offset = 0
    while True:
        resp = supabase.table("mathnet").select("id").range(offset, offset + EXISTING_IDS_PAGE - 1).execute()
        ids = [r["id"] for r in resp.data]
        existing.update(ids)
        if len(ids) < EXISTING_IDS_PAGE:
            break
        offset += EXISTING_IDS_PAGE
    return existing


def main():
    if not check_lm_studio():
        sys.exit(1)

    supabase_key = input("Enter Supabase service_role key: ").strip()
    if not supabase_key:
        print("ERROR: No key provided.")
        sys.exit(1)

    try:
        supabase = create_client(SUPABASE_URL, supabase_key)
    except Exception as e:
        print(f"ERROR: Failed to create Supabase client: {e}")
        sys.exit(1)

    print("Fetching existing IDs from Supabase...")
    try:
        existing_ids = get_existing_ids(supabase)
    except Exception as e:
        print(f"ERROR: Failed to fetch existing IDs: {e}")
        sys.exit(1)
    print(f"Found {len(existing_ids)} existing rows")

    print("Loading MathNet dataset from HuggingFace...")
    try:
        ds = load_dataset("ShadenA/MathNet", split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    total = len(ds)
    remaining = total - len(existing_ids)
    print(f"Loaded {total} problems, {remaining} new to process")

    inserted = 0
    skipped = 0
    already_exists = 0
    failed = 0
    batch = []

    for i, row in enumerate(ds):
        row_id = row.get("id")
        if row_id in existing_ids:
            already_exists += 1
            continue

        problem_text = row.get("problem_markdown") or ""
        if not problem_text.strip():
            skipped += 1
            continue

        parts = []
        if row.get("country"):
            parts.append(f"Country: {row['country']}")
        if row.get("competition"):
            parts.append(f"Competition: {row['competition']}")
        if row.get("language"):
            parts.append(f"Language: {row['language']}")
        if row.get("problem_type"):
            parts.append(f"Type: {row['problem_type']}")
        topics = row.get("topics_flat") or []
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")
        parts.append(problem_text)
        embed_text = "\n".join(parts)

        try:
            embedding = get_embedding(embed_text)
        except Exception as e:
            print(f"  Embedding failed for row {i} (id={row.get('id')}): {e}")
            failed += 1
            continue

        record = {
            "id": row["id"],
            "country": row.get("country"),
            "competition": row.get("competition"),
            "problem_markdown": problem_text,
            "solutions_markdown": json.dumps(row.get("solutions_markdown") or []),
            "topics_flat": json.dumps(row.get("topics_flat") or []),
            "language": row.get("language"),
            "problem_type": row.get("problem_type"),
            "final_answer": row.get("final_answer"),
            "embedding": embedding,
        }
        batch.append(record)

        if len(batch) >= BATCH_SIZE:
            try:
                supabase.table("mathnet").insert(batch).execute()
                inserted += len(batch)
                print(f"  [{inserted}/{remaining}] inserted ({inserted + already_exists + skipped + failed}/{total} processed)")
            except Exception as e:
                failed += len(batch)
                print(f"  Insert failed at batch ending row {i}: {e}")
            batch = []

        time.sleep(EMBED_DELAY)

    if batch:
        try:
            supabase.table("mathnet").insert(batch).execute()
            inserted += len(batch)
            print(f"  [{inserted}/{total}] inserted (final batch)")
        except Exception as e:
            failed += len(batch)
            print(f"  Final insert failed: {e}")

    print(f"\nDone! Inserted: {inserted}, Already existed: {already_exists}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
