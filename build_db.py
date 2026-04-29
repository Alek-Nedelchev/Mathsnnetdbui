"""Download MathNet from HuggingFace, embed via LM Studio (batched), compress images to WebP q=80, upload to Supabase with resume support."""

import json
import sys
import io
import time
import base64
import signal
import os
import re
import requests
from typing import Optional
from datasets import load_dataset
from supabase import create_client
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/embeddings")
EMBED_MODEL = "text-embedding-qwen3-embedding-4b@q8_0"
EMBED_DIMENSIONS = 2560
WEBP_QUALITY = 80
EMBED_BATCH_SIZE = 16
DB_BATCH_SIZE = 50
MAX_EMBED_RETRIES = 3
RETRY_BACKOFF = 2.0

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print("\nShutdown requested, finishing current batch...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def check_lm_studio() -> bool:
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


def validate_supabase_key(key: str) -> bool:
    if not key or len(key) < 20:
        return False
    if not key.startswith("eyJ"):
        return False
    parts = key.split(".")
    if len(parts) != 3:
        return False
    return True


def get_existing_ids(supabase) -> set:
    """Fetch all existing IDs from the database for resume support."""
    existing = set()
    offset = 0
    page_size = 1000
    while True:
        try:
            resp = supabase.table("mathnet").select("id").range(offset, offset + page_size - 1).execute()
            for row in resp.data:
                if row.get("id"):
                    existing.add(row["id"])
            if len(resp.data) < page_size:
                break
            offset += page_size
            if shutdown_requested:
                print("  (Shutdown requested, stopping ID fetch)")
                break
        except Exception as e:
            print(f"Warning: Failed to fetch existing IDs at offset {offset}: {e}")
            break
    return existing


def embed_single(text: str) -> Optional[list[float]]:
    """Embed a single text with retries."""
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
                return None
        except requests.HTTPError as e:
            if resp.status_code >= 500 and attempt < MAX_EMBED_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                print(f"  Server error {resp.status_code}, retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                return None
        except Exception:
            return None


def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed a batch of texts, returning list of embeddings or None for failures."""
    for attempt in range(MAX_EMBED_RETRIES):
        try:
            resp = requests.post(
                LM_STUDIO_URL,
                json={"model": EMBED_MODEL, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = [None] * len(texts)
            for item in data["data"]:
                idx = item.get("index", 0)
                emb = item.get("embedding")
                if emb and len(emb) == EMBED_DIMENSIONS:
                    embeddings[idx] = emb
            return embeddings
        except requests.ConnectionError:
            if attempt < MAX_EMBED_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                print(f"  Connection lost, retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                break
        except requests.HTTPError:
            if resp.status_code >= 500 and attempt < MAX_EMBED_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                print(f"  Server error {resp.status_code}, retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                break
        except Exception:
            break
    
    # Batch failed, try individually
    print(f"  Batch failed, retrying {len(texts)} items individually...")
    return [embed_single(t) for t in texts]


def encode_image_webp(pil_img: Image.Image) -> str:
    buffer = io.BytesIO()
    try:
        pil_img.save(buffer, format="WEBP", quality=WEBP_QUALITY)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        buffer.close()


def process_row(row: dict) -> Optional[dict]:
    problem_text = row.get("problem_markdown") or ""
    if not problem_text.strip():
        return None

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

    images = row.get("images") or []
    images_data = None
    num_images = 0
    has_images = False

    if images:
        valid_images = [img for img in images if img is not None]
        if valid_images:
            has_images = True
            num_images = len(valid_images)
            try:
                images_data = [
                    {"index": idx, "format": "webp", "data": encode_image_webp(img)}
                    for idx, img in enumerate(valid_images)
                ]
            except Exception as e:
                print(f"  Image encoding failed for {row.get('id')}: {e}")
                # Reset to consistent state on failure
                images_data = None
                has_images = False
                num_images = 0

    return {
        "embed_text": embed_text,
        "record": {
            "id": row["id"],
            "country": row.get("country"),
            "competition": row.get("competition"),
            "problem_markdown": problem_text,
            "solutions_markdown": row.get("solutions_markdown") or [],
            "topics_flat": row.get("topics_flat") or [],
            "language": row.get("language"),
            "problem_type": row.get("problem_type"),
            "final_answer": row.get("final_answer"),
            "has_images": has_images,
            "num_images": num_images,
            "images_data": images_data,
        },
    }


def main() -> None:
    if not check_lm_studio():
        sys.exit(1)

    if not SUPABASE_URL:
        print("ERROR: SUPABASE_URL not set. Check your .env file.")
        sys.exit(1)

    supabase_key = SUPABASE_SERVICE_ROLE_KEY
    if not supabase_key:
        supabase_key = input("Enter Supabase service_role key: ").strip()
    
    if not supabase_key:
        print("ERROR: No key provided.")
        sys.exit(1)
    
    if not validate_supabase_key(supabase_key):
        print("WARNING: Key format looks invalid (should be JWT starting with 'eyJ')")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            sys.exit(1)

    try:
        supabase = create_client(SUPABASE_URL, supabase_key)
    except Exception as e:
        print(f"ERROR: Failed to create Supabase client: {e}")
        sys.exit(1)

    print("Fetching existing IDs from Supabase for resume support...")
    existing_ids = get_existing_ids(supabase)
    print(f"Found {len(existing_ids)} existing rows in database")

    print("Loading MathNet dataset from HuggingFace...")
    try:
        ds = load_dataset("ShadenA/MathNet", split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    total = len(ds)
    print(f"Loaded {total} problems from dataset")

    # Phase 1: Process all rows (images + text prep), skip existing
    print("Processing rows (skipping existing IDs)...")
    to_process = []
    skipped_no_text = 0
    skipped_existing = 0
    skipped_no_id = 0

    for i, row in enumerate(ds):
        if shutdown_requested:
            print("  (Shutdown requested, stopping processing)")
            break

        row_id = row.get("id")
        
        # Fix 1: Check for None id separately
        if row_id is None:
            skipped_no_id += 1
            continue
            
        if row_id in existing_ids:
            skipped_existing += 1
            continue

        result = process_row(row)
        if result is None:
            skipped_no_text += 1
            continue
        to_process.append(result)

        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i + 1}/{total} dataset rows, {len(to_process)} new to process...")

    print(f"Prepared {len(to_process)} new rows ({skipped_existing} existing, {skipped_no_id} no id, {skipped_no_text} no problem text)")

    if len(to_process) == 0:
        print("\nNothing to insert - all rows already in database!")
        return

    # Phase 2: Sequential batch embedding and insert
    print(f"Embedding and inserting sequentially (batch={EMBED_BATCH_SIZE})...")
    inserted = 0
    failed = 0
    db_batch = []

    i = 0
    while i < len(to_process) and not shutdown_requested:
        batch_items = to_process[i:i + EMBED_BATCH_SIZE]
        texts = [item["embed_text"] for item in batch_items]
        records = [item["record"] for item in batch_items]

        embeddings = embed_batch(texts)
        
        # Handle individual successes/failures
        for rec, emb in zip(records, embeddings):
            if emb is None:
                print(f"  Failed to embed {rec['id']}")
                failed += 1
                continue
            rec["embedding_2560"] = emb
            db_batch.append(rec)

            if len(db_batch) >= DB_BATCH_SIZE:
                try:
                    supabase.table("mathnet").insert(db_batch).execute()
                    inserted += len(db_batch)
                    print(f"  [{inserted}/{len(to_process)}] inserted to DB")
                except Exception as e:
                    print(f"  DB insert failed for batch: {e}")
                    # Try individual inserts for this batch
                    for rec in db_batch:
                        try:
                            supabase.table("mathnet").insert(rec).execute()
                            inserted += 1
                        except Exception as e2:
                            print(f"    Failed to insert {rec['id']}: {e2}")
                            failed += 1
                db_batch = []

        i += len(batch_items)

    # Flush remaining DB batch
    if db_batch and not shutdown_requested:
        try:
            supabase.table("mathnet").insert(db_batch).execute()
            inserted += len(db_batch)
            print(f"  [{inserted}/{len(to_process)}] inserted (final batch)")
        except Exception as e:
            print(f"  Final DB insert failed: {e}")
            # Try individual
            for rec in db_batch:
                try:
                    supabase.table("mathnet").insert(rec).execute()
                    inserted += 1
                except Exception as e2:
                    print(f"    Failed to insert {rec['id']}: {e2}")
                    failed += 1

    if shutdown_requested:
        print(f"\nShutdown complete. Inserted: {inserted}, Failed: {failed}")
        print(f"Resume with: python build_db.py")
    else:
        print(f"\nDone! Inserted: {inserted}, Skipped existing: {skipped_existing}, Skipped no id: {skipped_no_id}, Skipped no text: {skipped_no_text}, Failed: {failed}")


if __name__ == "__main__":
    main()
