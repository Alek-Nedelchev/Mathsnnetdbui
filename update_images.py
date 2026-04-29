"""Find all MathNet problems with images and update the database with image metadata.

This script ONLY updates has_images, num_images, and images_data columns.
It does NOT modify any existing data (problem_markdown, solutions, embeddings, etc.)
"""

import json
import sys
import io
import base64
from datasets import load_dataset
from supabase import create_client
from PIL import Image

SUPABASE_URL = "https://reonrsbfjbzhebjrfeia.supabase.co"
BATCH_SIZE = 50


def get_existing_image_data(supabase) -> dict:
    """Get existing IDs and their current image-related columns from the database."""
    existing = {}
    offset = 0
    page_size = 1000
    
    while True:
        resp = supabase.table("mathnet").select("id,has_images,num_images").range(offset, offset + page_size - 1).execute()
        for row in resp.data:
            existing[row["id"]] = {
                "has_images": row.get("has_images"),
                "num_images": row.get("num_images")
            }
        if len(resp.data) < page_size:
            break
        offset += page_size
    
    return existing


def encode_image_to_base64(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 encoded PNG string."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main():
    supabase_key = input("Enter Supabase service_role key: ").strip()
    if not supabase_key:
        print("ERROR: No key provided.")
        sys.exit(1)

    try:
        supabase = create_client(SUPABASE_URL, supabase_key)
    except Exception as e:
        print(f"ERROR: Failed to create Supabase client: {e}")
        sys.exit(1)

    print("Fetching existing IDs with image data from Supabase...")
    try:
        existing_ids = get_existing_image_data(supabase)
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

    print(f"Loaded {len(ds)} problems")
    
    # Build a map of dataset IDs to check existence
    print("Building dataset ID index...")
    dataset_ids = set()
    for row in ds:
        dataset_ids.add(row["id"])
    
    # First pass: identify all IDs with images from the dataset
    print("Scanning for problems with images...")
    ids_with_images = []
    for i, row in enumerate(ds):
        images = row.get("images") or []
        num_images = len(images) if images else 0
        if num_images > 0:
            ids_with_images.append({
                "id": row["id"],
                "num_images": num_images,
                "images": images
            })

    total_images = sum(x["num_images"] for x in ids_with_images)
    print(f"Found {len(ids_with_images)} problems with images ({total_images} total images)")

    # Second pass: update database for IDs that have images but aren't marked correctly
    updated = 0
    skipped_correct = 0
    skipped_missing = 0  # ID not in DB (shouldn't happen if build_db.py was run)
    failed = 0

    for item in ids_with_images:
        row_id = item["id"]
        
        # Skip if this ID doesn't exist in the database
        if row_id not in existing_ids:
            skipped_missing += 1
            continue
        
        # Check if already correctly marked in DB
        existing = existing_ids[row_id]
        if existing.get("has_images") is True and existing.get("num_images") == item["num_images"]:
            skipped_correct += 1
            continue
        
        # Prepare image data
        try:
            images_data = [
                {
                    "index": idx,
                    "format": "png",
                    "data": encode_image_to_base64(img)
                }
                for idx, img in enumerate(item["images"]) if img
            ]
        except Exception as e:
            print(f"  Failed to encode images for {row_id}: {e}")
            failed += 1
            continue
        
        # Update ONLY the image-related columns for this specific row
        # Uses .eq() to target specific ID, then .update() to only change specified columns
        try:
            supabase.table("mathnet").update({
                "has_images": True,
                "num_images": item["num_images"],
                "images_data": json.dumps(images_data)
            }).eq("id", row_id).execute()
            updated += 1
            if updated % 10 == 0:
                print(f"  Updated {updated} records...")
        except Exception as e:
            failed += 1
            print(f"  Update failed for {row_id}: {e}")

    print(f"\nDone!")
    print(f"  Updated: {updated}")
    print(f"  Skipped (already correct): {skipped_correct}")
    print(f"  Skipped (ID not in DB): {skipped_missing}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
