#!/usr/bin/env python3
"""Build embeddings from MathNet dataset and upload to both ChromaDB and Qdrant."""

import json
import os
import base64
import numpy as np
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "moqsearch")

CHROMA_PATH = "./moqsearch_db"
DATASET_NAME = "ShadenA/MathNet"
EMBED_MODEL = "text-embedding-qwen3-embedding-4b"  # LM Studio model name
EMBED_DIMENSIONS = 2560
BATCH_SIZE = 128  # Embedding batch size for LM Studio
UPLOAD_BATCH_SIZE = 100  # Qdrant upload batch size

# Initialize LM Studio client
lm_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")


def check_lm_studio():
    """Check if LM Studio is running and accessible."""
    try:
        response = lm_client.embeddings.create(
            model=EMBED_MODEL,
            input=["test"]
        )
        return True
    except Exception as e:
        print(f"LM Studio connection failed: {e}")
        return False


def normalize_vector(v):
    """Normalize a vector to unit length for cosine similarity."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_embedding_lm_studio(texts):
    """Get embeddings from LM Studio local server."""
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        
        response = lm_client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        
        # Extract and normalize embeddings
        batch_embeddings = [
            normalize_vector(np.array(item.embedding)).tolist()
            for item in response.data
        ]
        all_embeddings.extend(batch_embeddings)
        
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    
    return all_embeddings


def format_list(value):
    """Format list fields for serialization."""
    if value is None or not value:
        return None
    if isinstance(value, list):
        return " | ".join(str(v) for v in value if v)
    return str(value)


def serialize_row(row):
    """Serialize a dataset row into text for embedding."""
    parts = []
    
    country = format_list(row.get("country"))
    competition = format_list(row.get("competition"))
    topics_flat = format_list(row.get("topics_flat"))
    problem_markdown = row.get("problem_markdown")
    solutions_markdown = row.get("solutions_markdown")
    
    if country:
        parts.append(f"Country: {country}")
    if competition:
        parts.append(f"Competition: {competition}")
    if topics_flat:
        parts.append(f"Topics: {topics_flat}")
    if problem_markdown:
        parts.append(f"Problem: {problem_markdown}")
    if solutions_markdown:
        parts.append(f"Solutions: {solutions_markdown}")
    
    return "\n".join(parts)


def extract_images(row):
    """Extract and encode images from row."""
    images_data = []
    
    for col in row.keys():
        if 'images' in col.lower():
            image_val = row.get(col)
            if image_val is None or image_val == "":
                continue
            
            try:
                # Handle different image formats
                if isinstance(image_val, bytes):
                    b64 = base64.b64encode(image_val).decode('utf-8')
                    images_data.append({"format": "png", "data": b64})
                elif isinstance(image_val, str) and len(image_val) > 100:
                    # Assume it's already base64 or a data URL
                    if image_val.startswith('data:image'):
                        # Extract base64 from data URL
                        b64 = image_val.split(',')[1] if ',' in image_val else image_val
                        fmt = image_val.split(';')[0].split('/')[-1] if '/' in image_val else 'png'
                        images_data.append({"format": fmt, "data": b64})
                    else:
                        # Raw base64 string
                        images_data.append({"format": "png", "data": image_val})
            except Exception as e:
                print(f"    Warning: Could not process image in {col}: {e}")
                continue
    
    return images_data


def get_metadata(row):
    """Extract metadata for storage."""
    metadata = {}
    
    # Basic metadata fields
    if row.get("language"):
        metadata["language"] = row["language"]
    if row.get("problem_type"):
        metadata["problem_type"] = row["problem_type"]
    if row.get("final_answer"):
        metadata["final_answer"] = str(row["final_answer"])
    if row.get("country"):
        metadata["country"] = row["country"] if isinstance(row["country"], str) else format_list(row["country"])
    if row.get("competition"):
        metadata["competition"] = row["competition"] if isinstance(row["competition"], str) else format_list(row["competition"])
    if row.get("topics_flat"):
        metadata["topics_flat"] = json.dumps(row["topics_flat"]) if isinstance(row["topics_flat"], list) else row["topics_flat"]
    
    # Image metadata
    images_data = extract_images(row)
    if images_data:
        metadata["images_data"] = json.dumps(images_data)
        metadata["has_images"] = True
        metadata["num_images"] = len(images_data)
    else:
        metadata["has_images"] = False
        metadata["num_images"] = 0
    
    return metadata


def build_and_upload():
    """Main function to build embeddings and upload to both databases."""
    
    # Check LM Studio connection
    print("Checking LM Studio connection...")
    if not check_lm_studio():
        print("ERROR: LM Studio is not running!")
        print("Please:")
        print("  1. Open LM Studio")
        print("  2. Load the Qwen3 embedding model")
        print("  3. Start the local server on port 1234")
        return
    print("LM Studio connected successfully!")
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("WARNING: QDRANT_URL or QDRANT_API_KEY not set. Will only create local ChromaDB.")
        upload_to_qdrant = False
    else:
        upload_to_qdrant = True
    
    # Connect to Qdrant
    qdrant_client = None
    if upload_to_qdrant:
        print(f"Connecting to Qdrant at {QDRANT_URL}...")
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Setup ChromaDB
    print(f"Setting up ChromaDB at {CHROMA_PATH}...")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Delete existing Chroma collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing Chroma collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    
    # Create Chroma collection with cosine similarity
    chroma_collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Delete existing Qdrant collection
    if upload_to_qdrant:
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing Qdrant collection '{COLLECTION_NAME}'")
        except Exception:
            pass
    
    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    total = len(dataset)
    print(f"Found {total} problems to process")
    
    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = dataset.select(range(batch_start, batch_end))
        
        print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(total//BATCH_SIZE)+1} ({batch_start+1}-{batch_end})...")
        
        # Prepare data
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in enumerate(batch):
            doc_id = str(row.get("id", f"problem_{batch_start + idx}"))
            doc_text = serialize_row(row)
            
            if doc_text:
                documents.append(doc_text)
                metadatas.append(get_metadata(row))
                ids.append(doc_id)
        
        if not documents:
            print("  No valid documents in batch, skipping...")
            continue
        
        # Generate embeddings
        print(f"  Generating embeddings for {len(documents)} documents...")
        try:
            embeddings = get_embedding_lm_studio(documents)
        except Exception as e:
            print(f"  ERROR: Failed to generate embeddings: {e}")
            continue
        
        # Add to ChromaDB
        print(f"  Adding to ChromaDB...")
        chroma_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        # Prepare Qdrant points
        if upload_to_qdrant:
            qdrant_points = []
            for i, (id, emb, meta, doc) in enumerate(zip(ids, embeddings, metadatas, documents)):
                # Prepare payload - flatten metadata, encode JSON fields as strings
                payload = {
                    "document": doc,
                    "country": meta.get("country"),
                    "competition": meta.get("competition"),
                    "language": meta.get("language"),
                    "problem_type": meta.get("problem_type"),
                    "final_answer": meta.get("final_answer"),
                    "has_images": meta.get("has_images", False),
                    "num_images": meta.get("num_images", 0),
                    "images_data": meta.get("images_data"),  # Already JSON string
                    "topics_flat": meta.get("topics_flat")  # Already JSON string
                }
                
                qdrant_points.append(PointStruct(
                    id=id,
                    vector=emb,
                    payload=payload
                ))
            
            # Upload to Qdrant in smaller batches
            print(f"  Uploading to Qdrant...")
            for i in range(0, len(qdrant_points), UPLOAD_BATCH_SIZE):
                batch_points = qdrant_points[i:i + UPLOAD_BATCH_SIZE]
                
                # Create collection on first batch
                if batch_start == 0 and i == 0:
                    qdrant_client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=VectorParams(size=EMBED_DIMENSIONS, distance=Distance.COSINE)
                    )
                
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch_points
                )
                print(f"    Uploaded {min(i + len(batch_points), len(qdrant_points))}/{len(qdrant_points)}")
        
        print(f"  ✓ Batch complete: {batch_end}/{total}")
    
    # Final stats
    print(f"\n{'='*60}")
    print("BUILD COMPLETE!")
    print(f"{'='*60}")
    print(f"Total problems processed: {total}")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"  Path: {CHROMA_PATH}")
    print(f"  Count: {chroma_collection.count()}")
    
    if upload_to_qdrant:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Qdrant collection: {COLLECTION_NAME}")
        print(f"  URL: {QDRANT_URL}")
        print(f"  Count: {collection_info.points_count}")
        print(f"  Distance: COSINE")
    
    print(f"\nNext steps:")
    print(f"  1. Test locally: python build_and_upload.py")
    print(f"  2. Deploy worker: cd worker && npm run deploy")
    print(f"  3. Set WORKER_URL in app.js")
    print(f"\nNote: OpenRouter is only needed for the worker to generate query embeddings.")
    print(f"      Set OPENROUTER_API_KEY in Cloudflare Worker secrets (not in .env).")


if __name__ == "__main__":
    build_and_upload()
