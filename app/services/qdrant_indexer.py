
import os
import json
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# It's recommended to move these settings to a .env file and use a library like python-dotenv to load them.
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "./elasticsearch_chunks")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME", "dcet_documents_metadata")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "neuralmind/bert-base-portuguese-cased")
BATCH_SIZE = int(os.getenv("QDRANT_INDEXER_BATCH_SIZE", 32))


def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {embedding_dim}")
        return model, embedding_dim
    except Exception as e:
        print(f"Error loading {EMBEDDING_MODEL}: {e}")
        return None, None


def initialize_qdrant_client():
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
        print("Connected to Qdrant successfully")
        return client
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return None


def create_collection(client, embedding_dim, recreate_if_exists=False):
    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        if collection_exists:
            if recreate_if_exists:
                client.delete_collection(COLLECTION_NAME)
            else:
                return True
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim, distance=models.Distance.COSINE)
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME, field_name="document_id", field_schema="keyword")
        client.create_payload_index(
            collection_name=COLLECTION_NAME, field_name="title", field_schema="text")
        client.create_payload_index(
            collection_name=COLLECTION_NAME, field_name="url_host", field_schema="keyword")
        client.create_payload_index(
            collection_name=COLLECTION_NAME, field_name="url_path_dir1", field_schema="keyword")
        return True
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False


def load_chunks_from_directory():
    chunks_data = []
    chunks_path = Path(CHUNKS_DIR)
    if not chunks_path.exists():
        return []
    doc_dirs = [d for d in chunks_path.iterdir() if d.is_dir()]
    for doc_dir in doc_dirs:
        chunk_files = sorted([f for f in doc_dir.iterdir(
        ) if f.name.startswith('chunk_') and f.suffix == '.txt'])
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                content = f.read()
            metadata_file = chunk_file.with_name(
                f"{chunk_file.stem}_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    chunk_metadata = json.load(f)
                full_metadata = {'content': content, **chunk_metadata}
                chunks_data.append(full_metadata)
    return chunks_data


def prepare_payload(chunk_data):
    payload = {
        'document_id': chunk_data.get('document_id', ''),
        'chunk_index': chunk_data.get('chunk_index', 0),
        'content': chunk_data.get('content', ''),
        'title': chunk_data.get('title', ''),
        'url': chunk_data.get('url', ''),
        'url_host': chunk_data.get('url_host', ''),
        'url_path': chunk_data.get('url_path', ''),
    }
    return payload


def embed_and_index_chunks(model, client, chunks_data):
    total_indexed = 0
    for i in tqdm(range(0, len(chunks_data), BATCH_SIZE), desc="Processing batches"):
        batch = chunks_data[i:i + BATCH_SIZE]
        texts = [chunk.get('content', '') for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        points = []
        for chunk_data, embedding in zip(batch, embeddings):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=prepare_payload(chunk_data)
            ))
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_indexed += len(points)
        except Exception as e:
            print(f"Error indexing batch: {e}")
            continue
    return total_indexed


def index_documents(recreate_collection=False):
    model, embedding_dim = load_embedding_model()
    if not model:
        return {"status": "error", "message": "Failed to load embedding model"}
    client = initialize_qdrant_client()
    if not client:
        return {"status": "error", "message": "Failed to connect to Qdrant"}
    if not create_collection(client, embedding_dim, recreate_if_exists=recreate_collection):
        return {"status": "error", "message": "Failed to create or access Qdrant collection"}
    chunks_data = load_chunks_from_directory()
    if not chunks_data:
        return {"status": "error", "message": "No chunks to process"}
    total_indexed = embed_and_index_chunks(model, client, chunks_data)
    return {"status": "complete", "indexed_chunks": total_indexed}
