
import requests
import json
import os
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = os.getenv("ES_PORT", "9200")
INDEX_NAME = os.getenv("ES_INDEX_NAME", "dcet-unifap-3")
OUTPUT_DIR = os.getenv("CHUNKS_DIR", "./elasticsearch_chunks")
BATCH_SIZE = int(os.getenv("DATA_LOADER_BATCH_SIZE", 100))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 512))

ESSENTIAL_METADATA_FIELDS = [
    'title', 'url', 'last_crawled_at', 'headings', 'url_path', 'url_host'
]
OPTIONAL_METADATA_FIELDS = [
    'links', 'url_scheme', 'url_port', 'url_path_dir1', 'url_path_dir2'
]


def create_output_directory():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def sentence_splitter(text):
    if not text or not text.strip():
        return []
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(text, max_size=MAX_CHUNK_SIZE):
    if not text or not text.strip():
        return []
    sentences = sentence_splitter(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(sentence) > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_size:
                    temp_chunk += word + " "
                else:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = word + " "
            if temp_chunk:
                current_chunk = temp_chunk
        else:
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def extract_metadata(source_data):
    metadata = {}
    for field in ESSENTIAL_METADATA_FIELDS:
        if field in source_data:
            value = source_data[field]
            metadata[field] = value[:5] if isinstance(value, list) else value
    for field in OPTIONAL_METADATA_FIELDS:
        if field in source_data:
            value = source_data[field]
            metadata[field] = value[:10] if isinstance(value, list) else value
    return metadata


def fetch_documents_batch(from_offset=0, size=BATCH_SIZE):
    url = f"http://{ES_HOST}:{ES_PORT}/{INDEX_NAME}/_search"
    headers = {"Content-Type": "application/json", "kbn-xsrf": "reporting"}
    payload = {"from": from_offset, "size": size,
               "query": {"match_all": {}}, "_source": True}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching documents: {e}")
        return None


def save_chunks_with_metadata(doc_id, metadata, chunks):
    doc_dir = os.path.join(OUTPUT_DIR, doc_id)
    Path(doc_dir).mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for idx, chunk in enumerate(chunks):
        chunk_metadata = {
            **metadata,
            "chunk_index": idx,
            "chunk_total": len(chunks),
            "chunk_text": chunk,
            "chunk_length": len(chunk),
            "document_id": doc_id
        }
        chunk_filename = f"chunk_{idx:04d}.txt"
        chunk_filepath = os.path.join(doc_dir, chunk_filename)
        try:
            with open(chunk_filepath, 'w', encoding='utf-8') as f:
                f.write(chunk)
            metadata_filename = f"chunk_{idx:04d}_metadata.json"
            metadata_filepath = os.path.join(doc_dir, metadata_filename)
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
            saved_count += 1
        except Exception as e:
            print(f"Error saving chunk {idx} of document {doc_id}: {e}")
    doc_metadata = {
        "document_id": doc_id,
        "total_chunks": len(chunks),
        "chunk_sizes": [len(c) for c in chunks],
        "max_chunk_size": MAX_CHUNK_SIZE,
        "original_metadata": metadata,
        "export_timestamp": datetime.now().isoformat()
    }
    doc_metadata_path = os.path.join(doc_dir, "_document_metadata.json")
    with open(doc_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(doc_metadata, f, indent=2, ensure_ascii=False)
    return saved_count


def export_all_documents():
    create_output_directory()
    initial_response = fetch_documents_batch(from_offset=0, size=1)
    if not initial_response:
        return {"status": "error", "message": "Failed to connect to Elasticsearch"}
    total_docs = initial_response['hits']['total']['value']
    exported_docs = 0
    total_chunks = 0
    from_offset = 0
    while from_offset < total_docs:
        response = fetch_documents_batch(
            from_offset=from_offset, size=BATCH_SIZE)
        if not response or 'hits' not in response:
            break
        hits = response['hits']['hits']
        if not hits:
            break
        for hit in hits:
            doc_id = hit['_id']
            source_data = hit.get('_source', {})
            body_content = source_data.get('body', '')
            if body_content:
                metadata = extract_metadata(source_data)
                chunks = chunk_by_sentences(body_content, MAX_CHUNK_SIZE)
                saved = save_chunks_with_metadata(doc_id, metadata, chunks)
                if saved > 0:
                    exported_docs += 1
                    total_chunks += saved
        from_offset += BATCH_SIZE
    global_metadata = {
        "index": INDEX_NAME,
        "total_documents": exported_docs,
        "total_chunks": total_chunks,
        "export_timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(OUTPUT_DIR, "_export_metadata.json"), 'w') as f:
        json.dump(global_metadata, f, indent=2, ensure_ascii=False)
    return {"status": "complete", "exported_docs": exported_docs, "total_chunks": total_chunks}
