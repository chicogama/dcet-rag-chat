
from fastapi import APIRouter, HTTPException
from app.services.data_loader import export_all_documents
from app.services.qdrant_indexer import index_documents
from app.schemas.rag import IngestResponse, IndexResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_data():
    """
    Endpoint to trigger the data ingestion process from Elasticsearch.
    """
    try:
        result = export_all_documents()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index", response_model=IndexResponse)
def index_data(recreate_collection: bool = False):
    """
    Endpoint to trigger the indexing process into Qdrant.
    """
    try:
        result = index_documents(recreate_collection)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
