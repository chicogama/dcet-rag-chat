from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class IngestResponse(BaseModel):
    status: str
    exported_docs: int
    total_chunks: int


class IndexResponse(BaseModel):
    status: str
    indexed_chunks: int
