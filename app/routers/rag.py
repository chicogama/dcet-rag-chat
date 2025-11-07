
from fastapi import APIRouter, HTTPException
from app.services.rag_service import RAGSystem
from app.schemas.rag import QuestionRequest, AnswerResponse

router = APIRouter()
rag_system = RAGSystem()


@router.post("/answer", response_model=AnswerResponse)
def answer_question(request: QuestionRequest):
    """
    Endpoint to receive a question and return an answer from the RAG system.
    """
    try:
        result = rag_system.answer_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
