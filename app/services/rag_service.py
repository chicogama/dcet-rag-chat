
import requests
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

# It's recommended to move these settings to a .env file and use a library like python-dotenv to load them.
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME", "dcet_documents_metadata")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "neuralmind/bert-base-portuguese-cased")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
TOP_K_RESULTS = int(os.getenv("RAG_TOP_K_RESULTS", 10))
API_KEY = os.getenv("QDRANT_API_KEY", "teste")

SYSTEM_PROMPT = '''Você é um assistente especializado do Departamento de Ciências Exatas e Tecnológicas (DCET) da Universidade Federal do Amapá (UNIFAP).

Sua função é responder perguntas sobre:
- Cursos de graduação (Engenharia, Arquitetura, Ciência da Computação, etc.)
- Editais, processos seletivos e programas de monitoria
- Estrutura departamental e corpo docente
- Eventos, reuniões e atividades acadêmicas
- Normas e regulamentos do departamento

Use as informações fornecidas no contexto para responder de forma precisa e completa. Se a informação não estiver disponível no contexto, indique isso claramente.

Seja profissional, objetivo e sempre em português do Brasil.'''


class RAGSystem:
    def __init__(self):
        self.model = self._load_embedding_model()

    def _load_embedding_model(self):
        try:
            model = SentenceTransformer(EMBEDDING_MODEL)
            return model
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return None

    def generate_query_embedding(self, query: str) -> List[float]:
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        embedding = self.model.encode(query, show_progress_bar=False)
        return embedding.tolist()

    def search_qdrant(self, query_vector: List[float], top_k: int = TOP_K_RESULTS) -> List[Dict]:
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/points/search"
        headers = {"Content-Type": "application/json", "api-key": API_KEY}
        payload = {"vector": query_vector, "limit": top_k,
                   "with_payload": True, "with_vector": False}
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json().get('result', [])
        except Exception as e:
            print(f"Error in Qdrant search: {e}")
            return []

    def format_context(self, search_results: List[Dict]) -> str:
        if not search_results:
            return "Nenhum contexto relevante encontrado."
        context_parts = []
        for idx, result in enumerate(search_results, 1):
            score = result.get('score', 0)
            content = result.get('payload', {}).get('content', '')
            context_parts.append(
                f"[Documento {idx} - Relevância: {score:.2f}]\n{content}\n")
        return "\n".join(context_parts)

    def generate_with_ollama(self, query: str, context: str) -> str:
        url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
        prompt = f'{SYSTEM_PROMPT}\n\nCONTEXTO:\n{context}\n\nPERGUNTA DO USUÁRIO:\n{query}\n\nRESPOSTA:'
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('response', 'Erro ao gerar resposta.')
        except requests.exceptions.Timeout:
            return "Error: Timeout while generating response."
        except Exception as e:
            return f"Error generating response: {e}"

    def answer_question(self, question: str):
        query_vector = self.generate_query_embedding(question)
        search_results = self.search_qdrant(query_vector)
        context = self.format_context(search_results)
        answer = self.generate_with_ollama(question, context)
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "num_sources": len(search_results)
        }
