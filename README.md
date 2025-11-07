### RAG CHAT DCET

# RAG System API

This project implements a Retrieval-Augmented Generation (RAG) system with a FastAPI backend and a Streamlit frontend. The system is designed to answer questions based on a collection of documents indexed in a Qdrant vector database.

## Features

- **FastAPI Backend**: A robust and fast API for handling data ingestion, indexing, and answering questions.
- **Streamlit Frontend**: A simple and intuitive user interface to interact with the RAG system.
- **Qdrant Integration**: Utilizes Qdrant as the vector store for efficient similarity search.
- **Modular Architecture**: The project is structured with separate routers for different functionalities (data and RAG).

## Requirements

- Python 3.x
- Conda (optional, but recommended for environment management)
- The dependencies listed in `requirements.txt`:
  - `fastapi`
  - `uvicorn`
  - `requests`
  - `sentence-transformers`
  - `qdrant-client`
  - `pydantic`
  - `python-dotenv`
  - `streamlit`

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd to-api
   ```

2. **Create and activate a Conda environment (recommended):**

   ```bash
   conda create --name rag-api python=3.12
   conda activate rag-api
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory of the project and add the necessary environment variables. For example:
   ```
   QDRANT_URL="http://localhost:6333"
   QDRANT_API_KEY="your-qdrant-api-key"
   EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
   ```

## Usage

1. **Run the FastAPI server:**

   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`.

2. **Run the Streamlit application:**
   In a separate terminal, run the following command:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   The Streamlit app will be available at `http://localhost:8501`.

## API Endpoints

The API provides the following endpoints:

### Data

- **`POST /data/ingest`**: Triggers the data ingestion process from the source.

  - **Response:**
    ```json
    {
      "status": "success",
      "message": "Data ingested successfully."
    }
    ```

- **`POST /data/index`**: Triggers the indexing process into Qdrant.
  - **Query Parameter:** `recreate_collection` (boolean, optional) - If `true`, it will delete the existing collection before indexing.
  - **Response:**
    ```json
    {
      "status": "success",
      "message": "Documents indexed successfully."
    }
    ```

### RAG

- **`POST /rag/answer`**: Receives a question and returns an answer from the RAG system.
  - **Request Body:**
    ```json
    {
      "question": "Your question here"
    }
    ```
  - **Response:**
    ```json
    {
      "answer": "The answer from the RAG system."
    }
    ```
