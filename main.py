import os
import uvicorn
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import requests
import tempfile
import json

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Gemini Imports ---
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

# Configure Gemini for direct API calls
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# --- Constants ---
VECTOR_STORE_DIR = "vector_store"

# --- Pydantic Models ---
class QARequest(BaseModel):
    documents: str  # URL of the document
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI(
    title="Document Q&A API - Gemini",
    description="Answers questions based on the provided document(s) using Gemini API.",
    version="1.0.0"
)

# --- Function: Download and Load Document ---
def load_document_from_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_ext = os.path.splitext(url.split('?')[0])[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_ext == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        elif file_ext == ".txt":
            loader = TextLoader(tmp_file_path)
        else:
            raise ValueError(f"Unsupported file format '{file_ext}'. Only PDF, DOCX, and TXT are supported.")

        docs = loader.load()
        os.remove(tmp_file_path)
        return docs
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {str(e)}")
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to load document: {str(e)}")

# --- Function: Create Vector Store ---
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # --- CORRECTED LINE: Explicitly pass the API key ---
    # This ensures LangChain's embedding model uses your key and doesn't search for ADC.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=None)
    return vector_store.as_retriever(search_kwargs={"k": 5})

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_run(request: QARequest):
    try:
        documents = load_document_from_url(request.documents)
        retriever = create_vector_store(documents)

        qa_pairs = []
        for q in request.questions:
            relevant_docs = retriever.get_relevant_documents(q)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            qa_pairs.append({"question": q, "context": context})

        prompt_parts = [
            "You are a helpful assistant that answers questions based ONLY on the provided context.",
            "You must return the answers strictly in the following JSON format:",
            '{"answers": ["answer for question 1", "answer for question 2", ...]}',
            "If the context does not contain the answer to a question, you should state 'Answer not found in context'.",
            "\n---",
            "CONTEXT AND QUESTIONS:",
        ]
        for i, pair in enumerate(qa_pairs, start=1):
            prompt_parts.append(f"\nQuestion {i}: {pair['question']}\nContext:\n{pair['context']}\n---")
        prompt_parts.append("Now, provide all the answers in the JSON format described above.")

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content("\n".join(prompt_parts))

        try:
            cleaned_response = response.text.strip().lstrip("```json").rstrip("```")
            result_json = json.loads(cleaned_response)
        except (json.JSONDecodeError, AttributeError):
            raise HTTPException(status_code=500, detail=f"Gemini response was not in the expected JSON format. Raw response: {response.text}")

        if "answers" not in result_json or not isinstance(result_json["answers"], list):
             raise HTTPException(status_code=500, detail=f"JSON from Gemini is missing the 'answers' list. Response: {result_json}")

        return QAResponse(answers=result_json["answers"])

    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {str(e)}")

# --- Run Server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)