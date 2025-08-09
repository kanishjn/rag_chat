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
import gc

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import google.generativeai as genai

# ---------------- Config ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"
VECTOR_STORE_DIR = "vector_store"

# ---------------- Pydantic Models ----------------
class QARequest(BaseModel):
    documents: str  # Document URL
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# ---------------- FastAPI App ----------------
app = FastAPI(
    title="Document Q&A API - Gemini (Optimized)",
    description="Memory-efficient, concise document Q&A using Gemini + Chroma",
    version="1.1.0"
)

# ---------------- Helpers ----------------
def load_document_from_url(url: str):
    try:
        response = requests.get(url, timeout=20)
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


def create_or_load_vector_store(documents=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    if os.path.exists(VECTOR_STORE_DIR) and not documents:
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embeddings
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=170)
        docs = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(
            docs, embeddings, persist_directory=VECTOR_STORE_DIR
        )

    return vector_store.as_retriever(search_kwargs={"k": 3})



# ---------------- API Endpoint ----------------
@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_run(request: QARequest):
    try:
        # Load and embed document only if new
        docs = load_document_from_url(request.documents)
        retriever = create_or_load_vector_store(docs)

        answers = []
        model = genai.GenerativeModel(MODEL_NAME)

        for q in request.questions:
            relevant_docs = retriever.invoke(q)
            if not relevant_docs:
                answers.append("Answer not found in context")
                continue

            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = (
                "You are a helpful assistant that answers questions using ONLY the provided context.\n"
                "Requirements:\n"
                " - Keep the answer in one short sentence (max 25 words).\n"
                " - If the context does not contain the answer, reply exactly: Answer not found in context\n"
                " - Return ONLY JSON in the form: {\"answer\": \"...\"}\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION: {q}\n\nProvide the JSON only."
            )

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 60
                }
            )

            raw_text = getattr(response, "text", "").strip()
            cleaned = raw_text.lstrip("```json").rstrip("```").strip()

            try:
                parsed = json.loads(cleaned)
                ans = parsed.get("answer", "").strip()
                if not ans:
                    ans = "Answer not found in context"
            except:
                ans = "Answer not found in context"

            answers.append(ans)

            # Free memory for each question
            gc.collect()

        return QAResponse(answers=answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
