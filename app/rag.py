import os
import uuid
import requests
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from app.config import CHROMA_PATH
from app.utils import Timer
import openai
from app.config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME

# Setup ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use a local lightweight embedding model for speed (<=300ms target)
# all-MiniLM-L6-v2 is fast and effective for standard English text
EMBED_MODEL = "all-MiniLM-L6-v2"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

collection = client.get_or_create_collection(
    name="data_knowledge",
    embedding_function=embed_fn
)

llm_client = openai.AzureOpenAI(
    azure_endpoint=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    api_version="2024-05-01-preview"
)

def ingest_document(url: str, source_name: str):
    """Task 3.2a: Ingest PDF, chunk, and store."""
    try:
        # Download
        response = requests.get(url)
        response.raise_for_status()
        temp_path = os.path.join("data", "temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(response.content)
            
        # Extract Text
        reader = PdfReader(temp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        os.remove(temp_path)
        
        # Chunking (Recursive-like sliding window)
        chunk_size = 500 # chars
        overlap = 50
        chunks = []
        metadatas = []
        ids = []
        
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk_text = full_text[i:i + chunk_size]
            chunks.append(chunk_text)
            metadatas.append({"source": source_name, "index": i})
            ids.append(str(uuid.uuid4()))
            
        # Store
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return {"status": "success", "chunks_count": len(chunks)}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def query_knowledge_base(query: str):
    """Task 3.2c: QA Endpoint with <300ms retrieval."""
    timer = Timer()
    timer.start()
    
    # 1. Retrieval (Fast local embedding + HNSW search)
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    retrieval_latency = timer.stop()
    
    context = ""
    citations = []
    
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            context += f"[Citation {i+1}, Source: {source}]: {doc}\n\n"
            citations.append(f"[{i+1}] {source}")
            
    # 2. Generation
    prompt = f"""
    You are a helpful assistant. Answer the user's question using ONLY the context below.
    If the answer is not in the context, say "I don't know".
    
    Context:
    {context}
    
    User Question: {query}
    """
    
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "retrieval_latency_ms": retrieval_latency,
        "citations": citations
    }

def run_automated_eval():
    """Task 3.2d: Automated Eval Script."""
    # Answer Key (Subset for demonstration)
    qa_pairs = [
        ("Who made the One Ring?", "Sauron"),
        ("Where was the One Ring made?", "Mount Doom"),
        ("Who is the bearer of the One Ring?", "Frodo"),
        ("What is the name of the wizard?", "Gandalf"),
        ("Who is Frodo's companion?", "Sam"),
        ("What is Sting?", "sword"),
        ("Where is Rivendell?", "valley"),
        ("Who is Aragorn?", "Ranger"),
        ("What is Legolas?", "Elf"),
        ("What is Gimli?", "Dwarf")
    ]
    
    hits = 0
    results = []
    
    for q, expected in qa_pairs:
        res = collection.query(query_texts=[q], n_results=5)
        retrieved_docs = " ".join(res['documents'][0]).lower()
        
        # Rough check: if key concept word is in retrieved chunks
        success = expected.lower() in retrieved_docs
        if success:
            hits += 1
        results.append({"question": q, "success": success})
        
    accuracy = hits / len(qa_pairs)
    return {"accuracy": accuracy, "details": results}
