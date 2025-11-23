# 

This repository contains AI Agents samples, implemented in Python using FastAPI, Streamlit, ChromaDB, and OpenAI/Azure.

## Project Structure
- `app/`: Core backend logic.
  - `chat.py`: Conversational core (Task 3.1).
  - `rag.py`: RAG engine with ingestion and eval (Task 3.2).
  - `agent.py`: Tool-calling agent (Task 3.3).
  - `coder.py`: Self-healing code generation loop (Task 3.4).
- `ui.py`: Streamlit dashboard (Stretch Goal).
- `Dockerfile` & `docker-compose.yml`: Infrastructure.

## Setup & Running
1. **Environment Variables**: The `.env` file is pre-populated with the credentials provided in for the task.

2. **Docker Run (Recommended)**:
   ```bash
   docker-compose up --build
   
Navigate to localhost:8501 to access UI and localhost:8000/docs to access backend APIs.
