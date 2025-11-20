import sqlite3
import json
import openai
from typing import List, Dict, Generator
from app.config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, SQLITE_PATH
from app.utils import count_tokens, calculate_cost, Timer

client = openai.AzureOpenAI(
    azure_endpoint=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    api_version="2024-05-01-preview"
)

def init_db():
    """Initialize SQLite database for chat history."""
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_message(role: str, content: str):
    """Save message and prune to keep last 10."""
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        # Prune: Keep only last 10 rows
        conn.execute("""
            DELETE FROM messages WHERE id NOT IN (
                SELECT id FROM messages ORDER BY id DESC LIMIT 10
            )
        """)

def get_history() -> List[Dict[str, str]]:
    """Retrieve last 10 messages."""
    try:
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.execute("SELECT role, content FROM messages ORDER BY id ASC")
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    except Exception:
        return []

def stream_chat(user_message: str) -> Generator[str, None, None]:
    """
    Task 3.1: Conversational Core
    - Streams response
    - Persists N=10 messages
    - Calculates telemetry (latency, tokens, cost)
    """
    init_db()
    
    # 1. Save User Message
    save_message("user", user_message)
    
    # 2. Prepare Context
    history = get_history()
    
    # 3. Metrics Setup
    timer = Timer()
    timer.start()
    
    # 4. Call LLM
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=history,
            stream=True
        )
        
        full_content = ""
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                # Yield partial content for streaming UI
                yield json.dumps({"type": "content", "data": content}) + "\n"
        
        # 5. Finalize Metrics
        latency_ms = timer.stop()
        save_message("assistant", full_content)
        
        prompt_tokens = sum(count_tokens(m['content']) for m in history)
        completion_tokens = count_tokens(full_content)
        cost = calculate_cost(prompt_tokens, completion_tokens)
        
        metrics = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "latency_ms": round(latency_ms)
        }
        
        yield json.dumps({"type": "metrics", "data": metrics}) + "\n"
        
    except Exception as e:
        yield json.dumps({"type": "error", "data": str(e)}) + "\n"
