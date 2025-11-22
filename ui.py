import streamlit as st
import requests
import json
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="AI Agents samples")
st.title("AI Agents samples")

tabs = st.tabs(["3.1 Conversational Core", "3.2 RAG", "3.3 Planning Agent", "3.4 Self-Healing Coder"])

# --- Task 3.1: Chat ---
with tabs[0]:
    st.header("Conversational Core & Telemetry")
    
    # Initialize or load chat history
    if "chat_history" not in st.session_state:
        try:
            # Fetch last 10 messages from the backend
            response = requests.get(f"{BACKEND_URL}/chat/history")
            response.raise_for_status()
            st.session_state.chat_history = response.json().get("history", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Could not load chat history: {e}")
            st.session_state.chat_history = []
            
    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "metrics" in msg:
                m = msg["metrics"]
                st.caption(f"Lat: {m['latency_ms']}ms | Cost: ${m['cost_usd']} | In: {m['prompt_tokens']} / Out: {m['completion_tokens']}")

    if prompt := st.chat_input("Type a message..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            box = st.empty()
            text = ""
            metrics = None
            
            with requests.post(f"{BACKEND_URL}/chat", json={"message": prompt}, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        if data["type"] == "content":
                            text += data["data"]
                            box.markdown(text + "▌")
                        elif data["type"] == "metrics":
                            metrics = data["data"]
                        elif data["type"] == "error":
                            st.error(data["data"])
                            
            box.markdown(text)
            if metrics:
                st.caption(f"Lat: {metrics['latency_ms']}ms | Cost: ${metrics['cost_usd']}")
                
            st.session_state.chat_history.append({"role": "assistant", "content": text, "metrics": metrics})

# --- Task 3.2: RAG ---
with tabs[1]:
    st.header("High-Performance RAG")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ingested Documents")
        try:
            docs = requests.get(f"{BACKEND_URL}/rag/ingested-docs").json()['documents']
            for doc in docs:
                st.success(doc)
        except Exception as e:
            st.error(f"Could not load ingested documents. Is the backend running? \n\n{e}")

    with col2:
        st.subheader("Automated Eval")
        if st.button("Run Top-5 Accuracy Test"):
            with st.spinner("Evaluating..."):
                res = requests.post(f"{BACKEND_URL}/rag/eval").json()
                st.metric("Retrieval Accuracy", f"{res['accuracy']*100:.1f}%")
                with st.expander("Details"):
                    st.json(res['details'])

    st.divider()
    st.subheader("QA Endpoint")
    q = st.text_input("Ask about the document:")
    if st.button("Search"):
        with st.spinner("Searching..."):
            res = requests.post(f"{BACKEND_URL}/rag/query", json={"query": q}).json()
            st.info(res['answer'])
            st.metric("Retrieval Latency", f"{res['retrieval_latency_ms']:.1f} ms")
            
            with st.expander("Citations"):
                for citation in res['citations']:
                    st.write(citation)

# --- Task 3.3: Agent ---
with tabs[2]:
    st.header("Autonomous Planning Agent")
    goal = st.text_area("Goal", "Plan a 2-day trip to Auckland for under NZ$500")
    if st.button("Run Planner"):
        with st.spinner("Agent is thinking (and calling tools)..."):
            res = requests.post(f"{BACKEND_URL}/agent", json={"prompt": goal}).json()
            
        st.subheader("Reasoning Logs")
        for log in res['logs']:
            st.code(log, language="text")
            
        st.subheader("Final Plan")
        st.markdown(res['plan'])

# --- Task 3.4: Coder ---
with tabs[3]:
    st.header("Self-Healing Code Assistant")
    task = st.text_input("Coding Task", "Write a python function to calculate fibonacci sequence recursively and test it.")
    if st.button("Generate Code"):
        box = st.empty()
        output = ""
        with requests.post(f"{BACKEND_URL}/coder", json={"prompt": task}, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    output += line.decode('utf-8') + "\n"
                    box.text(output)
