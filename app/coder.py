import os
import subprocess
import openai
import unittest
from app.config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, CODE_ENV_PATH

client = openai.AzureOpenAI(
    azure_endpoint=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    api_version="2024-05-01-preview"
)

def generate_and_heal_code(task_description: str):
    """Task 3.4: Self-Healing Code Assistant."""
    
    system_prompt = """
    You are a Python coding assistant. 
    1. Write a Python script that solves the user's task.
    2. The script MUST include `unittest` test cases in the same file.
    3. The script MUST include `if __name__ == '__main__': unittest.main()` to run the tests.
    4. Output ONLY the raw python code. No markdown backticks.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_description}
    ]
    
    max_attempts = 3
    
    for attempt in range(max_attempts):
        yield f"🔄 Attempt {attempt + 1}/{max_attempts}: Generating code..."
        
        # 1. Generate Code
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        code = response.choices[0].message.content
        
        # Clean markdown if present
        code = code.replace("```python", "").replace("```", "").strip()
        
        # 2. Write to Disk
        file_path = os.path.join(CODE_ENV_PATH, "solution.py")
        with open(file_path, "w") as f:
            f.write(code)
            
        # 3. Execute Tests
        yield f"🧪 Running tests..."
        proc = subprocess.run(
            ["python", file_path],
            capture_output=True,
            text=True
        )
        
        if proc.returncode == 0:
            yield "✅ Success! All tests passed."
            yield f"\nCode Output:\n{proc.stderr}" # unittest output usually goes to stderr
            yield f"\nFinal Code:\n{code}"
            return
        else:
            error_msg = proc.stderr
            yield f"❌ Tests Failed:\n{error_msg}"
            
            # 4. Feed back errors
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": f"The code failed with this error. Fix it:\n{error_msg}"})
            
    yield "⚠️ Failed to generate working code after 3 attempts."
