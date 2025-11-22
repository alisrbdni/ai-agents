import json
import openai
from app.config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME

client = openai.AzureOpenAI(
    azure_endpoint=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    api_version="2024-05-01-preview"
)

# Task 3.3a: External Tools
def search_flights(origin: str, destination: str, date: str):
    """Mock flight search API."""
    return json.dumps([
        {"airline": "AirNZ", "flight": "NZ101", "price": 150, "currency": "NZD", "time": "08:00"},
        {"airline": "JetStar", "flight": "JQ202", "price": 120, "currency": "NZD", "time": "14:00"}
    ])

def get_weather(location: str, date: str):
    """Mock weather API."""
    return json.dumps({
        "location": location,
        "date": date,
        "condition": "Partly Cloudy",
        "temperature": 18
    })

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights between cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["origin", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather forecast for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["location", "date"]
            }
        }
    }
]

def run_planning_agent(prompt: str):
    """Task 3.3: Autonomous Planning Agent with Tool Calling."""
    messages = [
        {"role": "system", "content": "You are a travel planner. You must plan within budget. Output the final result as a JSON itinerary.dont ask extra questions"},
        {"role": "user", "content": prompt}
    ]
    
    logs = []
    
    # Loop for multi-step reasoning
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)
                
                # Task 3.3b: Reasoning visible in logs
                log_entry = f"🛠 Tool Call: {func_name} | Args: {args}"
                logs.append(log_entry)
                
                if func_name == "search_flights":
                    result = search_flights(**args)
                elif func_name == "get_weather":
                    result = get_weather(args.get("location"), args.get("date"))
                else:
                    result = "Error: Tool not found"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        else:
            # Final response
            return msg.content, logs
