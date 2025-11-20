import time
import tiktoken
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_ai")

# Pricing constants (approximate for GPT-4o)
INPUT_COST_PER_1M = 5.00
OUTPUT_COST_PER_1M = 15.00

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    p_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_1M
    c_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    return round(p_cost + c_cost, 6)

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Returns elapsed time in milliseconds"""
        self.end_time = time.perf_counter()
        return (self.end_time - self.start_time) * 1000
