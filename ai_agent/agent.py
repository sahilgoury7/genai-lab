import os
import re
from pathlib import Path
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

client = genai.Client(api_key=api_key)

SAFE_DIR = (Path(__file__).parent / "files").resolve()


# Tool 1: Calculator
def calculate(operation: str, a: float, b: float) -> dict:
    if operation == "add":
        return {"result": a + b}
    elif operation == "subtract":
        return {"result": a - b}
    elif operation == "multiply":
        return {"result": a * b}
    elif operation == "divide":
        if b == 0:
            return {"error": "Cannot divide by zero"}
        return {"result": a / b}
    else:
        return {"error": f"Unknown operation: {operation}"}


# Tool 2: Safe File Reader
def read_file(filename: str) -> dict:
    target_path = (SAFE_DIR / filename).resolve()

    if not target_path.is_relative_to(SAFE_DIR):
        return {"error": "Access denied: cannot read outside files folder."}

    if not target_path.is_file():
        return {"error": f"File not found: {filename}"}

    content = target_path.read_text(encoding="utf-8")
    return {"filename": filename, "content": content}


# Tool 3: Free Web Search (Wikipedia API)
def web_search(query: str) -> dict:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
        "format": "json",
    }
    headers = {"User-Agent": "GenAILabAgent/1.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("query", {}).get("search", [])
    except requests.RequestException as err:
        return {"error": f"Search failed: {err}"}

    if not items:
        return {"error": f"No results found for: {query}"}

    results = []
    for item in items:
        clean_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
        results.append({
            "title": item.get("title", ""),
            "snippet": clean_snippet,
        })

    return {"query": query, "results": results}


# Tool Router: string naam se seedha Python function chalane ke liye
TOOL_MAP = {
    "calculate": calculate,
    "read_file": read_file,
    "web_search": web_search,
}

# Gemini ke liye 3 Tools ka Menu Card (Schemas)
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="calculate",
                description="Perform math calculations (add, subtract, multiply, divide) on two numbers.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The math operation to perform.",
                        },
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            ),
            types.FunctionDeclaration(
                name="read_file",
                description="Read a text file from the safe files folder (e.g. 'budget.txt').",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Name of the file to read"},
                    },
                    "required": ["filename"],
                },
            ),
            types.FunctionDeclaration(
                name="web_search",
                description="Search Wikipedia for facts, definitions, or general information.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query keywords"},
                    },
                    "required": ["query"],
                },
            ),
        ]
    )
]


SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools: calculate, read_file, and web_search. "
    "Use tools whenever needed instead of guessing. If a tool returns an error, handle it gracefully."
)


def run_agent(user_prompt: str, max_steps: int = 8) -> str:
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        )
    ]

    step = 0
    while step < max_steps:
        step += 1
        print(f"\n--- Step {step} ---")

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=messages,
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction=SYSTEM_PROMPT,
            ),
        )

        # 1. Agar koi tool call nahi -> Final Answer ready!
        if not response.function_calls:
            print(f"[Final Answer] {response.text}")
            return response.text or ""

        # 2. LLM ka tool-request history mein save karo
        messages.append(response.candidates[0].content)

        # 3. Is turn ke saare tool calls chalao
        tool_response_parts = []
        for tool_call in response.function_calls:
            name = tool_call.name
            args = dict(tool_call.args or {})
            print(f"[Tool Call] {name}({args})")

            try:
                if name in TOOL_MAP:
                    result = TOOL_MAP[name](**args)
                else:
                    result = {"error": f"Unknown tool: {name}"}
            except Exception as err:
                result = {"error": f"Tool execution error: {err}"}

            print(f"[Tool Result] {result}")
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=name,
                    response=result,
                )
            )

        # 4. Saare tool results wapas history mein jodo
        messages.append(types.Content(role="user", parts=tool_response_parts))

    print("\n[Stopped] Reached max steps without final answer.")
    return "Stopped: Max steps reached."


if __name__ == "__main__":
    test_questions = [
        "1. What is the capital of Japan? Answer in one line.",
        "2. Use web search to find who created Python programming language.",
        "3. Read 'budget.txt' and calculate the total cost of all three items.",
        "4. Read 'missing_file.txt' and tell me what happened.",
    ]

    for question in test_questions:
        print("\n" + "=" * 60)
        print("USER:", question)
        run_agent(question)


