import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

client = genai.Client(api_key=api_key)

# Part 2: Tool Function + Tool Schema (FunctionDeclaration)

# 1. Asli Python function (ye aapke laptop par chalega)
def add(a: float, b: float) -> float:
    return a + b


# 2. LLM ke liye Menu Card (Tool Schema)
add_declaration = types.FunctionDeclaration(
    name="add",
    description="Add two numbers together. Use this whenever the user asks to add numbers.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number"},
            "b": {"type": "number", "description": "The second number"},
        },
        "required": ["a", "b"],
    },
)

# 3. Gemini ke format mein tool list wrap karna
tools = [types.Tool(function_declarations=[add_declaration])]


# Part 3: Tools ke saath LLM call aur tool_call pakadna
user_prompt = "What is 25 + 37?"

messages = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_prompt)],
    )
]

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=messages,
    config=types.GenerateContentConfig(tools=tools),
)

if response.function_calls:
    tool_call = response.function_calls[0]
    print("LLM want tool name:")
    print("Tool Name:", tool_call.name)
    print("Arguments:", tool_call.args)



    # Part 4: Tool execute karna aur poori history wapas bhejna
    if tool_call.name == "add":
        result = add(**tool_call.args)
        print("Python ne tool chalaya, Result:", result)
    else:
        raise ValueError(f"Unknown tool: {tool_call.name}")

    # LLM ka tool_call response history mein jodo
    messages.append(response.candidates[0].content)

    # Tool ka result bhi history mein jodo
    messages.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=tool_call.name,
                    response={"result": result},
                )
            ],
        )
    )

    # Poori history ke saath 2nd call LLM ko bhejo
    final_response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=messages,
        config=types.GenerateContentConfig(tools=tools),
    )

    print("Final Answer:", final_response.text)
else:
    print("LLM ka direct jawab:", response.text)
