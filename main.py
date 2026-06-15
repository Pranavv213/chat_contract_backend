from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import json
import re

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"

class QueryRequest(BaseModel):
    contract_address: str
    abi: List[Dict[str, Any]]
    query: str

def get_view_functions(abi):
    """Extract view/pure functions from ABI"""
    functions = []
    for item in abi:
        if item.get('type') == 'function':
            state = item.get('stateMutability', '')
            if state in ['view', 'pure']:
                functions.append(item)
    return functions

def select_function_with_llm(query, functions):
    """Use local LLM to select the correct function and extract parameters"""
    
    if not functions:
        return {"function": "", "params": [], "confidence": 0}
    
    # Prepare function descriptions
    func_descriptions = []
    for func in functions:
        name = func.get('name')
        inputs = func.get('inputs', [])
        if inputs:
            params = []
            for inp in inputs:
                param_name = inp.get('name', 'param')
                param_type = inp.get('type')
                params.append(f"{param_name}:{param_type}")
            func_descriptions.append(f"{name}({', '.join(params)})")
        else:
            func_descriptions.append(f"{name}()")
    
    functions_text = ", ".join(func_descriptions)
    
    prompt = f"""You are a smart contract function router. Select a function and extract parameters.

User query: "{query}"

Available functions: {functions_text}

Return ONLY JSON. Examples:

Query: "total supply" -> {{"function": "totalSupply", "params": []}}
Query: "balance of 0x123" -> {{"function": "balanceOf", "params": ["0x123"]}}
Query: "name" -> {{"function": "name", "params": []}}
Query: "symbol" -> {{"function": "symbol", "params": []}}

Now return JSON for: "{query}"

Response:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 50}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '{}')
            print(f"LLM Response: {text}")
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                # Validate function exists
                if data.get('function') in [f.get('name') for f in functions]:
                    return data
                else:
                    # Try to find matching function
                    for func in functions:
                        if func.get('name').lower() == data.get('function', '').lower():
                            return {"function": func.get('name'), "params": data.get('params', [])}
        
        return {"function": "", "params": [], "confidence": 0}
        
    except Exception as e:
        print(f"LLM error: {e}")
        return {"function": "", "params": [], "confidence": 0}

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/query")
async def query_contract(request: QueryRequest):
    try:
        print(f"\n📝 Query: {request.query}")
        
        # Get view functions
        functions = get_view_functions(request.abi)
        if not functions:
            raise HTTPException(status_code=400, detail="No view/pure functions found")
        
        print(f"📚 Available functions: {[f.get('name') for f in functions]}")
        
        # Select function with LLM
        selection = select_function_with_llm(request.query, functions)
        
        return {
            "success": True,
            "function": selection.get("function", ""),
            "parameters": selection.get("params", []),
            "available_functions": [f.get('name') for f in functions]
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting backend on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)