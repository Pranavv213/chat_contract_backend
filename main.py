from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import json
import re
from web3 import Web3
from web3.middleware import geth_poa_middleware
import os

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
    rpc_url: str = "https://polygon-rpc.com"

class QueryResponse(BaseModel):
    success: bool
    function: str
    parameters: List[str]
    result: Any
    error: Optional[str] = None
    available_functions: List[str]

def get_view_functions(abi):
    """Extract view/pure functions from ABI"""
    functions = []
    for item in abi:
        if item.get('type') == 'function':
            state = item.get('stateMutability', '')
            if state in ['view', 'pure']:
                functions.append(item)
    return functions

def format_parameters(inputs):
    """Format function parameters for display"""
    if not inputs:
        return ""
    
    params = []
    for inp in inputs:
        name = inp.get('name', 'param')
        param_type = inp.get('type')
        params.append(f"{name}:{param_type}")
    return f"({', '.join(params)})"

def get_function_signature(func):
    """Get full function signature"""
    name = func.get('name')
    inputs = func.get('inputs', [])
    outputs = func.get('outputs', [])
    
    # Format parameters
    params = []
    for inp in inputs:
        param_name = inp.get('name', 'param')
        param_type = inp.get('type')
        params.append(f"{param_name}:{param_type}")
    
    # Format returns
    returns = []
    for out in outputs:
        return_type = out.get('type')
        return_name = out.get('name', 'output')
        returns.append(f"{return_name}:{return_type}")
    
    signature = f"{name}({', '.join(params)})"
    if returns:
        signature += f" returns ({', '.join(returns)})"
    
    return signature

def select_function_with_llm(query, functions):
    """Use local LLM to select the correct function and extract parameters"""
    
    if not functions:
        return {"function": "", "params": [], "confidence": 0}
    
    # Build comprehensive function descriptions
    function_list = []
    for func in functions:
        name = func.get('name')
        inputs = func.get('inputs', [])
        outputs = func.get('outputs', [])
        
        # Format input parameters with types
        if inputs:
            params = []
            for inp in inputs:
                param_name = inp.get('name', 'param')
                param_type = inp.get('type')
                params.append(f"{param_name}:{param_type}")
            params_str = f"({', '.join(params)})"
        else:
            params_str = "()"
        
        # Format return values
        if outputs:
            returns = []
            for out in outputs:
                return_type = out.get('type')
                returns.append(return_type)
            returns_str = f" returns ({', '.join(returns)})"
        else:
            returns_str = ""
        
        function_list.append(f"{name}{params_str}{returns_str}")
    
    functions_text = "\n".join([f"  • {f}" for f in function_list])
    
    prompt = f"""
You are an expert Smart Contract ABI Router and Function Selector.

Your task is to analyze a user's natural language query and determine the SINGLE BEST read-only function from the provided smart contract ABI that can answer the user's question.

The contract may be ANY type of contract, including but not limited to:

- ERC20
- ERC721
- ERC1155
- DAO
- Governance
- Staking
- Lending
- Marketplace
- Auction
- Event Management
- Ticketing
- Gaming
- Identity
- Real Estate
- Supply Chain
- DeFi
- Social Protocols
- Custom Smart Contracts

Never assume the contract follows a token standard.

================================================================
AVAILABLE FUNCTIONS
================================================================

{functions_text}

Each function may contain:
- Function Name
- Input Parameters
- Output Parameters
- State Mutability
- Documentation / Description (if available)

================================================================
USER QUERY
================================================================

"{query}"

================================================================
OBJECTIVE
================================================================

Determine:

1. Which function best answers the user's question.
2. Which parameter values should be supplied.
3. Whether enough information exists to confidently call that function.

You must understand the user's INTENT.

Do NOT simply match keywords.

Analyze:
- Function names
- Parameter names
- Return values
- Documentation
- Semantic meaning

================================================================
FUNCTION SELECTION RULES
================================================================

Select the function whose OUTPUT most directly answers the user's question.

GOOD EXAMPLES:

User:
"What is the token name?"

Function:
name()

--------------------------------------------------

User:
"Who owns NFT 42?"

Function:
ownerOf(42)

--------------------------------------------------

User:
"Show details of proposal 5"

Function:
getProposal(5)

--------------------------------------------------

User:
"How many attendees does event 10 have?"

Function:
getAttendanceCount(10)

--------------------------------------------------

User:
"Has address X voted?"

Function:
hasVoted(X)

--------------------------------------------------

User:
"What is the current APY?"

Function:
currentAPY()

================================================================
PARAMETER EXTRACTION RULES
================================================================

Extract parameter values directly from the query.

Examples:

"balance of 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"

Returns:

["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"]

--------------------------------------------------

"owner of token 55"

Returns:

["55"]

--------------------------------------------------

"proposal 17 status"

Returns:

["17"]

--------------------------------------------------

"event 42 details"

Returns:

["42"]

Rules:

- Preserve exact values
- Do not modify addresses
- Do not normalize values
- Do not invent missing values
- Extract parameters in the exact order required by the function

================================================================
MISSING PARAMETERS
================================================================

If the selected function requires parameters and the query does not provide all required values:

Return:

{{
    "function": null,
    "params": [],
    "confidence": 0,
    "reason": "Required parameter missing"
}}

Example:

Query:
"Who owns this NFT?"

Function:
ownerOf(tokenId)

Since tokenId is missing:

Return null.

================================================================
MULTIPLE POSSIBLE FUNCTIONS
================================================================

If multiple functions seem relevant:

Choose the function whose OUTPUT most directly answers the question.

Priority:

1. Exact answer
2. Strong semantic match
3. Least unrelated data
4. Fewest assumptions

================================================================
FUNCTION ELIMINATION RULES
================================================================

DO NOT choose a function if:

- Parameters are missing
- It requires guessing values
- It only partially answers the question
- It is unrelated to the query
- It is payable
- It changes contract state
- It performs writes

Prefer:

- view functions
- pure functions

================================================================
CONFIDENCE SCORING
================================================================

100 = exact semantic match
90-99 = extremely strong match
75-89 = strong match
50-74 = weak match
0 = insufficient information

If confidence < 50:

Return null.

================================================================
OUTPUT FORMAT
================================================================

Return ONLY valid JSON.

Successful Match:

{{
    "function": "functionName",
    "params": ["value1", "value2"],
    "confidence": 95,
    "reason": "Function directly answers the user's question"
}}

No Match:

{{
    "function": null,
    "params": [],
    "confidence": 0,
    "reason": "No suitable function found"
}}

================================================================
STRICT RULES
================================================================

- Output JSON only
- No markdown
- No code fences
- No comments
- No explanations outside JSON
- No hallucinated functions
- No hallucinated parameters
- Use ONLY functions present in the ABI
- Never guess IDs
- Never guess addresses
- Never guess proposal IDs
- Never guess token IDs
- Never guess event IDs
- Never guess usernames
- Never guess any missing value

Return EXACTLY ONE function.

JSON ONLY:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 150,
                    "top_p": 0.9,
                    "top_k": 40
                }
            },
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '{}').strip()
            print(f"LLM Raw Response: {text}")
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    function_name = data.get('function', '')
                    
                    # Verify function exists
                    if function_name:
                        # Find the function in ABI
                        matching_func = next((f for f in functions if f.get('name') == function_name), None)
                        if matching_func:
                            # Check if function expects parameters
                            expected_params = matching_func.get('inputs', [])
                            provided_params = data.get('params', [])
                            
                            # If function expects no parameters but params were provided, clear them
                            if len(expected_params) == 0:
                                provided_params = []
                            # If function expects parameters but none were provided, try to infer
                            elif len(expected_params) > 0 and len(provided_params) == 0:
                                # Try to extract from query
                                extracted = extract_parameters_from_query(query, expected_params)
                                if extracted:
                                    provided_params = extracted
                            
                            return {"function": function_name, "params": provided_params}
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
        
        # Fallback: Smart keyword matching
        query_lower = query.lower()
        keyword_score = {}
        
        for func in functions:
            func_name = func.get('name', '').lower()
            score = 0
            
            # Exact match
            if func_name == query_lower:
                score = 100
            # Function name in query
            elif func_name in query_lower:
                score = 50
            # Query in function name
            elif query_lower in func_name:
                score = 30
            
            # Additional keyword matching
            keywords = {
                'balance': ['balance', 'holdings', 'amount', 'how much'],
                'supply': ['supply', 'total', 'circulating'],
                'owner': ['owner', 'who owns'],
                'name': ['name', 'token name', 'what is the name'],
                'symbol': ['symbol', 'ticker'],
                'price': ['price', 'cost', 'value'],
                'rate': ['rate', 'apr', 'apy', 'interest'],
                'reserve': ['reserve', 'liquidity', 'tvl'],
                'proposal': ['proposal', 'vote', 'governance'],
                'stake': ['stake', 'staked', 'staking'],
                'reward': ['reward', 'earnings'],
                'level': ['level', 'rank'],
                'uri': ['uri', 'url', 'metadata'],
                'decimals': ['decimals', 'decimal places']
            }
            
            for key, words in keywords.items():
                if key in func_name:
                    for word in words:
                        if word in query_lower:
                            score += 20
                            break
            
            if score > 0:
                keyword_score[func.get('name')] = score
        
        # Get best match
        if keyword_score:
            best_func = max(keyword_score, key=keyword_score.get)
            # Check if the function expects parameters
            matching_func = next((f for f in functions if f.get('name') == best_func), None)
            params = []
            if matching_func and matching_func.get('inputs', []):
                # Try to extract parameters
                extracted = extract_parameters_from_query(query, matching_func.get('inputs', []))
                if extracted:
                    params = extracted
            return {"function": best_func, "params": params}
        
        return {"function": "", "params": []}
        
    except requests.exceptions.Timeout:
        print("LLM request timeout")
        return {"function": "", "params": []}
    except Exception as e:
        print(f"LLM error: {e}")
        return {"function": "", "params": []}

def extract_parameters_from_query(query, expected_params):
    """Extract parameters from query based on expected parameter types"""
    extracted = []
    
    for param in expected_params:
        param_type = param.get('type', '')
        param_name = param.get('name', '').lower()
        
        # Try to find address
        if param_type == 'address':
            address_match = re.search(r'0x[a-fA-F0-9]{40}', query)
            if address_match:
                extracted.append(address_match.group())
            else:
                # Check for "my" or "me" - could use a placeholder
                if 'my' in query.lower() or 'me' in query.lower():
                    extracted.append("0x0000000000000000000000000000000000000000")  # Placeholder
                else:
                    extracted.append("")
        # Try to find number
        elif param_type in ['uint256', 'int256', 'uint', 'int']:
            number_match = re.search(r'\b\d+\b', query)
            if number_match:
                extracted.append(number_match.group())
            else:
                extracted.append("")
        # Try to find string
        elif param_type == 'string':
            string_match = re.search(r'"([^"]*)"', query)
            if string_match:
                extracted.append(string_match.group(1))
            else:
                extracted.append("")
        # Try to find bytes
        elif param_type == 'bytes':
            bytes_match = re.search(r'0x[a-fA-F0-9]+', query)
            if bytes_match:
                extracted.append(bytes_match.group())
            else:
                extracted.append("")
        # Try to find boolean
        elif param_type == 'bool':
            if 'true' in query.lower():
                extracted.append("true")
            elif 'false' in query.lower():
                extracted.append("false")
            else:
                extracted.append("")
        else:
            extracted.append("")
    
    # Filter out empty strings
    return [p for p in extracted if p]

def call_contract_function(contract_address: str, abi: List[Dict], function_name: str, params: List[str], rpc_url: str):
    """Call the contract function and return the result"""
    try:
        # Initialize Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add middleware for PoA chains (like Polygon)
        if 'polygon' in rpc_url.lower() or 'matic' in rpc_url.lower():
            try:
                from web3.middleware import geth_poa_middleware
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except:
                pass
        
        if not w3.is_connected():
            raise Exception("Failed to connect to blockchain. Please check RPC URL.")
        
        # Create contract instance
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )
        
        # Get the function
        contract_function = getattr(contract.functions, function_name)
        
        # Get the function ABI to check parameter count
        function_abi = next((item for item in abi if item.get('name') == function_name and item.get('type') == 'function'), None)
        
        if not function_abi:
            raise Exception(f"Function '{function_name}' not found in ABI")
        
        # Check if function expects parameters
        expected_params = function_abi.get('inputs', [])
        
        # If function expects no parameters, call without any
        if len(expected_params) == 0:
            result = contract_function().call()
        else:
            # Prepare parameters - only use the ones we have
            converted_params = []
            for i, param in enumerate(params):
                if i < len(expected_params):
                    param_type = expected_params[i].get('type')
                    try:
                        if param_type == 'address':
                            converted_params.append(Web3.to_checksum_address(param))
                        elif param_type in ['uint256', 'int256', 'uint', 'int']:
                            converted_params.append(int(param))
                        elif param_type == 'bool':
                            converted_params.append(param.lower() == 'true')
                        else:
                            converted_params.append(param)
                    except Exception as e:
                        print(f"Error converting parameter {i}: {e}")
                        converted_params.append(param)
                else:
                    converted_params.append(param)
            
            # Call with the converted parameters
            if converted_params:
                result = contract_function(*converted_params).call()
            else:
                result = contract_function().call()
        
        # Format the result
        if isinstance(result, bytes):
            try:
                result = result.decode('utf-8')
            except:
                result = '0x' + result.hex()
        elif isinstance(result, int):
            # For large numbers, keep as int but ensure it's serializable
            result = result
        elif hasattr(result, 'items'):  # Tuple-like result
            result = dict(result)
        
        return result
        
    except Exception as e:
        raise Exception(f"Contract call failed: {str(e)}")

@app.get("/")
def root():
    return {
        "status": "running",
        "model": LLM_MODEL,
        "backend": "Universal Smart Contract Assistant",
        "supported_contracts": [
            "Tokens (ERC20, ERC721, ERC1155)",
            "DeFi Protocols (Lending, DEXes, Staking)",
            "Governance (DAO, Governor)",
            "Gaming & NFTs",
            "Oracles & Bridges",
            "ANY contract with view/pure functions"
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query_contract(request: QueryRequest):
    try:
        print(f"\n" + "="*70)
        print(f"📝 User Query: {request.query}")
        print(f"📍 Contract Address: {request.contract_address}")
        print(f"🔗 RPC URL: {request.rpc_url}")
        
        # Get view functions
        functions = get_view_functions(request.abi)
        if not functions:
            raise HTTPException(status_code=400, detail="No view/pure functions found in ABI. Contract must have view/pure functions to query.")
        
        print(f"📚 Total View Functions: {len(functions)}")
        print(f"🔍 First 10 functions: {[f.get('name') for f in functions[:10]]}")
        
        # Select function with LLM
        selection = select_function_with_llm(request.query, functions)
        
        if not selection.get("function"):
            available_funcs = [f.get('name') for f in functions[:20]]
            return QueryResponse(
                success=False,
                function="",
                parameters=[],
                result=None,
                error="Could not determine which function to call. Please rephrase your query.",
                available_functions=available_funcs
            )
        
        function_name = selection.get("function")
        params = selection.get("params", [])
        
        print(f"✅ Selected Function: {function_name}")
        print(f"📋 Parameters: {params}")
        
        # Call the contract function
        try:
            print(f"⛓️ Calling {function_name}() on blockchain...")
            result = call_contract_function(
                request.contract_address,
                request.abi,
                function_name,
                params,
                request.rpc_url
            )
            print(f"📊 Result: {result}")
            print("="*70)
            
            return QueryResponse(
                success=True,
                function=function_name,
                parameters=params,
                result=result,
                error=None,
                available_functions=[f.get('name') for f in functions]
            )
            
        except Exception as e:
            print(f"❌ Contract call error: {str(e)}")
            return QueryResponse(
                success=False,
                function=function_name,
                parameters=params,
                result=None,
                error=str(e),
                available_functions=[f.get('name') for f in functions]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 UNIVERSAL SMART CONTRACT ASSISTANT BACKEND")
    print("="*70)
    print(f"📍 Server: http://localhost:8000")
    print(f"🤖 Model: {LLM_MODEL}")
    print(f"🔗 Ollama: {OLLAMA_URL}")
    print("\n✅ SUPPORTED CONTRACT TYPES:")
    print("   • Tokens (ERC20, ERC721, ERC1155)")
    print("   • DeFi Protocols (Lending, DEXes, Staking)")
    print("   • Governance (DAO, Governor)")
    print("   • Gaming & NFTs")
    print("   • Oracles & Bridges")
    print("   • ANY contract with view/pure functions!")
    print("\n⚠️  Make sure Ollama is running:")
    print("   ollama pull llama3.2:3b")
    print("   ollama serve")
    print("\n✅ Backend ready! Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)