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
import copy

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

def sanitize_abi(abi):
    """Sanitize ABI to ensure all required fields are present"""
    sanitized = []
    for item in abi:
        sanitized_item = copy.deepcopy(item)
        
        if sanitized_item.get('type') == 'function':
            if 'outputs' not in sanitized_item:
                sanitized_item['outputs'] = []
            for output in sanitized_item.get('outputs', []):
                if 'type' not in output:
                    output['type'] = 'unknown'
            for input_param in sanitized_item.get('inputs', []):
                if 'type' not in input_param:
                    input_param['type'] = 'unknown'
                if 'name' not in input_param:
                    input_param['name'] = 'param'
        elif sanitized_item.get('type') in ['event', 'error']:
            for input_param in sanitized_item.get('inputs', []):
                if 'type' not in input_param:
                    input_param['type'] = 'unknown'
                if 'name' not in input_param:
                    input_param['name'] = 'param'
        
        sanitized.append(sanitized_item)
    return sanitized

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
        param_type = inp.get('type', 'unknown')
        params.append(f"{name}:{param_type}")
    return f"({', '.join(params)})"

def get_function_signature(func):
    """Get full function signature"""
    name = func.get('name')
    inputs = func.get('inputs', [])
    outputs = func.get('outputs', [])
    
    params = []
    for inp in inputs:
        param_name = inp.get('name', 'param')
        param_type = inp.get('type', 'unknown')
        params.append(f"{param_name}:{param_type}")
    
    returns = []
    for out in outputs:
        return_type = out.get('type', 'unknown')
        return_name = out.get('name', 'output')
        returns.append(f"{return_name}:{return_type}")
    
    signature = f"{name}({', '.join(params)})"
    if returns:
        signature += f" returns ({', '.join(returns)})"
    
    return signature

def extract_json_from_text(text):
    """Extract JSON from text, handling various formats"""
    # Try to find JSON in the text
    json_patterns = [
        r'\{[^{}]*\}',  # Simple JSON object
        r'```json\s*([\s\S]*?)\s*```',  # JSON in code block
        r'```\s*([\s\S]*?)\s*```',  # JSON in generic code block
        r'(\{[\s\S]*?\})',  # Multi-line JSON
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # If it's from a code block, use the captured group
                if isinstance(match, tuple):
                    match = match[0] if match else ''
                # Clean up the JSON string
                clean_json = match.strip()
                # Try to parse it
                data = json.loads(clean_json)
                return data
            except json.JSONDecodeError:
                continue
    
    # Try to find function name and parameters without JSON
    function_match = re.search(r'(?:function|Function)\s*["\']?(\w+)["\']?', text, re.IGNORECASE)
    if function_match:
        function_name = function_match.group(1)
        # Try to find parameters
        params_match = re.search(r'params?\s*:\s*\[([^\]]*)\]', text, re.IGNORECASE)
        params = []
        if params_match:
            params_str = params_match.group(1)
            # Extract quoted strings or numbers
            param_matches = re.findall(r'["\']([^"\']*)["\']|(\b\d+\b)', params_str)
            for p in param_matches:
                if p[0]:  # Quoted string
                    params.append(p[0])
                elif p[1]:  # Number
                    params.append(p[1])
        return {"function": function_name, "params": params}
    
    return None

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
        
        if inputs:
            params = []
            for inp in inputs:
                param_name = inp.get('name', 'param')
                param_type = inp.get('type', 'unknown')
                params.append(f"{param_name}:{param_type}")
            params_str = f"({', '.join(params)})"
        else:
            params_str = "()"
        
        if outputs:
            returns = []
            for out in outputs:
                return_type = out.get('type', 'unknown')
                returns.append(return_type)
            returns_str = f" returns ({', '.join(returns)})"
        else:
            returns_str = ""
        
        function_list.append(f"{name}{params_str}{returns_str}")
    
    functions_text = "\n".join([f"  • {f}" for f in function_list])
    
    prompt = f"""
You are an expert Smart Contract ABI Router and Function Selector with deep understanding of blockchain protocols, token standards, DeFi, NFTs, DAOs, and various contract types.

Your task is to analyze a user's natural language query and determine the SINGLE BEST read-only (view/pure) function from the provided smart contract ABI that can answer the user's question.

================================================================
AVAILABLE FUNCTIONS
================================================================

{functions_text}

================================================================
USER QUERY
================================================================

"{query}"

================================================================
CONTEXT & UNDERSTANDING
================================================================

The contract could be ANY type including:
- Tokens: ERC20, ERC721, ERC1155, ERC777, ERC4626
- DeFi: Uniswap, Aave, Compound, Curve, Balancer, Lido
- NFTs: OpenSea, Rarible, Foundation, Zora
- DAOs: Governor, Aragon, Snapshot, Safe
- Oracles: Chainlink, Pyth, Tellor
- Bridges: Wormhole, Axelar, LayerZero
- Gaming: Axie Infinity, The Sandbox, Decentraland
- Identity: ENS, Lens, Ceramic
- Real World Assets: RealT, Lofty
- L2s: Arbitrum, Optimism, Base, zkSync
- Wallets: Smart Contract Wallets, Multisig
- Custom: ANY custom smart contract

===============================================================
SMART PATTERN RECOGNITION
===============================================================

Function Name Patterns & Their Meanings:

TOKEN PATTERNS:
- balanceOf(address) → Query about balance, holdings, amount
- name() → Query about token name
- symbol() → Query about token symbol/ticker
- decimals() → Query about decimal places
- totalSupply() → Query about total supply, circulating supply
- allowance(address,address) → Query about spending allowance, approval
- getReserves() → Query about liquidity, reserves, pool size
- getAmountOut(uint,uint,uint) → Query about swap rates, output amounts
- tokenURI(uint256) → Query about NFT metadata, image, URL

NFT PATTERNS:
- ownerOf(uint256) → Query about NFT ownership, who owns
- tokenOfOwnerByIndex(address,uint256) → Query about tokens owned by address
- balanceOf(address) → Query about NFT count, how many NFTs
- getApproved(uint256) → Query about NFT approval
- isApprovedForAll(address,address) → Query about operator approval

DEFI PATTERNS:
- getPool(address,address) → Query about pool info, liquidity
- getExchangeRate() → Query about exchange rate, conversion rate
- getAPY() → Query about APY, interest rate, yield
- getTotalDeposits() → Query about TVL, total value locked
- getUserDeposit(address) → Query about user's deposit, position
- calculateRewards(address) → Query about rewards, earnings
- getPrice() → Query about price, value

GOVERNANCE PATTERNS:
- getProposal(uint256) → Query about proposal details
- getVoter(address) → Query about voting power, votes
- getDelegate(address) → Query about delegate, representation
- getProposalCount() → Query about number of proposals
- state(uint256) → Query about proposal status

QUERY PATTERNS:
- "my", "my balance", "my tokens" → Use functions taking no address or msg.sender
- "balance of [address]" → Look for balanceOf(address) or similar
- "owner of [id]" → Look for ownerOf(uint256) or similar
- "total supply", "supply" → Look for totalSupply()
- "price of", "how much" → Look for getPrice(), getAmountOut(), rate functions
- "info", "details", "status" → Look for getInfo(), getDetails() functions
- "list", "all", "get" → Look for getList(), getAll() functions
- "[noun] of [address]" → Look for functions with address parameters
- "[noun] of [id/number]" → Look for functions with uint parameters

===============================================================
PARAMETER EXTRACTION (CRITICAL)
===============================================================

Extract parameters with high precision:

ADDRESS PATTERNS:
- Ethereum address: 0x[a-fA-F0-9]{{40}}
- ENS name: *.eth (resolve to address if needed)
- "my", "me", "mine" → Use caller's address (msg.sender)

NUMBER PATTERNS:
- Integer: \\b\\d+\\b
- Decimal: \\b\\d+\\.\\d+\\b
- Token ID: "token [id]", "NFT [id]"
- Proposal ID: "proposal [id]"

STRING PATTERNS:
- Quoted text: "([^"]*)"
- Name: "name", "symbol" keywords

BOOL PATTERNS:
- "yes", "true", "enabled", "active" → true
- "no", "false", "disabled", "inactive" → false

BYTES PATTERNS:
- 0x[a-fA-F0-9]{{64}} (32 bytes)

===============================================================
FUNCTION SELECTION HIERARCHY
===============================================================

Priority 1: EXACT SEMANTIC MATCH
- Function name matches query intent exactly
- Example: "balanceOf" for "what is the balance"

Priority 2: STRONG SEMANTIC MATCH
- Function name partially matches query
- Example: "getBalance" for "balance"

Priority 3: PARAMETER MATCH
- Function parameters match extracted values
- Example: address parameter for address query

Priority 4: CONTEXTUAL MATCH
- Function fits the contract type context
- Example: NFT function for NFT contract

===============================================================
OUTPUT FORMAT - STRICT JSON ONLY
===============================================================

Return ONLY valid JSON. No explanations, no markdown, no comments.

SUCCESSFUL MATCH:
{{
    "function": "exact_function_name",
    "params": ["param1", "param2"],
    "confidence": 95,
    "reason": "Function directly answers the user's question"
}}

NO MATCH:
{{
    "function": null,
    "params": [],
    "confidence": 0,
    "reason": "No suitable function found"
}}

===============================================================
EXAMPLES ACROSS DIFFERENT CONTRACTS
===============================================================

ERC20 Token Contract:
User: "What is the balance of 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2?"
Functions: balanceOf(address), name(), symbol(), totalSupply()
Response: {{"function": "balanceOf", "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"], "confidence": 100}}

NFT Contract:
User: "Who owns token 123?"
Functions: ownerOf(uint256), balanceOf(address), tokenURI(uint256)
Response: {{"function": "ownerOf", "params": ["123"], "confidence": 100}}

Uniswap V2 Pair:
User: "What are the reserves?"
Functions: getReserves(), price0CumulativeLast(), price1CumulativeLast()
Response: {{"function": "getReserves", "params": [], "confidence": 100}}

Aave Lending Pool:
User: "What is my deposit amount?"
Functions: getUserAccountData(address), getReserveData(address), balanceOf(address)
Response: {{"function": "getUserAccountData", "params": [], "confidence": 95}}

Governance DAO:
User: "Status of proposal 42?"
Functions: state(uint256), getProposal(uint256), getVoter(address)
Response: {{"function": "state", "params": ["42"], "confidence": 95}}

Chainlink Price Feed:
User: "What is the current price?"
Functions: latestRoundData(), getAnswer(), getRoundData(uint80)
Response: {{"function": "latestRoundData", "params": [], "confidence": 100}}

Lens Protocol:
User: "Get my profile"
Functions: getProfile(address), getProfilesByOwner(address), getFollowers(address)
Response: {{"function": "getProfile", "params": [], "confidence": 95}}

ENS Registry:
User: "Get address for vitalik.eth"
Functions: resolver(bytes32), owner(bytes32), setSubnodeRecord()
Response: {{"function": "resolver", "params": ["vitalik.eth"], "confidence": 90}}

Custom Todo Contract:
User: "Todos of 0xdfe70B004f3e08fd81baC1626915590F6549ADBD"
Functions: getUserTodosByAddress(address), getUserTodos(), getTodoCount()
Response: {{"function": "getUserTodosByAddress", "params": ["0xdfe70B004f3e08fd81baC1626915590F6549ADBD"], "confidence": 100}}

===============================================================
CRITICAL RULES - DO NOT VIOLATE
===============================================================

1. NEVER hallucinate functions - use ONLY from provided list
2. NEVER invent parameters - extract from query or use empty string
3. NEVER guess IDs/addresses - only use what's in the query
4. ALWAYS preserve exact formatting (address case, number format)
5. ALWAYS return JSON only
6. For "my" queries, return empty params array (use msg.sender)
7. For public queries, include the address/number parameter
8. PREFER view/pure functions over payable/write functions
9. If multiple functions match, choose the most specific one
10. If confidence < 50, return null

===============================================================
YOUR RESPONSE (JSON ONLY - NO OTHER TEXT):
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
                    "num_predict": 200,
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
            
            # Try to extract JSON using our robust function
            parsed_data = extract_json_from_text(text)
            
            if parsed_data:
                function_name = parsed_data.get('function', '')
                params = parsed_data.get('params', [])
                
                # Verify function exists
                if function_name:
                    matching_func = next((f for f in functions if f.get('name') == function_name), None)
                    if matching_func:
                        # Ensure we have the right number of parameters
                        expected_params = matching_func.get('inputs', [])
                        if len(expected_params) == 0:
                            params = []
                        elif len(params) < len(expected_params):
                            # Try to extract missing parameters from query
                            extracted = extract_parameters_from_query(query, expected_params)
                            if extracted:
                                params = extracted
                        
                        print(f"✅ Selected function: {function_name}")
                        print(f"📋 Parameters: {params}")
                        return {"function": function_name, "params": params}
                    else:
                        print(f"⚠️ Function '{function_name}' not found in ABI")
                else:
                    print("⚠️ No function name extracted from LLM response")
            else:
                print("⚠️ Could not parse JSON from LLM response")
        
        # Fallback: Smart keyword matching
        print("🔄 Using fallback keyword matching...")
        query_lower = query.lower()
        keyword_score = {}
        
        for func in functions:
            func_name = func.get('name', '').lower()
            score = 0
            
            # Check if function name appears in query
            if func_name in query_lower:
                score += 50
            
            # Check for keyword matches
            keywords = {
                'balance': ['balance', 'holdings', 'amount'],
                'supply': ['supply', 'total'],
                'owner': ['owner', 'who owns'],
                'name': ['name', 'token name'],
                'symbol': ['symbol', 'ticker'],
                'todos': ['todo', 'todos', 'task'],
                'user': ['user', 'address', 'wallet'],
                'proposal': ['proposal', 'vote'],
                'stake': ['stake', 'staked'],
                'reward': ['reward', 'earnings']
            }
            
            for key, words in keywords.items():
                if key in func_name:
                    for word in words:
                        if word in query_lower:
                            score += 20
                            break
            
            if score > 0:
                keyword_score[func.get('name')] = score
        
        if keyword_score:
            best_func = max(keyword_score, key=keyword_score.get)
            matching_func = next((f for f in functions if f.get('name') == best_func), None)
            params = []
            if matching_func and matching_func.get('inputs', []):
                extracted = extract_parameters_from_query(query, matching_func.get('inputs', []))
                if extracted:
                    params = extracted
            print(f"✅ Fallback selected: {best_func}")
            return {"function": best_func, "params": params}
        
        print("❌ No function found")
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
                # Check for "my" or "me"
                if 'my' in query.lower() or 'me' in query.lower():
                    extracted.append("0x0000000000000000000000000000000000000000")
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
        # Sanitize the ABI before using it
        sanitized_abi = sanitize_abi(abi)
        
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
        
        # Create contract instance with sanitized ABI
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=sanitized_abi
        )
        
        # Get the function
        contract_function = getattr(contract.functions, function_name)
        
        # Get the function ABI to check parameter count
        function_abi = next((item for item in sanitized_abi if item.get('name') == function_name and item.get('type') == 'function'), None)
        
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
                    param_type = expected_params[i].get('type', '')
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
        if isinstance(result, (bytes, bytearray)):
            try:
                result = result.decode('utf-8')
            except:
                result = '0x' + result.hex()
        elif isinstance(result, int):
            result = result
        elif isinstance(result, (list, tuple)):
            result = list(result)
        elif hasattr(result, '_asdict'):
            result = result._asdict()
        elif isinstance(result, dict):
            pass
        else:
            result = str(result)
        
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
        print(f"🔍 Available functions: {[f.get('name') for f in functions[:10]]}")
        
        # Select function with LLM
        selection = select_function_with_llm(request.query, functions)
        
        if not selection or not selection.get("function"):
            available_funcs = [f.get('name') for f in functions[:20]]
            print("❌ No function selected")
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