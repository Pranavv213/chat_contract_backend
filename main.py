from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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
        
        # Try to get function documentation from the 'description' or 'details' field if exists
        description = func.get('description', '')
        if description:
            description = f" // {description}"
        
        function_list.append(f"{name}{params_str}{returns_str}{description}")
    
    functions_text = "\n".join([f"  • {f}" for f in function_list])
    
    # SUPER SOLID VAST PROMPT for ANY contract type
    prompt = f"""You are an expert smart contract function router. Your task is to analyze the user's natural language query and select the MOST APPROPRIATE view/pure function from the contract's ABI, extracting any parameters needed.

## CONTRACT TYPES YOU HANDLE:
- Token contracts (ERC20, ERC721, ERC1155) - name, symbol, totalSupply, balanceOf, ownerOf, tokenURI, etc.
- DeFi protocols - getReserveData, getUserAccountData, getPoolTokens, getDeposits, getBorrows, getLiquidity, getExchangeRate, getPrice, getAPY, getTVL, getTotalAssets, convertToShares, previewDeposit, etc.
- Lending platforms - getBorrowRate, getSupplyRate, getUserBorrowBalance, getUserSupplyBalance, getCollateralValue, getHealthFactor, getLoanToValue, getLiquidationThreshold, etc.
- DEXes/AMMs - getReserves, getAmountOut, getAmountIn, getQuote, getPoolInfo, getTokenBalances, getSwapFee, getLiquidityPosition, getTotalLiquidity, etc.
- Governance - proposalCount, getProposal, getVotes, delegates, getCurrentQuorum, getVotingDelay, getVotingPeriod, getProposalState, etc.
- Gaming - getPlayerStats, getCharacterInfo, getItemBalance, getLevel, getScore, getLeaderboard, getResourceAmount, etc.
- NFT marketplaces - getListingPrice, getFloorPrice, getRoyaltyInfo, getOffer, getCollectionStats, getSaleData, etc.
- Staking contracts - getUserStake, getRewardRate, getRewardAmount, getStakingBalance, getLockedAmount, getUnlockTime, getAPR, etc.
- Bridges - getMintAmount, getBurnAmount, getExchangeRate, getFee, getMinAmount, getMaxAmount, getSupportedChains, etc.
- Oracles - latestAnswer, getRoundData, latestRoundData, getPrice, getTimestamp, getDecimals, getDescription, etc.
- ANY other contract type with view/pure functions!

## AVAILABLE FUNCTIONS:

{functions_text}

## USER QUERY:
"{query}"

## EXTRACTION RULES:
1. **Function Selection**: Choose the function whose NAME and PURPOSE best matches the user's intent
2. **Parameter Extraction**: Extract values from the query that match function parameters:
   - Addresses: Start with "0x" followed by 40 hex characters OR ENS names (like vitalik.eth)
   - Numbers: Integers, decimals (e.g., 100, 100.5, 1000)
   - Strings: Text in quotes or clear text values
   - Booleans: true/false, yes/no, enabled/disabled
   - Bytes: 0x-prefixed hex strings
   - Account IDs: User IDs, token IDs, proposal IDs, etc.
3. **Smart Matching**:
   - "balance", "holdings", "how much", "amount" → balance functions
   - "name", "symbol", "decimals" → token metadata functions
   - "supply", "circulating", "minted" → supply functions
   - "owner", "who owns" → ownership functions
   - "rate", "fee", "price", "cost" → rate/price functions
   - "total locked", "TVL", "liquidity" → DeFi metrics
   - "proposal", "vote", "governance" → governance functions
   - "stake", "reward", "apr" → staking functions
   - "character", "player", "level" → gaming functions
   - "collection", "nft", "token id" → NFT functions

## RESPONSE FORMAT:
Return ONLY valid JSON with no additional text:

For functions WITHOUT parameters:
{{"function": "functionName", "params": []}}

For functions WITH parameters:
{{"function": "functionName", "params": ["param1_value", "param2_value"]}}

For partial matches where parameter is implied:
{{"function": "functionName", "params": ["inferred_value"]}}

## EXAMPLES BY CONTRACT TYPE:

### Token Contract (ERC20):
Query: "What is the total supply?" -> {{"function": "totalSupply", "params": []}}
Query: "balance of 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2" -> {{"function": "balanceOf", "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"]}}
Query: "my balance" -> {{"function": "balanceOf", "params": ["USER_ADDRESS_NEEDED"]}}
Query: "show me the name" -> {{"function": "name", "params": []}}
Query: "token symbol" -> {{"function": "symbol", "params": []}}
Query: "how many decimals" -> {{"function": "decimals", "params": []}}

### NFT Contract (ERC721):
Query: "owner of token 123" -> {{"function": "ownerOf", "params": ["123"]}}
Query: "token URI for token 456" -> {{"function": "tokenURI", "params": ["456"]}}
Query: "total supply of NFTs" -> {{"function": "totalSupply", "params": []}}
Query: "balance of 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2" -> {{"function": "balanceOf", "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"]}}
Query: "get approved for token 789" -> {{"function": "getApproved", "params": ["789"]}}

### Uniswap V2 Pool:
Query: "get reserves" -> {{"function": "getReserves", "params": []}}
Query: "token0 address" -> {{"function": "token0", "params": []}}
Query: "price of token0 in token1" -> {{"function": "getReserves", "params": []}}
Query: "factory address" -> {{"function": "factory", "params": []}}
Query: "pair name" -> {{"function": "name", "params": []}}

### Aave Lending Pool:
Query: "user account data for 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2" -> {{"function": "getUserAccountData", "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"]}}
Query: "reserve data for USDC" -> {{"function": "getReserveData", "params": ["0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"]}}
Query: "total liquidity" -> {{"function": "getTotalLiquidity", "params": []}}
Query: "APY for USDC" -> {{"function": "getReserveData", "params": ["0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"]}}

### Compound:
Query: "supply rate" -> {{"function": "supplyRatePerBlock", "params": []}}
Query: "borrow rate" -> {{"function": "borrowRatePerBlock", "params": []}}
Query: "total borrows" -> {{"function": "totalBorrows", "params": []}}
Query: "underlying asset" -> {{"function": "underlying", "params": []}}
Query: "exchange rate" -> {{"function": "exchangeRateStored", "params": []}}

### Governor Contract:
Query: "proposal count" -> {{"function": "proposalCount", "params": []}}
Query: "proposal 5 details" -> {{"function": "proposals", "params": ["5"]}}
Query: "voting period" -> {{"function": "votingPeriod", "params": []}}
Query: "quorum" -> {{"function": "quorum", "params": []}}
Query: "votes for address 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2" -> {{"function": "getVotes", "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"]}}

### Chainlink Oracle:
Query: "latest price" -> {{"function": "latestAnswer", "params": []}}
Query: "round data for round 123" -> {{"function": "getRoundData", "params": ["123"]}}
Query: "decimals" -> {{"function": "decimals", "params": []}}
Query: "price feed description" -> {{"function": "description", "params": []}}

### Staking Contract:
Query: "stake balance for 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2" -> {{"function": "balanceOf", "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"]}}
Query: "reward rate" -> {{"function": "rewardRate", "params": []}}
Query: "total staked" -> {{"function": "totalSupply", "params": []}}
Query: "APR" -> {{"function": "getAPR", "params": []}}

### Gaming/NFT Game:
Query: "player level for address 0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2" -> {{"function": "getPlayerLevel", "params": ["0x742d35Cc6634C0532925a3b844Bc9e7595f0b3f2"]}}
Query: "character 123 stats" -> {{"function": "getCharacterStats", "params": ["123"]}}
Query: "gold balance" -> {{"function": "getResource", "params": ["gold"]}}

## IMPORTANT NOTES:
- The contract may have hundreds of functions - choose the most semantically relevant
- If the query asks for "price" but there's no price function, look for "getAmountOut", "getQuote", "latestAnswer", etc.
- If address is implied (e.g., "my balance") but not provided, use "USER_ADDRESS_NEEDED" as placeholder
- Always return valid JSON - no explanations, no markdown, just the JSON object
- If no function matches, return {{"function": "", "params": []}}

## YOUR RESPONSE (JSON only):
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,  # Deterministic
                    "num_predict": 150,  # Enough for JSON response
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
            
            # Extract JSON - look for anything between { and }
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    # Validate function exists
                    function_name = data.get('function', '')
                    if function_name and function_name in [f.get('name') for f in functions]:
                        return data
                    else:
                        # Try to find matching function (case insensitive, partial match)
                        query_lower = query.lower()
                        for func in functions:
                            func_name = func.get('name', '').lower()
                            # Check if function name appears in query or vice versa
                            if func_name in query_lower or query_lower in func_name:
                                # Also check if we can extract parameters
                                params = data.get('params', [])
                                return {"function": func.get('name'), "params": params}
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
        
        # Fallback: Smart keyword matching
        query_lower = query.lower()
        
        # Priority keywords (higher score = better match)
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
                'balance': ['balance', 'holdings', 'amount'],
                'supply': ['supply', 'total', 'circulating'],
                'owner': ['owner', 'who owns'],
                'name': ['name', 'token name'],
                'symbol': ['symbol', 'ticker'],
                'price': ['price', 'cost', 'value'],
                'rate': ['rate', 'apr', 'apy', 'interest'],
                'reserve': ['reserve', 'liquidity', 'tvl'],
                'proposal': ['proposal', 'vote', 'governance'],
                'stake': ['stake', 'staked', 'staking'],
                'reward': ['reward', 'earnings'],
                'level': ['level', 'rank'],
                'uri': ['uri', 'url', 'metadata']
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
            return {"function": best_func, "params": []}
        
        return {"function": "", "params": [], "confidence": 0}
        
    except requests.exceptions.Timeout:
        print("LLM request timeout")
        return {"function": "", "params": [], "confidence": 0}
    except Exception as e:
        print(f"LLM error: {e}")
        return {"function": "", "params": [], "confidence": 0}

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

@app.post("/query")
async def query_contract(request: QueryRequest):
    try:
        print(f"\n" + "="*70)
        print(f"📝 User Query: {request.query}")
        print(f"📍 Contract Address: {request.contract_address}")
        
        # Get view functions
        functions = get_view_functions(request.abi)
        if not functions:
            raise HTTPException(status_code=400, detail="No view/pure functions found in ABI. Contract must have view/pure functions to query.")
        
        print(f"📚 Total View Functions: {len(functions)}")
        print(f"🔍 First 10 functions: {[f.get('name') for f in functions[:10]]}")
        
        # Select function with LLM
        selection = select_function_with_llm(request.query, functions)
        
        if not selection.get("function"):
            # Provide helpful error with suggestions
            available_funcs = [f.get('name') for f in functions[:20]]
            return {
                "success": False,
                "function": "",
                "parameters": [],
                "available_functions": available_funcs,
                "error": "Could not determine which function to call. Please rephrase your query or try one of these functions: " + ", ".join(available_funcs[:10])
            }
        
        print(f"✅ Selected Function: {selection.get('function')}")
        print(f"📋 Parameters: {selection.get('params')}")
        print("="*70)
        
        return {
            "success": True,
            "function": selection.get("function", ""),
            "parameters": selection.get("params", []),
            "available_functions": [f.get('name') for f in functions]
        }
        
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
    print("   ollama pull qwen3:8b")
    print("   ollama serve")
    print("\n✅ Backend ready! Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)