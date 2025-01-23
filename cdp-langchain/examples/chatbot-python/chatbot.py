import os
import sys
import time
import re
import json
from eth_utils import is_address
from datetime import datetime, timezone
import requests
from urllib.parse import unquote
from decimal import Decimal
import io
from PIL import Image
import base64

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit and Twitter Langchain Extensions
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from twitter_langchain import TwitterApiWrapper, TwitterToolkit
from cdp_langchain.tools import CdpTool
from pydantic import BaseModel, Field
from cdp import Wallet
from cdp.smart_contract import SmartContract

# Configure files to persist data
wallet_data_file = "wallet_data.txt"
last_check_file = "last_check_time.txt"

# NFT Contract configuration
NFT_CONTRACT_ADDRESS = "0x32f75546e56aEC829ce13A9b73d4ebb42bF56b9c"

# Add at the top with other constants
DEBUG_MODE = True
DUMMY_MENTIONS_FILE = "dummy_mentions.txt"
MENTION_MEMORY_FILE = "mention_memory.txt"

# Remove the hardcoded API key and get it from environment
etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
if not etherscan_api_key:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")

abi = [
    {
        "inputs": [{"internalType": "address", "name": "recipient", "type": "address"}],
        "name": "mintAndTransfer",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }
]

def get_transaction_data(tx_hash):
    url = f"https://api-sepolia.basescan.org/api?module=proxy&action=eth_getTransactionReceipt&txhash={tx_hash}&apikey={etherscan_api_key}"

    response = requests.get(url)
    data = response.json()

    last_log = data['result']['logs'][-1]
    int_value = int(last_log['topics'][3], 16)
    contract_address = last_log['address']

    return int_value, contract_address

MINT_MYNFT_PROMPT = """
This tool will mint a Xonin NFT and transfer it directly to the specified address by paying 0.001 ETH.
The NFT will be minted and transferred in a single transaction.
"""

class MintMyNftInput(BaseModel):
    """Input argument schema for mint XoninNFT action."""
    recipient_address: str = Field(
        ...,
        description="The address that will receive the minted NFT"
    )

def save_svg_to_png(token_id: int, svg_data: str):
    """Convert SVG to PNG using PIL."""
    # First save SVG to file
    svg_filename = f"nft_{token_id}.svg"
    with open(svg_filename, 'w') as f:
        f.write(svg_data)
        
    # Convert SVG data to PNG using PIL
    # First convert SVG data to base64
    svg_bytes = svg_data.encode('utf-8')
    svg_base64 = base64.b64encode(svg_bytes).decode('utf-8')
    
    # Create a data URL
    data_url = f"data:image/svg+xml;base64,{svg_base64}"
    
    # Open the image from the data URL
    response = requests.get(data_url)
    img = Image.open(io.BytesIO(response.content))
    
    # Save as PNG
    png_filename = f"nft_{token_id}.png"
    img.save(png_filename, 'PNG')
    
    return svg_filename, png_filename

def get_token_uri_and_svg(wallet: Wallet, contract_address: str, token_id: int) -> tuple[str, str, str]:
    """Get tokenURI and extract name and SVG from the response."""
    try:
        # Call tokenURI function
        token_uri = SmartContract.read(
            wallet.network_id,
            contract_address,
            abi=[{
                "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
                "name": "tokenURI",
                "outputs": [{"internalType": "string", "name": "", "type": "string"}],
                "stateMutability": "view",
                "type": "function"
            }],
            method="tokenURI",
            args={"tokenId": str(token_id)}
        )

        # Extract the JSON part after data:application/json;utf8,
        json_str = token_uri.split('data:application/json;utf8,')[1]
        json_data = json.loads(unquote(json_str))
        
        # Extract name and SVG
        name = json_data['name']
        svg_data = json_data['image'].split('data:image/svg+xml;utf8,')[1]
        
        return token_uri, name, svg_data

    except Exception as e:
        return None, None, f"Error getting token URI and SVG: {e!s}"

def mint_myNft(wallet: Wallet, recipient_address: str) -> str:
    """Mint a Xonin NFT and transfer it to the specified address."""
    price = Decimal("0.001")
  
    try:
        mint_invocation = wallet.invoke_contract(
            contract_address=NFT_CONTRACT_ADDRESS,
            abi=abi,
            method="mintAndTransfer",
            args={"recipient": recipient_address},
            amount=price,
            asset_id="eth",
        ).wait()

        token_id, nft_mint_address = get_transaction_data(mint_invocation.transaction.transaction_hash)
        
        # Get token URI, name and SVG after minting
        token_uri, name, svg_data = get_token_uri_and_svg(wallet, nft_mint_address, token_id)

        # Save both SVG and PNG
        svg_file, png_file = save_svg_to_png(token_id, svg_data)

        return (f"🎉 Successfully minted {name} for {recipient_address}!\n\n"
                f"🔗 Transaction: {mint_invocation.transaction.transaction_link}\n"
                f"🖼️ Your unique NFT has been saved to: {png_file}\n\n"
                f"#NFT #Xonin #OnChainArt")

    except Exception as e:
        return f"Error minting and transferring NFT: {e!s}"

GET_BALANCE_MYNFT_PROMPT = """
This tool will get the Xonin NFTs (ERC721 tokens) owned by the wallet for a specific NFT contract.

It takes the following inputs:
- contract_address: The NFT contract address to check
- address: (Optional) The address to check NFT balance for. If not provided, uses the wallet's default address
"""

class GetBalanceMyNftInput(BaseModel):
    """Input argument schema for get Xonin NFT balance action."""

    contract_address: str = Field(..., description="The NFT contract address to check balance for")
    address: str | None = Field(
        None,
        description="The address to check NFT balance for. If not provided, uses the wallet's default address",
    )


def get_balance_myNft(
    wallet: Wallet,
    contract_address: str,
    address: str | None = None,
) -> str:
    """Get Xonin NFT balance for a specific contract."""

    try:
        check_address = address if address is not None else wallet.default_address.address_id

        # First get the total number of tokens owned
        balance = SmartContract.read(
            wallet.network_id, 
            contract_address, 
            abi=abi, 
            method="balanceOf", 
            args={"owner": check_address}
        )

        if balance == 0:
            return f"Address {check_address} owns no NFTs in contract {contract_address}"
        else: print(f"Balance: {balance}")
        
        # Then get each token ID using tokenOfOwnerByIndex
        owned_tokens = []
        for i in range(balance):
            token_id = SmartContract.read(
                wallet.network_id,
                contract_address,
                abi=abi,
                method="tokenOfOwnerByIndex",
                args={
                    "owner": check_address,
                    "index": str(i)
                }
            )
            owned_tokens.append(token_id)

        token_list = ", ".join(str(token_id) for token_id in owned_tokens)
        return f"Address {check_address} owns {len(owned_tokens)} NFTs in contract {contract_address}.\nToken IDs: {token_list}"

    except Exception as e:
        return f"Error getting Xonin NFT balance for address {check_address} in contract {contract_address}: {e!s}"



####
def is_valid_mint_request(tweet_text):
    """Check if tweet contains valid mint request and extract address."""
    pattern = r"(?i)mint.*?to\s+(0x[a-fA-F0-9]+)"
    match = re.search(pattern, tweet_text)
    if match and is_address(match.group(1)):
        return match.group(1)
    return None

def process_mint_request(agent_executor, config, tweet_id, eth_address):
    """Process an NFT mint request."""
    try:
        # Get Twitter API wrapper from tools
        twitter_tool = next(
            tool for tool in agent_executor.tools 
            if isinstance(tool, TwitterToolkit)
        )
        twitter_api = twitter_tool.twitter_api_wrapper.api

        # Mint NFT and get response
        mint_response = mint_myNft(agent_executor.wallet, eth_address)
        
        # Extract SVG filename from response
        svg_file = re.search(r'nft_\d+\.svg', mint_response)
        if not svg_file:
            raise ValueError("Could not find SVG file in mint response")
        
        svg_file = svg_file.group(0)
        
        # Upload SVG media to Twitter
        media = twitter_api.media_upload(svg_file)
        media_id = media.media_id_string
        
        # Construct tweet with media
        tweet_text = mint_response
        twitter_api.update_status(
            status=tweet_text,
            in_reply_to_status_id=tweet_id,
            media_ids=[media_id]
        )
        
        # Clean up files
        try:
            os.remove(svg_file)
        except Exception as e:
            print(f"Warning: Could not clean up files: {e}")

    except Exception as e:
        error_msg = f"Error processing mint request: {str(e)}"
        print(error_msg)
        # Try to reply with error message
        try:
            twitter_api.update_status(
                status=f"Sorry, there was an error processing your mint request: {str(e)}",
                in_reply_to_status_id=tweet_id
            )
        except:
            print("Could not send error message tweet")

def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize Twitter wrapper using the existing TwitterApiWrapper
    values = {}
    twitter_wrapper = TwitterApiWrapper(**values)    
    twitter_toolkit = TwitterToolkit.from_twitter_api_wrapper(twitter_wrapper)
    twitter_tools = twitter_toolkit.get_tools()

    # Initialize CDP 
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    mintNftTool = CdpTool(
        name="mint_myNft",
        description=MINT_MYNFT_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        args_schema=MintMyNftInput,
        func=mint_myNft,
    )

    # Add after GET_BALANCE_MYNFT_PROMPT
    TRANSFER_MYNFT_PROMPT = """
    This tool will transfer a Xonin NFT (ERC-721) to another address.
    It takes the following inputs:
    - to_address: The address to transfer the NFT to
    - token_id: The ID of the NFT to transfer
    """

    class TransferMyNftInput(BaseModel):
        """Input argument schema for transfer Xonin NFT action."""
        to_address: str = Field(..., description="The address to transfer the NFT to")
        token_id: int = Field(..., description="The ID of the NFT to transfer")

    def transfer_myNft(
        wallet: Wallet,
        to_address: str,
        token_id: int,
    ) -> str:
        """Transfer a Xonin NFT to another address.

        Args:
            wallet (Wallet): The wallet to sign the transfer from
            to_address (str): The address to transfer the NFT to
            token_id (int): The ID of the NFT to transfer

        Returns:
            str: A message containing the transfer details
        """
        try:
            # Call transferFrom function
            transfer_invocation = wallet.invoke_contract(
                contract_address="0x15077415012b6f5a6F2842928886B51e0E2CB2D6",
                abi=abi,
                method="transferFrom",
                args={
                    "from": wallet.default_address.address_id,
                    "to": to_address,
                    "tokenId": str(token_id)
                }
            ).wait()

            return (f"Transferred Xonin NFT #{token_id} to {to_address} on network {wallet.network_id}.\n"
                    f"Transaction hash: {transfer_invocation.transaction.transaction_hash}\n"
                    f"Transaction link: {transfer_invocation.transaction.transaction_link}")

        except Exception as e:
            return f"Error transferring Xonin NFT: {e!s}"

    # Add in initialize_agent() after getBalanceMyNftTool
    transferNftTool = CdpTool(
        name="transfer_myNft",
        description=TRANSFER_MYNFT_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        args_schema=TransferMyNftInput,
        func=transfer_myNft,
    )

    # Combine tools from both toolkits
    tools = twitter_tools + [mintNftTool, transferNftTool]  + cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    agent_executor, config = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a helpful agent that can interact both onchain using the Coinbase Developer Platform AgentKit "
            "and with Twitter using Twitter API. You can perform blockchain operations and social media actions. "
            "For blockchain: If you ever need funds, you can request them from the faucet if you are on network ID "
            "'base-sepolia'. If not, you can provide your wallet details and request funds from the user. "
            "For Twitter: You can post tweets, read tweets, and interact with Twitter users. "
            "Before executing your first blockchain action, get the wallet details to see what network you're on. "
            "If there is a 5XX (internal) HTTP error code, ask the user to try again later. "
            "If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
        ),
    ), config

    return agent_executor, config, tools, twitter_wrapper  # Return twitter_wrapper as well

def get_account_details(tools):
    """Get Twitter account details to obtain account_id."""
    account_details_tool = next(
        tool for tool in tools 
        if tool.name == "account_details"
    )
    
    response = account_details_tool._run()
    print("Account details:", response)
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        
        data = json.loads(json_str)
        if "data" in data:
            return data["data"].get("id")
    except json.JSONDecodeError as e:
        print(f"Error parsing account details JSON: {e}")
    return None

def get_dummy_mentions():
    """Get mentions from dummy file for debugging."""
    if not os.path.exists(DUMMY_MENTIONS_FILE):
        print(f"Warning: {DUMMY_MENTIONS_FILE} not found")
        return []
        
    try:
        with open(DUMMY_MENTIONS_FILE, 'r') as f:
            data = json.load(f)
            if "data" in data:
                # Create author lookup from includes
                authors = {}
                if "includes" in data and "users" in data["includes"]:
                    for user in data["includes"]["users"]:
                        authors[user["id"]] = user["username"]
                
                # Add author username to each tweet
                tweets = data["data"]
                for tweet in tweets:
                    if "author_id" in tweet and tweet["author_id"] in authors:
                        tweet["author_username"] = authors[tweet["author_id"]]
                
                return tweets
    except json.JSONDecodeError as e:
        print(f"Error parsing dummy mentions file: {e}")
    except Exception as e:
        print(f"Error reading dummy mentions file: {e}")
    
    return []

def get_all_mentions(account_mentions_tool, account_id, max_results=10, since_id=None):
    """Get the latest mentions."""
    if DEBUG_MODE:
        print("DEBUG MODE: Reading from dummy mentions file")
        return get_dummy_mentions()
    
    # Add since_id and max_results to the API call if we have them
    params = {"account_id": account_id}
    if since_id:
        params["since_id"] = since_id
    if max_results:
        params["max_results"] = max_results
    
    response = account_mentions_tool._run(**params)
    print("Mentions response:", response)
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        
        data = json.loads(json_str)
        if "data" in data:
            # Create author lookup from includes
            authors = {}
            if "includes" in data and "users" in data["includes"]:
                for user in data["includes"]["users"]:
                    authors[user["id"]] = user["username"]
            
            # Add author username to each tweet
            tweets = data["data"]
            for tweet in tweets:
                if "author_id" in tweet and tweet["author_id"] in authors:
                    tweet["author_username"] = authors[tweet["author_id"]]
            
            return tweets
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
    
    return []

def is_valid_mint_request_with_feedback(tweet_text):
    """Check if tweet contains mint request and provide feedback."""
    # Case insensitive search for 'mint' and 'to' followed by an address
    pattern = r"(?i)mint.*?to\s+(0x[a-fA-F0-9]+)"
    match = re.search(pattern, tweet_text)
    if not match:
        return None, None
    
    address = match.group(1)
    if not is_address(address):
        return address, "invalid_address"  # Return the invalid address for the error message
    
    return address, "valid"

def check_eth_balance(agent_executor, config, address):
    """Check if address has non-zero ETH balance."""
    balance_prompt = f"Get the ETH balance of address {address}"
    
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=balance_prompt)]}, config
    ):
        if "agent" in chunk:
            response = chunk["agent"]["messages"][0].content
            # Look for balance in the response
            if "balance: 0" in response.lower() or "balance is 0" in response.lower():
                return False
            if "balance:" in response.lower() and "eth" in response.lower():
                return True
    return False

def send_error_reply(agent_executor, config, tweet_id, error_type, address=None, author=None, previous_tweet_id=None):
    """Send error reply tweet and return reply ID if successful."""
    greeting = f"Hey @{author}! " if author else ""
    
    if error_type == "invalid_address":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'{greeting}Sorry, the address {address} is not a valid Ethereum address. "
            "Please make sure to provide a valid address starting with 0x. You can always mint your NFT at https://xonin.vercel.app/.' Be creative in conveying this message!"
        )
    elif error_type == "zero_balance":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'{greeting}Sorry, the address {address} has 0 ETH balance. Please provide an active address. You can always mint your NFT at https://xonin.vercel.app/.'"
            f"Or more humorously like: '{greeting}Why so poor? Get some ETH first.' Be creative in conveying this message!"
        )
    elif error_type == "already_minted":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'{greeting}You have already minted an NFT (see tweet {previous_tweet_id}). "
            "This is limited to one NFT per user, don't be greedy! You can mint another one yourself at https://xonin.vercel.app/.' Be creative in conveying this message!"
        )
    
    reply_id = None
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=reply_prompt)]}, config
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].content)
        elif "tools" in chunk:
            response = chunk["tools"]["messages"][0].content
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                if "data" in data and "id" in data["data"]:
                    reply_id = data["data"]["id"]
            except:
                pass
    return reply_id

class MentionMemory:
    def __init__(self):
        self.processed_mentions = {}
        self.last_tweet_id = None
        self.load_memory()
    
    def load_memory(self):
        """Load processed mentions from file."""
        if os.path.exists(MENTION_MEMORY_FILE):
            try:
                with open(MENTION_MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.processed_mentions = data.get("mentions", {})
                    self.last_tweet_id = data.get("last_tweet_id")
            except json.JSONDecodeError:
                print("Error loading mention memory, starting fresh")
                self.processed_mentions = {}
                self.last_tweet_id = None
    
    def save_memory(self):
        """Save processed mentions to file."""
        data = {
            "mentions": self.processed_mentions,
            "last_tweet_id": self.last_tweet_id
        }
        with open(MENTION_MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_last_tweet_id(self, tweets):
        """Update the last tweet ID from a list of tweets."""
        if tweets:
            # Get the newest tweet ID
            newest_id = max(int(tweet["id"]) for tweet in tweets)
            # Update if it's newer than what we have
            if not self.last_tweet_id or int(newest_id) > int(self.last_tweet_id):
                self.last_tweet_id = str(newest_id)
                self.save_memory()

    def is_processed(self, tweet_id):
        """Check if a tweet has been processed."""
        return tweet_id in self.processed_mentions
    
    def add_mention(self, tweet_id, tweet_text, status, mint_success=False, tx_hash=None, reply_id=None, author=None, author_id=None):
        """Add a processed mention to memory."""
        # Don't save not_mint_request mentions
        if status == "not_mint_request":
            return
            
        mention_data = {
            "text": tweet_text,
            "status": status,
            "mint_success": mint_success,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Add author information
        if author:
            mention_data["author"] = {
                "username": author,
                "id": author_id
            }
        if tx_hash:
            mention_data["transaction_hash"] = tx_hash
        if reply_id:
            mention_data["reply_id"] = reply_id
            
        self.processed_mentions[tweet_id] = mention_data
        self.save_memory()

    def has_successful_mint(self, author_id):
        """Check if user has already successfully minted an NFT."""
        for tweet_id, mention in self.processed_mentions.items():
            if (mention.get("author", {}).get("id") == author_id and 
                mention.get("status") == "processed" and 
                mention.get("mint_success")):
                return tweet_id
        return None

def process_tweet(agent_executor, config, tweet, mention_memory, twitter_wrapper):
    """Process a single tweet."""
    tweet_id = tweet.get("id")
    tweet_text = tweet.get("text")
    
    if not tweet_text or not tweet_id:
        return False
    
    # Check if we've already processed this tweet
    if mention_memory.is_processed(tweet_id):
        return False
        
    # Get author info from the tweet data
    author = tweet.get("author_username")
    author_id = tweet.get("author_id")
    print(f"Processing tweet from @{author}")
    
    # Check if user has already minted successfully
    previous_tweet_id = mention_memory.has_successful_mint(author_id)
    if previous_tweet_id:
        print(f"User @{author} has already minted an NFT")
        reply_id = send_error_reply(
            agent_executor, 
            config, 
            tweet_id, 
            "already_minted", 
            author=author,
            previous_tweet_id=previous_tweet_id
        )
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "duplicate_request",
            author=author,
            author_id=author_id,
            reply_id=reply_id
        )
        return True
        
    # Check if it's a mint request and validate address
    address, status = is_valid_mint_request_with_feedback(tweet_text)
    
    if address is None:
        # Don't save not_mint_request mentions
        return False
        
    if status == "invalid_address":
        print(f"Invalid address found: {address}")
        reply_id = send_error_reply(agent_executor, config, tweet_id, "invalid_address", address, author)
        mention_memory.add_mention(
            tweet_id, 
            tweet_text, 
            "invalid_address", 
            author=author, 
            author_id=author_id,
            reply_id=reply_id
        )
        return True
        
    # Check ETH balance
    if not check_eth_balance(agent_executor, config, address):
        print(f"Zero balance address found: {address}")
        reply_id = send_error_reply(agent_executor, config, tweet_id, "zero_balance", address, author)
        mention_memory.add_mention(
            tweet_id, 
            tweet_text, 
            "zero_balance", 
            author=author, 
            author_id=author_id,
            reply_id=reply_id
        )
        return True
        
    # If we get here, address is valid and has balance
    print(f"Processing mint request for address: {address}")
    try:
        tx_hash = None
        reply_id = None
        
        # Process mint request and capture response
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=mint_prompt)]}, config
        ):
            if "tools" in chunk:
                response = chunk["tools"]["messages"][0].content
                # Try to extract transaction hash and reply ID
                try:
                    if "transaction hash" in response.lower():
                        tx_hash = re.search(r'0x[a-fA-F0-9]{64}', response).group(0)
                    if "post_tweet_reply" in response:
                        json_start = response.find('{')
                        json_end = response.rfind('}') + 1
                        json_str = response[json_start:json_end]
                        data = json.loads(json_str)
                        if "data" in data and "id" in data["data"]:
                            reply_id = data["data"]["id"]
                except:
                    pass
        
        mention_memory.add_mention(
            tweet_id, 
            tweet_text, 
            "processed", 
            mint_success=True,
            tx_hash=tx_hash,
            reply_id=reply_id,
            author=author,
            author_id=author_id
        )
    except Exception as e:
        print(f"Error minting NFT: {e}")
        mention_memory.add_mention(
            tweet_id, 
            tweet_text, 
            "mint_failed", 
            author=author,
            author_id=author_id
        )
    return True

def run_autonomous_mode(agent_executor, config, tools, twitter_wrapper, interval=3000):
    """Run the agent autonomously with specified intervals."""
    print("Starting autonomous mode with NFT minting capability...")
    print(f"Debug mode: {DEBUG_MODE}")
    
    account_id = '1413425385937809414'
    mention_memory = MentionMemory()
    
    # Get account_mentions tool
    account_mentions_tool = next(
        tool for tool in tools 
        if tool.name == "account_mentions"
    )
    
    while True:
        try:
            # Get mentions (either from API or dummy file)
            all_tweets = get_all_mentions(account_mentions_tool, account_id, max_results=10, since_id=mention_memory.last_tweet_id)
            mentions_found = False
            
            # Process all tweets
            for tweet in all_tweets:
                if process_tweet(agent_executor, config, tweet, mention_memory, twitter_wrapper):
                    mentions_found = True

            # Update last_tweet_id after processing
            mention_memory.update_last_tweet_id(all_tweets)

            if not mentions_found:
                print("No new mint requests found.")

            # Save memory state before waiting
            mention_memory.save_memory()
            print("Saved memory checkpoint...")

            # Wait before next check
            print(f"Waiting {interval} seconds before next check...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            mention_memory.save_memory()  # Final save before exiting
            sys.exit(0)
        except Exception as e:
            print(f"Error occurred: {e}")
            mention_memory.save_memory()  # Save on error too
            print("Saved memory checkpoint...")
            print("Waiting before retry...")
            time.sleep(interval)


# Chat Mode
def run_chat_mode(agent_executor, config):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")

        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")


def main():
    """Start the chatbot agent."""
    agent_executor, config, tools, twitter_wrapper = initialize_agent()  # Get twitter_wrapper from initialize_agent

    run_chat_mode(agent_executor=agent_executor, config=config)
    #run_autonomous_mode(agent_executor=agent_executor, config=config, tools=tools, twitter_wrapper=twitter_wrapper)


if __name__ == "__main__":
    print("Starting NFT Minting Agent...")
    main()
