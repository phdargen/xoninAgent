import os
import sys
import time
import re
import json
import random

from eth_utils import is_address
from datetime import datetime, timezone
import requests
from urllib.parse import unquote
from decimal import Decimal

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
from cdp.address import Address
from cdp.address_reputation import AddressReputation

from wand.image import Image

# Settings
# ---------

# Configure files to persist data
wallet_data_file = "wallet_data.txt"
last_check_file = "last_check_time.txt"

# NFT Contract configuration
network_id = os.getenv('NETWORK_ID')
NFT_CONTRACT_ADDRESS = "0x692E25F69857ceee22d5fdE61E67De1fcE7EA274" if network_id == "base-mainnet" else "0x32f75546e56aEC829ce13A9b73d4ebb42bF56b9c"
NFT_PRICE = Decimal("0.001") if network_id == "base-mainnet" else Decimal("0.001") # in ETH
REPUTATION_THRESHOLD = 20

DEBUG_MODE = True
DUMMY_MENTIONS_FILE = "dummy_mentions.txt"
MENTION_MEMORY_FILE = "mention_memory.txt"

etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
if not etherscan_api_key:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")
ETHERSCAN_URL = "https://api.basescan.org/api" if network_id == "base-mainnet" else "https://api-sepolia.basescan.org/api"

abi = [
    {
        "inputs": [{"internalType": "address", "name": "recipient", "type": "address"}],
        "name": "mintAndTransfer",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }
]

# Helper classes
# ---------
class MentionMemory:
    """Store and manage tweet mentions."""
    def __init__(self):
        self.memory = {"mentions": {}, "last_tweet_id": None}
        self.load_memory()

    def load_memory(self):
        """Load processed mentions from file."""
        if os.path.exists(MENTION_MEMORY_FILE):
            try:
                with open(MENTION_MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.memory = data
            except json.JSONDecodeError:
                print("Error loading mention memory, starting fresh")
                self.memory = {"mentions": {}, "last_tweet_id": None}
    
    def save_memory(self):
        """Save processed mentions to file."""
        with open(MENTION_MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def update_last_tweet_id(self, tweets):
        """Update the last tweet ID from a list of tweets."""
        if tweets:
            # Get the newest tweet ID
            newest_id = max(int(tweet["id"]) for tweet in tweets)
            # Update if it's newer than what we have
            if not self.memory["last_tweet_id"] or int(newest_id) > int(self.memory["last_tweet_id"]):
                self.memory["last_tweet_id"] = str(newest_id)
                self.save_memory()

    def is_processed(self, tweet_id):
        """Check if a tweet has been processed."""
        return tweet_id in self.memory["mentions"]
    
    def add_mention(self, tweet_id, tweet_text, status, mint_success=False, tx_hash=None, reply_id=None, author=None, author_id=None, minted_address=None):
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
        if minted_address:
            mention_data["minted_address"] = minted_address
            
        self.memory["mentions"][tweet_id] = mention_data
        self.save_memory()

    def has_successful_mint(self, author_id, address=None):
        """Check if author or address has already minted successfully."""
        for tweet_id, mention in self.memory["mentions"].items():
            if mention.get("mint_success"):
                # Check both author_id and minted_address
                if (mention.get("author_id") == author_id or 
                    (address and mention.get("minted_address", "").lower() == address.lower())):
                    return tweet_id
        return None

# Helper onchain functions
# ---------
def check_eth_balance(wallet: Wallet, address: str):
    """Check if address has non-zero ETH balance using CDP SDK."""
    try:
        # Create Address object for the given address
        addr = Address(
            network_id=wallet.network_id,
            address_id=address
        )
        # Get ETH balance directly using balance() method
        balance_eth = addr.balance("eth")
        print(f"ETH Balance for {address}: {balance_eth} ETH")
        return balance_eth > 0

    except Exception as e:
            print(f"Error checking ETH balance: {e}")
            return False

def check_reputation(address: str) -> AddressReputation:
    """Check if address reputation using CDP SDK."""
    addr = Address(
        network_id="base-mainnet",
        address_id=address
    )
    reputation = addr.reputation()
            
    return reputation

def get_transaction_data(tx_hash, max_retries=2, delay=15):
    """Get transaction data from etherscan with retries."""
    url = f"{ETHERSCAN_URL}?module=proxy&action=eth_getTransactionReceipt&txhash={tx_hash}&apikey={etherscan_api_key}"
    
    for attempt in range(max_retries):
        print(f"Getting transaction data for {tx_hash} from etherscan: {ETHERSCAN_URL} (Attempt {attempt + 1}/{max_retries})")

        response = requests.get(url)
        data = response.json()
        print(f"Transaction data: {data}")

        # Check if we have a valid result
        if data.get('result'):
            # Check transaction status (1 = success, 0 = failure)
            status = int(data['result'].get('status', '0'), 16)
            if status == 0:
                print("Transaction failed")
                if attempt < max_retries - 1:
                    print(f"Will retry mint in {delay} seconds...")
                    time.sleep(delay)
                    return None, None, False  
                return None, None, False

            # Transaction successful, get logs
            logs = data['result'].get('logs', [])
            if logs:
                try:
                    last_log = logs[-1]
                    if last_log.get('topics') and len(last_log['topics']) >= 4:
                        int_value = int(last_log['topics'][3], 16)
                        contract_address = last_log['address']
                        return int_value, contract_address, True
                except Exception as e:
                    print(f"Error parsing log data: {e}")

        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            print(f"Transaction data not ready yet, waiting {delay} seconds before retry...")
            time.sleep(delay)
        else:
            print("Max retries reached, transaction data not available")

    return None, None, False  

def save_svg_to_png(contract_address, token_id, svg_content) -> str:
    """
    Saves the given SVG content as a PNG file using ImageMagick via wand.

    Parameters:
        contract_address (str): The contract address for the file name.
        token_id (int): The identifier number for the token.
        svg_content (str): The SVG content as a string.
    """
    try:
        file_name = f"output_{contract_address}_{token_id}.png"
        
        # Ensure SVG content is properly formatted
        if not svg_content.strip().startswith('<svg'):
            print("Invalid SVG content")
            return None

        # Save svg content to a temporary file
        svg_file = f"output_{contract_address}_{token_id}.svg"
        with open(svg_file, "w") as f:
            f.write(svg_content)
        
        # Convert SVG to PNG 
        with Image(filename=svg_file, format='svg') as img:
            img.format = 'png'
            img.save(filename=file_name)
        
        print(f"SVG saved as PNG: {file_name}")
        return file_name
    except Exception as e:
        print(f"Error saving SVG to PNG: {e}")
        return None


# Mint nft functions
# ---------
MINT_MYNFT_PROMPT = """
This tool will mint a Xonin NFT and transfer it directly to the specified address by paying {NFT_PRICE} ETH.
The NFT will be minted and transferred in a single transaction.
"""

class MintMyNftInput(BaseModel):
    """Input argument schema for mint XoninNFT action."""
    recipient_address: str = Field(
        ...,
        description="The address that will receive the minted NFT"
    )

def get_token_uri_and_svg(wallet: Wallet, contract_address: str, token_id: int) -> tuple[str, str, str]:
    """Get tokenURI and extract name and SVG from the response."""
    print("Getting tokenURI and SVG from contract")
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
    try:
        print(f"Minting NFT for {recipient_address} from contract {NFT_CONTRACT_ADDRESS} on network {network_id} with price {NFT_PRICE} ETH")
        mint_invocation = wallet.invoke_contract(
            contract_address=NFT_CONTRACT_ADDRESS,
            abi=abi,
            method="mintAndTransfer",
            args={"recipient": recipient_address},
            amount=NFT_PRICE,
            asset_id="eth",
        ).wait()
        
        # Get transaction data
        return (f"🎉 Successfully minted NFT for {recipient_address}!\n\n"
                f"🔗 Transaction: {mint_invocation.transaction.transaction_link}\n"
                f"Transaction hash: {mint_invocation.transaction.transaction_hash}\n"
                )

    except Exception as e:
        return f"Error minting and transferring NFT: {e!s}"

# Twitter functions
# ---------

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

def process_mint_request(agent_executor, wallet: Wallet, config, tweet_id, eth_address, twitter_wrapper, author=None, reputation: AddressReputation=None):
    """Process an NFT mint request."""
    # Mint NFT
    print(f"Starting mint process for {eth_address}...")
    mint_result = mint_myNft(wallet, eth_address)
    print(f"Mint response: {mint_result}")
    
    # Extract transaction hash and link from the result
    txHash = re.search(r'Transaction hash: (0x[a-fA-F0-9]+)', mint_result)
    if not txHash:
        raise ValueError("Could not find transaction hash in mint response")
    txHash = txHash.group(1)
    
    txLink = re.search(r'Transaction: (https://[^\s\n]+)', mint_result)
    if not txLink:
        # Construct link if not found
        base_url = "https://basescan.org/tx/" if network_id == "base-mainnet" else "https://sepolia.basescan.org/tx/"
        txLink = base_url + txHash
    else:
        txLink = txLink.group(1)    
    print(f"Transaction hash: {txHash}")
    print(f"Transaction link: {txLink}")

    # Get transaction info
    token_id, contract_address, success = get_transaction_data(txHash)
    if not success:
        print("Transaction failed")
        return False, txHash, eth_address

    if token_id is None or contract_address is None:
        print("Could not get token data from transaction")
        return False, txHash, eth_address

    # Get token URI, name and SVG after minting
    token_uri, name, svg_data = get_token_uri_and_svg(wallet, contract_address, token_id)
    if not name or not svg_data:
        print("Could not get token URI and SVG")
        return False, txHash, eth_address

    print(f"Minted NFT: {name}")

    # Get Twitter API wrapper from tools list
    twitter_client = twitter_wrapper.v1_api

    # Upload media to Twitter
    png_file = save_svg_to_png(contract_address, token_id, svg_data)
    if not png_file:
        print("Failed to convert SVG to PNG")
    else:
        media = twitter_client.media_upload(png_file)
        if not media:
            print("Failed to upload media to Twitter")        
        else: 
            media_id = media.media_id_string
            print(f"Uploaded media to Twitter, ID: {media_id}")

    metric_msg = ""
    if reputation.score > 0:
        # Extract metrics from metadata
        metadata_str = str(reputation.metadata)
        key_metrics = {
            "total_transactions": "transactions",
            "unique_days_active": "days active",
            "token_swaps_performed": "token swaps",
            "smart_contract_deployments": "smart contracts deployed",
            "lend_borrow_stake_transactions": "lending/borrowing actions",
            "bridge_transactions_performed": "bridge transactions",
            "ens_contract_interactions": "ENS interactions"
        }
        
        # Find all metrics with values > 0
        positive_metrics = {}
        for key, label in key_metrics.items():
            match = re.search(f"{key}=(-?\d+)", metadata_str)
            if match:
                value = int(match.group(1))
                if value > 0:
                    positive_metrics[label] = value

        # Randomly choose one positive metric if any exist
        if positive_metrics:
            key, value = random.choice(list(positive_metrics.items()))
            print(f"Selected metric: {value} {key}")
            metric_msg = f"You may also use the following info to praise the user: {positive_metrics[key]} {key}."

    # Post reply with media
    greeting = f"@{author}! " if author else ""
    media_id_message = f"and attach the media ID: {media_id} " if media_id else ""
    reply_prompt = (
        f"Use post_tweet_reply {media_id_message} to reply to tweet {tweet_id} with a personalized message about the successful mint such as:\n"
        f"'Fuiyoh {greeting}, your onchain reputation score is {reputation.score}! That's so based! I minted {name} for you! Have fun with your fully onchain art on @base! Visit https://xonin.vercel.app/ to learn more about the project! Here's the transaction link: {txLink}.'"
        f"Be creative in conveying this message! For context, the score is between -100 (risky) and +100 (crypto-forward)."
        f"{metric_msg}"
    )
    print(f"Reply prompt: {reply_prompt}")

    print("Sending reply tweet...")
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=reply_prompt)]}, config
    ):
        if "tools" in chunk:
            response = chunk["tools"]["messages"][0].content
            print(f"Reply response: {response}")

    return True, txHash, eth_address  

def send_error_reply(agent_executor, config, tweet_id, error_type, address=None, author=None, reputation: AddressReputation=None):
    """Send error reply tweet and return reply ID if successful."""
    greeting = f"Hey @{author}! " if author else ""
    print(f"Sending error reply for {error_type}...")
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
            f"Or more humorously like: 'Haiyaa {author}, why so poor? Get some ETH first.' Be creative in conveying this message!"
        )
    elif error_type == "already_minted":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'{greeting}You have already minted an NFT. "
            "This is limited to one NFT per user, don't be greedy! You can mint another one yourself at https://xonin.vercel.app/.' Be creative in conveying this message!"
        )
    elif error_type == "low_reputation":
        metric_msg = ""
        if reputation.score > 0:
            # Extract metrics from metadata
            metadata_str = str(reputation.metadata)
            key_metrics = {
                "total_transactions": "transactions",
                "unique_days_active": "days active",
                "token_swaps_performed": "token swaps",
                "smart_contract_deployments": "smart contracts deployed",
                "lend_borrow_stake_transactions": "lending/borrowing actions",
                "bridge_transactions_performed": "bridge transactions",
                "ens_contract_interactions": "ENS interactions"
            }
            
            # Find all metrics and their values
            metrics = {}
            for key, label in key_metrics.items():
                match = re.search(f"{key}=(-?\d+)", metadata_str)  
                if match:
                    value = int(match.group(1))
                    metrics[label] = value

            # Randomly choose one metric 
            if metrics:
                key, value = random.choice(list(metrics.items()))
                print(f"Selected metric: {value} {key}")
                metric_msg = f"You may also use the following information to praise the user: {value} {key}."

        print(f"Metric message: {metric_msg}")

        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'Haiyaa @{author}, your onchain reputation score is only {reputation.score}. Why so low?"
            f"Sorry, no free NFT for you. You can always mint your own at https://xonin.vercel.app/.'"
            f"Be creative in conveying this message! For context, the score is between -100 (risky) and +100 (crypto-forward)."
            f"{metric_msg}."
        )

    reply_id = None
    print("Sending reply tweet...")
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=reply_prompt)]}, config
    ):
        if "tools" in chunk:
            response = chunk["tools"]["messages"][0].content
            print(f"Reply response: {response}")
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

def process_tweet(agent_executor, wallet: Wallet, config, tweet, mention_memory, twitter_wrapper):
    """Process a single tweet."""
    tweet_id = tweet.get("id")
    tweet_text = tweet.get("text")
    
    if not tweet_text or not tweet_id:
        return False
    
    # Check if already processed 
    if mention_memory.is_processed(tweet_id):
        return False
        
    # Get author info from tweet data
    author = tweet.get("author_username")
    author_id = tweet.get("author_id")
    print(f"Processing tweet from @{author}")
     
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
    if not check_eth_balance(wallet, address):
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

    # Check reputation
    reputation = check_reputation(address)
    print(f"Reputation score: {reputation.score}")
    print(f"Reputation metadata: {reputation.metadata}")

    if reputation.score < REPUTATION_THRESHOLD:
        print(f"Reputation score is too low: {reputation.score}")
        reply_id = send_error_reply(agent_executor, config, tweet_id, "low_reputation", address, author, reputation)
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "low_reputation",
            author=author,
            author_id=author_id,
            reply_id=reply_id
        )
        return True

    # Check if user or address has already minted successfully
    previous_tweet_id = mention_memory.has_successful_mint(author_id, address)
    if previous_tweet_id:
        print(f"User @{author} or address {address} has already minted an NFT")
        reply_id = send_error_reply(agent_executor, config, tweet_id, "already_minted", address, author, reputation)
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "duplicate_request",
            author=author,
            author_id=author_id,
            reply_id=reply_id
        )
        return True
       

    # Address is valid and has balance + reputation -> mint nft
    print(f"Processing mint request for address: {address}")
    
    try:
        mint_success, tx_hash, minted_address = process_mint_request(agent_executor, wallet, config, tweet_id, address, twitter_wrapper, author, reputation)
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "processed",
            mint_success=mint_success,
            tx_hash=tx_hash,
            minted_address=minted_address,  
            author=author,
            author_id=author_id
        )
        return True
    except Exception as e:
        print(f"Error in process_tweet: {e}")
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "error",
            mint_success=False,
            author=author,
            author_id=author_id
        )
        return False

# Init agent
# ---------
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

    wallet = agentkit.wallet
    print(f"Wallet: {wallet}")
    check_eth_balance(wallet, wallet.default_address.address_id)
    #score, metadata = check_reputation( wallet.default_address.address_id)

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

    # Combine tools from both toolkits
    tools = twitter_tools + [mintNftTool] # + cdp_toolkit.get_tools()

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

    return agent_executor, wallet, config, tools, twitter_wrapper  

# Running modes
# ---------
def run_autonomous_mode(agent_executor, wallet: Wallet, config, tools, twitter_wrapper, interval):
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
            all_tweets = get_all_mentions(account_mentions_tool, account_id, max_results=10, since_id=mention_memory.memory["last_tweet_id"])
            mentions_found = False
            
            # Process all tweets
            for tweet in all_tweets:
                if process_tweet(agent_executor, wallet, config, tweet, mention_memory, twitter_wrapper):
                    mentions_found = True

            # Update last_tweet_id after processing
            mention_memory.update_last_tweet_id(all_tweets)

            if not mentions_found:
                print("No new mint requests found.")

            # Save memory state before waiting
            mention_memory.save_memory()
            print("Saved memory checkpoint...")

            # Wait before next check
            if interval > 0:
                print(f"Waiting {interval} seconds before next check...")
                time.sleep(interval)
            else:
                exit(0)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            mention_memory.save_memory()  # Final save before exiting
            sys.exit(0)
        except Exception as e:
            print(f"Error occurred: {e}")
            mention_memory.save_memory()  # Save on error too
            print("Saved memory checkpoint...")
            print("Waiting before retry...")
            if interval > 0:
                time.sleep(interval)
            else:
                exit(0)

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

def main():
    """Start the chatbot agent."""
    agent_executor, wallet, config, tools, twitter_wrapper = initialize_agent()  # Get twitter_wrapper from initialize_agent

    #run_chat_mode(agent_executor=agent_executor, config=config)
    #run_autonomous_mode(agent_executor=agent_executor, wallet=wallet, config=config, tools=tools, twitter_wrapper=twitter_wrapper,interval=-1)
    save_svg_to_png("test", 33, "<svg width='500' height='500' viewBox='0 0 500 500' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'><radialGradient id='g0' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#83988e;stop-opacity:0.40'/><stop offset='100%' style='stop-color:#83988e;stop-opacity:1'/></radialGradient><radialGradient id='g1' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#e6f9bc;stop-opacity:0.37'/><stop offset='100%' style='stop-color:#e6f9bc;stop-opacity:1'/></radialGradient><radialGradient id='g2' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#bcdea5;stop-opacity:0.57'/><stop offset='100%' style='stop-color:#574951;stop-opacity:1'/></radialGradient><radialGradient id='g3' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#574951;stop-opacity:0.60'/><stop offset='100%' style='stop-color:#e6f9bc;stop-opacity:1'/></radialGradient><radialGradient id='g4' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#574951;stop-opacity:0.80'/><stop offset='100%' style='stop-color:#574951;stop-opacity:1'/></radialGradient><radialGradient id='g5' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#574951;stop-opacity:0.64'/><stop offset='100%' style='stop-color:#574951;stop-opacity:1'/></radialGradient><filter id='f1' width='200%' height='200%'><feOffset in='SourceGraphic' result='r' dx='0' dy='30' /><feGaussianBlur in='r' result='rb' stdDeviation='4'/><feMerge><feMergeNode in='rb' /><feMergeNode in='SourceGraphic' /></feMerge></filter><symbol id='p' viewBox='0 0 500 500'><path fill='url(#g0)' d='M250 250 q -21 -25 -20 -30 -16 -29 -26 -30 15 45 -14 47 -39 -48 -25 -31 -38 -22 43 48 -13 -21 -39 -34 -44 -20 31 21 -41 42 24 24 45 -41 -31 -21 -14 -11 -43 -27 -22 -19 -17 -45 49 18 -25 34 -46 25 -13 42 22 41 -33 -20 -44 18 -40 14 -13 -11 -15 46 -23 33 -17 32 -23 -30 -38 -30 -17 -37 -14 18 -22 35 -41 33 36 21 -40 -24 32 -31 35 -10 -23 34 -42 -44 -12 27 -32 41 -36 -11 34 -49 -23 35 10 -30 27 24 39 16 -22 -29 -43 -12 -20 -42 24 42 -30 30 -20 47 -26 35 -36 27 -23 27 13 18 12 20 -37 -47 -48 -49 15 -36 -15 -10 -27 37 36 -49 20 24 -23 -29 -42 -14 -46 30 39 -28 -25 11 31 36 -20 -14 -23 32 -36 45 -42 -20 -47 13 -46 -35 -47 -39 22 -30 49 34 -45 -33 -38 22 -27 -22 -48 48 12 -48 21 47 -26 23 30 29 -27 -10 -31 -25 -33 -40 18 35 29 27 -11 15 34 10 -34 -33 -36 27 40 -30 22 30 -46 -49'/><path fill='url(#g1)' d='M250 250 q 10 -31 -19 15 -33 -46 31 -46 -42 -11 31 -31 42 46 -28 11 -43 33 39 46 12 -28 -18 41 -32 14 30 47 13 -37 32 26 42 -14 43 -18 -14 30 -31 -20 -48 -47 -42 45 -30 37 32 -19 37 12 -31 -10 -49 -27 -22 -25 -17 -22 18 -39 -39 25 10 47 -24 11 19 -38 47 49 15 48 32 45 -44 -13 -31 49 31 29 -34 -45 31 -37 -40 -32 40 40 35 -29 -10 -25 45 13 -21 -35 -15 13 -19 -12 -23 21 -36 48 -37 -40 36 -13 -17 -27 -32 -36 30 -33 -48 -10 -30 34 12 33 -29 40 -18 -14 48 16 -25 -44 -35 -16 18 -36 -38 16 -27 -31 -47 -24 22 40 -33 34 -43 -44 -33 17 38 20 -35 -18 33 -16 47 26 21 47 -10 -47 21 -44 -36 -32 -31 -33 -31 17 -43 10 -39 -18 12 27 -27 -21 21 -22 38 13 14 -24 27 15 41 17 -26 -45 -38 -18 41 43 -44 -40 36 -39 32 32 -33 34 -48 -34 40 -20 10 46 33 -44 25 -24 -44 -16 -12 -16 -26 -32 -45 18 40 -33 35 -43 -47 -25 37 12 -13 -47 -41 -29 -18 -40 -41 -26 -49 24 -46 -14 -15 27 35 24 35 -31 -17 -15 -41 -39 -22 -45 -47 -44 -48 12 -12 -15 -10 -18 -41 44 -49 24 28 16 11 -28 -22 44 -23 -25 -24 -27 -17 -33 28 -21 34 -35 23 -23 19 44 -37 -18'/><path fill='url(#g2)' d='M250 250 t 37 -44 40 49 11 28 -48 -31 -15 -29 -31 32 30 39 -46 40 46 -33 -38 -46 -47 29 -33 11 10 28 14 42 49 11 -17 12 -46 25 35 12 21 16 -49 14 26 41 -21 29 -16 21 45 17 23 42 -21 -47 -16 25 36 -37 -13 39 28 43 24 11 -20 13 29 22 -41 14 48 29 -33 -26 -31 -19 46 49 20 27 34 -12 10 23 11 -21 -28 -24 36 32 -28 -28 -33 -40 13 -33 23 -30 -41 47 -35 32 31 41 49 39 -29 -40 43 38 16 -23 29 -39 -34 34 38 -12 27 21 23 42 -27 47 36 14 49 23 -41 24 41 30 36 -20 19 -47 24 20 -46 39 -34 18 17 -36 -49 10 32 11 -21 21 -22 30 43 -41 -12 25 -26 -40 43 12 31 48 27 12 -17 -47 45 -37 31 -24 -43 -29 -36 26 -17 17 -43 20 -19 -23 20 47 24 45 17 -12 37 -48 -13 46 26 -48 39 -37 45 45 -40 -13 34 43 -12 12 16 20 -39 -11 -42 27 48 27 29 -15 18 19 -34 47 -20 18 -30 16 -39 -32 16 35 -33 12 15 19'/><path fill='url(#g3)' d='M250 250 q -58 109 124 96 -56 184 -70 179 -119 -165 -67 -197 -176 -82 184 -141 190 120 -13 -151 -191 152 -167 -70 -162 -145 -182 -22 40 63 -71 103 -11 126 -150 -58 -181 86 -189 61 186 -142 -75 25 -142 40 -149 -172 -113 126 -118 -55 -76 115 158 -195 -88 -180 152 -189 -130 73 -130 179 -159 -114 -149 120 -138 -168 37 79 -18 180 -78 190 -142 141 -28 -69 -12 154 33 -26 -123 -84 -12 -33 -115 -54 121 98 -107 -130 -40 -42 -17 -124 108 -17 74 69 13 117 43 -111 69 -83 -10 -168 -115 -122 -14 -102 144 -156 -123 75 -148 -108 -116 61 96 110 19 57 37 -176 -143 -140 -83 -152 110 -128 -89 23 -123 48 -72 120 -124 131 36 162 -22 12 119 -16 -132 159 -195 -142 39 44 -154 -26 167 165 -55 -161 -162 -65 -74 33 -40 -178 152 64 -118 -175 -32 59 186 187 167 -50 -159 103 -102 -100 -59 -53 -139 -125 -197 105 66 63'/><path fill='url(#g4)' d='M250 250 q 42 49 -25 23 -22 -11 -23 -18 -32 21 27 -23 49 45 -27 11 -12 -47 41 42 29 -26 -32 34 25 35 -23 -45 27 16 16 -26 -28 25 11 -26 26 18 28 -47 32 20 -23 16 49 -15 -41 43 -16 20 12 -11 38 46 32 37 14 -39 10 32 38 39 23 -39 36 -45 39 11 -38 21 33 -35 26 41 -42 36 33 42 39 -47 45 -44 31 35 42 30 35 17 42 33 -40 12 -11 17 -21 -33 -39 -40 -36 12 -45 34 49 -33 -14 -24 28 -41 -42 13 -43 13 -11 -37 -19 19 28 36 -14 11 -15 23 11 44 -25 -31 -17 -40 39 19 16 -43 -44 -11 -28 49 -16 -46 18 43 32 -11 44 -40 -36 16 -33 -37 41 -21 44 22 -13 18 -32 -49 34 -30 -14 -13 -38 -35 19 46 38 36 -17 -13 20 -18 44 -41 -45 -32 21 37 43 34 -11 34 -46 -28 37 -47 -28 -34 11 11 -36 -42 -31 45 32 -39 -48 -38 -15 10 39 33 10 -21 25 26 16 -27 -24 -28 -45 -47 -26 -34 -22 -42 13 40 47 -12 44 -37 -21 30 10 -29 -48 23 -23 24 18 36 42 27 25 21 35 16 15 -44 -49 -13 -13 38 -10 -31 -26 -31 27 27 12 23 47 -30 -22 -16 -45 27 -36 35 -16 10 28 39 44 32 48 -40 10 26 16 -22 12 -46 47 34 -39 32 -31 38 -25 24'/><path fill='url(#g5)' d='M250 250 t 70 -196 84 -56 171 80 -17 -90 -44 -188 -96 -39 81 77 104 -43 198 -161 80 -153 -32 46 -12 52 131 127 128 154 -13 175 50 -25 -62 -131 -169 -45 82 -139 -83 -157 49 -13 32 -182 -122 -163 181 -70 47 -38 -14 123 -189 -62 -49 40 -115 199 102 28 -198 75 -85 34 113 124 53 -132 -110 -112 126 122 12 46 -122 155 21 -33 -29 -167 83 -127 150 -107 -124 -73 79 137 16 -81 78 -74 -194 -55 -127 39 118 124 -117 69 199 45 24 -29 -102 -20 -191 195 -72 94 -144 -126 -105 -169 113 -68 -80 -135 -175 -136 -187 -172 31 -54 -175 164 -69 -43 104 -196 42 129 151 -181 -50 -101 188 -163 -51 -46 -47 192 -176 -180 -182 141 -72 -139 88 -156 19 147 -14 63 -54 -156 127 18 -57 -154 -158 159 -18 -96 130 199 149 147 -110 -14 95 -64 -156 -92 151 -33 174 193 -113 -131 -42 -96 -172 -105 -127 -121 -36 -70 197 147'/></symbol><g fill='#3a111c'><rect width='500' height='500' /><use href='#p'/></g></svg>")

if __name__ == "__main__":
    print("Starting NFT Minting Agent...")
    main()
