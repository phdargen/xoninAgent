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

from web3 import Web3

import subprocess

# Settings
# ---------

# Config
wallet_data_file = "wallet_data.txt"
DEBUG_MODE = False
DUMMY_MENTIONS_FILE = "dummy_mentions.txt"
MENTION_MEMORY_FILE = "mention_memory.txt"
ADMIN_ID = '1340039893595074560'
ADMIN_NAME = "DukeOphir"

# NFT contract 
network_id = os.getenv('NETWORK_ID')
NFT_CONTRACT_ADDRESS = "0x692E25F69857ceee22d5fdE61E67De1fcE7EA274" if network_id == "base-mainnet" else "0x32f75546e56aEC829ce13A9b73d4ebb42bF56b9c"
NFT_PRICE = Decimal("0.001") if network_id == "base-mainnet" else Decimal("0.001") # in ETH
REPUTATION_THRESHOLD = 20

abi = [
    {
        "inputs": [{"internalType": "address", "name": "recipient", "type": "address"}],
        "name": "mintAndTransfer",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }
]

# Etherscan 
etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
if not etherscan_api_key:
    raise ValueError("ETHERSCAN_API_KEY environment variable is not set")
ETHERSCAN_URL = "https://api.basescan.org/api" if network_id == "base-mainnet" else "https://api-sepolia.basescan.org/api"

# Infura
infura_api = os.getenv('INFURA_API_KEY')
if not infura_api:
    raise ValueError("INFURA_API_KEY environment variable is not set")
w3 = Web3(Web3.HTTPProvider(infura_api))
w3.ens = w3.ens.from_web3(w3)

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
    
    def add_mention(self, tweet_id, tweet_text, status, mint_success=False, tx_hash=None, minted_address=None, minted_domain=None, minted_nft_name=None, author=None, author_id=None, reply_id=None):
        """Add a processed mention to memory."""
        # Don't save not_mint_request mentions
        if status == "not_mint_request":
            return
            
        mention_data = {
            "text": tweet_text,
            "status": status,
            "mint_success": mint_success,
            "tweet_success": bool(reply_id),  # Set based on whether we got a reply_id
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "author": {"username": author, "id": author_id} if author and author_id else None,
        }
        
        if tx_hash:
            mention_data["transaction_hash"] = tx_hash
        if minted_address:
            mention_data["minted_address"] = minted_address
        if minted_domain:
            mention_data["minted_domain"] = minted_domain
        if minted_nft_name:
            mention_data["minted_nft_name"] = minted_nft_name
        if reply_id:
            mention_data["reply_id"] = reply_id
            
        self.memory["mentions"][tweet_id] = mention_data
        self.save_memory()

    def has_successful_mint(self, author_id, address=None):
        """Check if author or address has already minted successfully."""
        for tweet_id, mention in self.memory["mentions"].items():
            if mention.get("mint_success"):
                # Check both author.id and minted_address
                author = mention.get("author", {})
                if (author.get("id") == author_id or 
                    (address and mention.get("minted_address", "").lower() == address.lower())):
                    return tweet_id
        return None

# Helper onchain functions
# ---------
def get_eth_balance(address: str):
    """Check if address has non-zero ETH balance using CDP SDK."""
    try:
        # Create Address object for the given address
        addr = Address(
            network_id="base-mainnet",
            address_id=address
        )
        # Get ETH balance
        balance_eth = addr.balance("eth")
        print(f"ETH Balance for {address}: {balance_eth} ETH")
        return balance_eth

    except Exception as e:
            print(f"Error checking ETH balance: {e}")
            return None

def check_reputation(address: str, max_retries=2, delay=30) -> AddressReputation:
    """Check if address reputation using CDP SDK."""
    for attempt in range(max_retries + 1): 
        try:
            addr = Address(
                network_id="base-mainnet",
                address_id=address
            )
            reputation = addr.reputation()
            print(f"Reputation for {address}: {reputation}. \n")
            return reputation
        except Exception as e:
            print(f"Error checking reputation (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:  
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return None

def resolve_ens(domain):
    """Resolve ENS domain to address using Base L2 resolver."""
    global w3  # Access the global w3 instance
    try:
        address = w3.ens.address(domain)
        print(f"Resolved {domain} to {address}")
        return address, domain
    except Exception as e:
        print(f"Error resolving ENS domain: {e}")
        return None, None

def get_transaction_data(tx_hash, max_retries=4, delay=25):
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
    Saves the given SVG content as a PNG file using rsvg-convert or inkscape as fallback.

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

        # Try rsvg-convert first
        try:
            subprocess.run(["rsvg-convert", "-o", file_name, svg_file], check=True)
            print(f"Converted SVG to PNG using rsvg-convert: {file_name}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If rsvg-convert fails, try inkscape
            try:
                subprocess.run(["inkscape", "-o", file_name, svg_file], check=True)
                print(f"Converted SVG to PNG using Inkscape: {file_name}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Both rsvg-convert and Inkscape failed: {e}")
                return None

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
    """Check if tweet contains an address or ENS domain and provide feedback."""
    
    # Split text into words and find where leading mentions end
    words = tweet_text.split()
    message_start = 0
    
    # Count leading mentions (words that start with @ and are separated by single space)
    for i, word in enumerate(words):
        if word.startswith('@'):
            message_start = i + 1
        else:
            break
    
    # Join the rest as the actual message
    actual_message = ' '.join(words[message_start:])
    
    # Check for tagged user in actual message (ignore @XoninNFT)
    tagged_user = None
    for match in re.finditer(r'@(\w+)', actual_message):
        username = match.group(1)
        if username != "XoninNFT":  # Skip XoninNFT mentions
            tagged_user = username
            break  # Take first non-XoninNFT tag

    # Search for 0x address or .eth domain
    patterns = [
        r"0x[a-fA-F0-9]{40}",  # ETH address
        r"\S+\.eth\b",         # .eth domain 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, actual_message)
        if match:
            # If ENS domain, try to resolve 
            address = match.group(0)            
            domain = None
            if '.eth' in address:
                try:
                    resolved, domain = resolve_ens(address)
                    if not resolved:
                        return address, domain, "invalid_address", tagged_user
                    address = resolved
                except Exception as e:
                    print(f"Error resolving ENS domain: {e}")
                    return address, domain, "invalid_address", tagged_user
            
            # Validate the address
            if not is_address(address):
                return address, domain, "invalid_address", tagged_user
                
            return address, domain, "valid", tagged_user
    
    return None, None, None, tagged_user

def process_mint_request(agent_executor, wallet: Wallet, config, tweet_id, eth_address, domain, twitter_wrapper, author=None, reputation: AddressReputation=None, tagged_user=None):
    """Process an NFT mint request."""

    print(f"Starting mint process for {eth_address}...")

    # Mint NFT
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
        return False, txHash, None

    if token_id is None or contract_address is None:
        print("Could not get token data from transaction")
        return False, txHash, None

    # Get token URI, name and SVG after minting
    token_uri, name, svg_data = get_token_uri_and_svg(wallet, contract_address, token_id)
    if not name or not svg_data:
        print("Could not get token URI and SVG")
        return False, txHash, None

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

    # Send reply with greeting 
    greeting = f"@{author}" if author else ""
    if author == ADMIN_NAME and tagged_user:
        greeting = f"@{tagged_user}" 

    # Select reputation metric to praise (optionally)
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
            metric_msg = ( f" Or use this info to praise the user: '{positive_metrics[key]} {key}' in addition to a message like:"
                           f"'Fuiyoh {greeting}, your @CoinbaseDev onchain reputation score is {reputation.score}!"
                           f" I minted {name} for you, an original piece of fully onchain art on @base: {txLink}!'"
            )

    # Post reply with media
    media_id_message = f"and attach the media_id: {media_id}" if media_id else ""
    reply_prompt = (
        f"Use post_tweet_reply {media_id_message} to reply to tweet {tweet_id} with a personalized message about the successful mint such as:\n"
        f"'Fuiyoh {greeting}, your @CoinbaseDev onchain reputation score is {reputation.score}! That's so based!"
        f" I minted {name} for you! Visit https://xonin.vercel.app/ to learn more! Have fun with your fully onchain art on @base: {txLink}.'"
        f"{metric_msg}"
        f" Be creative in conveying the message. If you get '403 Forbidden' error, try again ensuring the message is below 280 characters!"
    )
    print(f"Reply prompt: {reply_prompt}")

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

    return reply_id != None, txHash, reply_id, name

def send_error_reply(agent_executor, config, tweet_id, error_type, address=None, domain=None, author=None, reputation: AddressReputation=None, tagged_user=None):
    """Send error reply tweet and return reply ID if successful."""
    greeting = f"@{author}" if author else ""
    if author == ADMIN_NAME and tagged_user:
        greeting = f"@{tagged_user}" 
    
    print(f"Sending error reply for {error_type}...")
    if error_type == "invalid_address":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'Hey {greeting}! Sorry, the address {address} is not a valid. "
            "Please make sure to provide a valid eth address or ENS/basename. You can always mint your NFT at https://xonin.vercel.app/.' Be creative in conveying this message!"
        )
    elif error_type == "zero_balance":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'Hey {greeting}! Sorry, the address {address} has 0 ETH balance on @base. Please provide an active address. You can always mint your NFT at https://xonin.vercel.app/.'"
            f"Or more humorously like: 'Haiyaa {greeting}, why so poor? Get some ETH on @base first.' Be creative in conveying this message!"
        )
    elif error_type == "already_minted":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'Hey {greeting}! You have already minted an NFT. "
            "This is limited to 1 NFT per user or address, don't be greedy! You can mint another one yourself at https://xonin.vercel.app/.' Be creative in conveying this message!"
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
                metric_msg = f" You may also use this info to suggest the user how to improve the score: '{value} {key}' or just say: 'Go mint yourself at https://xonin.vercel.app/.'"
        print(f"Metric message: {metric_msg}")

        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"'Haiyaa {greeting}, your onchain reputation score is only {reputation.score}. Why so low?"
            f"Sorry, no free NFT for you.'"
            f"{metric_msg}."
            f"Be creative in conveying this message! If you get '403 Forbidden' error, try again ensuring the message is below 280 characters!"
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
    address, domain, status, tagged_user = is_valid_mint_request_with_feedback(tweet_text)
    
    if address is None:
        # Don't save not_mint_request mentions
        return False
        
    if status == "invalid_address":
        print(f"Invalid address found: {address}")
        reply_id = send_error_reply(agent_executor, config, tweet_id, "invalid_address", address, domain, author, None, tagged_user)
        mention_memory.add_mention(
            tweet_id, 
            tweet_text, 
            "invalid_address", 
            author=author, 
            author_id=author_id,
            minted_address=address,  
            minted_domain=domain,
            reply_id=reply_id
        )
        return True
        
    # Check ETH balance
    balance = get_eth_balance(address)
    if balance is None:
        return False
    if not balance > 0:
        print(f"Zero balance address found: {address}")
        reply_id = send_error_reply(agent_executor, config, tweet_id, "zero_balance", address, domain, author, None, tagged_user)
        mention_memory.add_mention(
            tweet_id, 
            tweet_text, 
            "zero_balance", 
            author=author, 
            author_id=author_id,
            minted_address=address,  
            minted_domain=domain,
            reply_id=reply_id
        )
        return True

    # Check reputation
    reputation = check_reputation(address)
    if reputation is None:
        print(f"Error checking reputation for address: {address}")
        exit(0)
    
    print(f"Reputation score: {reputation.score}")
    print(f"Reputation metadata: {reputation.metadata}")

    if reputation.score < REPUTATION_THRESHOLD:
        print(f"Reputation score is too low: {reputation.score}")
        reply_id = send_error_reply(agent_executor, config, tweet_id, "low_reputation", address, domain, author, reputation, tagged_user)
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "low_reputation",
            author=author,
            author_id=author_id,
            minted_address=address,  
            minted_domain=domain,
            reply_id=reply_id
        )
        return True

    # Check if user or address has already minted successfully
    if author_id != ADMIN_ID:
        print(f"Checking if user @{author} or address {address} has already minted an NFT")
        previous_tweet_id = mention_memory.has_successful_mint(author_id, address)
        if previous_tweet_id:
            print(f"User @{author} or address {address} has already minted an NFT")
            reply_id = send_error_reply(agent_executor, config, tweet_id, "already_minted", address, domain, author, reputation, tagged_user)
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
    print(f"Processing mint request for address: {address} and domain: {domain}")
    try:
        mint_success, tx_hash, reply_id, name = process_mint_request(agent_executor, wallet, config, tweet_id, address, domain, twitter_wrapper, author, reputation, tagged_user)
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "processed",
            mint_success=mint_success,
            tx_hash=tx_hash,
            minted_nft_name=name,
            minted_address=address,  
            minted_domain=domain,
            author=author,
            author_id=author_id,
            reply_id=reply_id
        )
        return True
    except Exception as e:
        print(f"Error in process_tweet: {e}")
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "error",
            mint_success=False,
            minted_address=address,  
            minted_domain=domain,
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

    balance = get_eth_balance(wallet.default_address.address_id)
    if balance < NFT_PRICE * Decimal("1.5") and network_id == "base-mainnet":
        print(f"Wallet balance is too low: {balance}. Please fund {wallet.default_address.address_id} with at least {NFT_PRICE * Decimal('1.5') - balance} ETH.")
        exit(0) 
    #check_reputation(wallet.default_address.address_id)
    #private_key = wallet.default_address.export()
    
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
    print(f"Network ID: {network_id}")

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
                    time.sleep(20)

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
    run_autonomous_mode(agent_executor=agent_executor, wallet=wallet, config=config, tools=tools, twitter_wrapper=twitter_wrapper,interval=-1)
    #save_svg_to_png("test", 34, "<svg width='500' height='500' viewBox='0 0 500 500' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'><radialGradient id='g0' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#f7af63;stop-opacity:0.39'/><stop offset='100%' style='stop-color:#f7af63;stop-opacity:1'/></radialGradient><radialGradient id='g1' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#633d2e;stop-opacity:0.61'/><stop offset='100%' style='stop-color:#633d2e;stop-opacity:1'/></radialGradient><radialGradient id='g2' r='1' spreadMethod='reflect'><stop offset='0%' style='stop-color:#ddd9ab;stop-opacity:0.57'/><stop offset='100%' style='stop-color:#ddd9ab;stop-opacity:1'/></radialGradient><filter id='f1' width='200%' height='200%'><feOffset in='SourceGraphic' result='r' dx='40' dy='40' /><feGaussianBlur in='r' result='rb' stdDeviation='3'/><feMerge><feMergeNode in='rb' /><feMergeNode in='SourceGraphic' /></feMerge></filter><symbol id='p' viewBox='0 0 500 500'><path fill='url(#g0)' d='M250 250 s 45 -35 -28 -42 -18 -33 -34 22 -37 -12 -20 -47 -35 -12 -32 18 -49 43 18 -24 -35 26 49 -24 -17 -47 31 19 -47 -49 32 43 -36 -16 -43 -37 -46 41 40 -25 45 11 -20 19 -28 -31 24 49 14 13 46 -39 -35 -32 -38 -36 -43 -38 -29 21 -48 15 -34 33 -22 -37 -15 40 -46 -16 18 41 33 -44 -33 44 -41 29 25 13 -26 31 -30 -23 -26 -29 49 38 49 15 43 19 -25 48 -22 -26 -29 45 11 -44 -26 -26 -10 35 -44 -39 -26 -29 33 -34 12 -37 -11 -16 48 23 -20 -45 -40 -15 -43 -33 29 -12 34 22 -34 25 -29 20 -14 -40 -19 16 -14 -20 -29 -33 -37 33 34 -13 -41 29 -25 47 -33 -24 -20 48 25 43 35 39 -25 -14 10 11 -48 47 -30 35 -42 29 -17 -33 -16 -38 46 -40 12 15 -33 -26 44 27 -16 23 -40 15 -17 33 -26 -16 -17 11 -27 31 38 -19 -38 -24 25 25 -36 35 18 20 21 22 34 -30 -43 15 33 -18 22 -14 19 34 -41 39 -25 -38 -40 -15 -38 -23 24 30'/><path fill='url(#g1)' d='M250 250 s 38 25 -11 27 -22 11 -30 -12 23 -33 28 15 14 -26 -47 -49 35 48 -14 30 -28 -11 22 -10 45 -48 25 -49 26 48 -23 -45 36 40 -29 -44 -36 35 -23 -12 -43 11 -18 34 45 43 17 -25 31 -23 -18 -45 22 -33 12 -27 -30 11 -42 28 32 -29 19 -29 -31 -42 -31 -24 31 -18 30 27 24 37 33 11 47 -23 24 18 47 44 14 19 49 38 -41 26 30 11 12 10 -30 28 -46 17 -31 -28 10 -14 43 -32 26 27 34 -31 11 31 26 -26 25 -37 29 -28 -35 13 -25 49 20 -41 -42 39 -26 19 10 -37 46 22 35 -11 35 34 21 -49 45 17 -30 26 21 -48 31 -47 -34 23 -30 31 -16 -49 33 -16 41 24 -13 -26 -36 -43 18 35 37 -14 21 22 46 -29 34 -13 -11 37 29 -36 39 24 -22 38 41 28 48 29 -30 -27 24 -43 -25 -11 15 23 -32 49 19 21 11 41 11 -45 -11 -36 -16 10 -26 -23 -35 43 30 13 -22 32 -34 17 26 -47 -22 -37 11 -40 46 11 -49 24 38 12 -49 -26 21 18 19 -41 22 -19 -19 -33 33 -12 -15 43 41 29'/><path fill='url(#g2)' d='M250 250 s 36 -29 32 -43 -159 -178 -29 33 83 -104 -144 -177 -41 -154 -105 -198 -156 174 -186 -19 123 -88 -181 -75 169 -144 195 -103 -121 -195 -35 -33 19 -126 -74 170 -52 70 -187 70 -108 75 91 -61 -12 -122 192 -75 -193 -15 169 -43 -125 -19 15 -145 63 49 -72 -66 142 -187 -153 108 -13 58 -13 59 99 -175 105 -191 -164 128 -142 50 -49 -167 147 -141 -139 -22 -85 29 20 -37 -39 -178 153 97 -184 -17 -187 -31 -30 -86 14 -35 -50 102 -34 -112 189 -167 -96 -197 -86 -21 -50 115 125 -162 183 -145 27 -44 184 -67 -186 -78 -52 -13 -88 59'/></symbol><g fill='#9cddc8'><rect width='500' height='500' /><use href='#p'/></g></svg>")
    
if __name__ == "__main__":
    print("Starting NFT Minting Agent...")
    main()
