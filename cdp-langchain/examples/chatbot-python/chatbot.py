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
import tweepy

import cairosvg

# Settings
# ---------

# Configure files to persist data
wallet_data_file = "wallet_data.txt"
last_check_file = "last_check_time.txt"

# NFT Contract configuration
network_id = os.getenv('NETWORK_ID')
NFT_CONTRACT_ADDRESS = "0x692E25F69857ceee22d5fdE61E67De1fcE7EA274" if network_id == "base" else "0x32f75546e56aEC829ce13A9b73d4ebb42bF56b9c"
NFT_PRICE = Decimal("0.001") if network_id == "base" else Decimal("0.0001") # in ETH

# Add at the top with other constants
DEBUG_MODE = True
DUMMY_MENTIONS_FILE = "dummy_mentions.txt"
MENTION_MEMORY_FILE = "mention_memory.txt"

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

# Helper classes
# ---------
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

def get_transaction_data(tx_hash):
    url = f"https://api-sepolia.basescan.org/api?module=proxy&action=eth_getTransactionReceipt&txhash={tx_hash}&apikey={etherscan_api_key}"

    print(f"Getting transaction data for {tx_hash} from etherscan")

    response = requests.get(url)
    data = response.json()

    last_log = data['result']['logs'][-1]
    int_value = int(last_log['topics'][3], 16)
    contract_address = last_log['address']

    return int_value, contract_address

# Helper functions
# ---------

def save_svg_to_png(file_number, svg_content) -> str:
    """
    Saves the given SVG content as a PNG file.

    Parameters:
        file_number (int): The identifier number for the file name.
        svg_content (str): The SVG content as a string.
    """
    file_name = f"output_{file_number}.png"
    cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=file_name)
    print(f"SVG saved as PNG: {file_name}")
    return file_name

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
        mint_invocation = wallet.invoke_contract(
            contract_address=NFT_CONTRACT_ADDRESS,
            abi=abi,
            method="mintAndTransfer",
            args={"recipient": recipient_address},
            amount=NFT_PRICE,
            asset_id="eth",
        ).wait()
        
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

def process_mint_request(agent_executor, wallet: Wallet, config, tweet_id, eth_address, twitter_wrapper, author=None):
    """Process an NFT mint request."""
    try:
        print(f"Starting mint process for {eth_address}...")
        
        # Mint NFT and get response
        mint_response = mint_myNft(wallet, eth_address)
        print(f"Mint response: {mint_response}")
        
        # Check if there was an error in minting
        if "Error" in mint_response:
            raise Exception(mint_response)

        # Get txHash from mint response
        txHash = re.search(r'Transaction hash: (\w+)', mint_response)
        if not txHash:
            raise ValueError("Could not find transaction hash in mint response")
        txHash = txHash.group(1)
        print(f"Transaction hash: {txHash}")

        # Get transaction link - update pattern to capture full URL
        txLink = re.search(r'Transaction: (https://[^\s\n]+)', mint_response)
        if not txLink:
            raise ValueError("Could not find transaction link in mint response")
        txLink = txLink.group(1)
        print(f"Transaction link: {txLink}")

        # Get transaction info
        token_id, nft_mint_address = get_transaction_data(txHash)
        print(f"Token ID: {token_id}, NFT Mint Address: {nft_mint_address}")

        # Get token URI, name and SVG after minting
        token_uri, name, svg_data = get_token_uri_and_svg(wallet, nft_mint_address, token_id)
        print(f"Minted NFT: {name}")

        if not svg_data:
            raise ValueError("Could not get SVG data for NFT")
                
        # Get Twitter API wrapper from tools list
        twitter_client = twitter_wrapper.v1_api

        # Upload media to Twitter
        png_file = save_svg_to_png(token_id, svg_data)
        if not os.path.exists(png_file):
            raise ValueError(f"PNG file {png_file} not found")
            
        media = twitter_client.media_upload(png_file)
        if not media:
            raise ValueError("Failed to upload media to Twitter")
            
        media_id = media.media_id_string
        print(f"Uploaded media to Twitter, ID: {media_id}")

        # Post reply with media
        greeting = f"@{author}! " if author else ""
        reply_prompt = (
            f"Use post_tweet_reply and attach the media ID: {media_id} to reply to tweet {tweet_id} with a personalized message about the successful mint such as:\n"
            f"{greeting} successfully minted {name}! Have fun with your fully onchain art on @base! Visit https://xonin.vercel.app/ to learn more about the project! Here's the transaction link: {txLink}.'"
            f"Be creative in conveying this message!"
        )

        print("Sending reply tweet...")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=reply_prompt)]}, config
        ):
            if "tools" in chunk:
                response = chunk["tools"]["messages"][0].content
                print(f"Reply response: {response}")

        return True, txHash

    except Exception as e:
        error_msg = f"Error processing mint request: {e}"
        print(error_msg)
        return False, None


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
            f"'{greeting}You have already minted an NFT. "
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
       

    # Address is valid and has balance -> mint nft
    print(f"Processing mint request for address: {address}")
    
    try:
        mint_success, tx_hash = process_mint_request(agent_executor, wallet, config, tweet_id, address, twitter_wrapper, author)
        mention_memory.add_mention(
            tweet_id,
            tweet_text,
            "processed",
            mint_success=mint_success,
            tx_hash=tx_hash,
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
            all_tweets = get_all_mentions(account_mentions_tool, account_id, max_results=10, since_id=mention_memory.last_tweet_id)
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
            time.sleep(interval)

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
    #save_svg_to_png(14, "<svg width='500' height='500' viewBox='0 0 500 500' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'><symbol id='c' viewBox='0 0 100 100'><circle cx='50' cy='50' r='25'></circle></symbol><symbol id='t' viewBox='0 0 100 100'><polygon points='0,100 100,100 50,0'></polygon></symbol><symbol id='b' viewBox='0 0 100 100'><rect width='500' height='50'></rect></symbol><filter id='f1' width='200%' height='200%'><feOffset in='SourceGraphic' result='r' dx='50' dy='20' /><feGaussianBlur in='r' result='rb' stdDeviation='0'/><feMerge><feMergeNode in='rb' /><feMergeNode in='SourceGraphic' /></feMerge></filter><symbol id='s'><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#aee239' fill-opacity='0.75' transform='translate(146 480)  scale(3.93)  rotate(56)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#8fbe00' fill-opacity='0.89' transform='translate(107 490)  scale(3.36)  rotate(291)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#aee239' fill-opacity='0.82' transform='translate(136 396)  scale(1.75)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#8fbe00' fill-opacity='0.79' transform='translate(393 342)  scale(3.82)  rotate(213)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#f9f2e7' fill-opacity='0.94' transform='translate(337 68)  scale(0.22)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#8fbe00' fill-opacity='0.91' transform='translate(41 264)  scale(1.20)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#8fbe00' fill-opacity='0.79' transform='translate(128 281)  scale(2.72)  rotate(306)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#8fbe00' fill-opacity='0.87' transform='translate(71 410)  scale(2.23)  rotate(232)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#8fbe00' fill-opacity='0.76' transform='translate(385 446)  scale(1.59)  rotate(246)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#8fbe00' fill-opacity='0.85' transform='translate(490 499)  scale(4.77)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#40c0cb' fill-opacity='0.66' transform='translate(350 202)  scale(2.63)  rotate(303)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#8fbe00' fill-opacity='0.93' transform='translate(341 78)  scale(2.32)  rotate(288)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#aee239' fill-opacity='0.69' transform='translate(196 46)  scale(0.1)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#f9f2e7' fill-opacity='0.87' transform='translate(154 314)  scale(1.68)  rotate(199)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#f9f2e7' fill-opacity='0.60' transform='translate(180 112)  scale(1.26)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#aee239' fill-opacity='0.84' transform='translate(337 232)  scale(0.49)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#f9f2e7' fill-opacity='0.79' transform='translate(454 75)  scale(1.56)  rotate(141)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#8fbe00' fill-opacity='0.78' transform='translate(47 158)  scale(4.5)  rotate(58)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#8fbe00' fill-opacity='0.97' transform='translate(145 447)  scale(2.77)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#aee239' fill-opacity='0.75' transform='translate(132 1)  scale(4.8)  rotate(92)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#8fbe00' fill-opacity='0.93' transform='translate(433 483)  scale(0.60)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#aee239' fill-opacity='0.69' transform='translate(362 205)  scale(3.90)  rotate(12)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#f9f2e7' fill-opacity='0.83' transform='translate(459 285)  scale(0.12)  rotate(22)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#f9f2e7' fill-opacity='0.72' transform='translate(25 80)  scale(2.21)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#aee239' fill-opacity='0.73' transform='translate(424 129)  scale(3.62)  rotate(100)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#8fbe00' fill-opacity='0.67' transform='translate(217 396)  scale(1.21)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#8fbe00' fill-opacity='0.98' transform='translate(133 478)  scale(3.29)  rotate(114)' > </use></g><g class='g2'><use xlink:href='#t' width='100' height='100'  fill='#aee239' fill-opacity='0.70' transform='translate(426 402)  scale(2.16)  rotate(137)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#8fbe00' fill-opacity='0.68' transform='translate(434 291)  scale(2.60)' > </use></g><g class='g1'><use xlink:href='#b' width='100' height='100'  fill='#40c0cb' fill-opacity='0.72' transform='translate(290 278)  scale(2.71)  rotate(248)' > </use></g><g class='g3'><use xlink:href='#c' width='100' height='100'  fill='#40c0cb' fill-opacity='0.68' transform='translate(467 482)  scale(1.92)' > </use></g></symbol><g fill='#00a8c6'><rect width='500' height='500' /><use href='#s' filter='url(#f1)'/></g></svg>")

if __name__ == "__main__":
    print("Starting NFT Minting Agent...")
    main()
