import os
import sys
import time
import re
import json
from eth_utils import is_address
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit and Twitter Langchain Extensions
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from twitter_langchain import TwitterApiWrapper, TwitterToolkit

# Configure files to persist data
wallet_data_file = "wallet_data.txt"
last_check_file = "last_check_time.txt"

# NFT Contract configuration
NFT_CONTRACT_ADDRESS = "0x4B9523186371F5a805d2EF882Cf0c6a52120deF8"

# Add at the top with other constants
DEBUG_MODE = False
DUMMY_MENTIONS_FILE = "dummy_mentions.txt"
MENTION_MEMORY_FILE = "mention_memory.txt"

def is_valid_mint_request(tweet_text):
    """Check if tweet contains valid mint request and extract address."""
    pattern = r"Mint nft to (0x[a-fA-F0-9]{40})"
    match = re.search(pattern, tweet_text)
    if match and is_address(match.group(1)):
        return match.group(1)
    return None

def process_mint_request(agent_executor, config, tweet_id, eth_address):
    """Process an NFT mint request."""
    # First, check wallet details and ensure we're on the right network
    wallet_prompt = "Get wallet details to check network."
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=wallet_prompt)]}, config
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].content)

    # Construct mint instruction with explicit reply requirement
    mint_prompt = (
        f"Follow these steps:\n"
        f"1. Mint an NFT from contract {NFT_CONTRACT_ADDRESS} to address {eth_address}\n"
        f"2. After successful minting, use the post_tweet_reply action to reply to tweet ID {tweet_id} "
        f"with the transaction hash and a personalized message to the user.\n"
        f"Make sure to use post_tweet_reply and not post_tweet. "
    )
    
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=mint_prompt)]}, config
    ):
        if "agent" in chunk:
            print(chunk["agent"]["messages"][0].content)
        elif "tools" in chunk:
            print(chunk["tools"]["messages"][0].content)

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
    
    # Combine tools from both toolkits
    tools = cdp_toolkit.get_tools() + twitter_tools

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
    pattern = r"Mint nft to (0x[a-fA-F0-9]+)"  # Changed to match any length hex
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
            f"{greeting}Sorry, the address {address} is not a valid Ethereum address. "
            "Please make sure to provide a valid address starting with 0x.'"
        )
    elif error_type == "zero_balance":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"{greeting}Sorry, the address {address} has 0 ETH balance. "
            "Please make sure to fund the address with some ETH before requesting an NFT mint.' "
            f"Or more humorously like: '{greeting}Why so poor? Get some ETH first.'"
        )
    elif error_type == "already_minted":
        reply_prompt = (
            f"Use post_tweet_reply to reply to tweet {tweet_id} with a message like:\n"
            f"{greeting}You have already received an NFT from us (see tweet {previous_tweet_id}). "
            "This is limited to one NFT per user, don't be greedy! '"
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
            all_tweets = get_all_mentions(account_mentions_tool, account_id, max_results=1, since_id=mention_memory.last_tweet_id)
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

    # For this NFT minting bot, we'll only run in autonomous mode
    run_autonomous_mode(agent_executor=agent_executor, config=config, tools=tools, twitter_wrapper=twitter_wrapper)


if __name__ == "__main__":
    print("Starting NFT Minting Agent...")
    main()
