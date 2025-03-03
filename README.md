# XoninBot

A X (Twitter) AI agent ([@XoninNFT](https://twitter.com/XoninNFT)) that analyzes user blockchain activity 
and awards unique generative NFTs based on their ([onchain reputation score](https://docs.cdp.coinbase.com/reputation/docs/welcome)).
This project connects to [Xonin](https://xonin.vercel.app/), a fully onchain generative art project ([GitHub](https://github.com/phdargen/onChainArt)),
and is built using [CDP AgentKit](https://github.com/coinbase/agentkit).

## Overview

XoninBot is an autonomous AI agent with the following features:

- **Twitter Integration**: Automatically monitors mentions and processes NFT mint requests
- **On-chain Analysis**: Evaluates wallet addresses to determine reputation score
- **Generative NFT**: Mints unique SVG-based NFTs
- **Address Resolution**: Support for ENS domains and direct Ethereum addresses
- **Autonomous Operation**: Fully automated workflow from Twitter interaction to NFT minting, with response messages containing transaction details and NFT preview images

The agents runs on a daily schedule via GitHub Actions (limited by X free tier API calls) and can also be triggered manually.

## NFT examples

<div align="center">
  <img src="examples/output_0x15077415012b6f5a6f2842928886b51e0e2cb2d6_43.png" width="250" alt="XoninNFT Example 1" />
  <img src="examples/output_0x15077415012b6f5a6f2842928886b51e0e2cb2d6_46.png" width="250" alt="XoninNFT Example 2" />
  <img src="examples/output_0xd58b1248d893f6dc0f93d7c1a12deed75bee3785_22.png" width="250" alt="XoninNFT Example 3" />
  <img src="examples/output_0xd58b1248d893f6dc0f93d7c1a12deed75bee3785_33.png" width="250" alt="XoninNFT Example 4" />
</div>

## Architecture

XoninBot uses:
- **LangChain**: For building the AI agent capabilities
- **CDP AgentKit**: Coinbase Developer Platform for blockchain interactions
- **Tweepy**: For X API integration
- **Pydantic**: For data validation
- **Poetry**: For dependency management

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   cd agentkit/cdp-langchain/examples/chatbot-python
   poetry install
   ```

3. Set required environment variables:
   ```
   NETWORK_ID=<blockchain_network>
   ETHERSCAN_API_KEY=<your_api_key>
   TWITTER_API_KEY=<your_api_key>
   TWITTER_API_SECRET=<your_api_secret>
   TWITTER_ACCESS_TOKEN=<your_access_token>
   TWITTER_ACCESS_TOKEN_SECRET=<your_access_token_secret>
   CDP_WALLET_DATA=<your_wallet_data>
   OPENAI_API_KEY=<your_openai_api_key>
   ```

## Usage

### Automated Mode
The bot runs automatically via GitHub Actions every day at 12:00 UTC.

### Manual Operation
You can run the bot manually with:
```bash
poetry install
cd agentkit/cdp-langchain/examples/chatbot-python
poetry run python chatbot.py
```

The script provides two modes:
1. Chat mode - for interactive testing
2. Autonomous mode - for monitoring Twitter mentions and processing requests

## Related Projects

- [Xonin Generative Art](https://xonin.vercel.app/) - View the generative art NFTs

## License

This project is based on CPD AgentKit, which is licensed under the Apache-2.0 license. 