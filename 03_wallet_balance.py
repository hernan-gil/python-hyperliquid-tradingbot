# 03_wallet_balance.py
import os
import ccxt
import ccxt.pro
import asyncio
from pprint import pprint
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Function to get API credentials based on environment variables
def get_credentials():
    if os.getenv('MAINNETOK') == True:
        api_key = os.getenv('API_KEY21')
        secret = os.getenv('SECRET21')
        wallet = os.getenv('WALLET21')
        url = os.getenv("MAINNET")
    else:
        api_key = os.getenv('API_KEYPRUEBA')
        secret = os.getenv('SECRETPRUEBA')
        wallet = os.getenv('WALLETPRUEBA')
        url = os.getenv("TESTNET")
    return api_key, secret, wallet, url

# Asynchronous function to print balance
async def print_balance(exchange, market_type):
    while True:
        try:
            balance = await exchange.fetchBalance({'type': market_type})
            pprint(balance)
            print(f'Balance of {market_type}:', balance)
            print(exchange.options[market_type])
            time.sleep(60)
        except ccxt.BaseError as e:
            print(type(e), e)
        except Exception as e:
            print(type(e), e)

async def main():
    # Get API credentials
    api_key, secret, wallet, url = get_credentials()

    # Initialize the exchange
    hyperliquid = ccxt.pro.hyperliquid({
        'apiKey': api_key,
        'privateKey': secret,
        'walletAddress': wallet,
        'urls': {
            'api': {
                'public': url,  # Change this URL based on actual testnet endpoint
            }
        },
        'options': {
            'defaultSlippage': 5  # Set default slippage (percentage)
        }
    })

    # Schedule tasks to print balance for different markets
    tasks = [
        asyncio.ensure_future(print_balance(hyperliquid, 'future')),
        asyncio.ensure_future(print_balance(hyperliquid, 'delivery')),
        asyncio.ensure_future(print_balance(hyperliquid, 'spot'))
    ]

    # Wait for all tasks to finish
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Run the event loop
    asyncio.run(main())