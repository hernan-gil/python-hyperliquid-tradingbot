#010_tradingbot_plantilla_01.py
import os
import sys
import ccxt.pro  # Biblioteca para interactuar con exchanges de criptomonedas
from pprint import pprint
from dotenv import load_dotenv
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
from backtesting.lib import crossover

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave secreta
if os.getenv('MAINNETOK') == "True":
    api_key = os.getenv('API_KEY21')
    secret = os.getenv('SECRET21')
    wallet = os.getenv('WALLET21')
    url = os.getenv("MAINNET")
else:
    api_key = os.getenv('API_KEYPRUEBA')
    secret = os.getenv('SECRETPRUEBA')
    wallet = os.getenv('WALLETPRUEBA')
    url = os.getenv("TESTNET")

print("API Key:", api_key)
print("Secret:", secret)
print("Wallet:", wallet)
print("URL   :", url)

# Especifica el símbolo de trading
symbol = 'BTC/USDC:USDC'

# Inicializar el exchange
hyperliquid = ccxt.pro.hyperliquid({
    'apiKey': api_key,
    'privateKey': secret,
    'walletAddress': wallet,
    'urls': {
        'api': {
            'public': url,  # Cambia esta URL según el endpoint de testnet real
        }
    },
    'options': {
        'defaultSlippage': 5  # Establece el deslizamiento predeterminado (en porcentaje)
    }
})

# Definir los límites
MAX_POSITIONS = 1
MAX_POSITION_VALUE = 0.001  # Este valor se usa para representar el valor de la posición máxima
INFINITE_TRADING = True
account_value = 0.0  # Este debería ser obtenido desde Hyperliquid dinámicamente
account_available = 0.0
leverage = 50
taker_fee = 0.0350
maker_fee = 0.0100

positions = {}  # Diccionario para almacenar información de las posiciones

async def get_latest_data(symbol, timeframe, limit):
    """Obtiene los últimos datos del mercado."""
    print(f"Llamando a fetch_ohlcv: Símbolo: {symbol}, timeframe: {timeframe}, limit: {limit}")
    ohlcv = await hyperliquid.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

def trading_decision(state):
    """Toma una decisión de trading basada en el estado predicho."""
    # Esto es un ejemplo simple. Ajusta según tu estrategia.
    if state in [0, 3, 6]:  # Estados alcistas
        return 'BUY'
    elif state in [1, 4, 5]:  # Estados bajistas
        return 'SELL'
    else:  # Estado neutral
        return 'HOLD'

def getparams(decision, price):
    if decision == 'SELL':
        params = {
            'stopLoss': {
                'triggerPrice': price,
                'price': price * 1.02,
            },
            'takeProfit': {
                'triggerPrice': price,
                'price': price * 0.98,
            },
        }
    else:
        params = {
            'stopLoss': {
                'triggerPrice': price,
                'price': price * 0.98,
            },
            'takeProfit': {
                'triggerPrice': price,
                'price': price * 1.02,
            },
        }
    return params

def process_data(data):
    #aqui se procesan los datos para emitir las señales para el bot
    df = data
    return df

def predict_state(data):
    #aqui se crea el state y se devuelve para que se cree el trade con la decision al llamar a execute_trade
    state = 'HOLD'
    return state

async def execute_trade(symbol, order_type, side, amount, price, params):
    """Ejecuta una operación de trading."""
    try:
        order = await hyperliquid.create_order(symbol, order_type, side, amount, price, params)
        print(f"Orden ejecutada: {side} {amount} de {symbol}")
        pprint(order)
        time.sleep(15)
    except Exception as e:
        print(f"Error al ejecutar la orden: {e}")

async def main():
    print("Iniciando la gestión del riesgo y el trading...")
    hyperliquid.set_sandbox_mode(True)  # Set sandbox mode for testing (recommended)
    await hyperliquid.load_markets()
    market = hyperliquid.market(symbol)

    # Define SMA Cross strategy parameters
    fast_window = 13
    slow_window = 32
    order_type = 'LIMIT'
    current_state = 'SELL'
    amount = 0.001
    side = 'buy'  # 'buy' | 'sell'
    ticker = await hyperliquid.fetch_ticker(symbol)
    # if order_type is 'market', then price is not needed
    price = ticker['last']
    # if order_type is 'limit', then set a price at your desired level
    if order_type == 'limit':
        price = bid_price * 0.95 if (side == 'buy') else ask_price * 1.05  # i.e. 5% from current price
    # set trigger price for stop-loss/take-profit to 2% from current price
    # (note, across different exchanges "trigger" price can be also mentioned with different synonyms, like "activation price", "stop price", "conditional price", etc. )
    stop_loss_trigger_price = (last_price if order_type == 'market' else price) * (0.98 if side == 'buy' else 1.02)
    take_profit_trigger_price = (last_price if order_type == 'market' else price) * (1.02 if side == 'buy' else 0.98)
    timeframe = '1d'
    task = asyncio.create_task(get_latest_data(symbol, timeframe, limit=100))
    # when symbol's price reaches your predefined "trigger price", stop-loss order would be activated as a "market order". but if you want it to be activated as a "limit order", then set a 'price' parameter for it
    params = {
        'stopLoss': {
            'triggerPrice': stop_loss_trigger_price,
            'price': stop_loss_trigger_price * 0.98,
        },
        'takeProfit': {
            'triggerPrice': take_profit_trigger_price,
            'price': take_profit_trigger_price * 0.98,
        },
    }
    while True:
        try:
            data = await task
           
            # Fetch latest data
            data = await get_latest_data(symbol, timeframe, limit=100)

            # Tomar una decisión de trading
            decision = ''

            # Calculate SMA indicators
            sma_fast = data['close'].rolling(window=fast_window).mean()
            sma_slow = data['close'].rolling(window=slow_window).mean()
            print(f"140. SMA lento: {sma_slow}  -  SMA Rapido: {sma_fast}")
            # Define trading decision based on SMA Cross
            current_state = 'HOLD'
            print(f"143. SMA lento: {sma_slow.iloc[-1]}  -  SMA Rapido: {sma_fast.iloc[-1]}")
            print(f"144. Rap sobre Len: {crossover(sma_fast.iloc[-1], sma_slow.iloc[-1])}")
            print(f"145. Len sobre Rap: {crossover(sma_slow.iloc[-1], sma_fast.iloc[-1])}")

            if sma_fast.iloc[-1] > sma_slow.iloc[-1] and current_state != 'BUY':
                print("Entro a current_state -> BUY")
                current_state = 'BUY'
            '''
            elif sma_slow.iloc[-1] > sma_fast.iloc[-1] and current_state != 'SELL':
                print("Entro a current_state -> SELL")
                current_state = 'SELL'
            '''

            decision = current_state
            print(f"Estado actual (SMA): {current_state}")

            # Execute the trade (adjust the amount according to your strategy)
            print(f"167. {symbol} - {order_type} - {side} - {amount} - {price} - {params}")
            if decision != 'HOLD':
                await execute_trade(symbol, order_type, current_state, amount, price, getparams(current_state, price))

            time.sleep(3600)  # Wait 1 hour before next iteration (adjust as needed)
        except Exception as e:
            print(f"Error al realizar una operación con Hyperliquid: {e}")
            time.sleep(60)  # Wait 1 minute before retrying
        finally:
            # loop.close()  # Uncomment this if needed
            await hyperliquid.close()

# Ejecuta el script principal
asyncio.run(main())
