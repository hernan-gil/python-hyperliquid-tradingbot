#010_tradingbot_plantilla_01.py
#004_hyperliquid_trading_bot
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
    hyperliquid.set_sandbox_mode(True)
    await hyperliquid.load_markets()
    market = hyperliquid.market(symbol)
    order_type = 'LIMIT'
    current_state = 'SELL'
    amount = 0.001
    side = 'buy'  # 'buy' | 'sell'
    ticker = await hyperliquid.fetch_ticker(symbol)
    last_price = ticker['last']
    ask_price = ticker['ask']
    bid_price = ticker['bid']
    # if order_type is 'market', then price is not needed
    price = None
    # if order_type is 'limit', then set a price at your desired level
    if order_type == 'limit':
        price = bid_price * 0.95 if (side == 'buy') else ask_price * 1.05  # i.e. 5% from current price
    # set trigger price for stop-loss/take-profit to 2% from current price
    # (note, across different exchanges "trigger" price can be also mentioned with different synonyms, like "activation price", "stop price", "conditional price", etc. )
    stop_loss_trigger_price = (last_price if order_type == 'market' else price) * (0.98 if side == 'buy' else 1.02)
    take_profit_trigger_price = (last_price if order_type == 'market' else price) * (1.02 if side == 'buy' else 0.98)
    print(f"Stop Loss Trigger price: {stop_loss_trigger_price} - take profit trigger price: {take_profit_trigger_price}")
    print("Cargando ordenes previas y variables del sistema...")
    timeframe = '1h'
    loop = asyncio.get_event_loop()
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
            processed_data = process_data(data)
           
            # Predecir el estado actual
            current_state = predict_state(processed_data)
            
            # Tomar una decisión de trading
            decision = trading_decision(current_state)
            
            print(f"Estado actual: {current_state}, Decisión: {decision}")
            # Ejecutar la operación (ajusta la cantidad según tu estrategia)
            if decision != 'HOLD':
                await execute_trade(symbol, order_type, decision, amount, price, getparams(decision, price))
            print(f"Consultar de nuevo los params: {params} de decision: {decision}")
            # Esperar antes de la próxima iteración
            time.sleep(3600) #3600 para una hora, para modificar cuando terminen las pruebas.
            
        except Exception as e:
            print(f"Error al realizar una operación con Hyperliquid: {e}")
            time.sleep(60)  # Esperar 1 minuto antes de intentar de nuevo
        finally:
            #loop.close()
            await hyperliquid.close()


# Ejecuta el script principal
asyncio.run(main())
