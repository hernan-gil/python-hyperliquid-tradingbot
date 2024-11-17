#015_sniper_bot.py
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

# Función para obtener el precio actual
async def get_current_price(symbol):
    ticker = await hyperliquid.fetch_ticker(symbol)
    return ticker['last']

# Función para ejecutar una orden de compra
async def execute_buy_order(symbol, amount):
    try:
        order = hyperliquid.create_market_buy_order(symbol, amount)
        return order['id']
    except Exception as e:
        print('Error al ejecutar la orden:', e)
        return None

# Función para ejecutar una orden de venta
def execute_sell_order(symbol, amount, order_id):
    try:
        order = hyperliquid.create_market_sell_order(symbol, amount)
        print('Orden de venta ejecutada:', order)
        hyperliquid.cancel_order(order_id)
    except Exception as e:
        print('Error al ejecutar la orden:', e)

# Lógica del bot sniper
async def sniper_bot(symbol, target_price, amount, take_profit, stop_loss):
    buy_order_id = None
    buy_price = None

    while True:
        try:
            current_price = await get_current_price(symbol)
            print(f"Precio actual: {current_price}, Precio objetivo: {target_price}")

            if not buy_order_id:
                if current_price <= target_price:
                    buy_order_id = await execute_buy_order(symbol, amount)
                    buy_price = current_price
                    print(f"Orden de compra ejecutada con ID: {buy_order_id}")
            else:
                profit_percent = ((current_price - buy_price) / buy_price) * 100
                loss_percent = -profit_percent
                print(f"Beneficio/Pérdida: {profit_percent:.2f}%")

                if profit_percent >= take_profit or loss_percent >= stop_loss:
                    execute_sell_order(symbol, amount, buy_order_id)
                    break

            time.sleep(5)
        except Exception as e:
            print(f"Error al realizar una operación con Hyperliquid: {e}")
            time.sleep(60)  # Esperar 1 minuto antes de intentar de nuevo
        finally:
            #loop.close()
            await hyperliquid.close()

# Configuración
# Especifica el símbolo de trading
symbol = 'BTC/USDC:USDC'
target_price = 87950
amount = 0.01
take_profit = 5  # Ganancia objetivo en porcentaje
stop_loss = 3   # Pérdida máxima permitida en porcentaje

# Ejecuta el script principal del sniper bot
asyncio.run(sniper_bot(symbol, target_price, amount, take_profit, stop_loss))
