import asyncio
import ccxt.pro
import os
from dotenv import load_dotenv

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
#print("Secret:", secret)
print("Wallet:", wallet)
print("URL   :", url)

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
        'slippage': 5  # Establece el deslizamiento predeterminado (en porcentaje)
    }
})

async def main():
    # Especifica el símbolo de trading
    symbol = 'BTC/USDC:USDC'
    amount = 0.01  # Cantidad que deseas vender/comprar

    try:
        # Conectar al exchange
        await hyperliquid.load_markets()
        
        # Obtener el último precio del mercado
        ticker = await hyperliquid.fetch_ticker(symbol)
        # Imprimir el ticker completo para depuración
        print("Ticker completo:", ticker)
        
        # Asegurarse de que el último precio es un número
        if 'last' not in ticker or ticker['last'] is None:
            raise ValueError("El último precio no se encontró en el ticker")
        
        last_price = ticker['last']
        
        # Asegúrate de que el precio es un número, explícitamente convertir a float
        print(f"Último precio obtenido: {last_price} (tipo: {type(last_price)})")
        
        # Crear la orden de mercado con el precio
        order = await hyperliquid.create_market_buy_order(symbol, amount, ) #create_market_order(symbol, 'buy', amount, {'price': float(last_price)})
        print("Orden ejecutada:", order)

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await hyperliquid.close()
'''
if hyperliquid.has['fetchOrders']:
    since = hyperliquid.milliseconds () - 86400000  # -1 day from now
    # alternatively, fetch from a certain starting datetime
    # since = hyperliquid.parse8601('2018-01-01T00:00:00Z')
    all_orders = []
    while since < hyperliquid.milliseconds ():
        symbol = None  # change for your symbol
        limit = 20  # change for your limit
        orders = await hyperliquid.fetch_orders(symbol, since, limit)
        if len(orders):
            since = orders[len(orders) - 1]['timestamp'] + 1
            all_orders += orders
        else:
            break

import time
if exchange.has['fetchTrades']:
    for symbol in exchange.markets:  # ensure you have called loadMarkets() or load_markets() method.
        print (symbol, exchange.fetch_trades (symbol))
'''

# Ejecutar la función principal
asyncio.run(main())