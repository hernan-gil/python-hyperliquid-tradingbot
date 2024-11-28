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
        
        # Crear la orden de mercado
        order = await hyperliquid.create_market_order(symbol, 'buy', amount)
        print("Orden ejecutada:", order)

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await hyperliquid.close()

# Ejecutar la función principal
asyncio.run(main())