import os
import sys
import ccxt.pro # Biblioteca para interactuar con exchanges de criptomonedas
from pprint import pprint
from dotenv import load_dotenv
import asyncio
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

# Cargar el modelo HMM y el scaler
model = joblib.load('../trained/hmm_model.pkl')
scaler = joblib.load('../trained/hmm_scaler.pkl')

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave secreta
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

print("API Key:", api_key)
print("Secret:", secret)
print("Wallet:", wallet)
print("URL   :", url)
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')
# Especifica el símbolo de trading
symbol = 'BTC/USDC:USDC'
symbolFromData = 'BTC/USDT'

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

async def get_latest_data(symbol, timeframe, limit):
    """Obtains the latest market data."""
    print(f"Values for calling fetch_ohlcv: Symbol: {symbol} timeframe: {timeframe} limit: {limit}")
    ohlcv = await hyperliquid.fetch_ohlcv(symbol, timeframe, limit=limit)

    # Check the data structure and adjust accordingly
    if isinstance(ohlcv, list):
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    elif isinstance(ohlcv, dict):
        df = pd.DataFrame.from_dict(ohlcv, orient='index', columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    else:
        raise ValueError("Unexpected data format from hyperliquid.fetch_ohlcv()")

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

def process_data(df):
    """Procesa los datos de la misma manera que en el entrenamiento."""
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=24).std()
    df['Volume_Change'] = df['volume'].pct_change()
    
    # Calcular ADX, ATR, Donchian y BB_WIDTH (asumiendo que tienes estas funciones definidas)

    df['ADX'] = calculate_adx(df['high'], df['low'], df['close'], window=14)

    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'], window=14)

    df['Donchian'] = (df['high'].rolling(window=20).max() + df['low'].rolling(window=20).min()) / 2

    df.fillna(value=0, inplace=True)
    
    #quitar inf de Volume_Change
    df['Volume_Change'] = np.where(np.isinf(df['Volume_Change']), np.nan, df['Volume_Change'])

    df = calculate_bollinger_bands(df)
    
    return df.dropna()

def calculate_adx(high, low, close, window=14):
    """
    Calcula el ADX (Average Directional Index).
    """
    # Calculate the directional movement
    up_move = high.diff()
    down_move = low.diff() * -1

    # Directional movement
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Calculate the True Range
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))

    # Smooth the DM and True Range
    smoothed_plus_dm = pd.Series(plus_dm).rolling(window=window).sum()
    smoothed_minus_dm = pd.Series(minus_dm).rolling(window=window).sum()
    smoothed_true_range = pd.Series(true_range).rolling(window=window).sum()

    # Directional indicators
    plus_di = 100 * (smoothed_plus_dm / smoothed_true_range)
    minus_di = 100 * (smoothed_minus_dm / smoothed_true_range)

    # Calculate ADX
    adx = 100 * (pd.Series(np.abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=window).mean())

    return adx


def calculate_atr(high, low, close, window=14):
    """
    Calcula el Average True Range (ATR).
    """
    # Calculate the True Range
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))

    # Calculate ATR
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calcula las Bandas de Bollinger y su ancho.
    
    :param data: DataFrame con una columna 'Close' para los precios de cierre
    :param window: Tamaño de la ventana para la media móvil (por defecto 20)
    :param num_std: Número de desviaciones estándar para las bandas (por defecto 2)
    :return: DataFrame con las columnas BB_Upper, BB_Lower, y BB_WIDTH añadidas
    """
    # Calcular la media móvil
    data['BB_MA'] = data['close'].rolling(window=window).mean()

    # Calcular la desviación estándar
    data['BB_STD'] = data['close'].rolling(window=window).std()
    
    # Calcular las bandas superior e inferior
    data['BB_Upper'] = data['BB_MA'] + (data['BB_STD'] * num_std)
    data['BB_Lower'] = data['BB_MA'] - (data['BB_STD'] * num_std)
    
    # Calcular el ancho de las bandas (normalizado por la media móvil)
    data['BB_WIDTH'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_MA']
    
    return data

def predict_state_new(data):
    """Predice el estado actual del mercado usando el modelo HMM."""

    features = ['Returns', 'Volatility', 'Volume_Change', 'ADX', 'ATR', 'Donchian', 'BB_WIDTH']

    # Verificar si todas las columnas existen
    missing_cols = set(features) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing_cols}")

    # Seleccionar las características y manejar valores faltantes
    X = data[features].copy()
    X.fillna(method='ffill', inplace=True)  # O cualquier otro método de imputación

    # Verificar si hay datos después de la imputación
    if X.empty:
        raise ValueError("No hay datos suficientes para realizar la predicción.")

    # **Asegurarse que el escalador tenga la misma dimensionalidad que X**
    if scaler.mean_.shape[0] != X.shape[1]:
        print("El escalador necesita ser reentrenado con los mismos datos que X.")
        scaler.fit(X)

    # Escalar los datos
    X_scaled = scaler.transform(X)

    # Verificar y manejar valores infinitos o NaN
    X_scaled[np.isinf(X_scaled)] = np.nan
    X_scaled = np.nan_to_num(X_scaled, copy=False)

    # Obtener el estado más reciente
    state = model.predict(X_scaled)[-1]
    return state

def predict_state(data):
    """Predice el estado actual del mercado usando el modelo HMM."""

    features = ['Returns', 'Volatility', 'Volume_Change']

    # Verificar si todas las columnas existen
    missing_cols = set(features) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing_cols}")

    # Seleccionar las características y manejar valores faltantes
    X = data[features].copy()
    X.fillna(method='ffill', inplace=True)  # O cualquier otro método de imputación

    # Verificar si hay datos después de la imputación
    if X.empty:
        raise ValueError("No hay datos suficientes para realizar la predicción.")
    
    # Escalar los datos de forma robusta
    X_scaled = scaler.transform(X)

    # Verificar y manejar valores infinitos o NaN
    X_scaled[np.isinf(X_scaled)] = np.nan
    X_scaled = np.nan_to_num(X_scaled, copy=False)

    # Obtener el estado más reciente
    state = model.predict(X_scaled)[-1]
    return state

def predict_state_old1(data):
    """Predice el estado actual del mercado usando el modelo HMM."""

    features = ['Returns', 'Volatility', 'Volume_Change', 'ADX', 'ATR', 'Donchian', 'BB_WIDTH']

    # Verificar si todas las columnas existen
    missing_cols = set(features) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing_cols}")

    # Seleccionar las características y manejar valores faltantes
    X = data[features].copy()
    X.fillna(method='ffill', inplace=True)  # O cualquier otro método de imputación

    # Verificar si hay datos después de la imputación
    if X.empty:
        raise ValueError("No hay datos suficientes para realizar la predicción.")

    # Escalar los datos
    X_scaled = scaler.transform(X)

    # Obtener el estado más reciente
    state = model.predict(X_scaled)[-1]
    return state

def predict_state_old(data):
    """Predice el estado actual del mercado usando el modelo HMM."""
    features = ['Returns', 'Volatility', 'Volume_Change', 'ADX', 'ATR', 'Donchian', 'BB_WIDTH']
    X = data[features].values
    X_scaled = scaler.transform(X)
    state = model.predict(X_scaled)[-1]  # Obtener el estado más reciente
    return state

def trading_decision(state):
    """Toma una decisión de trading basada en el estado predicho."""
    # Esto es un ejemplo simple. Ajusta según tu estrategia.
    if state in [0, 3, 6]:  # Estados alcistas
        return 'BUY'
    elif state in [1, 4, 5]:  # Estados bajistas
        return 'SELL'
    else:  # Estado neutral
        return 'HOLD'

async def execute_trade(symbol, order_type, side, amount, price, params):
    """Ejecuta una operación de trading."""
    try:
        order = await hyperliquid.create_order(symbol, order_type, side, amount, price, params)
        print(f"Orden ejecutada: {side} {amount} de {symbol}")
        pprint(order)

    except Exception as e:
        print(f"Error al ejecutar la orden: {e}")

async def main():
    print("Iniciando My First Hyperliquid Trading Bot - MFHLTB...")
    hyperliquid.set_sandbox_mode(True)
    await hyperliquid.load_markets()
    market = hyperliquid.market(symbol)
    #hyperliquid.verbose = True  # uncomment for debugging purposes if necessary
    order_type = 'limit'  # 'market' | 'limit'
    side = 'buy'  # 'buy' | 'sell'
    amount = 0.001  # how many contracts (see `market(symbol).contractSize` to find out coin portion per one contract)
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
    position_amount = market['contractSize'] * amount
    position_value = position_amount * last_price
    # log
    print('Going to open a position', 'for', amount, 'contracts worth', position_amount, market['base'], '~', position_value, market['settle'], 'using', side, order_type, 'order (', (hyperliquid.price_to_precision(symbol, price) if order_type == 'limit' else ''), '), using the following params:')
    #print(params)
    print('-----------------------------------------------------------------------')
    timeframe = '1h'
    loop = asyncio.get_event_loop()
    task = asyncio.create_task(get_latest_data(symbol, timeframe, limit=100))
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
                await execute_trade(symbol, order_type, decision, amount, price, params)
            
            # Esperar antes de la próxima iteración
            time.sleep(3600)
            
        except Exception as e:
            print(f"Error al realizar una operación con Hyperliquid: {e}")
            time.sleep(60)  # Esperar 1 minuto antes de intentar de nuevo
        finally:
            #loop.close()
            await hyperliquid.close()

asyncio.run(main())
