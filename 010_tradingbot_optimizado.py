#010_tradingbot_optimizado.py
import os
import sys
import ccxt.pro
from pprint import pprint
from dotenv import load_dotenv
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
from backtesting.lib import crossover
import logging
from collections import deque

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

class OptimizedTradingBot:
    def __init__(self):
        self.setup_exchange()
        self.setup_trading_params()
        self.data_cache = deque(maxlen=200)  # Cache circular para datos
        self.last_decision = None
        self.last_price_update = 0
        self.price_cache = {}
        
    def setup_exchange(self):
        """Configurar el exchange una sola vez"""
        # Obtener credenciales
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

        logger.info(f"Configurando exchange - API Key: {api_key[:10]}...")
        
        self.hyperliquid = ccxt.pro.hyperliquid({
            'apiKey': api_key,
            'privateKey': secret,
            'walletAddress': wallet,
            'urls': {
                'api': {
                    'public': url,
                }
            },
            'options': {
                'defaultSlippage': 5,
                'rateLimit': 100,  # Limitar requests
            }
        })
        
    def setup_trading_params(self):
        """Configurar parámetros de trading"""
        self.symbol = 'BTC/USDC:USDC'
        self.MAX_POSITIONS = 1
        self.MAX_POSITION_VALUE = 0.001
        self.INFINITE_TRADING = True
        self.leverage = 50
        self.taker_fee = 0.0350
        self.maker_fee = 0.0100
        self.timeframe = '1h'
        self.amount = 0.01
        self.positions = {}
        
        # Parámetros de optimización
        self.PRICE_UPDATE_INTERVAL = 30  # segundos
        self.DATA_UPDATE_INTERVAL = 300  # 5 minutos para datos OHLCV
        self.DECISION_COOLDOWN = 60  # 1 minuto entre decisiones

    async def initialize(self):
        """Inicializar el bot una sola vez"""
        try:
            self.hyperliquid.set_sandbox_mode(True)
            await self.hyperliquid.load_markets()
            self.market = self.hyperliquid.market(self.symbol)
            logger.info("Bot inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar: {e}")
            raise

    async def get_cached_price_data(self):
        """Obtener datos de precio con cache"""
        current_time = time.time()
        
        if current_time - self.last_price_update < self.PRICE_UPDATE_INTERVAL:
            if self.price_cache:
                return self.price_cache
        
        try:
            ticker = await self.hyperliquid.fetch_ticker(self.symbol)
            self.price_cache = {
                'last': ticker['last'],
                'ask': ticker['ask'],
                'bid': ticker['bid'],
                'timestamp': current_time
            }
            self.last_price_update = current_time
            return self.price_cache
        except Exception as e:
            logger.error(f"Error al obtener precios: {e}")
            return self.price_cache if self.price_cache else None

    async def get_latest_data_optimized(self, limit=50):
        """Obtener datos OHLCV optimizado con cache"""
        try:
            # Usar menos datos para análisis más rápido
            ohlcv = await self.hyperliquid.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            
            if not ohlcv:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Actualizar cache
            self.data_cache.extend(df.tail(10).to_dict('records'))
            
            return df
        except Exception as e:
            logger.error(f"Error al obtener datos OHLCV: {e}")
            return None

    def process_data_fast(self, data):
        """Procesamiento de datos optimizado"""
        if data is None or data.empty:
            return None
            
        # Cálculos vectorizados más rápidos
        try:
            df = data.copy()
            
            # Medias móviles simples (más rápidas que EMA complejas)
            df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            
            # RSI simplificado
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            logger.error(f"Error procesando datos: {e}")
            return data

    def predict_state_fast(self, data):
        """Predicción de estado optimizada"""
        if data is None or data.empty:
            return 'HOLD'
            
        try:
            latest = data.iloc[-1]
            
            # Lógica simple pero efectiva
            if latest['sma_5'] > latest['sma_10'] > latest['sma_20']:
                if latest['rsi'] < 70:  # No sobrecomprado
                    return 'BUY'
            elif latest['sma_5'] < latest['sma_10'] < latest['sma_20']:
                if latest['rsi'] > 30:  # No sobrevendido
                    return 'SELL'
                    
            return 'HOLD'
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return 'HOLD'

    def trading_decision_fast(self, state):
        """Decisión de trading rápida"""
        # Evitar decisiones repetitivas
        if state == self.last_decision:
            return 'HOLD'
            
        decision_map = {
            'BUY': 'buy',
            'SELL': 'sell',
            'HOLD': 'HOLD'
        }
        
        return decision_map.get(state, 'HOLD')

    def get_optimized_params(self, decision, price):
        """Parámetros optimizados para órdenes"""
        stop_loss_pct = 0.02  # 2%
        take_profit_pct = 0.02  # 2%
        
        if decision == 'sell':
            return {
                'stopLoss': {
                    'triggerPrice': price * (1 + stop_loss_pct),
                    'price': price * (1 + stop_loss_pct * 1.1),
                },
                'takeProfit': {
                    'triggerPrice': price * (1 - take_profit_pct),
                    'price': price * (1 - take_profit_pct * 0.9),
                },
            }
        else:  # buy
            return {
                'stopLoss': {
                    'triggerPrice': price * (1 - stop_loss_pct),
                    'price': price * (1 - stop_loss_pct * 1.1),
                },
                'takeProfit': {
                    'triggerPrice': price * (1 + take_profit_pct),
                    'price': price * (1 + take_profit_pct * 0.9),
                },
            }

    async def execute_trade_async(self, order_type, side, amount, price, params):
        """Ejecutar trade de forma asíncrona"""
        try:
            order = await self.hyperliquid.create_order(
                self.symbol, order_type, side, amount, price, params
            )
            logger.info(f"Orden ejecutada: {side} {amount} de {self.symbol}")
            self.last_decision = side.upper()
            return order
        except Exception as e:
            logger.error(f"Error ejecutando orden: {e}")
            return None

    async def run_optimized_loop(self):
        """Loop principal optimizado"""
        logger.info("Iniciando loop optimizado de trading...")
        
        # Variables para controlar actualizaciones
        last_data_update = 0
        last_decision_time = 0
        data = None
        
        while True:
            try:
                current_time = time.time()
                
                # Obtener precios (más frecuente)
                price_data = await self.get_cached_price_data()
                if not price_data:
                    await asyncio.sleep(5)
                    continue
                
                # Actualizar datos OHLCV menos frecuentemente
                if current_time - last_data_update > self.DATA_UPDATE_INTERVAL:
                    data = await self.get_latest_data_optimized(limit=50)
                    last_data_update = current_time
                
                if data is None:
                    await asyncio.sleep(10)
                    continue
                
                # Procesar datos y tomar decisión
                processed_data = self.process_data_fast(data)
                current_state = self.predict_state_fast(processed_data)
                decision = self.trading_decision_fast(current_state)
                
                logger.info(f"Estado: {current_state}, Decisión: {decision}, Precio: {price_data['last']}")
                
                # Ejecutar trade con cooldown
                if (decision != 'HOLD' and 
                    current_time - last_decision_time > self.DECISION_COOLDOWN):
                    
                    # Determinar precio y tipo de orden
                    order_type = 'limit'
                    if decision == 'buy':
                        price = price_data['bid'] * 0.999  # Precio ligeramente mejor
                    else:
                        price = price_data['ask'] * 1.001
                    
                    params = self.get_optimized_params(decision, price)
                    
                    # Ejecutar orden
                    order = await self.execute_trade_async(
                        order_type, decision, self.amount, price, params
                    )
                    
                    if order:
                        last_decision_time = current_time
                        # Esperar un poco después de ejecutar orden
                        await asyncio.sleep(5)
                
                # Sleep adaptativo basado en volatilidad
                sleep_time = 1 if current_state != 'HOLD' else 5
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error en loop principal: {e}")
                await asyncio.sleep(30)

    async def run(self):
        """Método principal para ejecutar el bot"""
        try:
            await self.initialize()
            await self.run_optimized_loop()
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
        except Exception as e:
            logger.error(f"Error crítico: {e}")
        finally:
            try:
                await self.hyperliquid.close()
                logger.info("Conexiones cerradas correctamente")
            except:
                pass

async def main():
    """Función principal"""
    bot = OptimizedTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())