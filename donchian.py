import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
#pip install yfinance

# Descarga los datos de Bitcoin desde Yahoo Finance
data = yf.download("BTC-USD", start="2020-01-01", end="2023-12-31")

# Calcula los canales de Donchian con un período de 20 días
data['upper_band'], data['middle_band'], data['lower_band'] = ta.donchian(close=data['Close'], length=20)

# Visualización de los canales de Donchian
plt.figure(figsize=(15, 8))
plt.plot(data.index, data['Close'], label='Precio de Cierre')
plt.plot(data.index, data['upper_band'], label='Banda Superior')
plt.plot(data.index, data['middle_band'], label='Banda Media')
plt.plot(data.index, data['lower_band'], label='Banda Inferior')
plt.legend()
plt.title('Canales de Donchian para Bitcoin')
plt.show()

# Ejemplo de estrategia: Comprar cuando el precio cierra por encima de la banda superior
data['buy_signal'] = np.where(data['Close'] > data['upper_band'], 1, 0)


