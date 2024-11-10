import pandas as pd
import talib as ta
import numpy as np

# Supongamos que tienes un DataFrame 'df' con tus datos de OHLCV (Open, High, Low, Close, Volume)
# df = pd.read_csv('tus_datos.csv')

# Parámetros del indicador
fast_length = 12
slow_length = 26
signal_smoothing = 9
k_length = 14
d_smoothing = 3
ema_length = 200

# Cálculo del MACD
df['macd'], df['macd_signal'], _ = ta.MACD(df['Close'], fastperiod=fast_length, slowperiod=slow_length, signalperiod=signal_smoothing)

# Cálculo del Estocástico
df['k'], df['d'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=k_length, slowk_period=d_smoothing, slowd_period=d_smoothing)

# Cálculo de la EMA
df['ema200'] = ta.EMA(df['Close'], timeperiod=ema_length)

# Condiciones de entrada
df['entry'] = ((df['macd'] > df['macd_signal']) & 
               (df['macd'] < 0) & 
               (df['macd_signal'] < 0) & 
               (df['k'] < 35) & 
               (df['Close'] < df['ema200']))

# Condiciones de salida
df['exit'] = ((df['macd'] < df['macd_signal']) & 
              (df['macd'] > 0) & 
              (df['macd_signal'] > 0) & 
              (df['k'] > 65))

# Inicialización de variables
position = 0  # 0 significa no hay posición, 1 significa posición larga
buy_signals = []
sell_signals = []

# Iterar sobre las filas del DataFrame
for index, row in df.iterrows():
    if row['entry'] and position == 0:
        buy_signals.append(index)
        position = 1
    elif row['exit'] and position == 1:
        sell_signals.append(index)
        position = 0

# Añadir las señales de compra y venta al DataFrame
df['buy_signal'] = np.nan
df['sell_signal'] = np.nan
df.loc[buy_signals, 'buy_signal'] = df['Close']
df.loc[sell_signals, 'sell_signal'] = df['Close']

# Mostrar el DataFrame con las señales
print(df[['Close', 'buy_signal', 'sell_signal']].dropna(how='all'))
