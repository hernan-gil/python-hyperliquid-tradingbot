import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('archive/btcusd_1-min_data_kaggle.csv')

# Convertir la columna 'TimeStamp' a datetime, especificando el formato si es necesario
#df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')  # Ajusta el formato si es diferente
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')  # Unit can be 's', 'ms', 'us', or 'ns'

# Guardar los cambios en un nuevo archivo CSV (opcional)
df.to_csv('archive/btcusd_1-min_data_kaggle_con_fecha.csv', index=False)