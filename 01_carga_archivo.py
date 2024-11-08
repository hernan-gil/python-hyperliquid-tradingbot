#01. Carga de Archivo

import pandas as pd

# Cargar los datos
df = pd.read_csv('archive/coin_Bitcoin.csv')

# Mostrar las primeras filas del dataframe
print(df.head())

# Resumen estadístico
print(df.describe())

# Comprobar si hay valores nulos
print(df.isnull().sum())

# Visualizar la tendencia de los precios
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Precio de Cierre', color='blue')
plt.title('Precio de Cierre de Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio en USD')
plt.legend()
plt.show()
