import pandas as pd

# Supongamos que has subido un archivo CSV llamado "data.csv"
data = pd.read_csv('/mnt/data/data.csv')

# Aquí podríamos mostrar las primeras filas del archivo para entender su estructura
data.head()
