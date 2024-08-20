
# Le indicamos la ruta donde queremos que se guarden las imagenes
import matplotlib.pyplot as plt
import os
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'proyecto_machine_learning_t13'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'imagenes', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Guardando figura', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


'''
PACF
'''

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

# Generar una serie temporal sintética
np.random.seed(42)
n = 100
time_series = np.cumsum(np.random.normal(size=n))

# Trazar la serie temporal
plt.plot(time_series)
plt.title('Serie Temporal Sintética')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
save_fig('serie temporal sintética')
plt.show()

# Calcular el PACF
lags = 20
pacf_values, confint = pacf(time_series, nlags=lags, alpha=0.05)

# Trazar el PACF
plt.figure(figsize=(10, 4))
plot_pacf(time_series, lags=lags, alpha=0.05)
plt.title('Función de autocorrelación parcial (PACF)')
plt.xlabel('Retraso')
plt.ylabel('Valor PACF')
save_fig('función de autocorrelación parcial (PACF)')
plt.show()

'''
Comprobación de estacionariedad
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generar una serie temporal sintética (no estacionaria)
np.random.seed(42)
n = 100
non_stationary_series = np.cumsum(np.random.normal(size=n))

# Trazar la serie temporal no estacionaria
plt.plot(non_stationary_series)
plt.title('Serie Temporal Sintética (no estacionaria)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
save_fig('serie temporal sintética (no estacionaria)')
plt.show()

# Comprobar visualmente la estacionariedad
time_series = pd.Series(time_series)
rolling_mean = time_series.rolling(window=10).mean()
rolling_std = time_series.rolling(window=10).std()

plt.plot(time_series, label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.legend()
plt.title('Comprobar visualmente la estacionariedad')
save_fig('comprobar visualmente la estacionariedad')
plt.show()

# Convertir serie en DataFrame
df = pd.DataFrame({'value': non_stationary_series})

# Eliminar la tendencia por diferenciación
df['stationary_series'] = df['value'].diff().dropna()

# Representar gráficamente la serie temporal diferenciada
plt.plot(df['stationary_series'])
plt.title('Series temporales diferenciadas (estacionarias)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
save_fig('series temporales diferenciadas (estacionarias)')
plt.show()

