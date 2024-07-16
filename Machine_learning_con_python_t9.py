
# Importar las bibliotecas necesarias
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar el conjunto de datos y definir columnas
csv = '../data/pima-indians-diabetes.csv'
column_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(csv, names=column_names)
array = dataframe.values
X = array[:,0:8] # Matriz de datos
y = array[:,8] # Matriz de objetivos

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8) # shuffle=False para que el entrenamiento siempre sea el mismo
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

model = LogisticRegression(solver='lbfgs')
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

# Le indicamos la ruta donde queremos que se guarde la imagen
import os
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'proyecto_machine_learning_t9'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'imagenes', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Guardando figura', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

sns.heatmap(cm, annot=True)
save_fig('Matriz de confusión')
plt.show()

array = [[38,5,0,0,0,0],
         [4,34,0,0,0,0],
         [0,6,42,0,0,0],
         [0,2,0,37,0,8],
         [0,0,0,0,41,13],
         [0,0,0,5,3,39]]
df_cm = pd.DataFrame(array, index = [i for i in 'ABCDEF'],
                     columns = [i for i in 'ABCDEF'])
plt.figure(figsize=(10,7))
sns.heatmap(df_cm, annot=True)
save_fig('Matriz de confusión 6 clases')
plt.show()

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

'''
CLASSIFICATION_REPORT
'''

# Importar las bibliotécas necesarias 'classification_report'
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar el conjunto de datos y definir columnas
csv = '../data/pima-indians-diabetes.csv'
column_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(csv, names=column_names)
array = dataframe.values
X = array[:,0:8] 
y = array[:,8] 

min_max_scaler = MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y,
                                                    train_size=.8, shuffle=False)
lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

'''
AREA BAJO LA CURVA (AUC)
'''

# Importar las bibliotecas necesarias
import warnings
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
csv = '../data/pima-indians-diabetes.csv'
dataframe = pandas.read_csv(csv)
dat = dataframe.values
X = dat[:,:-1]
y = dat[:,-1]
seed = 7
test_size = 0.3

#Dividir datos
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
model.fit(X_train, y_train)

# Predecir probabilidades
probs = model.predict_proba(X_test)

# Mantener la probabilidad solo para el resultado positivo
probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC - Test Set: %.2f%%' % (auc*100))

# Calcular curva ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
# Plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# Trazar la curva ROC para el modelo
plt.plot(fpr, tpr, marker='.')

plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC (Receiver Operating Characteristic)')

save_fig('Curva ROC')
plt.show()

'''
MÉTRICAS DE EVALUACIÓN DE LA REGRESIÓN
'''

# Importar las bibliotecas necesarias
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)

# Cargar el conjunto de datos sobre viviendas
X = california_housing.data
y = california_housing.target

print(X.columns)

california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.subplots_adjust(hspace=0.7, wspace=0.4)
save_fig('California Housing')
plt.show()

interes = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
print(california_housing.frame[interes].describe())

'''
MSE, RMSE, MAE
'''

# Importar las bibliotecas necesarias
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos sobre viviendas
X = california_housing.data
y = california_housing.target

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
model = linear_model.LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones sobre los datos de pruebas
y_pred = model.predict(X_test)

# Evaluar el modelo usando MAE, MSE, RMSE y R2
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Visualizar los resultados
plt.scatter(y_test, y_pred)
plt.xlabel('Precios Actuales')
plt.ylabel('Predicción de Precios')
plt.title('Precios Actuales vs Predicción de Precios')
save_fig('Precios actuales vs predicción de precios')
plt.show()

'''
R2 score
'''

# Importar las bibliotecas necesarias
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el conjunto de datos de diabetes
diabetes = datasets.load_diabetes()

# Para simplificar utilizar solo una función
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_y = diabetes.target

db_X_train, db_X_test, db_y_train, db_y_test = \
train_test_split(diabetes_X, diabetes_y,
                 train_size=.8, random_state=42)

min_max_scaler= MinMaxScaler()
db_X_train = min_max_scaler.fit_transform(db_X_train)
db_X_test = min_max_scaler.transform(db_X_test)

# Crear objeto de regresión lineal
regr = linear_model.LinearRegression()

# Entrenar al modelo usando el conjunto de entrenamiento
regr.fit(db_X_train, db_y_train)

# Hacer predicciones usando el conjunto del test
db_y_pred = regr.predict(db_X_test)

# Coeficientes, MSE y R2
print('Coeficientes:', regr.coef_)
print('MSE: %.2f' % mean_squared_error(db_y_test, db_y_pred))
print('R2: %.2f' % r2_score(db_y_test, db_y_pred))

# Visualizar los resultados
plt.scatter(db_X_test, db_y_test, color='black')
plt.plot(db_X_test, db_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

save_fig('Coeficientes, MSE y R2')
plt.show()

