
# Le indicamos la ruta donde queremos que se guarden las imagenes
import matplotlib.pyplot as plt
import os
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'proyecto_machine_learning_t11'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'imagenes', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Guardando figura', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

'''
Función cross_val_score
'''

# Importar las bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Configuración de k-NN con 1 vecino
model = KNeighborsClassifier(n_neighbors=1)

iris = load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos original en 2 partes iguales
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

# Ajuste y evaluación de dos modelos
y1_model = model.fit(X1, y1).predict(X2)
y2_model = model.fit(X2, y2).predict(X1)

print(accuracy_score(y1, y1_model), accuracy_score(y2, y2_model))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10)
print('Puntuación en cada fold: ', scores)
print('Puntuación media: ', scores.mean())
print('Desviación típica de las puntuaciones: ', scores.std())

from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
print(scores)

print('Putuación media: ', scores.mean())
print('Desviación típica de las puntuaciones: ', scores.std())

'''
Validación Cruzada Estratificada k-fold
'''

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Matriz de datos
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6], [7, 8],
             [9, 10], [9, 10], [11, 12], [11, 12], [1, 2],
             [3, 4], [5, 6], [7, 8], [4, 5], [5, 6], [7, 8],
             [4, 5]])

# Vector de respuesta
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)

# Print de los índices de los ejemplos en cada fold
for train_index, test_index in skf.split(X, y):
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

'''
Clase GridSearchCV
'''

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))

from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False]}

def make_data(N, err=1.0, rseed=1):
    # Muestreo aleatorio de datos
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)

import matplotlib.pyplot as plt
import seaborn; seaborn.set()

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X,y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
plt.xlim(-0.1, 1.0)
plt.ylim(-1, 12)

save_fig('GridSearchCV')
plt.show()

print(grid.best_params_)

import matplotlib.pyplot as plt

model = grid.best_estimator_

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test);
plt.axis(lim)
save_fig('GridSearchCV_best_estimator')
plt.show()

'''
Clase RandomizedSearchCV
'''

# Importar bibliotecas
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from timeit import default_timer as timer
from sklearn.svm import LinearSVC

def linear_SVC(x, y, param, kfold):
    param_grid = {'C':param}
    k = StratifiedKFold(n_splits=kfold)
    grid = GridSearchCV(LinearSVC(dual=False),
                        param_grid=param_grid,
                        cv=k, n_jobs=4, verbose=1)
    return grid.fit(x, y)

def Linear_SVC_Rand(x, y, param, kfold, n):
    param_grid = {'C':param}
    k = StratifiedKFold(n_splits=kfold)
    randsearch = RandomizedSearchCV(LinearSVC(dual=False),
                                    param_distributions=param_grid,
                                    cv=k, n_jobs=4, verbose=1, n_iter=n)
    return randsearch.fit(x, y)

from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

start = timer()
param = [i/1000 for i in range(1,1000)]
param1 = [i for i in range(1,101)]
param.extend(param1)

clf = Linear_SVC_Rand(x=x_train, y=y_train, param=param, kfold=3, n=100)

print('LinearSVC:')
print('Mejor precisión de CV: {}'.format(clf.best_score_))
print('Puntuación del test: {}'.format(clf.score(x_test, y_test)))
print('Mejores parámetros: {}'.format(clf.best_params_))
print()

duration = timer() - start
print('Tiempo de ejecución de RandomizedSearchCV: {}'.format(duration))

print('-----------')

# Una C alta significa más posibilidades de sobreajuste

start = timer()
param = [i/1000 for i in range(1,1000)]
param1 = [i for i in range(1,101)]
param.extend(param1)

clf = linear_SVC(x=x_train, y=y_train, param=param, kfold=3)

print('LinearSVC:')
print('Mejor precisión de CV: {}'.format(clf.best_score_))
print('Puntuación del test: {}'.format(clf.score(x_test, y_test)))
print('Mejores parámetros: {}'.format(clf.best_params_))
print()

duration = timer() -start
print('Tiempo de ejecución de GridSearchCV: {}'.format(duration))

import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# Obtener datos
digits = load_digits()
X, y = digits.data, digits.target

# Construir un clasificador
clf = RandomForestClassifier(n_estimators=20)

# Función de utilidad para comunicar las mejores puntuaciones
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Modelo con rango: {0}'.format(i))
            print('Puntuación media de validación: {0:.3f} (std: {1:.3f})'.format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print('Parámetros: {0}'.format(results['params'][candidate]))
            print('')

# Especificar los parámetros y las distribuciones de las que se tomarán las muestras
param_dist = {'max_depth': [3, None],
              'max_features': sp_randint(1, 11),
              'min_samples_split': sp_randint(2, 11),
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}

# Ejecutar búsqueda aleatoria
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

start = time()
random_search.fit(X, y)
print('RandomizedSearchCV tardó %.2f segundos para %d configuración de'
      ' parametros de candidatos.' % ((time() -start), n_iter_search))
report(random_search.cv_results_)

# Utilizar una cuadrícula completa para todos los usuarios
param_grid = {'max_depth': [3, None],
              'max_features': [1, 3, 10],
              'min_samples_split': [2, 3, 10],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}

# Ejecutar búsqueda en la cuadrícula
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print('GridSearchCV tardó %.2f segundos para %d configuración de'
      ' parametros de candidatos' 
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


