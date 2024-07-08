
# Importar las biliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos Pima (especificar ruta exacta del archivo .csv)
# Definir nombres de columnas para el conjunto de datos
column_names = ['Embarazos', 'Glucosa', 'Presión arterial',
                'Grosor de la piel', 'Insulina', 'IMC',
                'DiabetesPedigreeFunction', 'Edad','Resultado']
dataset = pd.read_csv('../data/pima-indians-diabetes.csv',
                      names=column_names, header=0)

print(dataset.hist())
print(dataset.head())
print(dataset.info())

# Dividir el conjunto de datos en características (X) y la variable objetivo (y)
X = dataset.drop('Resultado', axis=1)
y = dataset['Resultado']

# Dividir los datos en un conjunto de entrenamiento y uno de pruebas (por ejemplo, 70% entrenamiento, 30% pruebas)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Escala las funciones con StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de pruebas
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Presición: {accuracy * 100:.2f}%')
print('Matriz de confusión:\n', confusion)
print('Informe de clasificación:\n', classification_rep)

