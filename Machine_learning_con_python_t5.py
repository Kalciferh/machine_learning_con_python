
# Ruta para guardar las figuras
import os

PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'proyecto_machine_learning_t5'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'imagenes', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Guardando figura', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Un ejemplo sencillo

import matplotlib.pyplot as plt
import numpy as np

# Puntos de muestreo uniforme
t = np.arange(-20., +10., 0.2)

f = lambda x: (x+5)*(x+5)

plt.plot(t, f(t))
save_fig('Gradient Descent')
plt.show()

# Tasa de aprendizaje
rate = 0.01
df = lambda x: 2*(x+5) # Gradiente de nuestra función

def gradient_descent(initial_value: float, learning_rate: float):
    # Esto nos dice cuándo parar el algoritmo
    precision = 0.000001
    previous_step_size = 1
    cur_theta = initial_value
    # Número máximo de iteraciones
    max_iters = 1000
    iters = 0 # Contador de iteraciones
    loss_dict = dict()
    while previous_step_size > precision and iters < max_iters:
        prev_theta = cur_theta # Almacenar valor x actual en prev_theta
        cur_theta = cur_theta - learning_rate * df(prev_theta) # Descenso de Gradiente
        previous_step_size = abs(cur_theta - prev_theta) # Cambio en theta
        iters = iters + 1 # Recuento de iteraciones
        loss_dict[iters] = cur_theta
    return cur_theta, loss_dict

cur_theta, loss_dict = gradient_descent(initial_value=3, learning_rate=rate)
print('El mínimo local se produce en', cur_theta)

# Valores de pérdida del Descenso de Gradiente

import matplotlib.pyplot as plt

# Extraer claves (iteraciones) y valores (pérdidas) del diccionario
iterations = list(loss_dict.keys())
loss_values = list(loss_dict.values())

# Crear un gráfico de líneas
plt.plot(iterations, loss_values, marker='o', linestyle='-')
plt.title('Función de pérdida a lo largo de las iteraciones')
plt.xlabel('Iteración')
plt.ylabel('Valor de la pérdida')
plt.grid(True)

save_fig('Perdida Descenso de Gradiente')
plt.show()

# Generar algunos valores gaussianos

from numpy.random import randn
values = randn(10)

for initial_value in values:
    cur_theta, _ = gradient_descent(initial_value=3, learning_rate=rate)
    print(f'Empezando en {initial_value}, el mínimo local ocurre en {cur_theta}')

import numpy as np
import matplotlib.pyplot as plt

f1 = lambda x: (x ** 3)-(3 *(x ** 2))+7

# Obtén 500 numeros uniformemente espaciados entre -0.5 y 3
# (elegidos aribtrariamente para asegurar una curva pronunciada)
x = np.linspace(-0.5,3,500)

# Graficar la curva
plt.plot(x, f1(x))
save_fig('Gráfica de la función cúbica f(x)=x3-3x2+7')
plt.show()

from scipy import optimize

result = optimize.minimize_scalar(f1)
print(result.success) # Comprueba si el solucionador tuvo éxito

print('El mínimo de la función ocurre en x = %.2f' % result.x)

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

def f_paraboloide(x1, x2):
    return x1 ** 2 + x2 ** 2

x1 = np.linspace(-6, 6, 30)
x2 = np.linspace(-6, 6, 30)

X1, X2 = np.meshgrid(x1, x2)
Y = f_paraboloide(X1, X2)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, Y, 50, cmap='binary')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
save_fig('Paraboloide dos variables')
plt.show()

ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Superficie')
save_fig('Superficie paraboloide dos variables')
plt.show()

from scipy import optimize

def f2(x):
    return (x[0]**2 + x[1]**2)

print(optimize.minimize(f2, [2, -1], method='CG'))

# Ejemplo

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
def f(x1, x2):
    return np.sin(np.sqrt(x1**2 + x2**2))
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X1, X2 = np.meshgrid(x, y)
Y = f(X1, X2)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, Y, 50, cmap='binary')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
save_fig('Ejemplo')
plt.show()

from scipy import optimize

def f2(x):
    return(x[0]**2 + x[1]**2)

print(optimize.minimize(f2, [2, -1], method='CG'))

