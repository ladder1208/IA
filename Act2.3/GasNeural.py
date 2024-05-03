import random
import math
import matplotlib.pyplot as plt

# Función auxiliar para calcular la distancia euclidiana
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Inicializar los pesos dentro del anillo
weights = []
while len(weights) < 400:
    w = (random.random(), random.random())
    if 0.3**2 <= (w[0] - 0.5)**2 + (w[1] - 0.5)**2 <= 0.5**2:
        weights.append(w)

# Visualizar los pesos iniciales
circle_out = plt.Circle((0.5, 0.5), 0.5, color='k', fill=False)
circle_inn = plt.Circle((0.5, 0.5), 0.3, color='k', fill=False)
fig, ax = plt.subplots(facecolor='lightgray')
ax.add_patch(circle_out)
ax.add_patch(circle_inn)
ax.scatter(*zip(*weights), color='green', s=5)
ax.set_title("Pesos iniciales")
plt.show()

# Entrenamiento del modelo
lambda_i, lambda_f = 10, 0.01
epsilon_i, epsilon_f = 0.5, 0.05
i_max = 40000


# # Imprimir los pesos finales
# for w in weights:
#     print(w)