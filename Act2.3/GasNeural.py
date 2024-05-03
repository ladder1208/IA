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

for i in range(i_max):
    signal = (random.random(), random.random())
    if 0.3**2 <= (signal[0] - 0.5)**2 + (signal[1] - 0.5)**2 <= 0.5**2:
        # Calcular las distancias y ordenarlas
        distances = sorted([(euclidean_distance(signal, w), j) for j, w in enumerate(weights)])

        # Actualizar los pesos
        lambda_for_t = lambda_i * (lambda_f / lambda_i) ** ((i + 1) / i_max)
        epsilon_for_t = epsilon_i * (epsilon_f / epsilon_i) ** ((i + 1) / i_max)
        for k, (dist, j) in enumerate(distances):
            w_delta = epsilon_for_t * math.exp(-k / lambda_for_t)
            weights[j] = (
                weights[j][0] + w_delta * (signal[0] - weights[j][0]),
                weights[j][1] + w_delta * (signal[1] - weights[j][1])
            )

    # Visualizar los pesos en diferentes etapas del entrenamiento
    if i == 299 or i == 2499 or i == 39999:
        plt.close()
        circle_out = plt.Circle((0.5, 0.5), 0.5, color='k', fill=False)
        circle_inn = plt.Circle((0.5, 0.5), 0.3, color='k', fill=False)
        fig, ax = plt.subplots(facecolor='lightgray')
        ax.set_title(f"Iteración: {i+1}")
        ax.add_patch(circle_out)
        ax.add_patch(circle_inn)
        ax.scatter(*zip(*weights), color='green', s=5)
        ax.scatter(signal[0], signal[1], color='green', s=10)
        plt.show()
# # Imprimir los pesos finales
# for w in weights:
#     print(w)