
import random

const = 4
entrenamiento = []
prueba = []

training_patterns = []
training_outputs = []

patronesPrueba = []
output = []

xt = random.random()

for i in range(1127): 
    eq = const * xt * (1 - xt)
    xt = eq
    if i < 19:
        pass
    elif i < 1023:
        entrenamiento.append(eq)
    else:
        prueba.append(eq)

def process_data(data, patterns, outputs):
    for i in range(len(data) - 3):
        x1, x2, x3, result = data[i:i+4]
        x1sq = x1 ** 2
        x2sq = x2 ** 2
        x3sq = x3 ** 2
        patterns.append([x1, x2, x3, x1sq, x2sq, x3sq])
        outputs.append(result)

process_data(entrenamiento, training_patterns, training_outputs)
process_data(prueba, patronesPrueba, output)
