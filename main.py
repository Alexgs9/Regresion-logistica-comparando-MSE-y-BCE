#Hecho por Alexandro Gutierrez Serna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Carga el conjunto de datos
data = pd.read_csv("Employee.csv")

# Preprocesamiento de datos
# Codificación de variables categóricas con one-hot encoding
data = pd.get_dummies(data, columns=["Education", "City", "Gender", "EverBenched"], drop_first=True)

# Convertir columnas booleanas a enteros (True a 1, False a 0)
boolean_columns = ["Education_Masters", "Education_PHD", "City_New Delhi", "City_Pune", "Gender_Male", "EverBenched_Yes"]
data[boolean_columns] = data[boolean_columns].astype(int)

# Normaliza variables numéricas utilizando StandardScaler
scaler = StandardScaler()
numeric_columns = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

#Para ver todas las columnas de datos
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print(data)

# Divide los datos en conjuntos de entrenamiento y prueba
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Define la función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Inicializa los parámetros (pesos y sesgo)
num_features = len(data.columns) - 1  # Excluye la columna de LeaveOrNot
weights = np.random.randn(num_features)
bias = np.random.rand()

print("Número de características:", num_features)
print("Pesos iniciales:", weights)
print("Sesgo inicial:", bias)

# Define la función de regresión logística
def logistic_regression(features, weights, bias):
    z = np.dot(features, weights) + bias
    return sigmoid(z)


# Define la función de costo Mean Squared Error (MSE)
def mean_squared_error(predictions, labels):
    return ((predictions - labels) ** 2).mean()

# Define la función de costo Binary Cross-Entropy (BCE)
def binary_cross_entropy(predictions, labels):
    epsilon = 1e-15  # Pequeña constante para evitar divisiones por cero
    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # Clip para evitar logaritmos de cero y uno

            # Funcion de costo de entropia cruzada binaria
            #- 1/N sumatoria(i = 1, N) = [y(i) * log(ysombrero(i)) + (1 - y(i)) * log(1 - ysombrero(i))]
    return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

# Entrena el modelo
learning_rate = 0.1
num_iterations = 1000
train_features = data[['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain', 'Education_Masters', 'Education_PHD', 'City_New Delhi', 'City_Pune', 'Gender_Male', 'EverBenched_Yes']].values
train_labels = data["LeaveOrNot"].values

weights_mse = np.random.randn(train_features.shape[1])
bias_mse = np.random.rand()
weights_bce = np.random.randn(train_features.shape[1])
bias_bce = np.random.rand()

    #Predicciones utilizando el descenso del gradiente
for _ in range(num_iterations):
    predictions_mse = logistic_regression(train_features, weights_mse, bias_mse)
    predictions_bce = logistic_regression(train_features, weights_bce, bias_bce)

    dw_mse = (1/len(train_labels)) * np.dot(train_features.T, (predictions_mse - train_labels))
    db_mse = (1/len(train_labels)) * np.sum(predictions_mse - train_labels)
    weights_mse -= learning_rate * dw_mse
    bias_mse -= learning_rate * db_mse

    dw_bce = (1/len(train_labels)) * np.dot(train_features.T, (predictions_bce - train_labels))
    db_bce = (1/len(train_labels)) * np.sum(predictions_bce - train_labels)
    weights_bce -= learning_rate * dw_bce
    bias_bce -= learning_rate * db_bce

# Define una función para hacer predicciones
def predict(features, weights, bias):
    return (logistic_regression(features, weights, bias) >= 0.5).astype(int)

# Evalúa el modelo en el conjunto de prueba y calcula las funciones de costo
test_features = test_data.drop("LeaveOrNot", axis=1).values
test_labels = test_data["LeaveOrNot"].values
predictions_mse = predict(test_features, weights_mse, bias_mse)
predictions_bce = predict(test_features, weights_bce, bias_bce)

mse = mean_squared_error(predictions_mse, test_labels)
bce = binary_cross_entropy(predictions_bce, test_labels)

#El valor mas bajo que me ha dado es 0.26 [100000, 0.001]
print("Mean Squared Error (MSE):", mse)
#El valor mas bajo que me ha dado es 8.8 [100000, 0.001]
print("Binary Cross-Entropy (BCE):", bce)

# Calcula la precisión del modelo
accuracy_bce = accuracy_score(test_labels, predictions_bce)
accuracy_mse = accuracy_score(test_labels, predictions_mse)
print(f"Precisión del modelo BCE: {accuracy_bce}")
print(f"Precisión del modelo MSE: {accuracy_mse}")


#Grafica los valores reales vs los valores predichos error cuadratico medio
plt.figure(figsize=(8, 6))
plt.scatter(range(len(test_labels)), test_labels, label="Valores reales", color="blue", marker="o")
plt.scatter(range(len(predictions_mse)), predictions_mse, label="Valores predichos", color="red", marker="x")
plt.title("Valores reales vs. Valores predichos MSE")
plt.xlabel("Muestras")
plt.ylabel("Valores")
plt.legend()
plt.show()

#Grafica los valores reales vs los valores predichos error cuadratico medio
plt.figure(figsize=(8, 6))
plt.scatter(range(len(test_labels)), test_labels, label="Valores reales", color="blue", marker="o")
plt.scatter(range(len(predictions_bce)), predictions_bce, label="Valores predichos", color="red", marker="x")
plt.title("Valores reales vs. Valores predichos BCE")
plt.xlabel("Muestras")
plt.ylabel("Valores")
plt.legend()
plt.show()
