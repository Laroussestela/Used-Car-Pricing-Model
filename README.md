# 🚗 Vehicle Price Prediction - XGBRegressor vs LGBMRegressor

Este proyecto compara el rendimiento de dos modelos de regresión populares: **XGBRegressor** y **LGBMRegressor** para predecir el precio de vehículos usando un dataset de Kaggle. 

El objetivo es comparar los resultados de ambos modelos para determinar cuál es el más efectivo para predecir el precio de vehículos basándose en atributos como marca, modelo, año, kilometraje, tipo de combustible, entre otros.

## 🧠 Descripción

Este proyecto utiliza **XGBRegressor** (XGBoost) y **LGBMRegressor** (LightGBM), dos algoritmos de aprendizaje automático ampliamente utilizados, para predecir el precio de vehículos. Los resultados obtenidos de ambos modelos se comparan en base a la precisión de la predicción en un conjunto de datos de prueba.

## 📊 Dataset

El dataset utilizado proviene de la competencia [Playground Series - S4E9](https://www.kaggle.com/competitions/playground-series-s4e9/overview) en Kaggle. El conjunto de datos contiene varias características de vehículos, incluyendo:

- `Brand`
- `Model`
- `Model_year`
- `Milage`
- `Fuel_type`
- `Engine`
- `Transmition`
- `External color`
- `Internal color`
- `Accident`
- `Price`

## 🧑‍💻 Modelos

Se han probado dos modelos para la predicción del precio de los vehículos:

1. **XGBRegressor**: Implementado a partir de XGBoost, es un modelo basado en árboles de decisión que usa el enfoque de "gradient boosting".
2. **LGBMRegressor**: Basado en LightGBM, una implementación más eficiente de gradient boosting que es particularmente útil para datasets grandes.

## 📈 Resultados

A continuación se presentan algunos ejemplos de las predicciones realizadas por ambos modelos comparados con los precios reales y las diferencias:

| Precio real | Precio predicho | Diferencia |
|-------------|-----------------|-----------|
| 43000       | 42997.195925    | 2.804075  |
| 53500       | 53503.269418    | 3.269418  |
| 29500       | 29506.324012    | 6.324012  |
| 10750       | 10756.449301    | 6.449301  |
| 20000       | 19993.413099    | 6.586901  |
| 45515       | 45508.268955    | 6.731045  |
| 35000       | 34993.078886    | 6.921114  |
| 30989       | 30979.997198    | 9.002802  |
| 17500       | 17509.447614    | 9.447614  |
| 48000       | 47990.060157    | 9.939843  |

Como se puede observar, las predicciones están bastante cerca de los valores reales, con una diferencia mínima en todos los casos.

## 📊 Gráfica

La siguiente gráfica muestra la comparación entre el **precio real** y el **precio predicho**:

![image](https://github.com/user-attachments/assets/51da8bed-bae6-4955-bba2-3dd3b50c4a9b)

En la gráfica se puede observar la relación entre las predicciones de ambos modelos y los valores reales, destacando la precisión en las predicciones de precios de vehículos.

## 🛠 Requisitos

pip install -r requirements.txt
