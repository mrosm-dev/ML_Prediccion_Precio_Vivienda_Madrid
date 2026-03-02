# ML_Prediccion_Precio_Vivienda_Madrid

🏠 Estimación de Precios Inmobiliarios con Machine Learning
📌 Descripción del proyecto

Este proyecto desarrolla un modelo de Machine Learning supervisado para estimar el precio de publicación de viviendas a partir de datos estructurales y geográficos extraídos de anuncios inmobiliarios.

El objetivo es construir un sistema reproducible y preparado para producción que permita:

Estimar precios de mercado.

Detectar posibles anuncios sobrevalorados o infravalorados.

Analizar el impacto de variables estructurales y de ubicación en el precio.

El proyecto sigue un enfoque end-to-end, desde el análisis exploratorio hasta la optimización del modelo final.

🗂 Dataset

Cada fila representa un anuncio inmobiliario e incluye variables como:

Superficie en metros cuadrados

Número de dormitorios

Número de baños

Coordenadas geográficas (latitud, longitud)

Variables estructurales adicionales

Información energética

Distancias a puntos de interés

Algunas columnas presentan estructuras semiestructuradas (diccionarios), que fueron normalizadas antes del modelado.

La variable objetivo (y) es el precio de publicación en euros, extraído de la clave price dentro de la columna precio.

Se eliminaron variables derivadas del precio para evitar data leakage.

🔎 Análisis Exploratorio (EDA)

Durante el EDA se analizaron:

Distribución del precio (cola derecha pronunciada).

Presencia de valores nulos.

Distribución de variables numéricas.

Impacto preliminar de variables categóricas.

Correlaciones con Pearson y Spearman.

Matriz de correlación para evaluar colinealidad.

Se observó:

Relación monótona clara entre superficie y precio.

Incremento estructural del precio con mayor número de baños.

Influencia relevante de la ubicación.

Dado el sesgo del target, se evaluó el uso del logaritmo del precio.

🧠 Feature Engineering

Se aplicaron transformaciones estructurales y se incorporó un componente no supervisado:

Clustering Geográfico

Se aplicó K-Means sobre latitud y longitud para generar una variable cluster que actúa como proxy de zona.

Proceso:

Escalado con StandardScaler.

Selección de k mediante método del codo y Silhouette Score.

Integración como feature categórica dentro del pipeline.

Este enfoque permite capturar patrones espaciales sin introducir fuga de información.

⚙️ Pipeline y Arquitectura

El flujo completo se implementó utilizando:

Pipeline

ColumnTransformer

Transformadores personalizados

Validación cruzada

Se construyeron dos variantes de preprocesamiento:

Sin escalado (para modelos basados en árboles y boosting).

Con escalado (para modelos lineales y KNN).

Esto garantiza:

Reproducibilidad

Ausencia de data leakage

Preparación para producción

📊 Modelos Evaluados

Se entrenaron y compararon:

Decision Tree

Random Forest

XGBoost

LightGBM

CatBoost

KNN

Regresión Lineal

Métrica utilizada

Se utilizó MAPE (Mean Absolute Percentage Error), que mide el error relativo promedio.

Un MAPE de 0.15 implica un error medio del 15%.

🚀 Optimización

Se utilizó Optuna para optimizar hiperparámetros de:

XGBoost

LightGBM

CatBoost

Resultados finales:

XGBoost: 0.1475

LightGBM: 0.1549

CatBoost: 0.1594

XGBoost fue seleccionado como modelo final por:

Mejor MAPE.

Mejor alineación entre valores reales y predichos.

Menor tiempo de entrenamiento.

📈 Resultados Finales

El modelo final alcanza aproximadamente un 14.7% de error porcentual medio sobre el conjunto de validación.

El pipeline completo permite:

Entrenamiento reproducible.

Evaluación consistente.

Integración sencilla en entorno productivo.

🏗 Posibles Mejoras Futuras

Incorporar embeddings geográficos más avanzados.

Evaluar stacking o blending de modelos.

Incorporar información textual de la descripción mediante NLP.

Optimizar tiempos separando preprocesamiento estático.

🧩 Tecnologías Utilizadas

Python

Pandas

NumPy

Scikit-learn

XGBoost

LightGBM

CatBoost

Optuna

Matplotlib / Seaborn

🎯 Conclusión

Este proyecto demuestra un flujo completo de Machine Learning aplicado a un caso real de estimación inmobiliaria, integrando:

EDA estructurado

Feature engineering geográfico

Comparación sistemática de modelos

Optimización automatizada

Arquitectura modular y reproducible

El resultado es un modelo robusto, interpretable y preparado para escalar en un entorno real.
