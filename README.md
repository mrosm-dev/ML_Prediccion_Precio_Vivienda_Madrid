# ML_Prediccion_Precio_Vivienda_Madrid

# 🏠 Estimación de Precios Inmobiliarios con Machine Learning

Proyecto end-to-end de Machine Learning para la **predicción del precio de publicación de viviendas** a partir de datos estructurales y geográficos extraídos de anuncios inmobiliarios.

---

## 🎯 Objetivo

Construir un modelo supervisado de regresión capaz de:

- Estimar precios de mercado.
- Detectar anuncios potencialmente sobrevalorados o infravalorados.
- Analizar el impacto de variables estructurales y de ubicación.
- Implementar una arquitectura reproducible y preparada para producción.

---

## 🗂 Dataset

Cada fila representa un anuncio inmobiliario e incluye:

- 📐 Superficie (m²)  
- 🛏 Dormitorios  
- 🛁 Baños  
- 📍 Latitud y longitud  
- 🏢 Características estructurales  
- ⚡ Información energética  
- 📌 Distancias a puntos de interés  

La variable objetivo (`y`) es el **precio en euros**, extraído de la clave `price`.

Se eliminaron variables derivadas del precio para evitar **data leakage**.

---

## 🔎 Análisis Exploratorio (EDA)

Durante el análisis se evaluaron:

- Distribución del precio (cola derecha pronunciada).
- Valores nulos.
- Distribuciones numéricas.
- Impacto de variables categóricas.
- Correlaciones (Pearson y Spearman).
- Colinealidad entre variables.

Observaciones clave:

- Relación monótona clara entre superficie y precio.
- Incremento estructural del precio con mayor número de baños.
- Influencia significativa de la ubicación.
- Mejora en correlaciones tras aplicar log al target.

---

## 🧠 Feature Engineering

Se incorporaron transformaciones estructurales y un componente no supervisado.

### 📍 Clustering Geográfico

Se aplicó **K-Means** sobre latitud y longitud para generar una variable `cluster` como proxy de zona.

Proceso:

1. Escalado con `StandardScaler`.
2. Selección de `k` mediante método del codo y Silhouette Score.
3. Integración como feature categórica en el pipeline.

Esto permite capturar patrones espaciales sin introducir fuga de información.

---

## ⚙️ Arquitectura

El flujo completo se implementó con:

- `Pipeline`
- `ColumnTransformer`
- Transformadores personalizados
- Validación cruzada

Se construyeron dos variantes de preprocesamiento:

- Sin escalado → modelos basados en árboles y boosting.
- Con escalado → modelos lineales y KNN.

Beneficios:

- Reproducibilidad
- Modularidad
- Preparación para producción
- Sin data leakage

---

## 🤖 Modelos Evaluados

Se entrenaron y compararon:

- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  
- KNN  
- Regresión Lineal  

### 📊 Métrica utilizada

**MAPE (Mean Absolute Percentage Error)**  

---

## 🧩 Tecnologías Utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- LightGBM  
- CatBoost  
- Optuna  
- Matplotlib / Seaborn  

---

## 🧠 Conclusión

Este proyecto demuestra un flujo completo de Machine Learning aplicado a un caso real:

- EDA estructurado  
- Feature engineering geográfico  
- Comparación sistemática de modelos  
- Optimización automatizada  
- Arquitectura modular y reproducible  

El resultado es un modelo robusto, interpretable y preparado para escalar en entorno productivo.

El resultado es un modelo robusto, interpretable y preparado para escalar en un entorno real.
