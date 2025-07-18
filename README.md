# Forecasting the 2Y-10Y Interest Rate Spread: A Time Series Analysis with Market Indicators

## **Project Description**

This project focuses on the analysis and forecasting of the **2Y-10Y interest rate differential** (spread) time series, a key indicator of market expectations. The central objective is to model and predict the future behavior of this spread, integrating the VIX volatility index and the S&P 500 index as fundamental exogenous variables. The integration of these financial indicators aims to significantly enhance the predictive capability of the model, demonstrating the application of advanced time series analysis techniques in a financial context.

The initial approach includes data cleaning and visualization, stationarity evaluation, identification of the most suitable classic forecasting models, and validation of their performance.

## **Motivation**

The interest rate differential is often considered a key indicator of economic health and market expectations. The ability to predict its behavior is of great value for financial decision-making. This project stems from an interest in understanding the interconnections between the spread and other market variables, such as volatility (VIX) and overall market performance (S&P 500), and how they can be leveraged to generate more robust spread forecasts.

## **Methodology**

1.  **Exploratory Data Analysis (EDA):**
    * Visualization of time series to identify trends, seasonality, volatility, and breakpoints.
    * Analysis of cross-correlations between the spread and the VIX and S&P 500 variables.

2.  **Data Preparation:**
    * Handling missing values and transformations to stabilize variance (e.g., logarithms).
    * Evaluation of stationarity for the spread and exogenous variables using statistical tests (e.g., Augmented Dickey-Fuller) and differentiation.

3.  **Modeling and Forecasting (Classic Models):**
    * Implementation of **ARIMA** (AutoRegressive Integrated Moving Average) models for the spread.
    * Extension to **SARIMA** (Seasonal ARIMA) and, crucially, **SARIMAX** for the spread, incorporating VIX and S&P 500 as exogenous regressors to improve predictive capability.
    * Adjustment and optimization of model hyperparameters.

4.  **Validation and Evaluation:**
    * Splitting the dataset into training and validation sets to evaluate the predictive performance of the spread.
    * Calculation of common error metrics such as **RMSE** (Root Mean Square Error), **MAE** (Mean Absolute Error), and **MAPE** (Mean Absolute Percentage Error) for the spread forecast.
    * Analysis of model residuals to detect autocorrelation, heteroscedasticity, and other deviations from the white noise assumption.

## **Repository Structure**

├── GS10_GS2_Daily.csv

├── README.md

├── SARIMA.ipynb

├── SP500_diario_2000-2025.csv

├── VIX_diario_2000-2025.csv

├── WTI_diario_2000-2025.csv

## **Key Results**

* Non-stationarity of the spread and exogenous series was confirmed, with appropriate differentiations applied for modeling.
* The **SARIMAX** model, by incorporating VIX and S&P 500 as exogenous variables for spread forecasting, showed promising performance, although with variable results depending on the prediction horizon.
* Residual analysis of the spread model revealed the presence of heavy tails and heteroscedasticity, suggesting the need for more advanced models (such as **GARCH** or Machine Learning models) to capture the complexity of this type of financial series.

## **Technologies Used**

* **Python:** Main programming language.
* **Pandas & NumPy:** Data manipulation and analysis.
* **Matplotlib & Seaborn:** Data visualization.
* **Statsmodels:** Implementation of ARIMA and SARIMAX models.
* **Jupyter Notebook:** Development environment.

## **Next Steps**

* Explore **GARCH** models to model the volatility of the spread.
* Implement Machine Learning techniques such as **Random Forest**, **XGBoost**, or **LightGBM** for spread forecasting.
* Test recurrent neural network architectures (**RNN**, **LSTM**, **GRU**) to capture long-term dependencies and non-linearity in the spread series.
* Conduct more rigorous backtesting to evaluate model performance across different time periods.

## **How to Run the Project**

1.  Clone this repository:
    ```bash
    git clone [https://github.com/HGalletti/Forecasting-Spread-Treasury-Bonds.git](https://github.com/HGalletti/Forecasting-Spread-Treasury-Bonds.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open and run the notebooks in order to replicate the analysis.

----------------------------------------------------------------------------------------------------------------

# Pronóstico del Spread de Tasas de Interés 2Y-10Y: Un Análisis de Series de Tiempo con Indicadores de Mercado

## **Descripción del Proyecto**

Este proyecto se enfoca en el análisis y pronóstico de la serie de tiempo del **diferencial de tasas de interés 2Y-10Y** (spread), un indicador clave de las expectativas del mercado. El objetivo central es modelar y predecir el comportamiento futuro de este spread, integrando el índice de volatilidad VIX y el índice S&P 500 como variables exógenas fundamentales. La integración de estos indicadores financieros busca enriquecer significativamente la capacidad predictiva del modelo, demostrando la aplicación de técnicas avanzadas de análisis de series de tiempo en un contexto financiero.

El enfoque inicial incluye la limpieza y visualización de los datos, la evaluación de la estacionariedad, la identificación de los modelos de pronóstico clásicos más adecuados (ARIMA/SARIMA/SARIMAX) y la validación de su rendimiento.

## **Motivación**

El diferencial de tasas de interés a menudo se considera un indicador clave de la salud económica y las expectativas del mercado. La capacidad de predecir su comportamiento es de gran valor para la toma de decisiones financieras. Este proyecto surge del interés en comprender las interconexiones entre el spread y otras variables de mercado, como la volatilidad (VIX) y el rendimiento general del mercado (S&P 500), y cómo pueden ser aprovechadas para generar pronósticos más robustos del spread.

## **Metodología**

1.  **Análisis Exploratorio de Datos (EDA):**
    * Visualización de las series de tiempo para identificar tendencias, estacionalidad, volatilidad y puntos de quiebre.
    * Análisis de correlaciones cruzadas entre el spread y las variables VIX y S&P 500.

2.  **Preparación de los Datos:**
    * Manejo de valores faltantes y transformaciones para estabilizar la varianza (e.g., logaritmos).
    * Evaluación de la estacionariedad del spread y las variables exógenas utilizando pruebas estadísticas (e.g., Dickey-Fuller Aumentada) y diferenciación.

3.  **Modelado y Pronóstico (Modelos Clásicos):**
    * Implementación de modelos **ARIMA** (AutoRegressive Integrated Moving Average) para el spread.
    * Extensión a modelos **SARIMA** (Seasonal ARIMA) y, crucialmente, **SARIMAX** para el spread, incorporando el VIX y el S&P 500 como regresores externos para mejorar la capacidad predictiva.
    * Ajuste y optimización de hiperparámetros de los modelos.

4.  **Validación y Evaluación:**
    * División del conjunto de datos en entrenamiento y validación para evaluar el rendimiento predictivo del spread.
    * Cálculo de métricas de error comunes como **RMSE** (Root Mean Square Error), **MAE** (Mean Absolute Error) y **MAPE** (Mean Absolute Percentage Error) para el pronóstico del spread.
    * Análisis de los residuos del modelo para detectar autocorrelación, heterocedasticidad y otras desviaciones del supuesto de ruido blanco.

## **Estructura del Repositorio**

├── GS10_GS2_Daily.csv

├── README.md

├── SARIMA.ipynb

├── SP500_diario_2000-2025.csv

├── VIX_diario_2000-2025.csv

├── WTI_diario_2000-2025.csv

## **Resultados Clave**

* Se confirmó la no estacionariedad del spread y de las series exógenas, aplicando las diferenciaciones adecuadas para su modelado.
* El modelo **SARIMAX**, al incorporar el VIX y el S&P 500 como variables exógenas para el pronóstico del spread, mostró un rendimiento prometedor, aunque con resultados variables según el horizonte de predicción.
* El análisis de residuos del modelo del spread reveló la presencia de colas pesadas y heterocedasticidad, lo que sugiere la necesidad de modelos más avanzados (como **GARCH** o modelos de Machine Learning) para capturar la complejidad de este tipo de series financieras.

## **Tecnologías Utilizadas**

* **Python:** Lenguaje principal de programación.
* **Pandas & NumPy:** Manipulación y análisis de datos.
* **Matplotlib & Seaborn:** Visualización de datos.
* **Statsmodels:** Implementación de modelos ARIMA y SARIMAX.
* **Jupyter Notebook:** Entorno de desarrollo.

## **Próximos Pasos**

* Explorar modelos **GARCH** para modelar la volatilidad del spread.
* Implementar técnicas de Machine Learning como **Random Forest**, **XGBoost** o **LightGBM** para el pronóstico del spread.
* Probar arquitecturas de redes neuronales recurrentes (**RNN**, **LSTM**, **GRU**) para capturar dependencias a largo plazo y la no linealidad en la serie del spread.
* Realizar un backtesting más riguroso para evaluar el desempeño del modelo en diferentes periodos de tiempo.

## **Cómo Ejecutar el Proyecto**

1.  Clona este repositorio:
    ```bash
    git clone [https://github.com/HGalletti/Forecasting-Spread-Treasury-Bonds.git](https://github.com/HGalletti/Forecasting-Spread-Treasury-Bonds.git)
    ```
2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Abre y ejecuta los notebooks en orden para replicar el análisis.
