# Proyecto de Modelado y API con FastAPI

Este proyecto consta de la transcripción y ajuste de un modelo desde un archivo `.ipynb` a un archivo `.py`, la creación de una API con **FastAPI**, y su despliegue en un proveedor en la nube. Durante el desarrollo, se realizaron diversas optimizaciones y actualizaciones para asegurar el correcto funcionamiento de los tests y el despliegue.

## Eleccion del modelo
La elección de XGBoost sobre regresión logística se justifica por su capacidad para capturar relaciones no lineales entre las variables y el objetivo, lo que lo hace más adecuado para datos complejos. Además, maneja features categóricas mejor, especialmente cuando se combinan con codificación adecuada como One-Hot Encoding o Label Encoding y maneja de manera natural los missing values y puede ajustarse automáticamente a esos datos sin necesidad de imputarlos, lo cual es útil en escenarios con datos faltantes o ruidosos. Estas características hacen que XGBoost sea una opción más robusta y eficiente para modelos donde las relaciones entre las variables son más complejas. Ademas, se eligio el modelo con el balance de pesos debido a que había un mayor porcentaje de una variable que de la otra, lo que hacia que el modelo sea preciso solo para predecir una de ellas, con el balance se permite que el modelo aprenda de manera mas equilibrada a predecir ambas.
## Problemas Resueltos

1. Se resolvieron problemas con los parámetros de los gráficos `sns.barplot`, donde no se había definido correctamente qué variable era el eje **X** y cuál era el eje **Y**.
2. Durante la ejecución de los tests, se encontraron errores relacionados con módulos faltantes, por lo que se realizaron las siguientes actualizaciones:
   - Se instaló el paquete `httpx`.
   - Se actualizó el paquete `anyio`.
   - Se actualizó el paquete `Flask`.
   - Se instalo el paquete `loggin` para el manejo de log de errores en la api
3. Se realizó la implementación del CI/CD en el cual el CI se ejecuta al hacer push a cualquier de las 2 ramas para ejecutar los test del modelo y api. El CD se ejecuta solo al hacer push a main ya que despliega el modelo en GCP.
4. El DockerFile se llenó con toda la información y los paquetes necesarios para la ejecución de la API y el .dockerignore tiene todo lo no estrictamente necesario para poder ahorrar espacio a la hora de hacer build a la imagen.

## Paquetes Utilizados

A continuación se lista la versión final de todos los paquetes utilizados en este proyecto:

```bash
fastapi==0.114.0
Flask==3.0.3
Flask-BasicAuth==0.2.0
Flask-Cors==5.0.0
httpx==0.27.2
locust==1.6.0
matplotlib==3.7.5
mockito==1.2.2
numpy==1.26.4
pandas==2.2.2
pytest==6.2.5
pytest-cov==2.12.1
requests==2.32.3
scikit-learn==1.3.2
scipy==1.14.1
seaborn==0.12.2
uvicorn==0.15.0
xgboost==2.1.1

