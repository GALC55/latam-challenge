import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, ValidationError, conint
from typing import List
import xgboost as xgb
import pandas as pd
import logging
from enum import Enum
logger = logging.getLogger(__name__)

# Definir el esquema de los datos de vuelo
class OperaEnum(str, Enum):
    Aerolineas_Argentinas = "Aerolineas Argentinas"
    Grupo_LATAM = "Grupo LATAM"
    Sky_Airline = "Sky Airline"
    Copa_Air = "Copa Air"
    Latin_American_Wings = "Latin American Wings"

class TipoVueloEnum(str, Enum):
    I = "I"
    N = "N"

class FlightData(BaseModel):
    OPERA: OperaEnum
    TIPOVUELO: TipoVueloEnum
    MES: conint(ge=1, le=12)  # MES entre 1 y 12

class FlightRequest(BaseModel):
    flights: List[FlightData]


app = fastapi.FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes (ajusta según tus necesidades)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Cargar el modelo entrenado
model = xgb.XGBClassifier()
model.load_model("./modelo.xgb")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict")
async def post_predict(request: FlightRequest):
    # Transformación de los datos para que sean iguales a los del entrenamiento y en el mismo orden
    try:
        data = pd.DataFrame([{
            "OPERA": flight.OPERA,
            "TIPOVUELO": flight.TIPOVUELO,
            "MES": flight.MES
        } for flight in request.flights])

        x_pred = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        # Características utilizadas durante el entrenamiento
        topFeatures = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

        # Reindexación para asegurar que todas las columnas necesarias estén presentes
        x_pred = x_pred.reindex(columns=topFeatures, fill_value=0)

        # Asegurar que las columnas estén en el mismo orden que en el conjunto de entrenamiento
        predictions = model.predict(x_pred)

        return {"predict": predictions.tolist()}

    except ValidationError as e:
        # Manejo de la validación de datos (error 400)
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Logging de errores inesperados y devolver error 500
        logger.error(f"Error inesperado: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail="Error interno en la solicitud.")

