import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, ValidationError
from typing import List
import xgboost as xgb
import pandas as pd


# Definir el esquema de los datos de vuelo
class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


    @validator('OPERA')
    def validate_opera(cls, value):
        valid_opera = ["Aerolineas Argentinas", "Grupo LATAM", "Sky Airline", "Copa Air", "Latin American Wings"]
        if value not in valid_opera:
            raise ValueError(f'Operador desconocido: {value}')
        return value

    @validator('TIPOVUELO')
    def validate_tipovuelo(cls, value):
        if value not in ["I", "N"]:
            raise ValueError(f'Tipo de vuelo desconocido: {value}')
        return value

    @validator('MES')
    def validate_mes(cls, value):
        if not 1 <= value <= 12:
            raise ValueError(f'Mes fuera de rango: {value}')
        return value


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

# Cargar el modelo entrenado y el transformador
model = xgb.XGBClassifier()
model.load_model("./modelo.xgb")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: FlightRequest):
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

        # Las características que utilizaste durante el entrenamiento
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

        # Asegurarse de que todas las columnas necesarias estén en los datos de predicción
        # Si falta alguna columna, se agrega con valor 0
        for col in topFeatures:
            if col not in x_pred.columns:
                x_pred[col] = False

        # Asegurarse de que las columnas estén en el mismo orden que en el conjunto de entrenamiento
        x_pred = x_pred[topFeatures]

        predictions = model.predict(x_pred)

        return {"predict": predictions.tolist()}


    except ValidationError as e:

        # Maneja la validación de datos y devuelve un error 400 si ocurre una excepción de validación

        raise fastapi.HTTPException(status_code=400, detail=str(e))


    except Exception as e:

        # Maneja cualquier otra excepción y devuelve un error 400 genérico

        raise fastapi.HTTPException(status_code=400, detail="Error en la solicitud.")
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
