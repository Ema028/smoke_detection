from fastapi import FastAPI
from src.predict import predict
from pydantic import BaseModel, Field

app = FastAPI(title="Smoke Detection IoT",
              description="API para detecção de incêndio usando XGBoost",
              version="1.0.0")

class LeituraSensor(BaseModel):
    TVOCppb: float
    PressurehPa: float
    #python não permite variáveis com ponto, mas json aceita como chave
    PM1_0: float = Field(alias="PM1.0")

@app.get("/")
def home():
    return {"message": "Detector de Incêndio"}

@app.post("/predict")
def prever_incendio(dados: LeituraSensor):
    #by_alias=True -> chave volte a se chamar 'PM1.0'
    dicionario_dados = dados.model_dump(by_alias=True)
    resultado = predict(dicionario_dados)
    return resultado
