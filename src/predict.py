import joblib
import pandas as pd

modelo = joblib.load("models/xgb_reduzido.pkl")

def predict(data_dict):
    #assume chegada de dados só com as 3 colunas do modelo reduzido
    df = pd.DataFrame([data_dict])

    df = df.astype(float) #forçar float para garantir precisão do treino
    df.columns = df.columns.str.replace(r'[\[\]<]', '', regex=True)

    pred = modelo.predict(df)
    prob = modelo.predict_proba(df)[0][1]  #chance de ser classe 1(incêndio)

    #conversão de tipos por causa de typeerror no tradutor padrão de json
    return {"status": "alerta de incêndio" if pred[0] == 1 else "ok",
            "classe": int(pred[0]),
            "probabilidade_incendio": round(float(prob) * 100, 2)}
