from fastapi import FastAPI
import joblib
import pandas as pd

model = joblib.load(r"model\XGBClassifier.pkl")

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    proc_data = {}
    for k, v in data.items():
        proc_data[k] = [v]
        
    df = pd.DataFrame(proc_data)
    res = model.predict(df)
    print(type(res))
    return {"resul": res.tolist()}

