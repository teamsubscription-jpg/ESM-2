# HTTP server for RunPod load-balancing endpoints.
#
#   GET  /ping     health check polled by the RunPod load balancer
#   POST /predict  body: {"sequences": [...], "include": ["mean"], "repr_layers": [-1]}
#
# Started by the Dockerfile when ENDPOINT_TYPE=lb.

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from handler import MODEL_NAME, predict

app = FastAPI()


@app.get("/ping")
def ping():
    # The model is loaded when handler is imported, so once we serve requests we're ready.
    return {"status": "healthy", "model": MODEL_NAME}


# Sync route: FastAPI runs it in a thread pool, so /ping stays responsive during inference.
@app.post("/predict")
def predict_route(body: dict):
    result = predict(body)
    return JSONResponse(result, status_code=400 if "error" in result else 200)
