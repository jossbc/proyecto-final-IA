import logging

from fastapi import FastAPI

from routes.prediction_routes import router as prediction_router


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Sistema de Riesgo Crediticio",
    description="API para evaluar solicitudes de cerdito",
    version="1.0.0"
)

app.include_router(prediction_router)


@app.get("/")
def read_root():
    return {
        "message": "API de riesgo crediticio activa"
    }