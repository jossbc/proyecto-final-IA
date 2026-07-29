from fastapi import APIRouter

from controllers.prediction_controller import (
    get_model_metadata,
    predict_credit_risk
)
from models.credit_application import (
    CreditApplication,
    ModelMetadataResponse,
    PredictionResponse
)

router = APIRouter(
    prefix='/predict',
    tags=['Predicciones']
)


@router.get(
    "/metadata",
    response_model=ModelMetadataResponse
)
def metadata() -> ModelMetadataResponse:
    return get_model_metadata()


@router.post(
    '',
    response_model=PredictionResponse
)
def predict(
    application: CreditApplication
) -> PredictionResponse:
    return predict_credit_risk(application)
