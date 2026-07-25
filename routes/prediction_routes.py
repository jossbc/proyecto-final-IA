from fastapi import APIRouter

from controllers.prediction_controller import (
    predict_credit_risk
)
from models.credit_application import (
    CreditApplication,
    PredictionResponse
)

router = APIRouter(
    prefix='/predict',
    tags=['Predicciones']
)


@router.post(
    '',
    response_model=PredictionResponse
)
def predict(
    application: CreditApplication
) -> PredictionResponse:
    return predict_credit_risk(application)