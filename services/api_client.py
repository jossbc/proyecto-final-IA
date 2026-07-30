import os

import httpx
from dotenv import load_dotenv


load_dotenv()

FASTAPI_URL = os.getenv(
    'FASTAPI_URL',
    'http://127.0.0.1:8000'
).rstrip('/')

class PredictionApiError(RuntimeError):
    pass


def request_prediction(payload: dict) -> int:
    try:
        response = httpx.post(
            f'{FASTAPI_URL}/predict',
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        prediction = int(response.json()['risk_level'])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise PredictionApiError(
            'No fue posible comunicarse con la API de predicción.'
        ) from error

    if prediction not in (0, 1, 2):
        raise PredictionApiError(
            'La API devolvió una categoría de riesgo inválida.'
        )

    return prediction
