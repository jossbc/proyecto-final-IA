import logging
from datetime import datetime, timezone

import pandas as pd
from fastapi import HTTPException
from pymongo.errors import PyMongoError

from models.credit_application import (
    CreditApplication,
    PredictionResponse
)
from utils.mongo import collection


logger = logging.getLogger(__name__)


model_bundle = pd.read_pickle(
    'artifacts/credit_risk_model.pkl'
)

model = model_bundle['model']
feature_names = model_bundle['feature_names']
top_5_features = model_bundle['top_5_features']


def predict_credit_risk(
    application: CreditApplication
) -> PredictionResponse:
    try:
        application_data = application.model_dump()


        client_features = {
            feature: application_data[feature]
            for feature in feature_names
        }

        input_dataframe = pd.DataFrame(
            [client_features],
            columns=feature_names
        )

        prediction = int(
            model.predict(input_dataframe)[0]
        )

        if prediction not in (0, 1, 2):
            raise ValueError(
                'El modelo produjo una categoría inválida.'
            )

        prediction_record = {
            'user_id': application.user_id,
            'features': client_features,
            'prediction': prediction,
            'created_at': datetime.now(timezone.utc)
        }

        collection.insert_one(prediction_record)

        logger.info(
            'Predicción registrada para el usuario %s',
            application.user_id
        )

        return PredictionResponse(
            risk_level=prediction
        )

    except PyMongoError as error:
        logger.exception(
            'Error guardando la predicción en MongoDB'
        )

        raise HTTPException(
            status_code=503,
            detail='No fue posible guardar la evaluación.'
        ) from error

    except KeyError as error:
        logger.exception(
            'Falta una característica requerida por el modelo'
        )

        raise HTTPException(
            status_code=500,
            detail=(
                'La configuración del modelo no coincide '
                'con los datos de entrada.'
            )
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            'Error realizando la predicción'
        )

        raise HTTPException(
            status_code=500,
            detail='No fue posible realizar la predicción.'
        ) from error