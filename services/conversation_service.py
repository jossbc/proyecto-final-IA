from services.api_client import (
    PredictionApiError,
    request_prediction
)
from services.ollama_service import (
    OllamaServiceError,
    extract_applicant_data
)


RAW_REQUIRED_FIELDS = [
    "user_id",
    "age",
    "occupation_status",
    "years_employed",
    "annual_income",
    "credit_score",
    "credit_history_years",
    "savings_assets",
    "current_debt",
    "defaults_on_file",
    "delinquencies_last_2yrs",
    "derogatory_marks",
    "loan_intent",
    "loan_amount"
]

OCCUPATION_MAPPING = {
    "Employed": 1,
    "Student": 0,
    "Self-Employed": 2
}

LOAN_INTENT_MAPPING = {
    "Business": 0,
    "Home Improvement": 1,
    "Debt Consolidation": 2,
    "Education": 3,
    "Personal": 4,
    "Medical": 5
}

RISK_LABELS = {
    0: "Bajo",
    1: "Medio",
    2: "Alto"
}

FIELD_QUESTIONS = {
    "user_id": "¿Cómo te llamas o que identificador deseas usar?",
    "age": "¿Cual es tu edad?",
    "occupation_status": (
        "¿Actualmente eres empleado, estudiante o trabajador independiente?"
    ),
    "years_employed": "¿Cuántos años llevas trabajando?",
    "annual_income": "¿Cuál es tu ingreso anual aproximado?",
    "credit_score": "¿Cuál es tu puntaje crediticio?",
    "credit_history_years": (
        "¿Cuántos años de historial crediticio tienes?"
    ),
    "savings_assets": (
        "¿Cuánto tienes aproximadamente entre ahorros y activos?"
    ),
    "current_debt": "¿Cuánto debes actualmente?",
    "defaults_on_file": (
        "¿Has incumplido anteriormente el pago de algún crédito?"
    ),
    "delinquencies_last_2yrs": (
        "¿Cuántos atrasos de pago has tenido en los últimos dos años?"
    ),
    "derogatory_marks": (
        "¿Cuántas marcas negativas aparecen en tu historial crediticio?"
    ),
    "loan_intent": (
        "¿Para qué necesitas el préstamo: negocio, mejora de vivienda, "
        "consolidación de deudas, educación, uso personal o gastos médicos?"
    ),
    "loan_amount": "¿Qué cantidad deseas solicitar?"
}


def new_conversation_state() -> dict:
    return {
        "data": {},
        "pending_field": None,
        "completed": False
    }


def _missing_fields(data: dict) -> list[str]:
    return [
        field
        for field in RAW_REQUIRED_FIELDS
        if field not in data
    ]


def _question_for_missing_data(
    missing_fields: list[str]
) -> str:
    field = missing_fields[0]
    return FIELD_QUESTIONS[field]


def _prediction_payload(data: dict) -> dict:
    annual_income = float(data["annual_income"])

    return {
        "user_id": str(data["user_id"]),
        "age": int(data["age"]),
        "occupation_status": OCCUPATION_MAPPING[
            data["occupation_status"]
        ],
        "years_employed": float(data["years_employed"]),
        "annual_income": annual_income,
        "credit_score": int(data["credit_score"]),
        "credit_history_years": float(data["credit_history_years"]),
        "savings_assets": float(data["savings_assets"]),
        "current_debt": float(data["current_debt"]),
        "defaults_on_file": int(data["defaults_on_file"]),
        "delinquencies_last_2yrs": int(
            data["delinquencies_last_2yrs"]
        ),
        "derogatory_marks": int(data["derogatory_marks"]),
        "loan_intent": LOAN_INTENT_MAPPING[data["loan_intent"]],
        "loan_amount": float(data["loan_amount"]),
        "debt_to_income_ratio": round(
            float(data["current_debt"]) / annual_income,
            3
        ),
        "loan_to_income_ratio": round(
            float(data["loan_amount"]) / annual_income,
            3
        )
    }


def process_message(
    user_message: str,
    state: dict | None
) -> tuple[str, dict]:
    current_state = state or new_conversation_state()

    if current_state.get("completed"):
        return (
            "La evaluación ya terminó. Pulsa «Nueva evaluación» "
            "para analizar otra solicitud.",
            current_state
        )

    try:
        extracted = extract_applicant_data(
            user_message,
            current_state["data"],
            current_state.get("pending_field")
        )
        current_state["data"].update(extracted)
        current_state["pending_field"] = None

        missing = _missing_fields(current_state["data"])
        if missing:
            current_state["pending_field"] = missing[0]
            question = _question_for_missing_data(missing)
            return question, current_state

        payload = _prediction_payload(current_state["data"])
        prediction = request_prediction(payload)

        current_state["completed"] = True
        current_state["prediction"] = prediction

        result = (
            "## Resultado de la evaluación crediticia\n\n"
            f"**Categoría:** Riesgo {RISK_LABELS[prediction]} "
            f"({prediction})\n\n"
        )

        return result, current_state

    except (OllamaServiceError, PredictionApiError) as error:
        return str(error), current_state
    except (KeyError, TypeError, ValueError):
        return (
            "No pude interpretar uno de los datos. "
            "¿Puedes expresarlo nuevamente con una cifra concreta?",
            current_state
        )
