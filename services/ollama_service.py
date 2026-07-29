import os
import re
import unicodedata

import httpx
from langchain_core.prompts import ChatPromptTemplate

from models.credit_application import ExtractedApplicantData


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class OllamaServiceError(RuntimeError):
    pass


FLOAT_FIELDS = {
    "years_employed",
    "annual_income",
    "credit_history_years",
    "savings_assets",
    "current_debt",
    "loan_amount"
}

INTEGER_FIELDS = {
    "age",
    "credit_score",
    "delinquencies_last_2yrs",
    "derogatory_marks"
}


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )


def _fallback_pending_value(
    message: str,
    pending_field: str | None
) -> dict:
    if not pending_field:
        return {}

    normalized = _normalized_text(message).strip()
    number_match = re.search(r"-?\d+(?:[.,]\d+)?", normalized)

    if pending_field in FLOAT_FIELDS and number_match:
        value = float(number_match.group().replace(",", "."))
        return {pending_field: value}

    if pending_field in INTEGER_FIELDS and number_match:
        value = int(float(number_match.group().replace(",", ".")))
        return {pending_field: value}

    if pending_field == "defaults_on_file":
        if any(word in normalized for word in ("no", "nunca", "ninguno")):
            return {pending_field: 0}
        if any(word in normalized for word in ("si", "incumpli")):
            return {pending_field: 1}

    if pending_field in {
        "delinquencies_last_2yrs",
        "derogatory_marks"
    } and any(
        word in normalized
        for word in ("ninguno", "ninguna", "cero")
    ):
        return {pending_field: 0}

    if pending_field == "occupation_status":
        if "estudiante" in normalized or "student" in normalized:
            return {pending_field: "Student"}
        if "independiente" in normalized or "self-employed" in normalized:
            return {pending_field: "Self-Employed"}
        if "empleado" in normalized or "employed" in normalized:
            return {pending_field: "Employed"}

    if pending_field == "loan_intent":
        intent_terms = {
            "negocio": "Business",
            "business": "Business",
            "vivienda": "Home Improvement",
            "deuda": "Debt Consolidation",
            "educacion": "Education",
            "personal": "Personal",
            "medico": "Medical",
            "medical": "Medical"
        }
        for term, intent in intent_terms.items():
            if term in normalized:
                return {pending_field: intent}

    if pending_field == "user_id" and message.strip():
        return {pending_field: message.strip()}

    return {}


def _message_dicts(messages: list) -> list[dict[str, str]]:
    return [
        {
            "role": (
                "system"
                if message.type == "system"
                else "user"
            ),
            "content": str(message.content)
        }
        for message in messages
    ]


def _chat(
    messages: list[dict[str, str]],
    response_format: dict | str | None = None
) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    if response_format is not None:
        payload["format"] = response_format

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"]).strip()
    except (httpx.HTTPError, KeyError, TypeError) as error:
        raise OllamaServiceError(
            "No fue posible comunicarse con Ollama. "
            "Verifica que esté iniciado y que el modelo configurado exista."
        ) from error


def extract_applicant_data(
    user_message: str,
    collected_data: dict,
    pending_field: str | None = None
) -> dict:
    schema = ExtractedApplicantData.model_json_schema()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Eres un asistente que recopila datos para una evaluación crediticia.
Extrae únicamente información que el usuario haya dicho explícitamente.
No inventes, estimes ni completes datos ausentes.
Si un dato no aparece en el mensaje, devuélvelo como null.

Normaliza occupation_status exclusivamente a:
Employed, Student o Self-Employed.

Normaliza loan_intent exclusivamente a:
Business, Home Improvement, Debt Consolidation, Education, Personal o Medical.

Para defaults_on_file:
0 significa que nunca ha incumplido y 1 que sí ha incumplido.

Devuelve solamente un objeto que cumpla exactamente el esquema JSON.
            """.strip()
        ),
        (
            "human",
            """
Datos recopilados previamente:
{collected_data}

Dato solicitado en la pregunta anterior:
{pending_field}

Mensaje nuevo del usuario:
{user_message}

Interpreta una respuesta corta usando el dato solicitado en la pregunta anterior.
Extrae únicamente los datos nuevos o las correcciones expresas del mensaje nuevo.
            """.strip()
        )
    ])

    messages = prompt.format_messages(
        collected_data=collected_data,
        pending_field=pending_field or "ninguno",
        user_message=user_message
    )

    content = _chat(
        _message_dicts(messages),
        response_format=schema
    )

    extracted = ExtractedApplicantData.model_validate_json(content)
    extracted_data = extracted.model_dump(exclude_none=True)

    if pending_field not in extracted_data:
        fallback_data = _fallback_pending_value(
            user_message,
            pending_field
        )
        if fallback_data:
            validated_fallback = ExtractedApplicantData(
                **fallback_data
            )
            extracted_data.update(
                validated_fallback.model_dump(exclude_none=True)
            )

    return extracted_data


def explain_prediction(
    prediction: int,
    important_values: dict
) -> str:
    risk_labels = {
        0: "Bajo",
        1: "Medio",
        2: "Alto"
    }

    decisions = {
        0: "Aprobar la solicitud con condiciones estándar.",
        1: "Solicitar documentación adicional y evaluar nuevamente.",
        2: "Rechazar la solicitud y recomendar educación financiera."
    }

    credit_score = important_values["credit_score"]
    loan_ratio = important_values["loan_to_income_ratio"]
    debt_ratio = important_values["debt_to_income_ratio"]
    delinquencies = important_values["delinquencies_last_2yrs"]
    derogatory_marks = important_values["derogatory_marks"]

    if credit_score < 580:
        credit_observation = "muy bajo y es un factor desfavorable"
    elif credit_score < 600:
        credit_observation = "bajo y requiere cautela"
    elif credit_score < 687:
        credit_observation = "moderado y requiere atención"
    else:
        credit_observation = "favorable"

    if loan_ratio > 1.01:
        loan_observation = "elevada y es un factor desfavorable"
    elif loan_ratio >= 0.62:
        loan_observation = "moderada y requiere cautela"
    else:
        loan_observation = "baja y es un factor favorable"

    if debt_ratio >= 0.50:
        debt_observation = "muy elevada y es un factor desfavorable"
    elif debt_ratio >= 0.40:
        debt_observation = "elevada y requiere cautela"
    elif debt_ratio >= 0.30:
        debt_observation = "moderada"
    else:
        debt_observation = "baja y es un factor favorable"

    if delinquencies >= 3:
        delinquency_observation = "varios atrasos y es desfavorable"
    elif delinquencies >= 1:
        delinquency_observation = "presenta atrasos y requiere atención"
    else:
        delinquency_observation = "sin atrasos, lo cual es favorable"

    if derogatory_marks >= 2:
        marks_observation = "varias marcas y es desfavorable"
    elif derogatory_marks == 1:
        marks_observation = "una marca negativa y requiere atención"
    else:
        marks_observation = "sin marcas negativas, lo cual es favorable"

    grounded_observations = {
        "Puntaje crediticio": (
            f"{credit_score}: {credit_observation}"
        ),
        "Relación préstamo-ingreso": (
            f"{loan_ratio}: {loan_observation}"
        ),
        "Relación deuda-ingreso": (
            f"{debt_ratio}: {debt_observation}"
        ),
        "Atrasos de pago en los últimos dos años": (
            f"{delinquencies}: {delinquency_observation}"
        ),
        "Marcas negativas en el historial": (
            f"{derogatory_marks}: {marks_observation}"
        )
    }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Eres un asesor que comunica el resultado de una evaluación crediticia.
La categoría ya fue calculada por un modelo predictivo y no puedes cambiarla.
Redacta únicamente una justificación breve en español claro y respetuoso,
usando la categoría y las cinco características proporcionadas.
No menciones probabilidades, porcentajes, feature importance, SHAP, LIME,
algoritmos ni otros términos técnicos. No inventes información.
No escribas el número ni el nombre de la categoría.
No propongas, repitas ni cambies la decisión.
Limítate a explicar en dos o tres oraciones cómo los datos proporcionados
respaldan el resultado.
            """.strip()
        ),
        (
            "human",
            """
Categoría predicha: {prediction}
Significado: 0 = riesgo bajo, 1 = riesgo medio, 2 = riesgo alto.
Características principales del cliente:
{important_values}
            """.strip()
        )
    ])

    messages = prompt.format_messages(
        prediction=prediction,
        important_values=grounded_observations
    )

    justification = _chat(_message_dicts(messages))
    justification_sentences = re.split(
        r"(?<=[.!?])\s+",
        justification
    )
    justification = " ".join(
        sentence
        for sentence in justification_sentences
        if not re.search(
            r"\b(riesgo|categor[ií]a)\b",
            sentence,
            flags=re.IGNORECASE
        )
    ).strip()

    if not justification:
        justification = (
            "Los factores proporcionados combinan elementos favorables "
            "y aspectos que requieren atención adicional."
        )

    return (
        "## Resumen de la evaluación crediticia\n\n"
        f"**Categoría:** Riesgo {risk_labels[prediction]} "
        f"({prediction})\n\n"
        f"**Decisión:** {decisions[prediction]}\n\n"
        f"**Justificación:** {justification}"
    )
