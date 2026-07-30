from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreditApplication(BaseModel):
    user_id: str = Field(
        min_length=1,
        description='Identificador del cliente',
        examples=['cliente-001']
    )

    age: int = Field(gt=0, le=120)
    occupation_status: int = Field(ge=0, le=2)
    years_employed: float = Field(ge=0)
    annual_income: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)
    credit_history_years: float = Field(ge=0)
    savings_assets: float = Field(ge=0)
    current_debt: float = Field(ge=0)
    defaults_on_file: int = Field(ge=0, le=1)
    delinquencies_last_2yrs: int = Field(ge=0)
    derogatory_marks: int = Field(ge=0)
    loan_intent: int = Field(ge=0, le=5)
    loan_amount: float = Field(gt=0)
    debt_to_income_ratio: float = Field(ge=0)
    loan_to_income_ratio: float = Field(ge=0)


class PredictionResponse(BaseModel):
    risk_level: int = Field(
        ge=0,
        le=2,
        description='0 = Bajo, 1 = Medio, 2 = Alto'
    )


class ExtractedApplicantData(BaseModel):
    '''Datos que el LLM puede extraer de una conversación.'''

    model_config = ConfigDict(extra='forbid')

    user_id: str | None = Field(
        default=None,
        description='Nombre o identificador que el usuario indicó explícitamente.'
    )
    age: int | None = Field(default=None, gt=0, le=120)
    occupation_status: Literal[
        'Employed',
        'Student',
        'Self-Employed'
    ] | None = None
    years_employed: float | None = Field(default=None, ge=0)
    annual_income: float | None = Field(default=None, gt=0)
    credit_score: int | None = Field(default=None, ge=300, le=850)
    credit_history_years: float | None = Field(default=None, ge=0)
    savings_assets: float | None = Field(default=None, ge=0)
    current_debt: float | None = Field(default=None, ge=0)
    defaults_on_file: int | None = Field(default=None, ge=0, le=1)
    delinquencies_last_2yrs: int | None = Field(default=None, ge=0)
    derogatory_marks: int | None = Field(default=None, ge=0)
    loan_intent: Literal[
        'Business',
        'Home Improvement',
        'Debt Consolidation',
        'Education',
        'Personal',
        'Medical'
    ] | None = None
    loan_amount: float | None = Field(default=None, gt=0)
