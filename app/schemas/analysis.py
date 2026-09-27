from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalysisSource(BaseModel):
    chunk_id: str
    source_locator: str
    page_number: int | None
    similarity: float


class GroundedFinding(BaseModel):
    value: str
    basis: Literal["explicit", "inference"]
    sources: list[AnalysisSource] = Field(min_length=1)


class RawFinding(BaseModel):
    """Provider-side structure. Labels are validated and resolved by the service."""

    model_config = ConfigDict(extra="forbid")

    value: str
    basis: Literal["explicit", "inference"] = "explicit"
    source_labels: list[str] = Field(default_factory=list)


class RawContractAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_type: RawFinding | None = None
    parties: list[RawFinding] = Field(default_factory=list)
    effective_date: RawFinding | None = None
    expiration_date: RawFinding | None = None
    renewal_terms: list[RawFinding] = Field(default_factory=list)
    notice_periods: list[RawFinding] = Field(default_factory=list)
    payment_terms: list[RawFinding] = Field(default_factory=list)
    termination_conditions: list[RawFinding] = Field(default_factory=list)
    confidentiality_obligations: list[RawFinding] = Field(default_factory=list)
    intellectual_property_provisions: list[RawFinding] = Field(default_factory=list)
    non_compete_non_solicitation: list[RawFinding] = Field(default_factory=list)
    governing_law: RawFinding | None = None
    dispute_resolution: list[RawFinding] = Field(default_factory=list)
    major_obligations: list[RawFinding] = Field(default_factory=list)
    important_deadlines: list[RawFinding] = Field(default_factory=list)
    other_important_clauses: list[RawFinding] = Field(default_factory=list)
    attention_items: list[RawFinding] = Field(default_factory=list)


class ContractAnalysisResponse(BaseModel):
    document_id: str
    status: Literal["complete", "insufficient_context"]
    contract_type: GroundedFinding | None = None
    parties: list[GroundedFinding] = Field(default_factory=list)
    effective_date: GroundedFinding | None = None
    expiration_date: GroundedFinding | None = None
    renewal_terms: list[GroundedFinding] = Field(default_factory=list)
    notice_periods: list[GroundedFinding] = Field(default_factory=list)
    payment_terms: list[GroundedFinding] = Field(default_factory=list)
    termination_conditions: list[GroundedFinding] = Field(default_factory=list)
    confidentiality_obligations: list[GroundedFinding] = Field(default_factory=list)
    intellectual_property_provisions: list[GroundedFinding] = Field(default_factory=list)
    non_compete_non_solicitation: list[GroundedFinding] = Field(default_factory=list)
    governing_law: GroundedFinding | None = None
    dispute_resolution: list[GroundedFinding] = Field(default_factory=list)
    major_obligations: list[GroundedFinding] = Field(default_factory=list)
    important_deadlines: list[GroundedFinding] = Field(default_factory=list)
    other_important_clauses: list[GroundedFinding] = Field(default_factory=list)
    attention_items: list[GroundedFinding] = Field(default_factory=list)
