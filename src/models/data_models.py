from pydantic import BaseModel
from datetime import date
from typing import Optional, List
from enum import Enum


class Category(str, Enum):
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    ENVIRONMENTAL = "environmental"
    DATA_PRIVACY = "data_privacy"
    OTHER = "other"


class FederalRegulation(BaseModel):
    name: str
    citation: str
    description: str
    effective_date: date
    full_text: Optional[str] = None
    category: Category


class AgencyGuidance(BaseModel):
    title: str
    agency: str
    date_issued: date
    summary: str
    reference_number: str
    category: Category


class EnforcementAction(BaseModel):
    title: str
    agency: str
    date: date
    summary: str
    docket_number: str
    outcome: str
    category: Category


class ComplianceTopic(BaseModel):
    name: str
    description: str
    related_regulations: List[str]
    category: Category