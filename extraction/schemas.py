import datetime
from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["PERSON", "ORG", "COUNTRY", "COMPANY"]


class Entity(BaseModel):
    name: str
    type: EntityType
    role: str


class Claim(BaseModel):
    text: str
    source_url: str


class ExtractedEvent(BaseModel):
    title: str
    summary: str = Field(description="Neutral, at most two sentences")
    date: datetime.date | None = None
    location: str | None = None
    entities: list[Entity] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
