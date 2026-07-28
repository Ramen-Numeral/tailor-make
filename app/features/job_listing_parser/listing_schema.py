from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class Requirement(BaseModel):
    id: UUID = Field(default_factory=uuid4, frozen=True)
    text: str
    kind: Literal[
        "skill",
        "experience",
        "education",
        "certification",
        "responsibility",
        "other",
    ]
    importance: Literal["critical", "important", "supporting"] = "important"
    required: bool = True
    source_text: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_text(cls, value):
        """Accept common structured-output variants at the LLM boundary."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        supplied_id = normalized.get("id")
        if supplied_id is not None:
            try:
                UUID(str(supplied_id))
            except (TypeError, ValueError, AttributeError):
                # Provider-generated IDs are not evidence and a malformed one
                # must not invalidate an otherwise useful requirement.
                normalized.pop("id", None)
        text = normalized.get("text")
        if not isinstance(text, str) or not text.strip():
            for alternate in ("name", "source_text"):
                candidate = normalized.get(alternate)
                if isinstance(candidate, str) and candidate.strip():
                    normalized["text"] = candidate.strip()
                    break
        return normalized


class JobListing(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    workplace_type: Literal["remote", "hybrid", "onsite"] | None = None
    employment_type: str | None = None
    seniority: str | None = None
    years_of_experience: int | None = None
    compensation: str | None = None
    requirements: list[Requirement] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_root_shape(cls, value):
        """Recover common provider root shapes before strict validation."""
        if isinstance(value, list):
            return {"requirements": value}
        if isinstance(value, dict):
            normalized = dict(value)
            aliases = {
                "jobTitle": "title",
                "employmentType": "employment_type",
                "workplaceType": "workplace_type",
                "yearsOfExperience": "years_of_experience",
            }
            for provider_name, field_name in aliases.items():
                if (
                    field_name not in normalized
                    and provider_name in normalized
                ):
                    normalized[field_name] = normalized[provider_name]
            location = normalized.get("location")
            if isinstance(location, dict):
                normalized["location"] = _format_location(location)
            return normalized
        return value

    @model_validator(mode="after")
    def remove_interface_metadata(self) -> "JobListing":
        """Exclude job-board chrome accidentally emitted as requirements."""
        ignored_labels = {
            "estimated commute",
            "job address",
            "job type",
        }
        requirements = [
            requirement
            for requirement in self.requirements
            if not (
                requirement.kind == "other"
                and (
                    (requirement.source_text or "").strip().casefold()
                    in ignored_labels
                    or _looks_like_job_board_metadata(requirement)
                )
            )
        ]
        self.requirements = requirements
        return self


def _looks_like_job_board_metadata(requirement: Requirement) -> bool:
    value = " ".join(
        filter(None, (requirement.text, requirement.source_text))
    ).casefold()
    patterns = (
        "typically responds",
        "responds to applications",
        "responds to posts",
        "application response",
        "posted ",
        "days ago",
        "estimated commute",
        "job address",
        "profile insights",
        "hiring multiple",
        "urgently hiring",
    )
    return any(pattern in value for pattern in patterns)


def _format_location(value: dict) -> str | None:
    """Convert common structured address output into a display location."""
    locality = [
        value.get("city"),
        value.get("state") or value.get("region"),
    ]
    city_region = ", ".join(
        str(part).strip() for part in locality if str(part or "").strip()
    )
    postal = value.get("zip") or value.get("postal_code")
    if city_region and postal:
        return f"{city_region} {str(postal).strip()}"
    if city_region:
        return city_region
    address = value.get("street") or value.get("address")
    if address and postal:
        return f"{str(address).strip()} {str(postal).strip()}"
    return str(address or postal).strip() or None
