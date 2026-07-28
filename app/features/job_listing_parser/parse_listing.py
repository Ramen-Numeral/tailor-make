import logging
import re

from app.bootstrap import get_llm_clients
from app.infrastructure.llm.errors import LLMError
from app.infrastructure.cache import (
    content_key,
    get_cached,
    object_identity,
    set_cached,
)
from app.features.job_listing_parser.listing_schema import JobListing, Requirement
from app.utils.pydantic import (
    find_missing_fields,
    merge_pydantic_models,
)
from app.features.job_listing_parser.prompts import (
    get_job_listing_extraction_prompt,
    get_job_listing_retry_prompt,
)


REQUIRED_EXTRACTION_FIELDS = ("title", "requirements")
logger = logging.getLogger(__name__)


def parse_listing(
    job_listing: str,
    max_attempts: int = 3,
    minimum_attempts: int = 1,
) -> JobListing:
    """Extract structured information from a job listing."""
    if not job_listing.strip():
        raise ValueError("job_listing cannot be empty")

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not 1 <= minimum_attempts <= max_attempts:
        raise ValueError(
            "minimum_attempts must be between 1 and max_attempts"
        )

    parser = get_llm_clients().job_parser
    cache_key = content_key(
        "job_listing",
        object_identity(parser),
        job_listing,
        max_attempts,
        minimum_attempts,
    )
    cached = get_cached(cache_key)
    if cached is not None:
        return JobListing.model_validate(cached)

    deterministic = _extract_requirement_lines(job_listing)
    result: JobListing | None = (
        JobListing(requirements=deterministic)
        if len(deterministic) >= 3
        else None
    )
    last_provider_error: LLMError | None = None

    for attempt in range(1, max_attempts + 1):
        missing_fields = (
            find_missing_fields(result, REQUIRED_EXTRACTION_FIELDS)
            if result
            else []
        )

        prompt = (
            get_job_listing_retry_prompt(
                job_listing=job_listing,
                missing_fields=missing_fields,
            )
            if missing_fields
            else get_job_listing_extraction_prompt(job_listing)
        )

        try:
            response = parser.invoke_structured(
                prompt=prompt,
                schema=JobListing,
                temperature=0,
                trace_context=f"job_parser pass={attempt}/{max_attempts}",
            )
        except LLMError as error:
            last_provider_error = error
            logger.warning(
                "job_parser_provider_exhausted pass=%d/%d "
                "retained_requirements=%d error=%s",
                attempt,
                max_attempts,
                len(result.requirements) if result else 0,
                error,
            )
            break

        previous_count = len(result.requirements) if result else 0
        result = _merge_job_listings(result, response)
        added_count = len(result.requirements) - previous_count
        logger.info(
            "job_parser_pass_completed pass=%d/%d response_requirements=%d "
            "added_requirements=%d retained_requirements=%d",
            attempt,
            max_attempts,
            len(response.requirements),
            added_count,
            len(result.requirements),
        )

        missing_fields = find_missing_fields(
            result,
            REQUIRED_EXTRACTION_FIELDS,
        )

        if attempt < minimum_attempts:
            continue
        if (
            not missing_fields
            and (
                minimum_attempts == 1
                or added_count == 0
                or attempt == max_attempts
            )
        ):
            break

    if result is None:
        result = JobListing()
    if len(result.requirements) < 3 and len(job_listing) >= 600:
        llm_requirement_count = len(result.requirements)
        if len(deterministic) >= 3:
            result = result.model_copy(
                update={
                    "requirements": _merge_requirements(
                        result.requirements,
                        deterministic,
                    )
                }
            )
            logger.warning(
                "job_parser_low_yield_recovered llm_requirements=%d "
                "deterministic_requirements=%d retained_requirements=%d",
                llm_requirement_count,
                len(deterministic),
                len(result.requirements),
            )
    if not result.requirements and last_provider_error is not None:
        raise last_provider_error
    set_cached(cache_key, result.model_dump(mode="json"))
    return result


def _merge_job_listings(
    current: JobListing | None,
    new: JobListing,
) -> JobListing:
    """Merge scalar metadata and union requirements by normalized meaning."""
    if current is None:
        return new
    merged = merge_pydantic_models(
        current.model_copy(update={"requirements": []}),
        new.model_copy(update={"requirements": []}),
    )
    requirements = _merge_requirements(
        current.requirements,
        new.requirements,
    )
    return merged.model_copy(update={"requirements": requirements})


def _merge_requirements(
    current: list[Requirement],
    new: list[Requirement],
) -> list[Requirement]:
    retained: list[Requirement] = []
    seen: set[str] = set()
    for requirement in [*current, *new]:
        key = _requirement_key(requirement.text)
        if key in seen:
            continue
        seen.add(key)
        retained.append(requirement)
    return retained


def _requirement_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


_RELEVANT_HEADINGS = re.compile(
    r"^(?:requirements?|qualifications?|responsibilities|duties|"
    r"what you(?:'|’)ll do|what you will do|skills?|experience|education|"
    r"who you are|what we(?:'|’)re looking for)\s*:?$",
    re.IGNORECASE,
)
_IRRELEVANT_HEADINGS = re.compile(
    r"^(?:benefits?|perks?|compensation|about us|about the company|"
    r"equal opportunity|how to apply|application instructions?|"
    r"physical requirements?)\s*:?$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*•▪◦‣]|\d+[.)])\s+(.+?)\s*$")


def _extract_requirement_lines(job_listing: str) -> list[Requirement]:
    """Recover explicit bullets under requirement/duty headings."""
    relevant_section = False
    recovered: list[Requirement] = []
    for raw_line in job_listing.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _RELEVANT_HEADINGS.fullmatch(line):
            relevant_section = True
            continue
        if _IRRELEVANT_HEADINGS.fullmatch(line):
            relevant_section = False
            continue
        bullet = _BULLET.match(raw_line)
        if not bullet or not relevant_section:
            continue
        text = bullet.group(1).strip()
        if len(text) < 8 or _is_interface_line(text):
            continue
        recovered.append(
            Requirement(
                text=text,
                kind=_infer_kind(text),
                importance=(
                    "supporting"
                    if re.search(r"\b(?:preferred|bonus|nice to have)\b", text, re.I)
                    else "important"
                ),
                required=not bool(
                    re.search(r"\b(?:preferred|bonus|nice to have)\b", text, re.I)
                ),
                source_text=text,
            )
        )
    return _merge_requirements([], recovered)


def _is_interface_line(text: str) -> bool:
    lowered = text.casefold()
    return any(
        value in lowered
        for value in (
            "responds to applications",
            "responds to posts",
            "days ago",
            "estimated commute",
            "job address",
            "apply now",
            "profile insights",
        )
    )


def _infer_kind(text: str) -> str:
    lowered = text.casefold()
    if re.search(r"\b(?:degree|bachelor|master|phd|education)\b", lowered):
        return "education"
    if re.search(r"\b(?:certification|certified|pmp|license)\b", lowered):
        return "certification"
    if re.search(r"\b(?:years?|experience)\b", lowered):
        return "experience"
    if re.search(
        r"\b(?:proficiency|knowledge|skill|ability|python|sql|jira|excel)\b",
        lowered,
    ):
        return "skill"
    return "responsibility"
