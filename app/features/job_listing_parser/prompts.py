"""Compact prompts for structured job-listing extraction."""


def get_job_listing_extraction_prompt(job_listing: str) -> str:
    """Build the complete parser instruction as one user prompt."""
    return f"""Extract the job posting into the supplied JobListing schema.
Use only explicit evidence; never infer missing facts. Use null for unknown
scalars and [] when no requirements exist. Deduplicate requirements.

For each qualification or duty, create one concise requirement:
- text: the concise requirement name; this field is required on every record
- Split enumerated skills and technologies into separate requirement records
- kind: skill, experience, education, certification, responsibility, or other
- importance: critical, important, or supporting
- required: false only for clearly optional language such as preferred/bonus
- source_text: a short exact supporting excerpt
- Omit the id field; the application assigns stable UUIDs after extraction

Ignore benefits, marketing, legal boilerplate, and application instructions.
Ignore job-board activity such as posting age, response time, applicant count,
urgently hiring labels, commute estimates, and employer response statistics.
For experience ranges use the minimum stated years. Preserve meaningful
location and compensation wording. Return only a schema-valid JSON object.
Do not use "name", "level", or "min_years" in place of the required text field.
The root must be one JobListing object with a `requirements` array; never
return the requirements array by itself. Job type, job address, commute
estimates, benefits, and job-board interface labels are not requirements.

POSTING:
{job_listing}"""


def get_job_listing_retry_prompt(
    job_listing: str,
    missing_fields: list[str],
) -> str:
    """Build a compact retry prompt for essential missing fields."""
    return f"""Extract the complete JobListing again, focusing on explicit
evidence for: {', '.join(missing_fields)}. Do not guess; absent data stays null
or empty. Return only a schema-valid JSON object.

POSTING:
{job_listing}"""
