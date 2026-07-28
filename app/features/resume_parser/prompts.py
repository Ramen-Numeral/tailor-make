"""Prompts for lossless, factual resume extraction."""


def get_resume_extraction_prompt(resume_text: str) -> str:
    return f"""Extract this resume into the supplied ExtractedResume schema.

Rules:
- Copy facts only; never infer, embellish, rewrite, or add missing information.
- Preserve names, dates, metrics, technologies, URLs, and bullet meaning.
- Keep each role, school, project, and research entry separate.
- Put explicit skill groupings in skills. Do not derive skills from prose.
- Use null for unknown optional scalars and [] for absent lists.
- target_title is only an explicitly stated headline or desired role.
- A role must keep its explicit title, company, and start date.
- Education must keep its explicit degree and institution.
- Exclude page markers from extracted values.
- Return only a schema-valid JSON object.

RESUME:
{resume_text}"""
