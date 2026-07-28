"""Run the resume-tailoring MVP and save HTML and PDF outputs."""

from pathlib import Path

from app.features.agent.orchestrator import tailor_resume_agent
from app.features.job_listing_parser.listing_schema import JobListing
from app.features.keyword_evidence.schema import CoveragePlan
from app.features.pipeline import tailor_resume
from app.features.renderer.renderer import render_html, save_html, save_pdf
from app.features.validator.schema import AgentTraceEvent
from app.features.validator.validator import validate_resume_with_trace
from app.resume_schema.resume_schema import MutableResume
from config.resume.candidate_profile import build_resume
from config.settings import get_settings


JOB_LISTING = """
Job details
Pay
$65,000 - $115,000 a year
Job type
Full-time
&nbsp;
Benefits
Pulled from the full job description

Retirement plan
Vision insurance
Dental insurance
Disability insurance
Profit sharing
&nbsp;
Full job description
About Us

TherapyNotes is the go-to superhero for behavioral health Practice Management and EHR software! Our top-notch SaaS solution handles scheduling, billing, documenting, telehealth, and more so clinicians can focus on awesome patient care.

We're a dynamic team of pros who love to innovate and push the envelope, keeping our software cutting-edge. Join us, and let's revolutionize behavioral health software together while making a real difference!

About The Job

TherapyNotes is seeking a Software Developer to join our growing team. We are looking for a passionate engineer skilled in building scalable and responsive web applications and services using Angular and ASP.NET Core. The ideal candidate will have demonstrated experience in implementing robust APIs using event-based software design and adhering to Service-Oriented Architecture (SOA) principles. They should excel in a collaborative environment.

What You'll Do:

Perform full-stack development including front end, business logic, and data access layers.
Responsible for the entire development lifecycle from planning to release and support
Actively contribute to software architecture decisions, design strategies, and code reviews to ensure high-quality, scalable, and maintainable solutions
Collaborate closely with development team members and stakeholders
Maintain high standards, attention to detail, accuracy and completeness
What We're Looking For:

3 or more years of experience developing software in an Agile, team-based environment
1 or more years of experience developing responsive web applications
Expertise with Angular, ASP.NET Core, C#, JavaScript, TypeScript, CSS, SASS, and HTML
BS and/or MS in a technical discipline (Computer Science or Software Engineering required)
Strong understanding of OOP concepts and design patterns
Experience in building robust APIs and adhering to Service-Oriented Architecture (SOA) principles
Familiarity with event-based software design and event-driven architecture
Experience with PostgreSQL or other relational databases, and Entity Framework Core or similar object-relational mapping frameworks
Excellent problem solving and communication skills
What We Offer:

Competitive salary - $65,000-$115,000
Employer sponsored health, dental, vision, life, and disability insurance
Retirement plan with company contribution
Annual company profit sharing
Personal development/training budget
Open, collaborative work environment
Extensive 2-week onboarding plan
Comprehensive mentorship program
TherapyNotes LLC is an Equal Opportunity Employer and does not discriminate based on race, color, religion, sex, national origin, age, disability, genetic information, or any other protected status under federal, state, or local law. We are committed to providing a workplace free of discrimination and harassment. For more information about your rights under federal employment laws, please review the following:

Know Your Rights: Workplace Discrimination is Illegal
Family and Medical Leave Act (FMLA): Employee Rights Under FMLA
If you require a reasonable accommodation during the application process, please contact humanresources@therapynotes.com.
#LI-Remote
#LI-AC1
5/5/2026


Requirements

Benefits
&nbsp;
This employer uses Indeed, Inc. to generate an AI-powered summary evaluation of your application for this job. By applying, you agree that your application will include such a summary. If you do not agree, do not apply through Indeed. Instead, contact the employer directly to find another way to apply.
Accommodations
If you would like to request an accommodation, contact the employer by email at humanresources@therapynotes.com.
""".strip()


def print_trace(event: AgentTraceEvent) -> None:
    """Print one frontend-safe agent event as it happens."""
    context = " · ".join(
        part
        for part in (
            event.section,
            f"attempt {event.attempt}" if event.attempt is not None else None,
        )
        if part
    )
    heading = f"[{event.event_type}]"
    if context:
        heading = f"{heading} {context}"

    print(f"\n{heading} — {event.title}")
    print(event.summary)

    for observation in event.observations:
        print(f"  • {observation}")

    if event.action:
        print(f"  Action: {event.action}")

    for reason in event.decision_reasons:
        print(f"  Reason: {reason}")


def traced_quality_hook(
    job_listing: JobListing,
    resume: MutableResume,
) -> MutableResume:
    """Validate a resume while streaming the agent decision trace."""
    result = validate_resume_with_trace(
        job_listing,
        resume,
        trace_callback=print_trace,
    )
    return (
        result.resume
        if isinstance(result.resume, MutableResume)
        else MutableResume.model_validate(result.resume.model_dump())
    )


def print_coverage_plan(plan: CoveragePlan) -> None:
    """Print supported and unsupported job requirements with provenance."""
    print("\n[keyword_coverage] — Candidate evidence plan")
    for match in plan.requirement_matches:
        marker = {
            "supported": "✓",
            "partial": "≈",
            "unsupported": "✗",
        }[match.support]
        print(f"  {marker} {match.requirement_text}: {match.support}")
        for candidate in match.matches[:2]:
            evidence = candidate.evidence
            signals = []
            if candidate.exact_or_alias_match:
                signals.append("exact/alias")
            if candidate.bm25_rank is not None:
                signals.append(f"BM25 #{candidate.bm25_rank}")
            if candidate.cosine_score is not None:
                signals.append(f"cosine {candidate.cosine_score:.2f}")
            if candidate.judge_support is not None:
                signals.append(f"judge {candidate.judge_support}")
            print(
                f"      {evidence.section}.{evidence.field}: "
                f"{evidence.text} ({', '.join(signals)})"
            )
        if match.adjudication_reason:
            print(f"      Judge: {match.adjudication_reason}")

    if plan.assignments:
        print("  Required keyword assignments:")
        for assignment in plan.assignments:
            print(
                f"    • {assignment.section}: {assignment.keyword} "
                f"({assignment.importance})"
            )


def main() -> Path:
    source = build_resume()
    resume = source

    try:
        if get_settings().runtime.ai_detection_enabled:
            result = tailor_resume_agent(
                source,
                JOB_LISTING,
                trace_callback=print_trace,
                coverage_callback=print_coverage_plan,
            )
            resume = result.resume
        else:
            resume = tailor_resume(
                source,
                JOB_LISTING,
                coverage_hook=print_coverage_plan,
            )
    finally:
        html = render_html(resume)
        output_path = get_settings().io.pipeline_output_path
        output = save_html(html, output_path)
        pdf_output = save_pdf(html, output_path)
        print(f"Saved HTML: {output}")
        print(f"Saved PDF: {pdf_output}")

    return output


if __name__ == "__main__":
    main()
