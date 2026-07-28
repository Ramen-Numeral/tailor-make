"""Bounded action selection from deterministic attempt decisions."""

from app.features.agent.schema import AgentAction
from app.features.validator.schema import SectionValidationResult


def actions_for_section(
    result: SectionValidationResult,
) -> list[AgentAction]:
    """Translate attempt decisions into an auditable bounded action history."""
    actions: list[AgentAction] = []
    for attempt in result.attempts:
        decision = attempt.decision
        if decision.outcome == "retry":
            factual = any(
                failure.startswith(
                    (
                        "Metrics and numbers",
                        "Locked field",
                        "Skills and technologies",
                    )
                )
                or "Source item provenance" in failure
                for failure in decision.hard_failures
            )
            actions.append(
                AgentAction(
                    action="restore_facts" if factual else "rewrite_section",
                    section=result.section_name,
                    reason=decision.reasons[0],
                )
            )
        else:
            actions.append(
                AgentAction(
                    action="accept_section",
                    section=result.section_name,
                    reason=decision.reasons[0],
                )
            )
    if result.status == "best_attempt_selected":
        actions.append(
            AgentAction(
                action="keep_best_attempt",
                section=result.section_name,
                reason="The section retry budget was exhausted.",
            )
        )
    return actions
