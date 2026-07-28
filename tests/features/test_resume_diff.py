from app.features.agent.schema import PageTrimAction
from app.features.resume_diff.differ import build_resume_diffs, inline_diff
from app.resume_schema.resume_schema import MutableResume, ProfessionalSummaryItem
from config.resume.candidate_profile import build_resume


def test_inline_diff_emits_add_remove_and_equal_tokens() -> None:
    tokens = inline_diff(
        "Built an API.",
        "Built a reliable API.",
    )

    assert any(token.operation == "remove" for token in tokens)
    assert any(token.operation == "add" for token in tokens)
    assert "".join(
        token.text
        for token in tokens
        if token.operation != "remove"
    ) == "Built a reliable API."


def test_resume_diff_distinguishes_selection_rewrite_and_page_trim() -> None:
    source = build_resume()
    selected = MutableResume.model_validate(source.model_dump())
    selected.projects = selected.projects.model_copy(
        update={"items": selected.projects.items[:2]},
        deep=True,
    )

    final = selected.model_copy(deep=True)
    first_job = final.work_experience.items[0]
    rewritten_job = first_job.model_copy(
        update={
            "bullets": [
                "Built a reliable payment service.",
                *first_job.bullets[1:],
            ]
        },
        deep=True,
    )
    final.work_experience = final.work_experience.model_copy(
        update={
            "items": [
                rewritten_job,
                *final.work_experience.items[1:],
            ]
        },
        deep=True,
    )
    trimmed_project = final.projects.items[1]
    final.projects = final.projects.model_copy(
        update={"items": final.projects.items[:1]},
        deep=True,
    )

    diffs = build_resume_diffs(
        source,
        selected,
        final,
        page_trim_actions=[
            PageTrimAction(
                section="projects",
                item_id=trimmed_project.id,
                field="item",
                removed=trimmed_project.name,
                reason="Removed the lowest-ranked safe selected item.",
            )
        ],
    )

    assert any(
        diff.change_type == "rewritten"
        and diff.section == "work_experience"
        and diff.field == "bullets"
        and any(token.operation == "add" for token in diff.tokens)
        for diff in diffs
    )
    assert any(
        diff.change_type == "trimmed_for_page_limit"
        and diff.item_id == trimmed_project.id
        for diff in diffs
    )
    assert any(
        diff.change_type == "not_selected"
        and diff.item_id == source.projects.items[2].id
        for diff in diffs
    )


def test_field_page_trim_is_labeled_separately_from_rewrite() -> None:
    source = build_resume()
    selected = MutableResume.model_validate(source.model_dump())
    final = selected.model_copy(deep=True)
    item = final.skills.items[-1]
    removed_skill = item.skills[-1]
    updated_item = item.model_copy(
        update={"skills": item.skills[:-1]},
        deep=True,
    )
    final.skills = final.skills.model_copy(
        update={
            "items": [
                *final.skills.items[:-1],
                updated_item,
            ]
        },
        deep=True,
    )

    diffs = build_resume_diffs(
        source,
        selected,
        final,
        page_trim_actions=[
            PageTrimAction(
                section="skills",
                item_id=item.id,
                field="skills",
                removed=removed_skill,
                reason="Removed the lowest-ranked non-required skill.",
            )
        ],
    )

    skill_diff = next(
        diff
        for diff in diffs
        if diff.item_id == item.id and diff.field == "skills"
    )
    assert skill_diff.change_type == "trimmed_for_page_limit"
    assert skill_diff.original[-1] == removed_skill
    assert skill_diff.final == item.skills[:-1]


def test_empty_placeholder_item_is_not_reported_as_added() -> None:
    source = MutableResume.model_validate(build_resume().model_dump())
    source.summary = source.summary.model_copy(
        update={"items": []},
        deep=True,
    )
    selected = source.model_copy(deep=True)
    final = source.model_copy(deep=True)
    placeholder = ProfessionalSummaryItem(content=None)
    final.summary = final.summary.model_copy(
        update={"items": [placeholder]},
        deep=True,
    )

    diffs = build_resume_diffs(source, selected, final)

    assert all(diff.item_id != placeholder.id for diff in diffs)


def test_nonempty_new_item_is_still_reported_as_added() -> None:
    source = MutableResume.model_validate(build_resume().model_dump())
    source.summary = source.summary.model_copy(
        update={"items": []},
        deep=True,
    )
    selected = source.model_copy(deep=True)
    final = source.model_copy(deep=True)
    summary = ProfessionalSummaryItem(content="Evidence-grounded summary.")
    final.summary = final.summary.model_copy(
        update={"items": [summary]},
        deep=True,
    )

    diffs = build_resume_diffs(source, selected, final)

    added = next(diff for diff in diffs if diff.item_id == summary.id)
    assert added.change_type == "added"
    assert added.final == '{"content": "Evidence-grounded summary."}'
