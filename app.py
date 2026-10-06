from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Sequence
from uuid import uuid4

import duckdb
import streamlit as st

from src.analytics.analysis_agent import AnalysisAgent, AnalysisResult
from src.ui.charts import create_auto_chart
from src.ui.components import (
    initialize_session_state,
    render_chat_history,
    render_data_overview,
    render_previews,
    render_schema,
)
from src.ui.data_loader import UploadDataError, load_uploaded_files
from src.utils.config import BASE_DIR

SESSION_DATABASE_DIR = BASE_DIR / "data" / "processed" / "ui_sessions"
MAX_HISTORY_MESSAGES = 20
MAX_DISPLAY_ROWS = 100
MAX_CHART_ROWS = 500


def main() -> None:
    st.set_page_config(
        page_title="Business Analytics Copilot",
        page_icon=":material/query_stats:",
        layout="wide",
    )
    initialize_session_state()
    st.title("Business Analytics Copilot")
    st.caption("Ask business questions and explore the evidence behind each answer.")

    with st.sidebar:
        st.header("Your data")
        uploaded_files = st.file_uploader(
            "Upload CSV or XLSX files",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            key=f"business_files_{st.session_state.upload_widget_version}",
            help="Each file is loaded as a table. Similar file names are disambiguated automatically.",
        )
        if st.button(
            "Load data",
            type="primary",
            disabled=not uploaded_files,
            icon=":material/upload_file:",
        ):
            _load_uploaded_data(uploaded_files)

        if st.session_state.database_path:
            st.divider()
            if st.button("Clear data and reset session", icon=":material/delete_sweep:"):
                _reset_session()

        if st.session_state.uploaded_tables:
            st.divider()
            render_data_overview(st.session_state.uploaded_tables)
            render_schema(st.session_state.uploaded_tables)
        else:
            st.info("Upload one or more CSV or XLSX files to begin.")

    if st.session_state.uploaded_tables:
        render_previews(st.session_state.uploaded_tables)
    else:
        st.info("Load business data from the sidebar to start an analysis.")

    render_chat_history(st.session_state.question_history)
    question = st.chat_input(
        "Ask a business question about the loaded data",
        key=f"business_question_{st.session_state.chat_widget_version}",
        submit_mode="disable",
    )
    if question is not None:
        _analyze_question(question)

    latest_result = st.session_state.latest_result
    if latest_result is not None:
        _render_analysis(latest_result)


def _load_uploaded_data(uploaded_files: Sequence[BinaryIO]) -> None:
    session_id = st.session_state.copilot_session_id
    candidate_path = _new_database_path(session_id)
    try:
        tables = load_uploaded_files(uploaded_files, candidate_path)
    except (
        UploadDataError,
        OSError,
        ValueError,
        duckdb.Error,
    ) as exc:
        _remove_session_database(candidate_path, session_id)
        st.error(f"Data could not be loaded: {exc}")
        return

    previous_path = st.session_state.database_path
    st.session_state.database_path = str(candidate_path)
    st.session_state.uploaded_tables = tables
    st.session_state.question_history = []
    st.session_state.latest_result = None
    st.session_state.upload_widget_version += 1
    st.session_state.chat_widget_version += 1
    if previous_path:
        try:
            _remove_session_database(Path(previous_path), session_id)
        except OSError as exc:
            st.warning(f"The previous session database could not be removed: {exc}")
    st.success(f"Loaded {len(tables)} table(s) into your session database.")
    st.rerun()


def _analyze_question(question: str) -> None:
    question = question.strip()
    if not question:
        st.warning("Enter a business question before analyzing.")
        return
    if not st.session_state.database_path:
        st.warning("Upload and load business data before asking a question.")
        return

    st.session_state.question_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Planning analysis and checking the data…"):
            result = AnalysisAgent(
                db_path=st.session_state.database_path,
                debug=False,
            ).answer_question(
                _build_analysis_question(
                    question,
                    st.session_state.question_history[:-1],
                )
            )
        if result.answer:
            st.markdown(_markdown_safe_answer(result.answer))
        if result.success:
            st.caption(
                "Answer checked against query evidence."
                if result.answer_verified
                else "Answer evidence could not be verified."
            )
        else:
            st.error(
                result.error
                or "The analysis could not be completed with the available data."
            )

    assistant_content = result.answer or (
        f"The analysis could not be completed: {result.error or 'No answer was returned.'}"
    )
    st.session_state.question_history.append(
        {
            "role": "assistant",
            "content": assistant_content,
            "verification": result.answer_verified,
        }
    )
    st.session_state.question_history = st.session_state.question_history[
        -MAX_HISTORY_MESSAGES:
    ]
    st.session_state.latest_result = result


def _render_analysis(result: AnalysisResult) -> None:
    st.divider()
    st.header("Supporting analysis")
    if result.plan is not None:
        st.caption(
            f"Plan: {result.plan.intent.replace('_', ' ')}"
            + (f" · Metric: {result.plan.metric}" if result.plan.metric else "")
        )

    successful_steps = [
        step for step in result.steps if step.error is None and not step.data.empty
    ]
    if successful_steps:
        chart_step = successful_steps[-1]
        st.subheader(chart_step.goal)
        chart_title = chart_step.goal.strip().rstrip(".")
        try:
            chart = create_auto_chart(
                chart_step.data.head(MAX_CHART_ROWS),
                question=result.question,
                title=chart_title,
            )
        except (KeyError, TypeError, ValueError) as exc:
            st.info(f"Visualization not available for this result: {exc}")
        else:
            if chart.figure is None:
                st.info(chart.reason or "Visualization not available for this result.")
            else:
                if len(chart_step.data) > MAX_CHART_ROWS:
                    st.caption(
                        f"Chart uses the first {MAX_CHART_ROWS} rows of "
                        f"{len(chart_step.data):,} returned rows."
                    )
                st.plotly_chart(
                    chart.figure,
                    alt=f"{chart_title}, based on the query results",
                    width="stretch",
                )

    with st.expander("Supporting query results", expanded=True):
        for step in result.steps:
            st.markdown(f"**Step {step.number}: {step.goal}**")
            if step.error:
                st.error(f"Step failed: {step.error}")
                continue
            st.caption(
                f"{len(step.data):,} row(s) returned · Validation {step.validation}"
            )
            if step.data.empty:
                st.info("This query returned no rows.")
            else:
                st.dataframe(
                    step.data.head(MAX_DISPLAY_ROWS),
                    hide_index=True,
                    alt=f"Results for analysis step {step.number}: {step.goal}",
                )
                if len(step.data) > MAX_DISPLAY_ROWS:
                    st.caption(
                        f"Showing the first {MAX_DISPLAY_ROWS} of {len(step.data):,} rows."
                    )

    with st.expander("Analysis trace", expanded=False):
        if result.plan is not None:
            st.markdown("**Plan**")
            st.json(result.plan.as_dict())
        for step in result.steps:
            with st.container(border=True):
                st.markdown(f"**Step {step.number} · {step.goal}**")
                if step.sql:
                    st.code(step.sql, language="sql")
                st.write(f"Validation: {step.validation}")
                execution_status = (
                    "NOT RUN"
                    if step.validation != "PASSED"
                    else "FAILED"
                    if step.error
                    else "SUCCESS"
                )
                st.write(f"Execution: {execution_status}")
                st.write(f"Rows returned: {len(step.data):,}")
                if step.error:
                    st.caption(step.error)

    with st.expander("SQL queries used", expanded=False):
        for step in result.steps:
            st.markdown(f"**Step {step.number}: {step.goal}**")
            if step.sql:
                st.code(step.sql, language="sql")
            else:
                st.caption("No SQL was generated for this step.")


def _new_database_path(session_id: str) -> Path:
    SESSION_DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DATABASE_DIR / f"{session_id}_{uuid4().hex}.duckdb"


def _build_analysis_question(
    question: str,
    history: list[dict[str, object]],
) -> str:
    if len(history) < 2:
        return question
    previous_user, previous_assistant = history[-2:]
    if previous_user.get("role") != "user" or previous_assistant.get("role") != "assistant":
        return question

    context_lines = [
        "Conversation context is only for interpreting this follow-up. "
        "Run fresh analysis against the active database, and ground every factual "
        "claim in the new query results.",
        f"Previous question: {previous_user.get('content', '')}",
    ]
    if previous_assistant.get("verification") is True:
        context_lines.append(
            f"Previous evidence-verified answer: {previous_assistant.get('content', '')}"
        )
    else:
        context_lines.append(
            "The previous response was not evidence-verified; do not rely on its claims."
        )
    context_lines.append(f"Current follow-up question: {question}")
    return "\n".join(context_lines)


def _markdown_safe_answer(answer: str) -> str:
    return answer.replace("$", "&#36;")


def _remove_session_database(path: Path, session_id: str) -> None:
    resolved_path = path.resolve()
    if (
        resolved_path.parent != SESSION_DATABASE_DIR.resolve()
        or not resolved_path.name.startswith(f"{session_id}_")
        or resolved_path.suffix != ".duckdb"
    ):
        raise ValueError("Refusing to remove a database outside this Streamlit session.")
    resolved_path.unlink(missing_ok=True)
    resolved_path.with_name(f"{resolved_path.name}.wal").unlink(missing_ok=True)


def _reset_session() -> None:
    database_path = Path(st.session_state.database_path)
    try:
        _remove_session_database(database_path, st.session_state.copilot_session_id)
    except OSError as exc:
        st.error(f"The session database could not be removed: {exc}")
        return
    st.session_state.database_path = None
    st.session_state.uploaded_tables = []
    st.session_state.question_history = []
    st.session_state.latest_result = None
    st.session_state.upload_widget_version += 1
    st.session_state.chat_widget_version += 1
    st.rerun()


if __name__ == "__main__":
    main()
