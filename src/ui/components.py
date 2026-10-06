from __future__ import annotations

from typing import Any
from uuid import uuid4

import streamlit as st

from src.data.schema import get_expected_relationships


def initialize_session_state() -> None:
    """Set up session-local database, upload, and conversation state."""
    st.session_state.setdefault("copilot_session_id", uuid4().hex)
    st.session_state.setdefault("database_path", None)
    st.session_state.setdefault("uploaded_tables", [])
    st.session_state.setdefault("question_history", [])
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("upload_widget_version", 0)
    st.session_state.setdefault("chat_widget_version", 0)


def render_data_overview(tables: list[dict[str, Any]]) -> None:
    st.subheader("Data overview")
    total_rows = sum(int(table["row_count"]) for table in tables)
    total_columns = sum(int(table["column_count"]) for table in tables)
    st.caption(
        f"{len(tables)} table(s) · {total_rows:,} rows · {total_columns:,} columns"
    )

    for table in tables:
        st.markdown(f"**{table['table_name']}**")
        st.caption(
            f"{table['source_name']} · {int(table['row_count']):,} rows · "
            f"{int(table['column_count'])} columns"
        )


def render_schema(tables: list[dict[str, Any]]) -> None:
    with st.expander("Database schema", expanded=False):
        for table in tables:
            st.markdown(f"**{table['table_name']}**")
            st.dataframe(
                table["schema"],
                hide_index=True,
                alt=f"Columns and database types for {table['table_name']}",
            )

        relationships = _available_relationships(tables)
        st.markdown("**Known relationships**")
        if relationships:
            for source, target in relationships.items():
                st.write(f"`{source}` → `{target}`")
        else:
            st.caption("No matching known relationships were found in the loaded schema.")


def render_previews(tables: list[dict[str, Any]]) -> None:
    with st.expander("Sample rows", expanded=False):
        for table in tables:
            st.markdown(f"**{table['table_name']}**")
            st.dataframe(
                table["preview"],
                hide_index=True,
                alt=f"First 10 sample rows from {table['table_name']}",
            )
            for warning in table["warnings"]:
                st.caption(f"Data quality note: {warning}")


def render_chat_history(history: list[dict[str, str]]) -> None:
    for item in history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("verification") is not None:
                label = (
                    "Evidence verified"
                    if item["verification"]
                    else "Evidence verification failed"
                )
                st.caption(label)


def _available_relationships(tables: list[dict[str, Any]]) -> dict[str, str]:
    available = {
        str(table["table_name"]): {
            str(column["column_name"]) for column in table["schema"]
        }
        for table in tables
    }
    return {
        source: target
        for source, target in get_expected_relationships().items()
        if _reference_exists(source, available) and _reference_exists(target, available)
    }


def _reference_exists(reference: str, tables: dict[str, set[str]]) -> bool:
    table, separator, column = reference.partition(".")
    return bool(separator and column in tables.get(table, set()))
