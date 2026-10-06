from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ChartType = Literal["line", "bar", "horizontal_bar", "scatter"]


@dataclass(frozen=True)
class ChartResult:
    figure: go.Figure | None
    chart_type: ChartType | None
    reason: str | None = None


def create_auto_chart(
    dataframe: pd.DataFrame,
    *,
    question: str = "",
    title: str = "Analysis result",
) -> ChartResult:
    """Select and build a safe Plotly chart from a result DataFrame."""
    if dataframe.empty:
        return ChartResult(None, None, "Visualization not available for an empty result.")

    numeric_columns = [
        str(column)
        for column in dataframe.select_dtypes(include="number").columns
    ]
    non_numeric_columns = [
        str(column)
        for column in dataframe.columns
        if column not in numeric_columns
    ]
    if not numeric_columns:
        return ChartResult(None, None, "Visualization not available for this result.")
    metric_columns = [
        column for column in numeric_columns if "rank" not in column.casefold()
    ]
    metric_column = metric_columns[0] if metric_columns else numeric_columns[0]

    time_column = _find_time_column(dataframe, non_numeric_columns)
    if time_column:
        value_column = metric_column
        chart_data = dataframe[[time_column, value_column]].copy()
        parsed = _parse_time_values(chart_data[time_column])
        chart_data[time_column] = parsed
        chart_data = chart_data.dropna(subset=[time_column]).sort_values(time_column)
        if not chart_data.empty:
            figure = px.line(
                chart_data,
                x=time_column,
                y=value_column,
                markers=True,
                title=title,
                labels={time_column: _label(time_column), value_column: _label(value_column)},
            )
            return ChartResult(figure, "line")

    if non_numeric_columns:
        category_column = non_numeric_columns[0]
        if _is_ranking(question, dataframe, category_column, numeric_columns):
            figure = px.bar(
                dataframe,
                x=metric_column,
                y=category_column,
                orientation="h",
                title=title,
                labels={
                    metric_column: _label(metric_column),
                    category_column: _label(category_column),
                },
            )
            figure.update_layout(yaxis={"categoryorder": "total ascending"})
            return ChartResult(figure, "horizontal_bar")

        figure = px.bar(
            dataframe,
            x=category_column,
            y=metric_column,
            title=title,
            labels={
                category_column: _label(category_column),
                metric_column: _label(metric_column),
            },
        )
        return ChartResult(figure, "bar")

    if len(numeric_columns) >= 2 and len(dataframe) >= 3:
        x_column, y_column = numeric_columns[:2]
        figure = px.scatter(
            dataframe,
            x=x_column,
            y=y_column,
            title=title,
            labels={x_column: _label(x_column), y_column: _label(y_column)},
        )
        return ChartResult(figure, "scatter")

    return ChartResult(None, None, "Visualization not available for this result.")


def _find_time_column(
    dataframe: pd.DataFrame,
    non_numeric_columns: list[str],
) -> str | None:
    date_names = re.compile(r"(date|time|month|quarter|period)", re.IGNORECASE)
    for column in non_numeric_columns:
        series = dataframe[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return column
        if not date_names.search(column):
            continue
        if _parse_time_values(series).notna().sum() >= max(1, int(len(series) * 0.7)):
            return column
    return None


def _parse_time_values(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def _is_ranking(
    question: str,
    dataframe: pd.DataFrame,
    category_column: str,
    numeric_columns: list[str],
) -> bool:
    ranking_question = bool(
        re.search(r"\b(top|rank|highest|lowest|most|least|best|worst)\b", question, re.I)
    )
    return (
        ranking_question
        or "rank" in category_column.casefold()
        or any("rank" in column.casefold() for column in numeric_columns)
        or any("rank" in str(column).casefold() for column in dataframe.columns)
    )


def _label(column: str) -> str:
    return column.replace("_", " ").strip().title()
