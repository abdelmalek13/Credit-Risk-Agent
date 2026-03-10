"""Plotly chart helper utilities for the analytics agent."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLOR_PALETTE = px.colors.qualitative.Set2


def default_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply a consistent theme to any plotly figure."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white",
        font=dict(size=12),
        margin=dict(l=60, r=40, t=60, b=60),
        height=500,
    )
    return fig


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    orientation: str = "v",
    color: str | None = None,
    text: str | None = None,
) -> go.Figure:
    """Create a styled bar chart."""
    fig = px.bar(
        df, x=x, y=y, orientation=orientation, color=color, text=text,
        color_discrete_sequence=COLOR_PALETTE,
    )
    if text:
        fig.update_traces(textposition="outside")
    return default_layout(fig, title)


def histogram(
    df: pd.DataFrame,
    x: str,
    title: str = "",
    nbins: int = 50,
    color: str | None = None,
) -> go.Figure:
    """Create a styled histogram."""
    fig = px.histogram(
        df, x=x, nbins=nbins, color=color,
        color_discrete_sequence=COLOR_PALETTE,
    )
    return default_layout(fig, title)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: str | None = None,
) -> go.Figure:
    """Create a styled scatter plot."""
    fig = px.scatter(
        df, x=x, y=y, color=color,
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.6,
    )
    return default_layout(fig, title)


def heatmap(corr_matrix: pd.DataFrame, title: str = "") -> go.Figure:
    """Create a correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    return default_layout(fig, title)


def pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str = "",
) -> go.Figure:
    """Create a styled pie chart."""
    fig = px.pie(
        df, names=names, values=values,
        color_discrete_sequence=COLOR_PALETTE,
    )
    return default_layout(fig, title)
