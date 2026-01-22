"""
Nigeria Macroeconomic Analytics Dashboard
GDP • CPI • Company PAT Analysis

Author: Ukaba Jimmy
Description:
Interactive Streamlit dashboard that ingests cleaned SQLite data,
performs macroeconomic analytics, visualizations, and generates
PDF investment reports with email delivery.
"""

# =========================================================
# CORE PYTHON & DATA LIBRARIES
# =========================================================
import os
import sqlite3
import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd

# =========================================================
# VISUALIZATION LIBRARIES
# =========================================================
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# STREAMLIT
# =========================================================
import streamlit as st

# =========================================================
# PDF REPORT GENERATION (REPORTLAB)
# =========================================================
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# =========================================================
# EMAIL (SMTP)
# =========================================================
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import logging

# =========================================================
# WARNING & DISPLAY CONFIG
# =========================================================
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# =========================================================
# STREAMLIT APP CONFIG
# =========================================================

# =========================================================
# DATABASE CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "Nigeria_Economy_Performance.db"

GDP_TABLE = "GDP_clean"
CPI_TABLE = "CPI_clean"
PAT_TABLE = "COMPANY_PAT_clean"

# =========================================================
# SECTOR CONFIG
# =========================================================
MAIN_SECTORS = [
    "Agriculture",
    "Manufacturing",
    "Services"
]

# =========================================================
# KPI CONFIG (USED ACROSS TABS)
# =========================================================
KPI_METRICS = {
    "Total GDP (₦)": "₦",
    "Avg GDP Growth (%)": "%",
    "Top GDP Sector": "",
    "GDP Volatility (%)": "%",
    "Avg Inflation Rate (%)": "%",
    "Total PAT (₦)": "₦",
    "Avg PAT Growth (%)": "%"
}


# =========================================================
# EMAIL CONFIG (STREAMLIT SECRETS)
# =========================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = st.secrets.get("SENDER_EMAIL", "")
SENDER_PASSWORD = st.secrets.get("SENDER_PW", "")
DEFAULT_RECIPIENTS = (
    st.secrets.get("RECIPIENTS", "").split(",")
    if st.secrets.get("RECIPIENTS")
    else []
)


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def clean_NEA_dataset(
    sqlite_db_path: str = str(DB_PATH),
    run_tables=("CPI", "COMPANY_PAT", "GDP"),
    save_to_sqlite: bool = False
    ):
    """
    ETL pipeline for NEA Data
    """
    
    logger.info("═" * 80)
    logger.info("NEA DATASET CLEANING PIPELINE")
    logger.info("═" * 80)

    # ------------------- CONFIG ------------------------------------
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    # ------------------- CHECK DATABASE EXISTS ----------------------
    if not Path(sqlite_db_path).exists():
        raise FileNotFoundError(f"SQLite database not found at {sqlite_db_path}")
    conn = sqlite3.connect(sqlite_db_path)

    try:
        # ----------------------EXTRACTION PHASE-----------------------
        logger.info("EXTRACTING DATA FROM DATABASE...")
        logger.info("=" * 80)
        

        
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        available_tables = [row[0] for row in cursor.fetchall()]

        tables_to_load = [t for t in run_tables if t in available_tables]

        if not tables_to_load:
            raise RuntimeError(f"No requested tables found. Available: {available_tables}")

        dataframes = {}
        for table in tables_to_load:
            df = pd.read_sql_query(f"SELECT * FROM [{table}]", conn)  # [] to handle reserved names
            dataframes[table] = df
            logger.info(f"Extracted table {table} with shape {df.shape}")
            

        # ----------------------TRANSFORMATION PHASE-------------------
        logger.info("TRANSFORMING DATA...")
        logger.info("=" * 80)
        
        
        cleaned_tables = {}
        # Example transformation for COMPANY_PAT data
        if "COMPANY_PAT" in dataframes:
            logger.info("Cleaning COMPANY_PAT data...")
            

            df = dataframes["COMPANY_PAT"].copy()
            df.drop(df.tail(3).index, inplace=True)

            # Step 1: create new column names from row 0
            new_columns = ['Company'] + [f'PAT {int(col)}' for col in df.iloc[0, 1:]]

            # Step 2: assign new column names
            df.columns = new_columns

            # Step 3: drop the first row (it was header info)
            df = df.iloc[1:].reset_index(drop=True)
            logger.info("Cleaned COMPANY_PAT data")
            cleaned_tables["COMPANY_PAT"] = df

        if "CPI" in dataframes:
            logger.info("Cleaning CPI data...")
            df1 = dataframes["CPI"].copy().reset_index(drop=True)
            logger.info("Cleaned CPI data")
            # Rename the first column
            df1 = df1.rename(columns={
                "CPI Breakdown for Nigeria": "CPI_Component"
            })

            # Remove ' Dec' from all other columns (year columns)
            df1.columns = [col.replace(" Dec", "") if "Dec" in col else col for col in df1.columns]

            # Save cleaned table
            cleaned_tables["CPI"] = df1


        if "GDP" in dataframes:
            logger.info("Cleaning GDP data...")
            df2 = dataframes["GDP"].copy().reset_index(drop=True)
            # Step 1: Use row 0 as column names
            df2.columns = df2.iloc[0]

            # Step 2: Drop the header row from data
            df2 = df2.iloc[1:].reset_index(drop=True)

            # Step 3: Keep only SECTOR and YEAR columns
            df2 = df2.loc[:, ["SECTOR","GDP SUB SECTOR", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015"]]
            df2["SECTOR"] = df2["SECTOR"].ffill()
                        # Rename columns for SQL-safe usage
            df2 = df2.rename(columns={
                "SECTOR": "Sector",
                "GDP SUB SECTOR": "Sub_sector"
            })
            logger.info("Cleaned GDP data")
            cleaned_tables["GDP"] = df2
                    

        if save_to_sqlite:
            logger.info("Saving cleaned tables back to database...")
            for table_name, df_clean in cleaned_tables.items():
                df_clean.to_sql(table_name + "_clean", conn, if_exists="replace", index=False)
                logger.info(f"Saved {table_name}_clean with shape {df_clean.shape}")
                

        logger.info("ETL pipeline finished.")

        return cleaned_tables
    
    finally:
        conn.close()

# ------------------- RUN THE PIPELINE ------------------

@st.cache_data(show_spinner="Loading cleaned data...")
def load_cleaned_data(db_path):
    return clean_NEA_dataset(
        sqlite_db_path=db_path,
        run_tables=("COMPANY_PAT", "CPI", "GDP"),
        save_to_sqlite=True
    )
cleaned_data = load_cleaned_data(str(DB_PATH))

# =========================================================
# PREPARE PAT DATA (WIDE → LONG) [RUN ONCE]
# =========================================================

df_pat_raw = cleaned_data["COMPANY_PAT"].copy()

df_pat_long = df_pat_raw.melt(
    id_vars=["Company"],
    value_vars=[
        "PAT 2015", "PAT 2016", "PAT 2017", "PAT 2018", "PAT 2019",
        "PAT 2020", "PAT 2021", "PAT 2022", "PAT 2023", "PAT 2024"
    ],
    var_name="Year",
    value_name="PAT_Value"
)

# Extract numeric year
df_pat_long["Year"] = (
    df_pat_long["Year"]
    .str.extract(r"(\d{4})")
    .astype(int)
)

# Clean PAT values
df_pat_long["PAT_Value"] = (
    df_pat_long["PAT_Value"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Sort for time-series operations
df_pat_long = df_pat_long.sort_values(["Company", "Year"]).reset_index(drop=True)

# Compute YoY PAT growth (%)
df_pat_long["pat_growth_pct"] = (
    df_pat_long
    .groupby("Company")["PAT_Value"]
    .pct_change() * 100
)

    
#===========================================================================================================================
# --------------------------------------ANALYTICS AND VISUALIZATION--------------------------------------------


    
def gdp_yoy_growth_analytics(cleaned_data):
    """
    Computes Year-on-Year GDP growth by Sector and Sub-sector
    and returns a transformed dataframe and interactive Plotly chart.
    """

    # -------------------------------
    # Transform GDP data (wide → long)
    # -------------------------------
    df = cleaned_data["GDP"].copy()

    df_long = df.melt(
        id_vars=["Sector", "Sub_sector"],
        value_vars=[
            "2015", "2016", "2017", "2018", "2019",
            "2020", "2021", "2022", "2023", "2024"
        ],
        var_name="Year",
        value_name="GDP_Value"
    )

    # Data type conversions
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["GDP_Value"] = (
        df_long["GDP_Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Sort values
    df_long = df_long.sort_values(
        by=["Sector", "Sub_sector", "Year"]
    )

    # -------------------------------
    # Compute YoY GDP growth (%)
    # -------------------------------
    df_long["yoy_gdp_growth_pct"] = (
        df_long
        .groupby(["Sector", "Sub_sector"])["GDP_Value"]
        .pct_change() * 100
    )

    # -------------------------------
    # Build interactive Plotly chart
    # -------------------------------
    fig = go.Figure()

    sub_sectors = df_long["Sub_sector"].unique()

    for sub in sub_sectors:
        df_sub = df_long[df_long["Sub_sector"] == sub]

        fig.add_trace(
            go.Scatter(
                x=df_sub["Year"],
                y=df_sub["yoy_gdp_growth_pct"],
                mode="lines+markers",
                name=sub,
                visible=True,
                hovertemplate=(
                    "Year: %{x}<br>"
                    "YoY Growth: %{y:.2f}%<br>"
                    "GDP Value: ₦%{customdata[0]:,.2f}<br>"
                    "Sector: %{customdata[1]}"
                ),
                customdata=df_sub[["GDP_Value", "Sector"]].values
            )
        )

    # -------------------------------
    # Dropdown (All + Individual)
    # -------------------------------
    buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": [True] * len(sub_sectors)}]
        )
    ]

    for i, sub in enumerate(sub_sectors):
        visibility = [False] * len(sub_sectors)
        visibility[i] = True

        buttons.append(
            dict(
                label=sub,
                method="update",
                args=[{"visible": visibility}]
            )
        )

    fig.update_layout(
        title="YoY GDP Growth (%) by Sub-sector (2015–2024)",
        xaxis_title="Year",
        yaxis_title="YoY GDP Growth (%)",
        template="plotly_white",
        legend_title="Sub-sector",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top"
            )
        ]
    )

    return df_long, fig


def plot_top_5_subsectors_yoy_growth_2024(df_long):
    """
    Plots Top 5 Sub-sectors by YoY GDP Growth in 2024.
    Returns a Plotly figure.
    """

    # Filter for 2024
    df_2024 = df_long[df_long["Year"] == 2024]

    # Rank sub-sectors by YoY GDP growth
    top_subsectors = (
        df_2024[["Sector", "Sub_sector", "yoy_gdp_growth_pct"]]
        .sort_values("yoy_gdp_growth_pct", ascending=False)
        .head(5)
    )

    fig = px.bar(
        top_subsectors,
        x="yoy_gdp_growth_pct",
        y="Sub_sector",
        color="Sector",
        orientation="h",
        text="yoy_gdp_growth_pct",
        title="Top 5 Sub-sectors by YoY GDP Growth in 2024"
    )

    fig.update_layout(
        xaxis_title="YoY GDP Growth (%)",
        yaxis_title="Sub-sector",
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white",
        height=400
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    return fig

def top_gdp_subsectors_2024_analytics(cleaned_data):
    """
    Identifies top 5 GDP contributing sub-sectors for 2024
    and returns a dataframe and horizontal bar chart.
    """

    # -------------------------------
    # Transform GDP data (wide → long)
    # -------------------------------
    df = cleaned_data["GDP"].copy()

    df_long = df.melt(
        id_vars=["Sector", "Sub_sector"],
        value_vars=[
            "2015", "2016", "2017", "2018", "2019",
            "2020", "2021", "2022", "2023", "2024"
        ],
        var_name="Year",
        value_name="GDP_Value"
    )

    # Data cleaning
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["GDP_Value"] = (
        df_long["GDP_Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # -------------------------------
    # Filter for 2024 & rank top 5
    # -------------------------------
    df_2024 = df_long[df_long["Year"] == 2024]

    top_subsectors_gdp = (
        df_2024[["Sector", "Sub_sector", "GDP_Value"]]
        .sort_values("GDP_Value", ascending=False)
        .head(5)
    )

    # -------------------------------
    # Plot horizontal bar chart
    # -------------------------------
    fig = px.bar(
        top_subsectors_gdp,
        x="GDP_Value",
        y="Sub_sector",
        color="Sector",
        orientation="h",
        text="GDP_Value",
        title="Top 5 Sub-sectors by GDP Contribution in 2024"
    )

    fig.update_layout(
        xaxis_title="GDP Contribution (₦)",
        yaxis_title="Sub-sector",
        yaxis={"categoryorder": "total ascending"},  # largest on top
        template="plotly_white",
        height=450
    )

    fig.update_traces(
        texttemplate="₦%{text:,.2f}",
        textposition="outside"
    )

    return top_subsectors_gdp, fig


def avg_yoy_gdp_growth_analytics(cleaned_data):
    """
    Computes average YoY GDP growth (%) by sector and sub-sector
    for the period 2016–2024 and returns dataframe + plot.
    """

    # -------------------------------
    # Transform GDP data (wide → long)
    # -------------------------------
    df = cleaned_data["GDP"].copy()

    df_long = df.melt(
        id_vars=["Sector", "Sub_sector"],
        value_vars=[
            "2015", "2016", "2017", "2018", "2019",
            "2020", "2021", "2022", "2023", "2024"
        ],
        var_name="Year",
        value_name="GDP_Value"
    )

    # Data cleaning
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["GDP_Value"] = (
        df_long["GDP_Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # -------------------------------
    # Compute YoY GDP growth (%)
    # -------------------------------
    df_long = df_long.sort_values(
        by=["Sector", "Sub_sector", "Year"]
    )

    df_long["yoy_gdp_growth_pct"] = (
        df_long
        .groupby(["Sector", "Sub_sector"])["GDP_Value"]
        .pct_change() * 100
    )

    # -------------------------------
    # Average YoY growth (2016–2024)
    # -------------------------------
    subsector_growth = (
        df_long[df_long["Year"] > 2015]
        .groupby(["Sector", "Sub_sector"])["yoy_gdp_growth_pct"]
        .mean()
        .reset_index(name="avg_yoy_growth_pct")
        .sort_values("avg_yoy_growth_pct", ascending=False)
    )

    # -------------------------------
    # Plot horizontal bar chart
    # -------------------------------
    fig = px.bar(
        subsector_growth,
        x="avg_yoy_growth_pct",
        y="Sub_sector",
        color="Sector",
        orientation="h",
        title="Average YoY GDP Growth by Sector (2016–2024)",
        labels={"avg_yoy_growth_pct": "Average YoY Growth (%)"}
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Average YoY GDP Growth (%)",
        yaxis_title="Sub-sector",
        height=600
    )

    fig.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside"
    )

    return subsector_growth, fig


def gdp_subsector_contribution_analytics(df_long):
    """
    Total GDP contribution by sub-sector (2015–2024)
    """

    # Aggregate total GDP contribution
    subsector_contribution = (
        df_long
        .groupby(["Sector", "Sub_sector"])["GDP_Value"]
        .sum()
        .reset_index(name="total_gdp_2015_2024")
        .sort_values("total_gdp_2015_2024", ascending=False)
    )

    # Create horizontal bar chart
    fig = px.bar(
        subsector_contribution,
        x="total_gdp_2015_2024",
        y="Sub_sector",
        color="Sector",
        orientation="h",
        text="total_gdp_2015_2024",
        title="Total GDP Contribution by Sub-sector (2015–2024)",
        labels={"total_gdp_2015_2024": "Total GDP Contribution"}
    )

    fig.update_layout(
        template="plotly_white",
        yaxis={"categoryorder": "total ascending"},  # largest on top
        height=500
    )

    return subsector_contribution, fig


def gdp_growth_vs_contribution_analytics(
    subsector_growth,
    subsector_contribution
):
    """
    Bubble chart showing sub-sector performance:
    Average YoY GDP Growth vs Total GDP Contribution (2015–2024)
    """

    # Merge growth and contribution data
    bubble_df = subsector_growth.merge(
        subsector_contribution,
        on=["Sector", "Sub_sector"],
        how="inner"
    )

    # Create bubble chart
    fig = px.scatter(
        bubble_df,
        x="avg_yoy_growth_pct",
        y="total_gdp_2015_2024",
        size="total_gdp_2015_2024",
        color="Sector",
        hover_name="Sub_sector",
        size_max=60,
        title="Sub-sector Performance: Growth vs Contribution (2015–2024)",
        labels={
            "avg_yoy_growth_pct": "Average YoY Growth (%)",
            "total_gdp_2015_2024": "Total GDP Contribution"
        }
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Average YoY GDP Growth (%)",
        yaxis_title="Total GDP Contribution",
        height=550
    )

    return bubble_df, fig

def sector_gdp_share_analytics(sector_summary):
    """
    Pie chart showing sectoral share of total GDP (2015–2024)
    """

    fig = px.pie(
        sector_summary,
        values="total_gdp",
        names="Sector",
        title="Sectoral Share of Total GDP (2015–2024)"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    return sector_summary, fig

# ===========================================================================================================================
def plot_yoy_cpi_inflation(df_cpi):
    """
    Computes YoY CPI growth (%) and returns an interactive Plotly figure
    with dropdown filtering by CPI component.
    """

    # Sort by component and year
    df_cpi = df_cpi.sort_values(["CPI_Component", "Year"]).copy()

    # Compute YoY CPI growth (%)
    df_cpi["yoy_Cpi_growth_pct"] = (
        df_cpi
        .groupby("CPI_Component")["CPI_Value"]
        .pct_change() * 100
    )

    fig = go.Figure()

    cpi_components = df_cpi["CPI_Component"].unique()

    # Add a trace for each CPI component
    for comp in cpi_components:
        df_comp = df_cpi[df_cpi["CPI_Component"] == comp]

        fig.add_trace(
            go.Scatter(
                x=df_comp["Year"],
                y=df_comp["yoy_Cpi_growth_pct"],
                mode="lines+markers",
                name=comp,
                visible=True,
                hovertemplate=(
                    "Year: %{x}<br>"
                    "YoY Inflation: %{y:.2f}%<br>"
                    "CPI Value: %{customdata[0]}<extra></extra>"
                ),
                customdata=df_comp[["CPI_Value"]].values
            )
        )

    # Dropdown buttons
    buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": [True] * len(cpi_components)}]
        )
    ]

    for i, comp in enumerate(cpi_components):
        visibility = [False] * len(cpi_components)
        visibility[i] = True

        buttons.append(
            dict(
                label=comp,
                method="update",
                args=[{"visible": visibility}]
            )
        )

    # Layout
    fig.update_layout(
        title="Year-on-Year CPI Inflation Rate (%) by Component (2015–2024)",
        xaxis_title="Year",
        yaxis_title="YoY Inflation Rate (%)",
        template="plotly_white",
        legend_title="CPI Component",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top"
            )
        ]
    )

    return fig


def plot_cumulative_cpi_growth(df_cpi):
    """
    Computes cumulative CPI growth (%) by component (2015–2024)
    and returns a horizontal bar Plotly figure.
    """

    # Prepare cumulative CPI growth data
    cpi_growth = (
        df_cpi
        .sort_values(["CPI_Component", "Year"])
        .groupby("CPI_Component")
        .agg(
            start_cpi=("CPI_Value", "first"),
            end_cpi=("CPI_Value", "last"),
            avg_yoy_inflation=("yoy_Cpi_growth_pct", "mean")
        )
        .reset_index()
    )

    # Calculate cumulative growth %
    cpi_growth["cumulative_growth_pct"] = (
        (cpi_growth["end_cpi"] / cpi_growth["start_cpi"]) - 1
    ) * 100

    # Sort for visualization
    cpi_growth = cpi_growth.sort_values(
        "cumulative_growth_pct",
        ascending=True
    )

    # Build figure
    fig = px.bar(
        cpi_growth,
        x="cumulative_growth_pct",
        y="CPI_Component",
        orientation="h",
        title="CPI Components with Highest Cost Increase (2015–2024)",
        labels={
            "cumulative_growth_pct": "Cumulative CPI Growth (%)",
            "CPI_Component": "CPI Component"
        },
        text=cpi_growth["cumulative_growth_pct"].round(1)
    )

    fig.update_layout(
        template="plotly_white",
        yaxis=dict(title=""),
        xaxis=dict(title="Cumulative CPI Growth (%)")
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "Component: %{y}<br>"
            "Cumulative Growth: %{x:.2f}%<br>"
            "Avg YoY Inflation: %{customdata[0]:.2f}%<extra></extra>"
        ),
        customdata=cpi_growth[["avg_yoy_inflation"]].values
    )

    return fig
#========================================================================================================
# Company performance and sector relationship

def plot_pat_vs_sector_gdp(df_pat_long, df_long):
    """
    Company PAT vs Sector/Sub-sector GDP
    """

    df = df_pat_long.merge(
        df_long,
        on=["Sector", "Sub_sector", "Year"],
        how="left"
    )

    fig = go.Figure()

    for company in df["Company"].unique():
        df_c = df[df["Company"] == company]

        fig.add_trace(
            go.Scatter(
                x=df_c["Year"],
                y=df_c["PAT_Value"],
                mode="lines+markers",
                name=f"{company} PAT",
                hovertemplate="Year: %{x}<br>PAT: ₦%{y:,.0f}<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_c["Year"],
                y=df_c["GDP_Value"],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{df_c['Sub_sector'].iloc[0]} GDP",
                hovertemplate="Year: %{x}<br>GDP: ₦%{y:,.0f}<extra></extra>"
            )
        )

    fig.update_layout(
        title="Company PAT vs Sector/Sub-sector GDP",
        xaxis_title="Year",
        yaxis_title="₦ Value (Millions)",
        template="plotly_white"
    )

    return fig


def plot_pat_vs_cpi(df_pat_long, df_cpi):
    """
    Company PAT vs Overall Inflation
    """

    df = df_pat_long.merge(df_cpi, on="Year", how="left")
    cpi_all = df[df["CPI_Component"] == "All Items"]

    fig = go.Figure()

    for company in df["Company"].unique():
        df_c = df[df["Company"] == company]

        fig.add_trace(
            go.Scatter(
                x=df_c["Year"],
                y=df_c["PAT_Value"],
                mode="lines+markers",
                name=f"{company} PAT"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=cpi_all["Year"],
            y=cpi_all["CPI_Value"],
            mode="lines+markers",
            name="Overall CPI",
            yaxis="y2"
        )
    )

    fig.update_layout(
        title="Company PAT vs Inflation (CPI)",
        xaxis_title="Year",
        yaxis_title="PAT (₦ Millions)",
        yaxis2=dict(
            title="CPI Index",
            overlaying="y",
            side="right"
        ),
        template="plotly_white"
    )

    return fig

import plotly.graph_objects as go

def plot_pat_yoy_growth(df_pat_long):
    """
    Computes Year-on-Year PAT growth (%) for each company
    from an already-long PAT dataframe.
    """

    df = df_pat_long.copy()

    # Ensure correct sorting
    df = df.sort_values(["Company", "Year"])

    # Compute YoY PAT growth (%)
    if "pat_growth_pct" not in df.columns:
        df["pat_growth_pct"] = (
            df
            .groupby("Company")["PAT_Value"]
            .pct_change() * 100
        )

    fig = go.Figure()

    for company in df["Company"].unique():
        df_c = df[df["Company"] == company]

        fig.add_trace(
            go.Scatter(
                x=df_c["Year"],
                y=df_c["pat_growth_pct"],
                mode="lines+markers",
                name=company,
                hovertemplate=(
                    "Company: %{text}<br>"
                    "Year: %{x}<br>"
                    "PAT Growth: %{y:.2f}%<br>"
                    "PAT Value: ₦%{customdata[0]:,.0f}<extra></extra>"
                ),
                text=df_c["Company"],
                customdata=df_c[["PAT_Value"]].values
            )
        )

    fig.update_layout(
        title="Annual PAT Growth (%) by Company",
        xaxis_title="Year",
        yaxis_title="PAT Growth (%)",
        template="plotly_white",
        legend_title="Company"
    )

    return fig



# =========================================================
# COMPANY → SECTOR / SUB-SECTOR MAPPING
# =========================================================

company_sector_map = pd.DataFrame({
    "Company": [
        "Dangote Cement",
        "BUA Cement",
        "MTN Nigeria",
        "Airtel Africa",
        "Nestle Nigeria",
        "Unilever Nigeria",
        "GTCO",
        "Zenith Bank"
    ],
    "Sector": [
        "Manufacturing",
        "Manufacturing",
        "Information & Communication",
        "Information & Communication",
        "Manufacturing",
        "Manufacturing",
        "Financial Services",
        "Financial Services"
    ],
    "Sub_sector": [
        "Cement",
        "Cement",
        "Telecommunications",
        "Telecommunications",
        "Food & Beverages",
        "Food & Beverages",
        "Banking",
        "Banking"
    ]
})

# Merge sector info into PAT long data
df_pat_long = df_pat_long.merge(
    company_sector_map,
    on="Company",
    how="left"
)




def plot_pat_vs_gdp_and_cpi(df_pat_long, df_long, df_cpi):
    """
    Returns:
    fig_pat_gdp  -> Company PAT vs Sector/Sub-sector GDP
    fig_pat_cpi  -> Company PAT vs Overall CPI
    """

    # -------------------------------
    # PAT vs Sector/Sub-sector GDP
    # -------------------------------
    df_pat_gdp = df_pat_long.merge(
        df_long,  # GDP long table
        on=["Sector", "Sub_sector", "Year"],
        how="left"
    )

    fig_pat_gdp = go.Figure()
    companies = df_pat_gdp["Company"].unique()

    for company in companies:
        df_comp = df_pat_gdp[df_pat_gdp["Company"] == company]

        # PAT line
        fig_pat_gdp.add_trace(
            go.Scatter(
                x=df_comp["Year"],
                y=df_comp["PAT_Value"],
                mode="lines+markers",
                name=f"{company} PAT",
                hovertemplate="Year: %{x}<br>PAT: %{y:,.0f}<extra></extra>"
            )
        )

        # Sector GDP line
        sector = df_comp["Sector"].iloc[0]
        sub_sector = df_comp["Sub_sector"].iloc[0]

        fig_pat_gdp.add_trace(
            go.Scatter(
                x=df_comp["Year"],
                y=df_comp["GDP_Value"],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{sector} - {sub_sector} GDP",
                hovertemplate="Year: %{x}<br>GDP: %{y:,.0f}<extra></extra>"
            )
        )

    fig_pat_gdp.update_layout(
        title="Company PAT vs Sector/Sub-sector GDP (2015–2024)",
        xaxis_title="Year",
        yaxis_title="Value (₦ Millions)",
        template="plotly_white",
        legend_title="Legend"
    )

    # -------------------------------
    # PAT vs Overall CPI
    # -------------------------------
    df_pat_cpi = df_pat_long.merge(
        df_cpi,
        on="Year",
        how="left"
    )

    fig_pat_cpi = go.Figure()

    for company in companies:
        df_comp = df_pat_cpi[df_pat_cpi["Company"] == company]

        fig_pat_cpi.add_trace(
            go.Scatter(
                x=df_comp["Year"],
                y=df_comp["PAT_Value"],
                mode="lines+markers",
                name=f"{company} PAT",
                hovertemplate="Year: %{x}<br>PAT: %{y:,.0f}<extra></extra>"
            )
        )

    # CPI (secondary axis)
    cpi_all = df_pat_cpi[df_pat_cpi["CPI_Component"] == "All Items"]

    fig_pat_cpi.add_trace(
        go.Scatter(
            x=cpi_all["Year"],
            y=cpi_all["CPI_Value"],
            mode="lines+markers",
            name="Overall Inflation (CPI)",
            yaxis="y2",
            hovertemplate="Year: %{x}<br>CPI: %{y:.2f}<extra></extra>"
        )
    )

    fig_pat_cpi.update_layout(
        title="Company PAT vs Overall Inflation (2015–2024)",
        xaxis_title="Year",
        yaxis_title="PAT (₦ Millions)",
        yaxis2=dict(
            title="CPI Value",
            overlaying="y",
            side="right"
        ),
        template="plotly_white",
        legend_title="Legend"
    )

    return fig_pat_gdp, fig_pat_cpi


def plot_investment_insight(df_invest):
    """
    Returns a bubble chart for investment decision-making
    """

    fig = px.scatter(
        df_invest,
        x="sector_gdp",
        y="avg_pat",
        size=df_invest["total_pat"].abs(),
        color="Sector",
        hover_name="Company",
        labels={
            "sector_gdp": "Average Sector GDP",
            "avg_pat": "Average PAT",
            "total_pat": "Cumulative PAT"
        },
        title="Investment Insight: Companies by PAT and Sector GDP"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig

def calculate_kpis(df_long, df_cpi, df_pat):
    """
    Calculates key macroeconomic KPIs for the dashboard.

    Parameters:
    - df_long : GDP long-format dataframe
    - df_cpi  : CPI long-format dataframe
    - df_pat  : Company PAT long-format dataframe

    Returns:
    - dict: KPI metrics
    """

    # ---- GDP KPIs ----
    total_gdp = df_long["GDP_Value"].sum()
    avg_gdp_growth = df_long["yoy_gdp_growth_pct"].mean()

    top_sector = (
        df_long
        .groupby("Sector")["GDP_Value"]
        .sum()
        .idxmax()
    )

    gdp_volatility = df_long["yoy_gdp_growth_pct"].std()

    # ---- CPI KPIs ----
    
    df_cpi = df_cpi.sort_values(["CPI_Component", "Year"])

    df_cpi["yoy_inflation_pct"] = (
        df_cpi
        .groupby("CPI_Component")["CPI_Value"]
        .pct_change() * 100
    )
    avg_inflation = (df_cpi[df_cpi["CPI_Component"] == "All Items"]
    ["yoy_inflation_pct"]
    .mean()
    )

    # ---- PAT KPIs ----
    total_pat = df_pat["PAT_Value"].sum()
    avg_pat_growth = df_pat["yoy_pat_growth_pct"].mean()

    # ---- KPI Dictionary ----
    kpi_metrics = {
        "Total GDP (₦)": total_gdp,
        "Avg GDP Growth (%)": avg_gdp_growth,
        "Top GDP Sector": top_sector,
        "GDP Volatility (%)": gdp_volatility,
        "Avg Inflation Rate (%)": avg_inflation,
        "Total PAT (₦)": total_pat,
        "Avg PAT Growth (%)": avg_pat_growth,
    }

    return kpi_metrics

#===========================================================================================================================


# =====================================================================================================================
#==========================Streamlit ui phase=====================================================
# ------------------------------ Streamlit UI phase ---------------------------------------------
# GDP long dataframe
# ================= KPI DATA PREP =================

# GDP analytics
df_long, _ = gdp_yoy_growth_analytics(cleaned_data)

# CPI melt (must exist)
df_cpi = cleaned_data["CPI"].melt(
    id_vars=["CPI_Component"],
    var_name="Year",
    value_name="CPI_Value"
)
df_cpi["Year"] = df_cpi["Year"].astype(int)
df_cpi["CPI_Value"] = pd.to_numeric(df_cpi["CPI_Value"], errors="coerce")

# PAT melt (must exist)
df_pat = cleaned_data["COMPANY_PAT"].melt(
    id_vars=["Company"],
    var_name="Year",
    value_name="PAT_Value"
)
df_pat["Year"] = df_pat["Year"].str.replace("PAT ", "").astype(int)
df_pat["PAT_Value"] = pd.to_numeric(df_pat["PAT_Value"], errors="coerce")

df_pat = df_pat.sort_values(["Company", "Year"])
df_pat["yoy_pat_growth_pct"] = (
    df_pat.groupby("Company")["PAT_Value"].pct_change() * 100
)

# ================= KPI CALC =================
kpi_values = calculate_kpis(df_long, df_cpi, df_pat)

# ------------------------------ Streamlit UI phase ---------------------------------------------
st.set_page_config(
    page_title="Nigeria Economic Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------ App Title & Description ---------------------------------------
st.title("Nigeria Economic Analytics Dashboard")
st.markdown("**GDP • Inflation (CPI) • Company PAT Analysis (2015–2024)**")
st.caption("Author: Ukaba Jimmy • Interactive economic insights & investment analytics")
st.divider()

# ------------------------------ KPI CALCULATION -----------------------------------------------
# Assumes these are already created earlier:
# df_long  -> GDP long dataframe (with yoy_gdp_growth_pct)
# df_cpi   -> CPI long dataframe
# df_pat   -> PAT long dataframe (with yoy_pat_growth_pct)

kpi_values = calculate_kpis(df_long, df_cpi, df_pat)

# ------------------------------ KPI DISPLAY ---------------------------------------------------
st.subheader("📌 Key Economic Indicators")

kpi_cols = st.columns(len(kpi_values))

for col, (kpi_name, kpi_value) in zip(kpi_cols, kpi_values.items()):
    
    unit = KPI_METRICS.get(kpi_name, "")
    
    # Formatting logic
    if isinstance(kpi_value, (int, float)):
        if "₦" in unit:
            display_value = f"₦{kpi_value:,.0f}"
        elif "%" in unit:
            display_value = f"{kpi_value:.2f}%"
        else:
            display_value = f"{kpi_value:,.2f}"
    else:
        display_value = str(kpi_value)

    col.metric(
        label=kpi_name,
        value=display_value
    )

st.divider()

#==================================================================BE BACK

# ================================
# SIDEBAR: DATABASE UPLOAD
# ================================
st.sidebar.header("📂 Data Source")

uploaded_db = st.sidebar.file_uploader(
    "Upload Nigeria Economy SQLite Database (.db)",
    type=["db", "sqlite", "sqlite3"]
)

# Session state guards
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if uploaded_db is not None:
    with st.spinner("🔄 Loading & cleaning database..."):

        # Save uploaded DB to temp path
        temp_db_path = BASE_DIR / "uploaded_temp.db"
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_db.getbuffer())

        # Run ETL pipeline
        cleaned_data = clean_NEA_dataset(
            sqlite_db_path=str(temp_db_path),
            run_tables=("COMPANY_PAT", "CPI", "GDP"),
            save_to_sqlite=True
        )

        # Validation
        if not cleaned_data or any(df.empty for df in cleaned_data.values()):
            st.sidebar.error("❌ ETL failed or returned empty tables.")
        else:
            st.session_state.cleaned_data = cleaned_data
            st.session_state.data_loaded = True
            st.sidebar.success("✅ Database cleaned & loaded successfully")

# Stop app if no data
if not st.session_state.data_loaded:
    st.info("👈 Upload a Nigeria Economy SQLite database from the sidebar to begin.")
    st.stop()

# Bring cleaned data into app scope
cleaned_data = st.session_state.cleaned_data


# ================================
# DATA PREVIEW (CLEANED TABLES)
# ================================
st.subheader("🧾 Cleaned Data Preview")

tab1, tab2, tab3 = st.tabs(["📊 GDP", "📈 CPI", "🏦 Company PAT"])

with tab1:
    st.markdown("**GDP (Top 5 Rows)**")
    st.dataframe(cleaned_data["GDP"].head(), use_container_width=True)

with tab2:
    st.markdown("**CPI (Top 5 Rows)**")
    st.dataframe(cleaned_data["CPI"].head(), use_container_width=True)

with tab3:
    st.markdown("**Company PAT (Top 5 Rows)**")
    st.dataframe(cleaned_data["COMPANY_PAT"].head(), use_container_width=True)

st.divider()
# ================================
# SIDEBAR: SECTOR FILTER
# ================================
st.sidebar.header("🎛️ Filters")

# Extract sectors dynamically from GDP table
available_sectors = (
    cleaned_data["GDP"]["Sector"]
    .dropna()
    .unique()
    .tolist()
)

# Add "All Sectors" option
sector_options = ["All Sectors"] + sorted(available_sectors)

selected_sector = st.sidebar.selectbox(
    "Select Economic Sector",
    sector_options
)
# ================================
# APPLY SECTOR FILTER
# ================================
if selected_sector == "All Sectors":
    gdp_filtered = cleaned_data["GDP"].copy()
else:
    gdp_filtered = cleaned_data["GDP"][
        cleaned_data["GDP"]["Sector"] == selected_sector
    ].copy()

# ================================
# SIDEBAR: SECTOR FILTER
# ================================
st.sidebar.header("🎯 Sector Filter")

available_sectors = sorted(cleaned_data["GDP"]["Sector"].unique())

selected_sector = st.sidebar.selectbox(
    "Select Sector",
    options=["All"] + available_sectors
)

# ================================
# APPLY SECTOR FILTER
# ================================
filtered_cleaned_data = cleaned_data.copy()

if selected_sector != "All":
    filtered_cleaned_data["GDP"] = cleaned_data["GDP"][
        cleaned_data["GDP"]["Sector"] == selected_sector
    ]


# ================================
# MAIN ANALYTICS TABS
# ================================
tab_gdp, tab_cpi, tab_relationships, tab_insights = st.tabs(
    ["📊 GDP Analysis", "📈 Inflation (CPI)", "🔗 Economic Relationships", "💡 Insights & Recommendations"]
)
with tab_gdp:
    st.subheader("📊 Gross Domestic Product (GDP) Performance")

    st.markdown("""
    **Overview:**  
    This section analyzes Nigeria’s GDP performance across sectors and sub-sectors.
    All visuals dynamically respond to the selected sector in the sidebar.
    """)

    # ================================
    # YoY GDP Growth
    # ================================
    df_long, fig_gdp_yoy = gdp_yoy_growth_analytics(filtered_cleaned_data)
    st.plotly_chart(fig_gdp_yoy, use_container_width=True)

    # ================================
    # Top 5 Fastest Growing Sub-sectors (2024)
    # ================================
    st.markdown("### 🚀 Top 5 Fastest Growing Sub-sectors (2024)")
    fig_top_yoy = plot_top_5_subsectors_yoy_growth_2024(df_long)
    st.plotly_chart(fig_top_yoy, use_container_width=True)

    # ================================
    # Top GDP Contributors (2024)
    # ================================
    st.markdown("### 🏭 Top 5 GDP-Contributing Sub-sectors (2024)")
    top_sub_gdp, fig_top_gdp = top_gdp_subsectors_2024_analytics(filtered_cleaned_data)
    st.plotly_chart(fig_top_gdp, use_container_width=True)

    # ================================
    # Average YoY GDP Growth
    # ================================
    st.markdown("### 📈 Average GDP Growth by Sub-sector (2016–2024)")
    subsector_growth, fig_avg_growth = avg_yoy_gdp_growth_analytics(filtered_cleaned_data)
    st.plotly_chart(fig_avg_growth, use_container_width=True)

    # ================================
    # Total GDP Contribution (2015–2024)
    # ================================
    st.markdown("### 🧮 Total GDP Contribution by Sub-sector (2015–2024)")
    subsector_contribution, fig_contribution = gdp_subsector_contribution_analytics(df_long)
    st.plotly_chart(fig_contribution, use_container_width=True)

    # ================================
    # Growth vs Contribution (Bubble)
    # ================================
    st.markdown("### 🫧 Growth vs Contribution Analysis")
    bubble_df, fig_bubble = gdp_growth_vs_contribution_analytics(
        subsector_growth,
        subsector_contribution
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ================================
    # SECTORAL GDP SHARE (PIE CHART)
    # ================================
    st.markdown("### 🧩 Sectoral Share of Total GDP (2015–2024)")

    sector_summary = (
        df_long
        .groupby("Sector")["GDP_Value"]
        .sum()
        .reset_index(name="total_gdp")
    )

    sector_summary, fig_sector_pie = sector_gdp_share_analytics(sector_summary)
    st.plotly_chart(fig_sector_pie, use_container_width=True)

    # ================================
    # INSIGHT
    # ================================
    st.info("""
    **Policy Insight:**  
    - High-growth, low-contribution sub-sectors signal emerging opportunities  
    - Large but slow-growing sectors may require productivity reforms  
    - Sectoral GDP dominance shifts highlight structural economic changes
    """)
# ================================
# TAB 2: CPI & INFLATION ANALYSIS
# ================================
with tab_cpi:

    st.subheader("📈 Inflation & Consumer Price Index (CPI) Trends")

    st.markdown("""
    **Overview:**  
    This section analyzes Nigeria’s inflation dynamics across CPI components
    from 2015–2024, highlighting both short-term inflation volatility and
    long-term cost accumulation affecting households and businesses.
    """)

    # ------------------------------------------------
    # SAFETY: Ensure YoY CPI is computed ONCE
    # ------------------------------------------------
    df_cpi = df_cpi.sort_values(["CPI_Component", "Year"]).copy()

    if "yoy_Cpi_growth_pct" not in df_cpi.columns:
        df_cpi["yoy_Cpi_growth_pct"] = (
            df_cpi
            .groupby("CPI_Component")["CPI_Value"]
            .pct_change() * 100
        )

    # ------------------------------------------------
    # CHART 1: YoY CPI Inflation
    # ------------------------------------------------
    st.markdown("### 📊 Year-on-Year CPI Inflation by Component")

    fig_cpi_yoy = plot_yoy_cpi_inflation(df_cpi) 
    
    st.plotly_chart(fig_cpi_yoy,use_container_width=True,
    key="cpi_yoy_chart"
    )

    # ------------------------------------------------
    # CHART 2: Cumulative CPI Growth
    # ------------------------------------------------
    st.markdown("### 🔺 Cumulative CPI Cost Increase (2015–2024)")

    fig_cpi_cum = plot_cumulative_cpi_growth(df_cpi)
    st.plotly_chart(fig_cpi_cum, use_container_width=True)

    # ------------------------------------------------
    # RISK INTERPRETATION
    # ------------------------------------------------
    st.subheader("""
    **Inflation Risk Insight:**  
    Part 2: CPI & Inflation Analysis — Interpretation & Insights
## 1 Year-on-Year Inflation Rate by CPI Component

Using the CPI data from 2015 to 2024, I computed the year-on-year (YoY) inflation rate for each CPI component by measuring the percentage change in CPI values relative to the previous year.

The results show that inflation in Nigeria is uneven across components. Essential consumption items—particularly Food & Non-Alcoholic Beverages, Transport, and Housing & Utilities—consistently recorded higher YoY inflation, often exceeding 12–18% in peak inflation years. In contrast, components such as Education, Recreation, and Communication exhibited comparatively lower and more stable inflation rates.

This pattern indicates that inflationary pressure is concentrated in non-discretionary spending categories, which households cannot easily substitute or avoid.

## 2 CPI Components with the Highest Cost Increase (2015–2024)

Over the 10-year period, cumulative CPI analysis shows that:

Food & Non-Alcoholic Beverages experienced the highest cumulative cost increase, rising by well over 200% between 2015 and 2024.

Transport followed closely, driven by fuel price adjustments, exchange-rate depreciation, and logistics costs.

Housing, Water & Energy also recorded persistent cost increases due to electricity tariff changes and rising construction input costs.

These components account for a large share of household expenditure, meaning their inflationary impact is both deep and widespread.

## 3 CPI vs GDP: What Happens When Prices Rise?

Comparing CPI trends with GDP growth and contribution reveals an important structural insight.

Sectors aligned with high-inflation CPI components—such as Agriculture (Food) and Transport Services—often continued to expand in GDP contribution, even during inflationary periods. This suggests that demand for these goods and services remains strong regardless of rising prices.
 
However, higher prices did not always translate into higher real growth. In many cases, GDP growth was driven by price effects rather than volume expansion, meaning consumers were paying more but not necessarily consuming more.

## 4 Are Rising Prices Hurting or Helping Certain Sectors?

Rising prices appear to benefit producers in essential sectors in nominal terms, as revenues increase alongside prices. For example:

Food-related sub-sectors maintained positive GDP growth despite high inflation.

Transport services continued to expand due to inelastic demand.

On the other hand, inflation hurts consumption-driven and discretionary sectors, where higher prices suppress demand. Manufacturing sub-sectors reliant on imported inputs also face cost-push inflation, reducing profit margins and competitiveness.

5Sector Resilience Despite Inflation
Some sectors demonstrate clear resilience in the face of rising prices:
Agriculture remains resilient due to constant food demand.
Services linked to essential needs continue to grow despite cost pressures.
Certain financial and telecom-related services show stable performance due to strong demand elasticity.
These sectors are better positioned to pass rising costs to consumers without significant demand loss.

📌 Key Takeaway

Inflation in Nigeria over the past decade has been structural rather than temporary, with essential CPI components driving long-term cost increases. While rising prices support nominal GDP in some sectors, they simultaneously erode purchasing power and constrain real economic welfare. The most resilient sectors are those tied to essential consumption, while discretionary and import-dependent sectors remain most vulnerable.
    """)

    st.divider()

# ================================
# TAB 3: COMPANY PAT & MACRO LINKAGES
# ================================


with tab_relationships:

    st.subheader("💼 Company Performance, Profitability & Macroeconomic Linkages")

    st.markdown("""
    **Overview:**  
    This section evaluates company-level profitability using Profit After Tax (PAT),
    examining growth trends over time and how firm performance interacts with
    sectoral GDP growth and overall inflation dynamics.
    """)

    st.divider()

    # ------------------------------------------------
    # CHART 1: YoY PAT Growth
    # ------------------------------------------------
    st.markdown("### 📈 Year-on-Year PAT Growth by Company")

    fig_pat_yoy = plot_pat_yoy_growth(df_pat_long)
    st.plotly_chart(
        fig_pat_yoy,
        use_container_width=True,
        key="pat_yoy_unique"
    )

    st.info("""
    **Interpretation:**  
    Companies with relatively smooth upward PAT trends demonstrate financial
    resilience, while sharp year-to-year swings indicate exposure to macroeconomic
    shocks, inflation pressures, or operational instability.
    """)

    st.divider()

    # ------------------------------------------------
    # CHART 2: PAT vs Sector GDP
    # ------------------------------------------------
    st.markdown("### 🏭 PAT vs Sector / Sub-sector GDP")

    fig_pat_gdp = plot_pat_vs_sector_gdp(df_pat_long, df_long)

    st.plotly_chart(
        fig_pat_gdp,
        use_container_width=True,
        key="pat_vs_gdp_unique"
    )

    st.divider()

    # ------------------------------------------------
    # CHART 3: PAT vs Inflation
    # ------------------------------------------------
    st.markdown("### 📉 PAT vs Overall Inflation (CPI)")

    fig_pat_cpi = plot_pat_vs_cpi(df_pat_long, df_cpi)

    st.plotly_chart(
        fig_pat_cpi,
        use_container_width=True,
        key="pat_vs_cpi_unique"
    )

    st.warning("""
    **Key Economic Insight:**  
    If PAT growth aligns closely with sector GDP growth, profitability is supported
    by real economic expansion. If PAT rises mainly during high inflation periods,
    earnings growth may reflect pricing effects rather than productivity gains.
    """)

with tab_insights:
    st.subheader("📊 Inflation, Corporate Performance & Investment Insights")

    # ======================================================
    # CPI & INFLATION DYNAMICS
    # ======================================================
    st.markdown("## 🔹 CPI & Inflation Dynamics")

    st.markdown("""
    **Key Question:**  
    👉 *Where is inflation coming from, and who is most affected by rising prices?*
    """)

    st.markdown("### 📈 Inflation Trend Overview")

    st.markdown("""
    Inflation in Nigeria has been **persistent and structural**, rather than temporary.

    Year-on-year CPI inflation accelerated notably in the **post-2020 period**, driven primarily by:
    - **Food price inflation**
    - **Energy and transport costs**
    - **Exchange-rate depreciation**
    """)

    st.markdown("### 🧺 Component-Level Insights")

    st.markdown("""
    - **Food-related CPI components** recorded the **highest cumulative price increases** over the decade.
      - This disproportionately impacts **low- and middle-income households**.
      - Food inflation also feeds directly into **wage pressure and social instability**.

    - **Transport and energy-linked components** showed **sharp spikes rather than smooth trends**.
      - These spikes align closely with **fuel subsidy changes** and **global oil price shocks**.

    - **Core inflation components (non-food)** were relatively more stable but still trended upward,
      indicating **broad-based inflationary pressure** across the economy.
    """)

    st.markdown("### 🔗 CPI vs GDP Comparison")

    st.markdown("""
    When CPI trends are compared with GDP growth:
    - Periods of rising GDP did **not always coincide with real economic improvement**.
    - In several years, **nominal GDP growth masked declining real purchasing power**.
    """)

    st.info("""
    **Key Insight:**  
    Inflation in Nigeria has increasingly become **cost-driven rather than demand-driven**,
    limiting the effectiveness of traditional monetary tightening alone.
    """)

    st.divider()

    # ======================================================
    # COMPANY PERFORMANCE & PAT
    # ======================================================
    st.markdown("## 🔹 Company Performance, PAT & Macroeconomic Linkages")

    st.markdown("""
    **Key Question:**  
    👉 *Are companies benefiting from economic growth, or merely surviving inflation?*
    """)

    st.markdown("### 💼 PAT Growth Patterns")

    st.markdown("""
    Company-level **Profit After Tax (PAT)** showed **significant divergence** across firms:
    - Some companies maintained **steady and consistent profitability growth**
    - Others experienced **sharp volatility**, especially during periods of high inflation
    """)

    st.markdown("### 🏭 PAT vs Sector GDP")

    st.markdown("""
    - Companies operating in **high-growth service sub-sectors** generally saw PAT move
      **in line with sector GDP**, indicating **real, demand-driven profitability**.

    - In **manufacturing-related firms**:
      - PAT growth was often **disconnected from sector GDP**
      - Rising input and energy costs **compressed margins**, despite sector output growth
    """)

    st.markdown("### 📉 PAT vs Inflation")

    st.markdown("""
    - Several firms reported **PAT increases during high-inflation periods**.
      - This suggests **price pass-through**, not productivity improvements.
    - Firms without strong pricing power showed **margin erosion** as costs outpaced revenues.
    """)

    st.markdown("### 💡 Investment Insight (Bubble Chart Interpretation)")

    st.markdown("""
    - **Top-right quadrant firms** (high PAT + strong sector GDP) represent
      **sustainable investment opportunities**.
    - **Large bubbles** signal **long-term earnings consistency**, not short-term spikes.
    - Firms in strong sectors but with weak PAT performance point to
      **execution or cost-structure challenges**, not sector weakness.
    """)

    st.divider()

    # ======================================================
    # SYNTHESIS
    # ======================================================
    st.markdown("## 🧠 Synthesis: The Big Picture")

    st.markdown("""
    Nigeria’s economy is growing, but **not evenly**.  
    Profits are rising, but **not always productively**.  
    Inflation is high, but **not neutral in its impact**.

    Growth is increasingly driven by **services and pricing power**,  
    rather than **industrial efficiency or productivity gains**.
    """)

    st.divider()

    # ======================================================
    # RECOMMENDATIONS
    # ======================================================
    st.markdown("## ✅ Recommendations")

    st.markdown("### 🏛️ For Policymakers")

    st.markdown("""
    - **Target food inflation aggressively**
      - Invest in agricultural storage, logistics, and processing
      - Reduce post-harvest losses rather than relying on imports

    - **Shift from consumption-led to productivity-led growth**
      - Prioritise power, transport, and industrial inputs
      - Support manufacturing firms beyond credit — focus on **cost reduction**

    - **Align monetary and fiscal policy**
      - Inflation driven by supply shocks requires **structural solutions**,
        not interest rate hikes alone
    """)

    st.markdown("### 💰 For Investors")

    st.markdown("""
    - Prioritise firms with **PAT growth aligned to sector GDP**
      - This signals **real demand**, not inflation-driven profits

    - Be cautious of firms growing profits **only during high inflation**
      - These gains may reverse once price pressures ease

    - Favour **consistency over volatility**
      - Large cumulative PAT with low volatility indicates resilience
    """)

    st.markdown("### 🚀 Sectors for Further Investment Focus")

    st.markdown("""
    **✅ High Priority**
    - Telecommunications
    - Financial Services
    - Digital-enabled Services
    - Consumer Staples with strong pricing power

    **⚠️ Selective / Conditional**
    - Manufacturing (only firms with energy efficiency & local sourcing)
    - Food processing (rather than raw production)

    **❌ Higher Risk**
    - Cost-heavy, import-dependent manufacturing
    - Firms exposed to FX risk without natural hedges
    """)
