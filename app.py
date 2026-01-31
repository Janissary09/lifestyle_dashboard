# ============================================================
# 1. IMPORTS
# ============================================================

from dash import Dash, dcc, html, Input, Output, State, callback, callback_context, no_update
import dash
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ============================================================
# 2. DATA LOAD
# ============================================================

df = pd.read_csv("health_lifestyle_dataset.csv") 

# ============================================================
# 3. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# 3.1 Gender Mapping
# ------------------------------------------------------------
def map_gender(g):
    if isinstance(g, str):
        g_lower = g.lower()
        if g_lower.startswith("m"):
            return "Männlich"
        if g_lower.startswith("f"):
            return "Weiblich"
    return "Andere"

df["gender_de"] = df["gender"].apply(map_gender)

# ------------------------------------------------------------
# 3.2 BMI Categorization
# ------------------------------------------------------------
def categorize_bmi(bmi):
    if pd.isna(bmi):
        return None
    if bmi < 18.5:
        return "Untergewicht"
    elif bmi < 25:
        return "Normalgewicht"
    elif bmi < 30:
        return "Übergewicht"
    else:
        return "Adipositas"

df["bmi_kategorie"] = df["bmi"].apply(categorize_bmi)

# ------------------------------------------------------------
# 3.3 Age Limits & Slider Setup
# ------------------------------------------------------------
age_min = int(df["age"].min())
age_max = int(df["age"].max())

# Anzeige-Bereich für Slider: feste 10er Schritte
age_min_display = 18
age_max_display = 79
age_marks = {age: str(age) for age in range(age_min_display, age_max_display + 1, 10)}

# ------------------------------------------------------------
# 3.4 Numeric Columns
# ------------------------------------------------------------
numeric_cols = [
    "age",
    "bmi",
    "daily_steps",
    "sleep_hours",
    "water_intake_l",
    "calories_consumed",
    "resting_hr",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "disease_risk",
]

# ------------------------------------------------------------
# 3.5 Column Labels
# ------------------------------------------------------------
col_labels = {
    "age": "Alter",
    "bmi": "BMI",
    "daily_steps": "Tägliche Schritte",
    "sleep_hours": "Schlafdauer (Stunden)",
    "water_intake_l": "Wasseraufnahme (L/Tag)",
    "calories_consumed": "Kalorienaufnahme (kcal/Tag)",
    "resting_hr": "Ruhepuls (bpm)",
    "systolic_bp": "Systolischer Blutdruck",
    "diastolic_bp": "Diastolischer Blutdruck",
    "cholesterol": "Cholesterin",
    "disease_risk": "Krankheitsrisiko (%)",
}

# ------------------------------------------------------------
# 3.6 Disease Risk Scaling
# ------------------------------------------------------------
disease_scale_factor = 100 if df["disease_risk"].max() <= 1.5 else 1

# ============================================================
# 4. RADAR NORMALIZATION
# ============================================================

# ------------------------------------------------------------
# 4.1 Radar Variables
# ------------------------------------------------------------
radar_cols = [
    "daily_steps",
    "water_intake_l",
    "sleep_hours",
    "calories_consumed",
    "resting_hr",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "family_history",
    "disease_risk",
]

# ------------------------------------------------------------
# 4.2 Radar Min / Max Calculation
# ------------------------------------------------------------
radar_min = {}
radar_max = {}

for col in radar_cols:
    if col == "family_history":
        series = df[col] * 100.0
    elif col == "disease_risk":
        series = df[col] * disease_scale_factor
    else:
        series = df[col]
    if len(series) > 0:
        radar_min[col] = float(series.min())
        radar_max[col] = float(series.max())
    else:
        radar_min[col] = 0.0
        radar_max[col] = 1.0

# ------------------------------------------------------------
# 4.3 Radar Group Value Function
# ------------------------------------------------------------
def radar_group_value(d, col):
    #Gruppenmittelwert für Radar auf 0–1 skalieren.
    if len(d) == 0:
        return 0.0

    if col == "family_history":
        raw = d[col].mean() * 100.0  # Prozent
    elif col == "disease_risk":
        raw = d[col].mean() * disease_scale_factor  # 0–100
    else:
        raw = d[col].mean()

    mn = radar_min.get(col, 0.0)
    mx = radar_max.get(col, 1.0)
    if mx <= mn or pd.isna(raw):
        return 0.0
    return float((raw - mn) / (mx - mn))

# ============================================================
# 5. FILTER & HELPER FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# 5.1 Main Data Filter
# ------------------------------------------------------------
def filter_dataframe(data, age_range, gender, smoker, alcohol, bmi_cat):
    dff = data.copy()

    # Alter: 10-Jahres-Intervalle wie [18, 28], [28, 38] ...
    if age_range is not None and len(age_range) == 2:
        low, high = age_range
        # letztes Intervall inklusiv, sonst oberes Ende exklusiv
        if high >= age_max_display:
            dff = dff[(dff["age"] >= low) & (dff["age"] <= high)]
        else:
            dff = dff[(dff["age"] >= low) & (dff["age"] < high)]

    # Geschlecht
    if gender and gender != "Alle":
        dff = dff[dff["gender_de"] == gender]

    # Raucherstatus
    if smoker and smoker != "Alle":
        if smoker == "Raucher":
            dff = dff[dff["smoker"] == 1]
        elif smoker == "Nichtraucher":
            dff = dff[dff["smoker"] == 0]

    # Alkoholkonsum
    if alcohol and alcohol != "Alle":
        if alcohol == "Mit Alkohol":
            dff = dff[dff["alcohol"] == 1]
        elif alcohol == "Ohne Alkohol":
            dff = dff[dff["alcohol"] == 0]

    # BMI-Kategorie
    if bmi_cat and bmi_cat != "Alle":
        dff = dff[dff["bmi_kategorie"] == bmi_cat]

    return dff

# ------------------------------------------------------------
# 5.2 Filter from Stored Dict (Group A / B)
# ------------------------------------------------------------
def filter_from_dict(data, filt_dict):
    #Filter auf Basis eines gespeicherten Filter-Sets (Gruppe A/B).
    if filt_dict is None:
        return data.iloc[0:0]  # leeres DataFrame
    return filter_dataframe(
        data,
        age_range=filt_dict.get("age_range"),
        gender=filt_dict.get("gender"),
        smoker=filt_dict.get("smoker"),
        alcohol=filt_dict.get("alcohol"),
        bmi_cat=filt_dict.get("bmi_cat"),
    )

# ------------------------------------------------------------
# 5.3 Formatting Helpers
# ------------------------------------------------------------
def format_value(val, decimals=1, suffix=""):
    if val is None or pd.isna(val):
        return "-"
    return f"{val:.{decimals}f}{suffix}"


def mean_or_percent(series, as_percent=False):
    if len(series) == 0:
        return None
    val = series.mean()
    if as_percent:
        return val * 100.0
    return val

# ============================================================
# 6. APP INITIALIZATION
# ============================================================

app = Dash(__name__)

# ============================================================
# 7. LAYOUT
# ============================================================

app.layout = html.Div(

    # --------------------------------------------------------
    # 7.0 GLOBAL PAGE STYLE
    # --------------------------------------------------------
    style={
        "margin": "20px",
        "fontFamily": "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
        "fontSize": "16px",
    },
    children=[

        # ----------------------------------------------------
        # 7.1 PAGE TITLE
        # ----------------------------------------------------
        
        html.Div(
            style={
                "backgroundColor": "#edf1ee",  # Äusserer Rahmen (grau)
                "padding": "12px",
                "borderRadius": "12px",
                "boxShadow": "0px 4px 10px rgba(0, 0, 0, 0.15)",
                "marginBottom": "25px",
            },
            children=[
                html.Div(
                    style={
                        # Innerer Rahmen mit stärker gesättigtem Farbverlauf
                        "background": "linear-gradient(90deg, #0DEE19, #FFB347, #F26C6C)",
                        "padding": "22px",
                        "borderRadius": "10px",
                    },
                    children=[
                        html.H1(
                            "Lifestyle, Heart & Risk Indicators Dashboard",
                            style={
                                "margin": "0 0 6px 0",
                                "fontSize": "30px",
                                "fontWeight": "600",
                                "lineHeight": "1.3",
                            },
                        ),
                        html.P(
                            "An interactive dashboard for exploring lifestyle, heart, and risk indicators across different groups.",
                            style={
                                "margin": "0",
                                "fontSize": "15px",
                                "fontWeight": "400",
                                "lineHeight": "1.5",
                                "color": "#4a5560",
                            },
                        ),
                    ],
                )
            ],
        ),
        
        # ----------------------------------------------------
        # 7.2 GLOBAL STORES (STATE MANAGEMENT)
        # ----------------------------------------------------
        dcc.Store(id="group-a-store"),
        dcc.Store(id="group-b-store"),
        dcc.Store(id="radar-mode-store", data="lifestyle"),
        dcc.Store(id="selected-kpi", data=None),
        dcc.Store(id="filtered-df-store"),
        
        

        # ====================================================
        # 7.3 TOP SECTION
        #     Filter + KPI + Gauge
        # ====================================================
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            children=[

                # -----------------------------------------------
                # 7.3.1 FILTER PANEL (LEFT)
                # -----------------------------------------------
                # ------------------------------------------------------------
                # FILTERGRUPPE – Komplett und funktionssicher
                # ------------------------------------------------------------
                html.Div(
                    style={
                        "backgroundColor": "#edf1ee",  # Warmer Grauton
                        "padding": "16px",
                        "borderRadius": "12px",
                        "boxShadow": "0px 4px 10px rgba(0, 0, 0, 0.15)",
                        "width": "320px",
                    },
                    children=[

                        # ---------------- Titel ----------------
                        html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "marginBottom": "14px",
                        },
                        children=[

                            html.H3(
                                "Filtergruppe",
                                style={
                                    "margin": "0",
                                    "fontSize": "24px",
                                    "fontWeight": "600",
                                    "lineHeight": "1.35",
                                },
                            ),
                            
                            # ------------------------------------------------------------
                            # INFO BUTTON – Erklärung zu Filtergruppe & Gruppenvergleich
                            # ------------------------------------------------------------
                            html.Button(
                                "ℹ️",
                                id="filter-info-btn",
                                n_clicks=0,
                                style={
                                    "border": "none",
                                    "background": "transparent",
                                    "color": "#0B52EC",
                                    "fontSize": "20px",
                                    "cursor": "pointer",
                                    "padding": "0 4px",
                                },
                            ),
                        ],
                    ),

                        # ---------------- Altersbereich ----------------
                        html.Label(
                            "Alter",
                            style={"fontSize": "16px", "fontWeight": "600", "lineHeight": "1.4"},
                        ),
                        dcc.RangeSlider(
                            id="age-slider",
                            min=18,
                            max=79,
                            step=1,
                            value=[18,79],
                            marks={
                                18: {"label": "18", "style": {"fontSize": "16px", "fontWeight": "400"}},
                                28: {"label": "28", "style": {"fontSize": "16px", "fontWeight": "400"}},
                                38: {"label": "38", "style": {"fontSize": "16px", "fontWeight": "400"}},
                                48: {"label": "48", "style": {"fontSize": "16px", "fontWeight": "400"}},
                                58: {"label": "58", "style": {"fontSize": "16px", "fontWeight": "400"}},
                                68: {"label": "68", "style": {"fontSize": "16px", "fontWeight": "400"}},
                                79: {"label": "79", "style": {"fontSize": "16px", "fontWeight": "400"}},
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),

                        html.Br(),

                        # ---------------- Geschlecht ----------------
                        html.Label(
                            "Geschlecht",
                            style={"fontSize": "16px", "fontWeight": "600", "lineHeight": "1.4"},
                        ),
                        dcc.Dropdown(
                            id="gender-dropdown",
                            options=[
                                {"label": "Alle", "value": "Alle"},
                                {"label": "Männlich", "value": "Männlich"},
                                {"label": "Weiblich", "value": "Weiblich"},
                            ],
                            value="Alle",
                            clearable=False,
                            style={
                                "fontSize": "16px",
                                "fontWeight": "400",
                                "lineHeight": "1.4",
                            },
                        ),

                        html.Br(),

                        # ---------------- Rauchstatus ----------------
                        html.Label(
                            "Rauchstatus",
                            style={"fontSize": "16px", "fontWeight": "600", "lineHeight": "1.4"},
                        ),
                        dcc.Dropdown(
                            id="smoking-dropdown",
                            options=[
                                {"label": "Alle", "value": "Alle"},
                                {"label": "Raucher", "value": "Raucher"},
                                {"label": "Nichtraucher", "value": "Nichtraucher"},
                            ],
                            value="Alle",
                            clearable=False,
                            style={
                                "fontSize": "16px",
                                "fontWeight": "400",
                                "lineHeight": "1.4",
                            },
                        ),

                        html.Br(),

                        # ---------------- Alkoholkonsum ----------------
                        html.Label(
                            "Alkoholkonsum",
                            style={"fontSize": "16px", "fontWeight": "600", "lineHeight": "1.4"},
                        ),
                        dcc.Dropdown(
                            id="alcohol-dropdown",
                            options=[
                                {"label": "Alle", "value": "Alle"},
                                {"label": "Ja", "value": "Ja"},
                                {"label": "Nein", "value": "Nein"},
                            ],
                            value="Alle",
                            clearable=False,
                            style={
                                "fontSize": "16px",
                                "fontWeight": "400",
                                "lineHeight": "1.4",
                            },
                        ),

                        html.Br(),

                        # ---------------- BMI-Kategorie ----------------
                        html.Label(
                            "BMI-Kategorie",
                            style={"fontSize": "16px", "fontWeight": "600", "lineHeight": "1.4"},
                        ),
                        dcc.Dropdown(
                            id="bmi-dropdown",
                            options=[
                                {"label": "Alle", "value": "Alle"},
                                {"label": "Untergewicht", "value": "Untergewicht"},
                                {"label": "Normalgewicht", "value": "Normalgewicht"},
                                {"label": "Übergewicht", "value": "Übergewicht"},
                                {"label": "Adipositas", "value": "Adipositas"},
                            ],
                            value="Alle",
                            clearable=False,
                            style={
                                "fontSize": "16px",
                                "fontWeight": "400",
                                "lineHeight": "1.4",
                            },
                        ),

                        # ====================================================
                        # GRUPPENVERGLEICH – Innerer separater Rahmen
                        # ====================================================
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "14px",
                                "borderRadius": "10px",
                                "boxShadow": "0px 2px 6px rgba(0, 0, 0, 0.1)",
                                "marginTop": "20px",
                            },
                            children=[

                                html.H4(
                                    "Gruppenvergleich",
                                    style={
                                        "marginTop": "0",
                                        "marginBottom": "6px",
                                        "fontSize": "24px",
                                        "fontWeight": "600",
                                        "lineHeight": "1.35"
                                    },
                                ),

                                html.P(
                                    "Aktivieren Sie den Vergleich, um zwei unterschiedliche Gruppen gegenüberzustellen.",
                                    style={
                                        "fontSize": "16px",
                                        "marginTop": "0",
                                        "marginBottom": "12px",
                                        "opacity": "0.8",
                                        "fontWeight": "400"
                                    },
                                ),

                                dcc.Checklist(
                                    id="group-compare-toggle",
                                    options=[
                                        {
                                            "label": html.Span(
                                                "Gruppenvergleich aktivieren",
                                                style={
                                                    "fontSize": "20px",
                                                    "fontWeight": "600",
                                                    "fontWeight": "500"
                                                },
                                            ),
                                            "value": "on",
                                        }
                                    ],
                                    value=[],
                                ),

                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "10px",
                                        "marginTop": "10px",
                                        "fontSize": "20px",
                                        "fontWeight": "600",
                                        "lineHeight": "1.4",
                                    },
                                    children=[

                                        # -------- Gruppe A --------
                                        html.Button(
                                            "Als Gruppe A festlegen",
                                            id="set-group-a",
                                            style={
                                                "flex": "1",
                                                "backgroundColor": "#0B52EC",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "6px",
                                                "padding": "8px",
                                                "fontWeight": "400",
                                                "cursor": "pointer",
                                            },
                                        ),

                                        # -------- Gruppe B --------
                                        html.Button(
                                            "Als Gruppe B festlegen",
                                            id="set-group-b",
                                            style={
                                                "flex": "1",
                                                "backgroundColor": "#719C0D",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "6px",
                                                "padding": "8px",
                                                "fontWeight": "400",
                                                "cursor": "pointer",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                
                # ============================================================
                # KPI BEREICH
                # ============================================================

                html.Div(
                    style={
                        "backgroundColor": "#edf1ee",
                        "padding": "18px",
                        "borderRadius": "12px",
                        "boxShadow": "0px 4px 10px rgba(0,0,0,0.15)",
                        "flex": "1",
                    },
                    children=[

                        # ----------------------------------------------------
                        # KPI ÜBERSCHRIFT + HINWEIS
                        # ---------------------------------------------------- 

                        html.H3(
                            "Kernkennzahlen – Überblick",
                            style={
                                "fontSize": "24px",
                                "fontWeight": "600",
                                "lineHeight": "1.35",
                                "margin": "0 0 8px 0",
                            },
                        ),

                        html.P(
                            "Wählen Sie einen Bereich (Lifestyle, Heart oder Risk), um die jeweiligen Kennzahlen anzuzeigen.",
                            style={
                                "margin": "0 0 12px 0",
                                "fontSize": "15px",
                                "fontWeight": "400",
                                "lineHeight": "1.5",
                            },
                        ),

                        # ----------------------------------------------------
                        # MODE BANNER (Lifestyle / Heart / Risk)
                        # ----------------------------------------------------
                        html.Div(
                            style={
                                "display": "flex",
                                "borderRadius": "10px",
                                "overflow": "hidden",
                                "marginBottom": "10px",
                            },
                            children=[

                                html.Button(
                                    "Lifestyle",
                                    id="btn-lifestyle",
                                    n_clicks=0,
                                    style={
                                        "flex": "1",
                                        "backgroundColor": "#0DEE19",
                                        "border": "none",
                                        "padding": "10px",
                                        "fontSize": "18px",
                                        "fontWeight": "550",
                                        "cursor": "pointer",
                                    },
                                ),

                                html.Button(
                                    "Heart",
                                    id="btn-heart",
                                    n_clicks=0,
                                    style={
                                        "flex": "1",
                                        "backgroundColor": "#FFB347",
                                        "border": "none",
                                        "padding": "10px",
                                        "fontSize": "18px",
                                        "fontWeight": "550",
                                        "cursor": "pointer",
                                    },
                                ),

                                html.Button(
                                    "Risk",
                                    id="btn-risk",
                                    n_clicks=0,
                                    style={
                                        "flex": "1",
                                        "backgroundColor": "#F26C6C",
                                        "border": "none",
                                        "padding": "10px",
                                        "fontSize": "18px",
                                        "fontWeight": "550",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                        ),
                        
                        # ----------------------------------------------------
                        # KPI KARTEN (DYNAMISCH – klickbar)
                        # ----------------------------------------------------
                        html.Div(
                            id="kpi-indicator-container",
                            style={"marginBottom": "20px"},
                        ),

                        # ----------------------------------------------------
                        # KPI ANALYSE (DETAILANSICHT)
                        # ----------------------------------------------------
                        html.Div(
                            children=[
                            
                                html.P(
                                    "Klicken Sie auf eine Kennzahl, um im unteren Diagramm die Abweichung zum Gesamtmittel darzustellen.",
                                    style={
                                        "margin": "8px 0 12px 0",
                                        "fontSize": "16px",
                                        "fontWeight": "400",
                                        "lineHeight": "1.5",
                                    },
                                ),

                                html.Div(
                                    id="kpi-analysis-container",
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        # ====================================================
        # 7.4 MIDDLE SECTION
        #     Distribution Analysis
        # ====================================================
        html.Div(
            style={
                "marginTop": "20px",
                "backgroundColor": "#edf1ee",
                "padding": "18px",
                "borderRadius": "12px",
                "boxShadow": "0px 4px 10px rgba(0,0,0,0.15)",
            },
            children=[

                # ------------------------------------------------
                # 7.4.0 Abschnittstitel & Erklärung
                # ------------------------------------------------
                html.H3(
                    "Detaillierte Betrachtung der ausgewählten Gruppe",
                    style={
                        "marginTop": "0",
                        "marginBottom": "6px",
                        "fontSize": "24px",
                        "lineHeight": "1.35",
                        "fontWeight": "600",
                    },
                ),
                html.P(
                    "Hier sehen Sie, wie sich die ausgewählte Gruppe "
                    "für die gewählte Variable verteilt.",
                    style={
                        "fontSize": "16px",
                        "fontWeight": "400",
                        "opacity": "0.75",
                        "marginBottom": "16px",
                    },
                ),

                # ------------------------------------------------
                # 7.4.1 Distribution Variable Selector
                # ------------------------------------------------
                html.Div(
                    children=[
                        html.Label(
                            "Variable auswählen",
                            style={
                                "fontSize": "16px",
                                "fontWeight": "500",
                                "lineHeight": "1.4",
                            },
                        ),
                        dcc.Dropdown(
                            id="dist-variable",
                            options=[
                                {"label": col_labels[col], "value": col}
                                for col in numeric_cols
                            ],
                            value="daily_steps",
                            clearable=False,
                            style={
                                "maxWidth": "320px",
                                "fontSize": "16px",
                                "fontWeight": "400",
                                "lineHeight": "1.4",
                            },
                        ),
                    ],
                    style={"marginBottom": "18px"},
                ),

                # ------------------------------------------------
                # 7.4.2 Visualisierungen
                # ------------------------------------------------
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "20px",
                        "flexWrap": "wrap",
                    },
                    children=[

                        # --------------------------------------------
                        # 7.4.2a Histogramm – Überblick
                        # --------------------------------------------
                        html.Div(
                            style={
                                "flex": "1",
                                "minWidth": "350px",
                                "backgroundColor": "white",
                                "padding": "14px",
                                "borderRadius": "10px",
                                "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)",
                            },
                            children=[
                                html.H4(
                                    "Verteilung innerhalb der Gruppe",
                                    style={
                                        "marginTop": "0",
                                        "marginBottom": "4px",
                                        "fontSize": "20px",      
                                        "fontWeight": "600",
                                        "lineHeight": "1.35",
                                    },
                                ),
                                html.P(
                                    "Zeigt, wie häufig bestimmte Werte "
                                    "in der ausgewählten Gruppe vorkommen.",
                                    style={
                                        "fontSize": "16px",      
                                        "fontWeight": "400",
                                        "lineHeight": "1.5",
                                        "opacity": "0.85",
                                        "marginBottom": "8px",
                                    },
                                ),
                                dcc.Graph(
                                    id="histogram-main",
                                    style={"height": "300px"},
                                ),
                            ],
                        ),

                        # --------------------------------------------
                        # 7.4.2b Punktverteilung – Einzelwerte
                        # --------------------------------------------
                        html.Div(
                            style={
                                "flex": "1",
                                "minWidth": "350px",
                                "backgroundColor": "white",
                                "padding": "14px",
                                "borderRadius": "10px",
                                "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)",
                            },
                            children=[
                                html.H4(
                                    "Einzelne Werte innerhalb der Gruppe",
                                    style={
                                        "marginTop": "0",
                                        "marginBottom": "4px",
                                        "fontSize": "20px",      
                                        "fontWeight": "600",
                                        "lineHeight": "1.35",
                                    },
                                ),
                                html.P(
                                    "Jeder Punkt steht für eine Person "
                                    "und zeigt die tatsächlichen Messwerte.",
                                    style={
                                        "fontSize": "16px",   
                                        "fontWeight": "400",
                                        "lineHeight": "1.5",
                                        "opacity": "0.85",
                                        "marginBottom": "8px",
                                    },
                                ),
                                dcc.Graph(
                                    id="boxplot-main",  # wird später als Beeswarm genutzt
                                    style={"height": "300px"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ====================================================
        # 7.5 BOTTOM SECTION
        #     Scatter + Radar
        # ====================================================
        html.Div(
            style={
                "display": "flex",
                "gap": "20px",
                "marginTop": "20px",
                "flexWrap": "wrap",
            },
            children=[

                # -----------------------------------------------
                # 7.5.1 Scatter Plot – Variablenbeziehung
                # -----------------------------------------------
                html.Div(
                    style={
                        "flex": "1",
                        "minWidth": "350px",
                        "backgroundColor": "#f7f7f7",
                        "border": "1px solid #ddd",
                        "borderRadius": "10px",
                        "padding": "12px",
                        "boxShadow": "0px 2px 6px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.H3(
                            "Zusammenhang zwischen zwei Variablen",
                            style={
                                "fontSize": "24px",     
                                "fontWeight": "600",
                                "lineHeight": "1.35",
                                "opacity": "0.85",
                                "marginBottom": "10px",
                            },
                        ),
                        html.P(
                            "Zeigt die Beziehung zwischen zwei ausgewählten Variablen innerhalb der aktuellen Gruppe.",
                            style={
                                "fontSize": "16px",     
                                "fontWeight": "400",
                                "lineHeight": "1.5",
                                "opacity": "0.85",
                                "marginBottom": "10px",
                            },
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "10px", "marginBottom": "8px"},
                            children=[
                                html.Div(
                                    style={"flex": 1},
                                    children=[
                                        html.Label("X-Achse", style={"fontSize": "16px", "fontWeight": "500"}),
                                        dcc.Dropdown(
                                            id="scatter-x-main",
                                            options=[
                                                {"label": col_labels[col], "value": col}
                                                for col in numeric_cols
                                            ],
                                            value="daily_steps",  # default
                                            clearable=False,
                                            style={"fontSize": "16px", "fontWeight": "400", "lineHeight": "1.4"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"flex": 1},
                                    children=[
                                        html.Label("Y-Achse", style={"fontSize": "16px", "fontWeight": "500"}),
                                        dcc.Dropdown(
                                            id="scatter-y-main",
                                            options=[
                                                {"label": col_labels[col], "value": col}
                                                for col in numeric_cols
                                            ],
                                            value="resting_hr",  # default
                                            clearable=False,
                                            style={"fontSize": "16px", "fontWeight": "400", "lineHeight": "1.4"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="scatter-main",
                            style={
                                "height": "320px",
                                "backgroundColor": "#f7f7f7",
                            },
                        ),
                    ],
                ),

                # -----------------------------------------------
                # 7.5.2 Radar Plot – Gruppenprofil
                # -----------------------------------------------
                html.Div(
                    style={
                        "flex": "1",
                        "minWidth": "350px",
                        "backgroundColor": "#f7f7f7",
                        "border": "1px solid #ddd",
                        "borderRadius": "10px",
                        "padding": "12px",
                        "boxShadow": "0px 2px 6px rgba(0,0,0,0.08)",
                    },
                    children=[
                        html.H3(
                            "Profilvergleich der Gruppen",
                            style={
                                "marginBottom": "4px",
                                "fontSize": "24px",    
                                "fontWeight": "600",
                                "lineHeight": "1.35",
                            },
                        ),
                        html.P(
                            "Wählen Sie eine Kennzahlengruppe aus, um das Profil der Gruppen über mehrere Kennzahlen hinweg zu vergleichen.",
                            style={
                                "fontSize": "16px",    
                                "fontWeight": "400",
                                "opacity": "0.85",
                                "lineHeight": "1.5",
                                "marginBottom": "10px",
                            },
                        ),

                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "10px",
                                "marginBottom": "10px",
                            },
                            children=[
                                html.Button(
                                    "Lifestyle",
                                    id="btn-radar-lifestyle",
                                    n_clicks=0,
                                    style={
                                        "flex": 1,
                                        "backgroundColor": "#5cb85c",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "10px",
                                        "fontSize": "16px",    
                                        "fontWeight": "400",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Heart",
                                    id="btn-radar-heart",
                                    n_clicks=0,
                                    style={
                                        "flex": 1,
                                        "backgroundColor": "#f0ad4e",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px",
                                        "fontSize": "16px",     
                                        "fontWeight": "400",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "Risk",
                                    id="btn-radar-risk",
                                    n_clicks=0,
                                    style={
                                        "flex": 1,
                                        "backgroundColor": "#d9534f",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px",
                                        "fontSize": "16px",    
                                        "fontWeight": "400",
                                        "borderRadius": "6px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                        ),

                        dcc.Graph(
                            id="radar-graph",
                            style={
                                "height": "320px",
                                "backgroundColor": "#f7f7f7",
                            },
                        ),
                    ],
                ),
                # ============================================================
                # INFO MODAL – Filtergruppe & Gruppenvergleich
                # ============================================================

                html.Div(
                    id="filter-info-modal",
                    style={
                        "display": "none",  
                        "position": "fixed",
                        "top": "0",
                        "left": "0",
                        "width": "100vw",
                        "height": "100vh",
                        "backgroundColor": "rgba(0, 0, 0, 0.45)",
                        "zIndex": "1000",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                    },
                    children=[

                        # ----------------------------------------------------
                        # MODAL CONTENT BOX
                        # ----------------------------------------------------
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "24px",
                                "borderRadius": "12px",
                                "width": "440px",
                                "boxShadow": "0px 10px 30px rgba(0,0,0,0.25)",
                            },
                            children=[

                                # ---------------- Titel ----------------
                                html.H4(
                                    "Filtergruppe & Gruppenvergleich",
                                    style={
                                        "marginTop": "0",
                                        "marginBottom": "12px",
                                        "fontSize": "20px",
                                        "fontWeight": "600",
                                    },
                                ),

                                # ---------------- Erklärung Filtergruppe ----------------
                                html.P(
                                    "Die Filtergruppe beschreibt die aktuell ausgewählte Personengruppe. "
                                    "Sie wird anhand der gewählten Filter wie Alter, Geschlecht, Rauchstatus, "
                                    "Alkoholkonsum und BMI definiert.",
                                    style={
                                        "fontSize": "15px",
                                        "lineHeight": "1.5",
                                        "marginBottom": "10px",
                                    },
                                ),

                                html.P(
                                    "Alle Visualisierungen im Dashboard beziehen sich standardmäßig "
                                    "auf diese Filtergruppe.",
                                    style={
                                        "fontSize": "15px",
                                        "lineHeight": "1.5",
                                        "marginBottom": "14px",
                                    },
                                ),

                                html.Hr(),

                                # ---------------- Erklärung Gruppenvergleich ----------------
                                html.P(
                                    "Der Gruppenvergleich ist optional und dient dazu, "
                                    "zwei unterschiedliche Filterzustände gezielt miteinander zu vergleichen.",
                                    style={
                                        "fontSize": "15px",
                                        "lineHeight": "1.5",
                                        "marginBottom": "10px",
                                    },
                                ),

                                html.P(
                                    "Dazu kann eine Filterkonfiguration als Gruppe A "
                                    "und eine andere als Gruppe B gespeichert werden. "
                                    "Ist der Vergleich aktiviert, werden beide Gruppen "
                                    "parallel in den entsprechenden Diagrammen dargestellt.",
                                    style={
                                        "fontSize": "15px",
                                        "lineHeight": "1.5",
                                    },
                                ),

                                # ---------------- Close Button ----------------
                                html.Button(
                                    "Schließen",
                                    id="close-filter-info",
                                    style={
                                        "marginTop": "18px",
                                        "backgroundColor": "#0B52EC",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px 14px",
                                        "borderRadius": "6px",
                                        "fontSize": "14px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                        ),
                    ],
                )

            ],
        )
    ],
)

# ============================================================
# 8. CALLBACKS
# ============================================================

# ------------------------------------------------------------
# 8.0 INFO MODAL TOGGLE – Filtergruppe & Gruppenvergleich
# ------------------------------------------------------------
@callback(
    Output("filter-info-modal", "style"),
    Input("filter-info-btn", "n_clicks"),
    Input("close-filter-info", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_filter_info(n_open, n_close):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Modal aç
    if trigger_id == "filter-info-btn":
        return {
            "display": "flex",
            "position": "fixed",
            "top": "0",
            "left": "0",
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": "rgba(0,0,0,0.45)",
            "zIndex": "1000",
            "alignItems": "center",
            "justifyContent": "center",
        }

    # Modal kapat
    return {"display": "none"}

# ------------------------------------------------------------
# 8.1 KPI-RENDERING (Lifestyle / Heart / Risk)
# ------------------------------------------------------------
@callback(
    Output("kpi-indicator-container", "children"),
    Input("btn-lifestyle", "n_clicks"),
    Input("btn-heart", "n_clicks"),
    Input("btn-risk", "n_clicks"),
    Input("group-compare-toggle", "value"),
    Input("group-a-store", "data"),
    Input("group-b-store", "data"),
    Input("age-slider", "value"),
    Input("gender-dropdown", "value"),
    Input("smoking-dropdown", "value"),
    Input("alcohol-dropdown", "value"),
    Input("bmi-dropdown", "value"),
)
def render_kpis(
    n_lifestyle, n_heart, n_risk,
    compare, group_a_store, group_b_store,
    age_range, gender, smoker, alcohol, bmi_cat
):
    ctx = callback_context

    # --------------------------------------------------------
    # Aktiven Modus bestimmen
    # --------------------------------------------------------
    if not ctx.triggered:
        mode = "lifestyle"
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        mode = {
            "btn-lifestyle": "lifestyle",
            "btn-heart": "heart",
            "btn-risk": "risk",
        }.get(trigger_id, "lifestyle")

    # --------------------------------------------------------
    # Gefiltertes DataFrame (aktuelle Filter)
    # --------------------------------------------------------
    dff = filter_dataframe(df, age_range, gender, smoker, alcohol, bmi_cat)
    if len(dff) == 0:
        return html.P("Keine Daten für die aktuelle Filtergruppe.")

    compare_on = compare is not None and "on" in compare
    has_groups = compare_on and group_a_store and group_b_store

    if has_groups:
        dfa = filter_from_dict(df, group_a_store)
        dfb = filter_from_dict(df, group_b_store)

    # --------------------------------------------------------
    # KPI-Definitionen je Modus
    # --------------------------------------------------------
    if mode == "lifestyle":
        title = html.H4(
            "Lifestyle Indicators",
            style={
                "fontSize": "20px",      
                "fontWeight": "600",     
                "lineHeight": "1.4",
                "margin": "0 0 8px 0",
            },
        )
        accent_color = "#0DEE19"
        cols = [
            ("Steps", "daily_steps"),
            ("Water Intake (L)", "water_intake_l"),
            ("Sleep Duration (h)", "sleep_hours"),
            ("Calories Intake", "calories_consumed"),
        ]

    elif mode == "heart":
        title = html.H4(
            "Cardiovascular Indicators",
            style={
                "fontSize": "20px",
                "fontWeight": "600",
                "lineHeight": "1.4",
                "margin": "0 0 8px 0",
            },
        )
        accent_color = "#FFB347"
        cols = [
            ("Resting Pulse (bpm)", "resting_hr"),
            ("Systolic Blood Pressure", "systolic_bp"),
            ("Diastolic Blood Pressure", "diastolic_bp"),
        ]

    else:  # risk
        title = html.H4(
            "Risk Indicators",
            style={
                "fontSize": "20px",
                "fontWeight": "600",
                "lineHeight": "1.4",
                "margin": "0 0 8px 0",
            },
        )
        accent_color = "#F26C6C"
        cols = [
            ("Cholesterol", "cholesterol"),
            ("Family Risk (%)", "family_history"),
            ("Disease Risk (%)", "disease_risk"),
        ]

    # --------------------------------------------------------
    # KPI-Karten erzeugen
    # --------------------------------------------------------
    kpi_cards = []

    for label, col in cols:

        if has_groups:
            val_a = dfa[col].mean()
            val_b = dfb[col].mean()

            if col == "family_history":
                val_a *= 100
                val_b *= 100
            if col == "disease_risk":
                val_a *= disease_scale_factor
                val_b *= disease_scale_factor

            value_block = html.Div(
                children=[
                    html.Div(
                        f"A: {format_value(val_a)}",
                        style={
                            "color": "#0B52EC",
                            "marginTop": "10px",
                            "textAlign": "right",
                            "fontSize": "24px",   
                            "fontWeight": "600",
                            "lineHeight": "1.2",
                        },
                    ),
                    html.Div(
                        f"B: {format_value(val_b)}",
                        style={
                            "color": "#719C0D",
                            "marginTop": "10px",
                            "textAlign": "right",
                            "fontSize": "24px",   
                            "fontWeight": "600",
                            "lineHeight": "1.2",
                        },
                    ),
                ]
            )
        else:
            val = dff[col].mean()
            if col == "family_history":
                val *= 100
            if col == "disease_risk":
                val *= disease_scale_factor

            value_block = html.Div(
                format_value(val),
                style={
                    "margin": "10px 0 0 0",
                    "textAlign": "right",   
                    "fontSize": "24px",     
                    "fontWeight": "600",
                    "lineHeight": "1.2",
                },
            )

        kpi_cards.append(
            html.Div(
                id={"type": "kpi-card", "kpi": label},
                n_clicks=0,
                style={
                    "backgroundColor": "white",
                    "borderTop": f"6px solid {accent_color}",
                    "padding": "10px 16px 16px 16px",
                    "borderRadius": "8px",
                    "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)",
                    "cursor": "pointer",
                },
                children=[
                    html.Div(
                        label, 
                        style={
                            "textAlign": "center",
                            "fontSize": "18px",
                            "fontWeight": "400",
                            "lineHeight": "1.4",
                            "margin": "0 0 2px 0",
                        },
                    ),
                    value_block,
                ],
            )
        )

    return html.Div(
        children=[
            html.H4(title, style={"marginBottom": "12px"}),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({len(cols)}, 1fr)",
                    "gap": "12px",
                },
                children=kpi_cards,
            ),
        ]
    )

# ------------------------------------------------------------
# 8.2 KPI-AUSWAHL (Klick auf KPI-Karte)
# ------------------------------------------------------------
@callback(
    Output("selected-kpi", "data"),
    Input({"type": "kpi-card", "kpi": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_kpi(n_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return eval(triggered_id)["kpi"]

# ------------------------------------------------------------
# 8.3 KPI-ANALYSE – Referenzbasierter Vergleich zum Gesamtmittel
# ------------------------------------------------------------
@callback(
    Output("kpi-analysis-container", "children"),
    Input("selected-kpi", "data"),
    Input("group-compare-toggle", "value"),
    Input("age-slider", "value"),
    Input("gender-dropdown", "value"),
    Input("smoking-dropdown", "value"),
    Input("alcohol-dropdown", "value"),
    Input("bmi-dropdown", "value"),
    State("group-a-store", "data"),
    State("group-b-store", "data"),
)
def render_kpi_analysis(
    selected_kpi,
    compare,
    age_range,
    gender,
    smoking,
    alcohol,
    bmi,
    group_a_store,
    group_b_store,
):
    # --------------------------------------------------------
    # 1) KPI Mapping & Fallback
    # --------------------------------------------------------
    KPI_COLUMN_MAP = {
        "Steps": "daily_steps",
        "Water Intake (L)": "water_intake_l",
        "Sleep Duration (h)": "sleep_hours",
        "Calories Intake": "calories_consumed",
        "Resting Pulse (bpm)": "resting_hr",
        "Systolic Blood Pressure": "systolic_bp",
        "Diastolic Blood Pressure": "diastolic_bp",
        "Cholesterol": "cholesterol",
        "Family Risk (%)": "family_history",
        "Disease Risk (%)": "disease_risk",
    }

    # selected_kpi yoksa default al
    kpi_label = selected_kpi or "Steps"
    col = KPI_COLUMN_MAP.get(kpi_label)

    if col not in df.columns:
        return html.Div()

    # --------------------------------------------------------
    # 2) Global Referenz
    # --------------------------------------------------------
    global_mean = df[col].mean()

    # --------------------------------------------------------
    # 3) Aktuelle Filtergruppe (IMMER)
    # --------------------------------------------------------
    dff = filter_dataframe(
        df,
        age_range,
        gender,
        smoking,
        alcohol,
        bmi,
    )

    if not dff.empty and not dff[col].dropna().empty:
        delta_current = dff[col].mean() - global_mean
    else:
        delta_current = 0

    # --------------------------------------------------------
    # 4) Figure Setup
    # --------------------------------------------------------
    fig = go.Figure()

    # Referenzlinie (Gesamtmittel)
    fig.add_vline(
        x=0,
        line_color="#444",
        line_width=2,
        line_dash="dot",
    )

    # Aktuelle Filtergruppe
    if not compare:
        fig.add_trace(
            go.Scatter(
                x=[delta_current],
                y=["Aktuelle Filtergruppe"],
                mode="markers",
                marker=dict(size=16, color="#111"),
                hovertemplate=(
                    "Aktuelle Filtergruppe<br>"
                    "Δ = %{x:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # --------------------------------------------------------
    # 5) Optionale Vergleichsgruppen
    # --------------------------------------------------------
    delta_a = None
    delta_b = None

    if compare and group_a_store:
        dfa = filter_from_dict(df, group_a_store)
        if not dfa.empty and not dfa[col].dropna().empty:
            delta_a = dfa[col].mean() - global_mean
            fig.add_trace(
                go.Scatter(
                    x=[delta_a],
                    y=["Gruppe A"],
                    mode="markers",
                    marker=dict(size=16, color="#0B52EC"),
                    hovertemplate=(
                        "Gruppe A<br>"
                        "Δ = %{x:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    if compare and group_b_store:
        dfb = filter_from_dict(df, group_b_store)
        if not dfb.empty and not dfb[col].dropna().empty:
            delta_b = dfb[col].mean() - global_mean
            fig.add_trace(
                go.Scatter(
                    x=[delta_b],
                    y=["Gruppe B"],
                    mode="markers",
                    marker=dict(size=16, color="#719C0D"),
                    hovertemplate=(
                        "Gruppe B<br>"
                        "Δ = %{x:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    # --------------------------------------------------------
    # 6) Dynamischer Zoom (stabil & sicher)
    # --------------------------------------------------------
    deltas = [delta_current]
    if delta_a is not None:
        deltas.append(delta_a)
    if delta_b is not None:
        deltas.append(delta_b)

    max_dev = max(abs(d) for d in deltas)
    pad = max_dev * 1.5 + 0.01

    fig.update_layout(
        height=220,
        margin=dict(l=40, r=40, t=20, b=30),
        xaxis=dict(
            range=[-pad, pad],
            title="Abweichung vom Gesamtmittel",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(
                size=14,
                color="#333",),
        ),
    )

    # --------------------------------------------------------
    # 7) Output Container
    # --------------------------------------------------------
    return html.Div(
        [
            html.H4(
                f"Detailanalyse: {kpi_label}",
                style={
                    "fontSize": "20px",
                    "fontWeight": "600",
                    "marginBottom": "8px",
                },
            ),
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": True},
            ),
            html.P(
                "Die Grafik zeigt die Abweichung des Mittelwerts der aktuell "
                "gefilterten Gruppe vom Gesamtmittel. "
                "Optional können Vergleichsgruppen (A/B) ergänzt werden.",
                style={
                    "fontSize": "16px",
                    "fontWeight": "400",
                    "opacity": "0.85",
                    "marginTop": "6px",
                    "lineHeight": "1.5"
                },
            ),
        ]
    )

# ------------------------------------------------------------
# 8.4 GRUPPENSPEICHER (Gruppe A / Gruppe B festlegen)
# ------------------------------------------------------------
@callback(
    Output("group-a-store", "data"),
    Output("group-b-store", "data"),
    Input("set-group-a", "n_clicks"),
    Input("set-group-b", "n_clicks"),
    State("age-slider", "value"),
    State("gender-dropdown", "value"),
    State("smoking-dropdown", "value"),
    State("alcohol-dropdown", "value"),
    State("bmi-dropdown", "value"),
    prevent_initial_call=True,
)
def store_groups(
    n_a, n_b,
    age_range, gender, smoker, alcohol, bmi_cat
):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    current_filters = {
        "age_range": age_range,
        "gender": gender,
        "smoker": smoker,
        "alcohol": alcohol,
        "bmi_cat": bmi_cat,
    }

    if trigger == "set-group-a":
        return current_filters, no_update

    if trigger == "set-group-b":
        return no_update, current_filters

    raise PreventUpdate

# ------------------------------------------------------------
# 8.5 DISTRIBUTION ANALYSIS
#     Histogram + Horizontal Raincloud (Middle Section)
# ------------------------------------------------------------
@callback(
    Output("histogram-main", "figure"),
    Output("boxplot-main", "figure"),
    Input("dist-variable", "value"),
    Input("group-compare-toggle", "value"),
    Input("group-a-store", "data"),
    Input("group-b-store", "data"),
    Input("age-slider", "value"),
    Input("gender-dropdown", "value"),
    Input("smoking-dropdown", "value"),
    Input("alcohol-dropdown", "value"),
    Input("bmi-dropdown", "value"),
)
def update_distribution_analysis(
    variable, compare, group_a_store, group_b_store,
    age_range, gender, smoker, alcohol, bmi_cat
):

    compare_on = compare is not None and "on" in compare

    dff = filter_dataframe(
        df,
        age_range,
        gender,
        smoker,
        alcohol,
        bmi_cat,
    )

    # ----------------------------
    # HISTOGRAM
    # ----------------------------
    hist_fig = go.Figure()

    if compare_on and group_a_store and group_b_store:
        dfa = filter_from_dict(df, group_a_store)
        dfb = filter_from_dict(df, group_b_store)

        hist_fig.add_histogram(
            x=dfa[variable],
            marker_color="#0B52EC",
            opacity=0.6,
        )
        hist_fig.add_histogram(
            x=dfb[variable],
            marker_color="#719C0D",
            opacity=0.6,
        )
    else:
        hist_fig.add_histogram(
            x=dff[variable],
            marker_color="#93C5FD",
        )

    hist_fig.update_layout(
        barmode="overlay",
        xaxis_title=col_labels.get(variable, variable),
        yaxis_title="Anzahl Personen",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,   
    )

    # ----------------------------
    # RAINCLOUD
    # ----------------------------
    rain_fig = go.Figure()

    def sample_points(data, n=1500):
        if len(data) > n:
            return data.sample(n=n, random_state=42)
        return data

    def add_raincloud(data, color, y_pos):
        # Violin
        rain_fig.add_trace(
            go.Violin(
                x=data,
                y=[y_pos] * len(data),
                orientation="h",
                side="positive",
                width=0.6,
                line_color=color,
                fillcolor=color,
                opacity=0.35,
                showlegend=False,
            )
        )

        # Box
        rain_fig.add_trace(
            go.Box(
                x=data,
                y=[y_pos] * len(data),
                orientation="h",
                marker_color=color,
                boxpoints=False,
                showlegend=False,
            )
        )

        # Punkte (gesampled)
        pts = sample_points(data)
        rain_fig.add_trace(
            go.Scatter(
                x=pts,
                y=[y_pos] * len(pts),
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.6),
                showlegend=False,
            )
        )

    if compare_on and group_a_store and group_b_store:
        dfa = filter_from_dict(df, group_a_store)
        dfb = filter_from_dict(df, group_b_store)

        add_raincloud(dfa[variable], "#0B52EC", "A")
        add_raincloud(dfb[variable], "#719C0D", "B")

        rain_fig.update_yaxes(
            categoryorder="array",
            categoryarray=["B", "A"],
        )
    else:
        add_raincloud(dff[variable], "#93C5FD", "G")

    rain_fig.update_layout(
        xaxis_title=col_labels.get(variable, variable),
        yaxis_visible=False,
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        showlegend=False,   
    )

    return hist_fig, rain_fig


# ------------------------------------------------------------
# 8.6 SCATTER ANALYSIS
#     Variable Relationship + Trend Line
# ------------------------------------------------------------
@callback(
    Output("scatter-main", "figure"),
    Input("scatter-x-main", "value"),
    Input("scatter-y-main", "value"),
    Input("group-compare-toggle", "value"),
    Input("group-a-store", "data"),
    Input("group-b-store", "data"),
    Input("age-slider", "value"),
    Input("gender-dropdown", "value"),
    Input("smoking-dropdown", "value"),
    Input("alcohol-dropdown", "value"),
    Input("bmi-dropdown", "value"),
)
def update_scatter(
    x_var, y_var, compare,
    group_a_store, group_b_store,
    age_range, gender, smoker, alcohol, bmi_cat
):

    fig = go.Figure()

    if x_var is None or y_var is None:
        return fig

    compare_on = compare is not None and "on" in compare

    def prepare_data(data):
        data = data.dropna(subset=[x_var, y_var])
        if len(data) > 2000:
            data = data.sample(n=2000, random_state=42)
        return data

    def add_scatter_with_trend(data, color):
        data = prepare_data(data)

        fig.add_trace(
            go.Scatter(
                x=data[x_var] + np.random.normal(0, 0.05, len(data)),
                y=data[y_var] + np.random.normal(0, 0.5, len(data)),
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.25),
                showlegend=False,   
            )
        )

        if len(data) > 1:
            m, b = np.polyfit(data[x_var], data[y_var], 1)
            x_vals = np.array([data[x_var].min(), data[x_var].max()])
            y_vals = m * x_vals + b

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=color, width=3),
                    showlegend=False,
                )
            )

    if compare_on and group_a_store and group_b_store:
        dfa = filter_from_dict(df, group_a_store)
        dfb = filter_from_dict(df, group_b_store)

        add_scatter_with_trend(dfa, "#0B52EC")
        add_scatter_with_trend(dfb, "#719C0D")

    else:
        dff = filter_dataframe(
            df,
            age_range,
            gender,
            smoker,
            alcohol,
            bmi_cat,
        )
        add_scatter_with_trend(dff, "#93C5FD")

    fig.update_layout(
        xaxis_title=col_labels.get(x_var, x_var),
        yaxis_title=col_labels.get(y_var, y_var),
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode=False,
        showlegend=False,      
    )

    return fig


# ------------------------------------------------------------
# 8.7 RADAR ANALYSIS
#     Profile Comparison (Lifestyle / Heart / Risk)
# ------------------------------------------------------------
@callback(
    Output("radar-graph", "figure"),
    Input("btn-radar-lifestyle", "n_clicks"),
    Input("btn-radar-heart", "n_clicks"),
    Input("btn-radar-risk", "n_clicks"),
    Input("group-a-store", "data"),
    Input("group-b-store", "data"),
    Input("age-slider", "value"),
    Input("gender-dropdown", "value"),
    Input("smoking-dropdown", "value"),
    Input("alcohol-dropdown", "value"),
    Input("bmi-dropdown", "value"),
)
def update_radar(
    n_life, n_heart, n_risk,
    group_a_store, group_b_store,
    age_range, gender, smoker, alcohol, bmi_cat
):

    ctx = callback_context
    fig = go.Figure()

    KPI_GROUPS = {
        "btn-radar-lifestyle": {
            "Steps": "daily_steps",
            "Sleep": "sleep_hours",
            "Water": "water_intake_l",
            "Calories": "calories_consumed",
        },
        "btn-radar-heart": {
            "Resting HR": "resting_hr",
            "Systolic BP": "systolic_bp",
            "Diastolic BP": "diastolic_bp",
        },
        "btn-radar-risk": {
            "Cholesterol": "cholesterol",
            "Family Risk": "family_history",
            "Disease Risk": "disease_risk",
        },
    }

    # ----------------------------
    # Default: Lifestyle
    # ----------------------------
    trigger = "btn-radar-lifestyle"

    if ctx.triggered:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id in ["btn-radar-lifestyle", "btn-radar-heart", "btn-radar-risk"]:
            trigger = triggered_id

    kpis = KPI_GROUPS.get(trigger)
    if kpis is None:
        return fig

    categories = list(kpis.keys())

    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    # ----------------------------
    # Gruppenvergleich ODER Einzelgruppe
    # ----------------------------
    if group_a_store and group_b_store:
        dfa = filter_from_dict(df, group_a_store)
        dfb = filter_from_dict(df, group_b_store)

        vals_a, vals_b = [], []
        for col in kpis.values():
            norm = normalize(df[col])
            vals_a.append(norm.loc[dfa.index].mean())
            vals_b.append(norm.loc[dfb.index].mean())

        fig.add_trace(
            go.Scatterpolar(
                r=vals_a + [vals_a[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line_color="#0B52EC",
                name="Gruppe A",
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=vals_b + [vals_b[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line_color="#719C0D",
                name="Gruppe B",
            )
        )

        show_legend = True

    else:
        dff = filter_dataframe(
            df,
            age_range,
            gender,
            smoker,
            alcohol,
            bmi_cat,
        )

        vals = []
        for col in kpis.values():
            norm = normalize(df[col])
            vals.append(norm.loc[dff.index].mean())

        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line_color="#93C5FD",
                showlegend=False,   
            )
        )

        show_legend = False

    # ----------------------------
    # Layout
    # ----------------------------
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 1],
            )
        ),
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=show_legend,
    )

    return fig


# ------------------------------------------------------------
# Start
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
