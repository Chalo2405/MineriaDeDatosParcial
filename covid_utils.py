from pathlib import Path
import json
import urllib.request
import warnings

import pandas as pd


warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

DATA_PATH = Path("dataset_covid_unido.csv")
OUTPUT_DIR = Path("outputs")
US_STATES_GEOJSON = Path("us_states.geojson")
US_STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/python-visualization/"
    "folium-example-data/main/us_states.json"
)


REGIONS = {
    "Connecticut": "Northeast",
    "Maine": "Northeast",
    "Massachusetts": "Northeast",
    "New Hampshire": "Northeast",
    "Rhode Island": "Northeast",
    "Vermont": "Northeast",
    "New Jersey": "Northeast",
    "New York": "Northeast",
    "Pennsylvania": "Northeast",
    "Illinois": "Midwest",
    "Indiana": "Midwest",
    "Michigan": "Midwest",
    "Ohio": "Midwest",
    "Wisconsin": "Midwest",
    "Iowa": "Midwest",
    "Kansas": "Midwest",
    "Minnesota": "Midwest",
    "Missouri": "Midwest",
    "Nebraska": "Midwest",
    "North Dakota": "Midwest",
    "South Dakota": "Midwest",
    "Delaware": "South",
    "District of Columbia": "South",
    "Florida": "South",
    "Georgia": "South",
    "Maryland": "South",
    "North Carolina": "South",
    "South Carolina": "South",
    "Virginia": "South",
    "West Virginia": "South",
    "Alabama": "South",
    "Kentucky": "South",
    "Mississippi": "South",
    "Tennessee": "South",
    "Arkansas": "South",
    "Louisiana": "South",
    "Oklahoma": "South",
    "Texas": "South",
    "Arizona": "West",
    "Colorado": "West",
    "Idaho": "West",
    "Montana": "West",
    "Nevada": "West",
    "New Mexico": "West",
    "Utah": "West",
    "Wyoming": "West",
    "Alaska": "West",
    "California": "West",
    "Hawaii": "West",
    "Oregon": "West",
    "Washington": "West",
    "American Samoa": "Territory",
    "Guam": "Territory",
    "Northern Mariana Islands": "Territory",
    "Puerto Rico": "Territory",
    "Virgin Islands": "Territory",
    "Diamond Princess": "Other",
    "Grand Princess": "Other",
}


STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def cargar_y_limpiar_datos() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    # Esta fila es un agregado nacional de recuperados, no un estado/territorio.
    df = df[df["Province_State"] != "Recovered"].copy()

    fecha_date = pd.to_datetime(df["Date"], errors="coerce")
    fecha_update = pd.to_datetime(df["Last_Update"], errors="coerce").dt.normalize()
    df["Fecha"] = fecha_date.fillna(fecha_update)
    df = df.dropna(subset=["Fecha"]).sort_values(["Province_State", "Fecha"])

    df["Region"] = df["Province_State"].map(REGIONS).fillna("Other")
    df["Codigo_Estado"] = df["Province_State"].map(STATE_ABBR)
    df["Casos_Nuevos"] = (
        df.groupby("Province_State")["Confirmed"].diff().fillna(0).clip(lower=0)
    )
    df["Muertes_Nuevas"] = (
        df.groupby("Province_State")["Deaths"].diff().fillna(0).clip(lower=0)
    )

    df.to_csv(OUTPUT_DIR / "dataset_covid_limpio.csv", index=False)
    return df


def ultimo_registro_por_estado(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("Fecha").groupby("Province_State", as_index=False).tail(1)


def ultimo_registro_con_columnas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    return (
        df.dropna(subset=columnas)
        .sort_values("Fecha")
        .groupby("Province_State", as_index=False)
        .tail(1)
    )


def cargar_geojson_estados() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not US_STATES_GEOJSON.exists():
        urllib.request.urlretrieve(US_STATES_GEOJSON_URL, US_STATES_GEOJSON)

    with US_STATES_GEOJSON.open(encoding="utf-8") as file:
        return json.load(file)
