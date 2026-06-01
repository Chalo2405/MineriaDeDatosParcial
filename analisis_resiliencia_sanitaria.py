import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import json
from pathlib import Path

import branca.colormap as cm
import folium
from folium import plugins
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from covid_utils import (
    OUTPUT_DIR,
    STATE_ABBR,
    cargar_geojson_estados,
    cargar_y_limpiar_datos,
    ultimo_registro_con_columnas,
    ultimo_registro_por_estado,
)


FEATURES_CLUSTER_BASE = [
    "Incident_Rate",
    "Testing_Rate",
    "Case_Fatality_Ratio",
]

FEATURES_STRESS = [
    "peak_case_share_pct",
    "peak_incidence_per_100k",
    "window_fatality_ratio",
]

FEATURES_REPORTE = FEATURES_CLUSTER_BASE + FEATURES_STRESS

PROFILE_ORDER = {
    "Alta incidencia / testing medio-bajo": 1,
    "Testing muy alto / letalidad media-baja": 2,
    "Letalidad alta / testing medio": 3,
    "Incidencia menor / letalidad baja": 4,
}

PROFILE_COLORS = {
    1: "#2563eb",
    2: "#16a34a",
    3: "#dc2626",
    4: "#9333ea",
}


def construir_metricas_resiliencia() -> pd.DataFrame:
    df = cargar_y_limpiar_datos()
    estados_usa = set(STATE_ABBR) - {"District of Columbia"}

    ultimo = ultimo_registro_por_estado(df)
    ultimo = ultimo[ultimo["Province_State"].isin(estados_usa)].dropna(
        subset=["Codigo_Estado", "Incident_Rate", "Case_Fatality_Ratio"]
    )
    testing = ultimo_registro_con_columnas(df, ["Testing_Rate"])[
        ["Province_State", "Testing_Rate"]
    ]
    ultimo = (
        ultimo.drop(columns=["Testing_Rate"], errors="ignore")
        .merge(testing, on="Province_State", how="left")
        .copy()
    )
    ultimo["Testing_Rate_Original"] = ultimo["Testing_Rate"]
    ultimo["Testing_Rate_Imputado"] = ultimo["Testing_Rate"].isna() | (
        ultimo["Testing_Rate"] <= 1000
    )
    mediana_testing = ultimo.loc[
        ~ultimo["Testing_Rate_Imputado"], "Testing_Rate"
    ].median()
    ultimo.loc[ultimo["Testing_Rate_Imputado"], "Testing_Rate"] = mediana_testing

    # Estimamos poblacion a partir de Confirmed e Incident_Rate.
    # Incident_Rate = Confirmed / Population * 100000.
    ultimo["Population_Estimated"] = (
        ultimo["Confirmed"] / ultimo["Incident_Rate"] * 100_000
    )

    semanal = (
        df[df["Province_State"].isin(ultimo["Province_State"])]
        .groupby(["Province_State", pd.Grouper(key="Fecha", freq="W")])
        .agg(
            cases_week=("Casos_Nuevos", "sum"),
            deaths_week=("Muertes_Nuevas", "sum"),
        )
        .reset_index()
    )

    filas = []
    for estado, grupo in semanal.groupby("Province_State"):
        grupo = grupo.sort_values("Fecha").reset_index(drop=True)
        total_casos = grupo["cases_week"].sum()
        if total_casos <= 1000:
            continue

        idx_pico = grupo["cases_week"].idxmax()
        pico = grupo.loc[idx_pico]
        ventana = grupo.iloc[idx_pico : idx_pico + 5]
        registro = ultimo[ultimo["Province_State"] == estado]
        if registro.empty:
            continue

        registro = registro.iloc[0]
        casos_ventana = ventana["cases_week"].sum()
        muertes_ventana = ventana["deaths_week"].sum()

        filas.append(
            {
                "Province_State": estado,
                "Region": registro["Region"],
                "Codigo_Estado": registro["Codigo_Estado"],
                "Population_Estimated": registro["Population_Estimated"],
                "Testing_Rate": registro["Testing_Rate"],
                "Testing_Rate_Original": registro["Testing_Rate_Original"],
                "Testing_Rate_Imputado": registro["Testing_Rate_Imputado"],
                "Incident_Rate": registro["Incident_Rate"],
                "Case_Fatality_Ratio": registro["Case_Fatality_Ratio"],
                "peak_week_date": pico["Fecha"],
                "peak_week_cases": pico["cases_week"],
                "peak_case_share_pct": pico["cases_week"] / total_casos * 100,
                "peak_incidence_per_100k": (
                    pico["cases_week"] / registro["Population_Estimated"] * 100_000
                ),
                "window_cases_5w": casos_ventana,
                "window_deaths_5w": muertes_ventana,
                "window_fatality_ratio": (
                    muertes_ventana / casos_ventana * 100 if casos_ventana > 0 else 0
                ),
            }
        )

    metricas = pd.DataFrame(filas).dropna(subset=FEATURES_REPORTE)
    return metricas


def nombrar_perfiles(datos: pd.DataFrame) -> dict[int, str]:
    resumen = datos.groupby("Cluster_Modelo")[FEATURES_CLUSTER_BASE].mean()
    pendientes = set(resumen.index.tolist())
    etiquetas = {}

    cluster_testing_alto = resumen["Testing_Rate"].idxmax()
    etiquetas[cluster_testing_alto] = "Testing muy alto / letalidad media-baja"
    pendientes.remove(cluster_testing_alto)

    cluster_letalidad_alta = resumen.loc[list(pendientes), "Case_Fatality_Ratio"].idxmax()
    etiquetas[cluster_letalidad_alta] = "Letalidad alta / testing medio"
    pendientes.remove(cluster_letalidad_alta)

    cluster_incidencia_alta = resumen.loc[list(pendientes), "Incident_Rate"].idxmax()
    etiquetas[cluster_incidencia_alta] = "Alta incidencia / testing medio-bajo"
    pendientes.remove(cluster_incidencia_alta)

    for cluster in pendientes:
        etiquetas[cluster] = "Incidencia menor / letalidad baja"

    return etiquetas


def asignar_clusters(metricas: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    scaler = StandardScaler()
    matriz = scaler.fit_transform(metricas[FEATURES_CLUSTER_BASE])
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)

    datos = metricas.copy()
    datos["Cluster_Modelo"] = modelo.fit_predict(matriz)
    etiquetas = nombrar_perfiles(datos)
    datos["Perfil_Nombre"] = datos["Cluster_Modelo"].map(etiquetas)
    datos["Perfil_ID"] = datos["Perfil_Nombre"].map(PROFILE_ORDER)
    datos["Perfil_Label"] = datos.apply(
        lambda row: f"Cluster {int(row['Perfil_ID'])}: {row['Perfil_Nombre']}",
        axis=1,
    )
    datos.to_csv(OUTPUT_DIR / "metricas_resiliencia_sanitaria.csv", index=False)

    resumen = (
        datos.groupby("Perfil_Label")[FEATURES_REPORTE]
        .mean()
        .round(2)
        .reset_index()
    )
    resumen.to_csv(OUTPUT_DIR / "resumen_clusters_resiliencia.csv", index=False)
    return datos


def crear_scatter_resiliencia(datos: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))

    min_share = datos["peak_case_share_pct"].min()
    max_share = datos["peak_case_share_pct"].max()

    def escalar_tamanio(serie: pd.Series) -> pd.Series:
        normalizado = (serie - min_share) / (max_share - min_share)
        return 90 + normalizado * 760

    for perfil_id, grupo in datos.sort_values("Perfil_ID").groupby("Perfil_ID"):
        ax.scatter(
            grupo["Incident_Rate"],
            grupo["Case_Fatality_Ratio"],
            s=escalar_tamanio(grupo["peak_case_share_pct"]),
            color=PROFILE_COLORS[perfil_id],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.8,
            label=grupo["Perfil_Label"].iloc[0],
        )

    ax.axvline(datos["Incident_Rate"].median(), color="#94a3b8", linestyle="--", linewidth=1.5)
    ax.axhline(datos["Case_Fatality_Ratio"].median(), color="#94a3b8", linestyle="--", linewidth=1.5)

    handles_cluster, labels_cluster = ax.get_legend_handles_labels()
    legend_cluster = ax.legend(
        handles_cluster,
        labels_cluster,
        title="Perfil epidemiologico base",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(legend_cluster)

    valores_share = [
        datos["peak_case_share_pct"].quantile(0.25),
        datos["peak_case_share_pct"].quantile(0.50),
        datos["peak_case_share_pct"].quantile(0.75),
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#94a3b8",
            markeredgecolor="white",
            markersize=(escalar_tamanio(pd.Series([valor])).iloc[0] ** 0.5) / 1.45,
            label=f"{valor:.1f}%",
        )
        for valor in valores_share
    ]
    ax.legend(
        handles=size_handles,
        title="% casos en peor semana",
        loc="lower right",
        frameon=True,
    )

    ax.set_title("Clustering epidemiologico base + magnitud de la semana explosiva")
    ax.set_xlabel("Incidencia acumulada (casos por 100,000 habitantes aprox.)")
    ax.set_ylabel("Letalidad acumulada (%)")
    ax.margins(x=0.05, y=0.18)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resiliencia_scatter_clusters.png", dpi=170)
    plt.close(fig)


def crear_boxplot_resiliencia(datos: pd.DataFrame) -> None:
    orden = [
        f"Cluster {PROFILE_ORDER[nombre]}: {nombre}" for nombre in PROFILE_ORDER
    ]
    grupos = [
        datos.loc[datos["Perfil_Label"] == perfil, "window_fatality_ratio"].values
        for perfil in orden
    ]

    fig, ax = plt.subplots(figsize=(13, 6.2))
    box = ax.boxplot(
        grupos,
        patch_artist=True,
        labels=[
            "Alta incidencia\n/testing medio-bajo",
            "Testing muy alto\n/letalidad media-baja",
            "Letalidad alta\n/testing medio",
            "Incidencia menor\n/letalidad baja",
        ],
        medianprops={"color": "#111827", "linewidth": 2},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    for patch, perfil_id in zip(box["boxes"], PROFILE_COLORS):
        patch.set_facecolor(PROFILE_COLORS[perfil_id])
        patch.set_alpha(0.72)

    for i, valores in enumerate(grupos, start=1):
        ax.scatter(
            [i] * len(valores),
            valores,
            color="#0f172a",
            s=24,
            alpha=0.55,
            zorder=3,
        )

    ax.set_title("Prueba de estres: letalidad durante la ventana critica por perfil base")
    ax.set_ylabel("Letalidad en peor semana + 4 semanas posteriores (%)")
    ax.set_xlabel("Perfil epidemiologico base")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resiliencia_boxplot_letalidad_ventana.png", dpi=170)
    plt.close(fig)


def crear_heatmap_perfiles(datos: pd.DataFrame) -> None:
    resumen = datos.groupby("Perfil_Label")[FEATURES_REPORTE].mean()
    orden = [
        f"Cluster {PROFILE_ORDER[nombre]}: {nombre}" for nombre in PROFILE_ORDER
    ]
    resumen = resumen.reindex(orden)
    normalizado = (resumen - resumen.min()) / (resumen.max() - resumen.min())

    fig, ax = plt.subplots(figsize=(14, 6.2))
    im = ax.imshow(normalizado.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(FEATURES_REPORTE)))
    ax.set_xticklabels(
        [
            "Incidencia acumulada",
            "Testing rate",
            "Letalidad acumulada",
            "% casos en peor semana",
            "Pico semanal por 100k",
            "Letalidad ventana 5 sem.",
        ],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(normalizado.index)))
    ax.set_yticklabels(normalizado.index)
    ax.set_title("Union entre perfil epidemiologico base y estres por semana explosiva")

    for i in range(normalizado.shape[0]):
        for j in range(normalizado.shape[1]):
            ax.text(
                j,
                i,
                f"{resumen.iloc[i, j]:,.2f}",
                ha="center",
                va="center",
                color="white" if normalizado.iloc[i, j] > 0.62 else "#0f172a",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Intensidad relativa")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resiliencia_heatmap_perfiles.png", dpi=170)
    plt.close(fig)


def crear_mapa_resiliencia(datos: pd.DataFrame) -> None:
    geojson_estados = cargar_geojson_estados()
    datos_por_estado = datos.set_index("Codigo_Estado").to_dict(orient="index")

    for feature in geojson_estados["features"]:
        codigo = feature.get("id")
        registro = datos_por_estado.get(codigo)
        props = feature["properties"]

        if registro is None:
            props.update(
                {
                    "cluster": "Sin datos",
                    "region": "Sin datos",
                    "incident": "Sin datos",
                    "fatality_global": "Sin datos",
                    "peak_share": "Sin datos",
                    "peak_incidence": "Sin datos",
                    "fatality_window": "Sin datos",
                    "testing": "Sin datos",
                    "peak_date": "Sin datos",
                }
            )
            continue

        props.update(
            {
                "perfil_id": int(registro["Perfil_ID"]),
                "cluster": registro["Perfil_Label"],
                "region": registro["Region"],
                "incident": f"{registro['Incident_Rate']:,.0f}",
                "fatality_global": f"{registro['Case_Fatality_Ratio']:.2f}%",
                "peak_share": f"{registro['peak_case_share_pct']:.2f}%",
                "peak_incidence": f"{registro['peak_incidence_per_100k']:,.0f}",
                "fatality_window": f"{registro['window_fatality_ratio']:.2f}%",
                "testing": f"{registro['Testing_Rate']:,.0f}",
                "peak_date": pd.to_datetime(registro["peak_week_date"]).strftime("%Y-%m-%d"),
            }
        )

    def estilo_estado(feature: dict) -> dict:
        perfil = feature["properties"].get("perfil_id")
        return {
            "fillColor": "#d9d9d9" if perfil is None else PROFILE_COLORS[perfil],
            "color": "#ffffff",
            "weight": 1.1,
            "fillOpacity": 0.84,
        }

    def estilo_resaltado(_: dict) -> dict:
        return {"fillOpacity": 0.97, "weight": 2.6, "color": "#111827"}

    mapa = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("CartoDB Voyager", name="Mapa claro").add_to(mapa)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(mapa)

    folium.GeoJson(
        geojson_estados,
        name="Clusters de resiliencia sanitaria",
        style_function=estilo_estado,
        highlight_function=estilo_resaltado,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "name",
                "cluster",
                "region",
                "incident",
                "testing",
                "fatality_global",
                "peak_date",
                "peak_share",
                "peak_incidence",
                "fatality_window",
            ],
            aliases=[
                "Estado:",
                "Perfil base:",
                "Region:",
                "Incidencia acumulada:",
                "Testing rate:",
                "Letalidad acumulada:",
                "Fecha peor semana:",
                "% casos en peor semana:",
                "Pico semanal por 100k:",
                "Letalidad ventana 5 semanas:",
            ],
            sticky=True,
            labels=True,
            style=(
                "background-color: white; color: #111827; font-family: Arial; "
                "font-size: 13px; padding: 10px; border-radius: 6px; "
                "box-shadow: 0 2px 12px rgba(0,0,0,0.18);"
            ),
        ),
    ).add_to(mapa)

    leyenda = """
    <div style="
        position: fixed; bottom: 32px; right: 24px; z-index: 9999;
        background: rgba(255,255,255,0.94); padding: 12px 14px;
        border-radius: 10px; box-shadow: 0 6px 24px rgba(15,23,42,0.18);
        font-family: Arial, sans-serif; font-size: 13px; color: #111827;">
        <b>Clusters epidemiologicos base</b><br>
        <span style="color:#2563eb;">■</span> Alta incidencia / testing medio-bajo<br>
        <span style="color:#16a34a;">■</span> Testing muy alto / letalidad media-baja<br>
        <span style="color:#dc2626;">■</span> Letalidad alta / testing medio<br>
        <span style="color:#9333ea;">■</span> Incidencia menor / letalidad baja<br>
    </div>
    """
    titulo = """
    <div style="
        position: fixed; top: 18px; left: 60px; z-index: 9999;
        background: rgba(255,255,255,0.94); padding: 14px 18px;
        border-radius: 10px; box-shadow: 0 6px 24px rgba(15,23,42,0.18);
        font-family: Arial, sans-serif; max-width: 520px;">
        <div style="font-size: 18px; font-weight: 700; color: #111827;">
            Perfil base + prueba de estres por semana explosiva
        </div>
        <div style="font-size: 13px; color: #374151; margin-top: 4px;">
            El color muestra el cluster base; el tooltip muestra como resistio cada estado durante su peor semana.
        </div>
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(titulo))
    mapa.get_root().html.add_child(folium.Element(leyenda))
    plugins.Fullscreen(position="topright").add_to(mapa)
    plugins.MiniMap(toggle_display=True, position="bottomleft").add_to(mapa)
    folium.LayerControl(collapsed=True).add_to(mapa)
    mapa.save(OUTPUT_DIR / "mapa_resiliencia_clusters.html")


def crear_dashboard_interactivo(datos: pd.DataFrame) -> None:
    def limpiar_para_json(valor):
        if isinstance(valor, dict):
            return {str(clave): limpiar_para_json(contenido) for clave, contenido in valor.items()}
        if isinstance(valor, list):
            return [limpiar_para_json(contenido) for contenido in valor]
        if pd.isna(valor):
            return None
        return valor

    df = cargar_y_limpiar_datos()
    estados_validos = set(datos["Province_State"])
    geojson_estados = cargar_geojson_estados()

    series_semanales = (
        df[df["Province_State"].isin(estados_validos)]
        .groupby(["Province_State", pd.Grouper(key="Fecha", freq="W")])
        .agg(
            cases_week=("Casos_Nuevos", "sum"),
            deaths_week=("Muertes_Nuevas", "sum"),
        )
        .reset_index()
    )
    series_semanales["Fecha"] = series_semanales["Fecha"].dt.strftime("%Y-%m-%d")

    columnas_tabla = [
        "Fecha",
        "Confirmed",
        "Deaths",
        "Incident_Rate",
        "Testing_Rate",
        "Case_Fatality_Ratio",
        "Casos_Nuevos",
        "Muertes_Nuevas",
    ]

    filas_estado = {}
    timeline_estado = {}
    for estado in sorted(estados_validos):
        filas = (
            df[df["Province_State"] == estado]
            .sort_values("Fecha")
            .tail(14)[columnas_tabla]
            .copy()
        )
        filas["Fecha"] = filas["Fecha"].dt.strftime("%Y-%m-%d")
        filas_estado[estado] = filas.sort_values("Fecha", ascending=False).round(3).to_dict(
            orient="records"
        )
        timeline_estado[estado] = (
            series_semanales[series_semanales["Province_State"] == estado]
            .sort_values("Fecha")[["Fecha", "cases_week", "deaths_week"]]
            .round(2)
            .to_dict(orient="records")
        )

    datos_estado = {}
    for registro in datos.sort_values("Province_State").to_dict(orient="records"):
        estado = registro["Province_State"]
        datos_estado[estado] = {
            "state": estado,
            "abbr": registro["Codigo_Estado"],
            "region": registro["Region"],
            "profileId": int(registro["Perfil_ID"]),
            "profile": registro["Perfil_Label"],
            "profileName": registro["Perfil_Nombre"],
            "color": PROFILE_COLORS[int(registro["Perfil_ID"])],
            "metrics": {
                "Incident_Rate": float(registro["Incident_Rate"]),
                "Testing_Rate": float(registro["Testing_Rate"]),
                "Case_Fatality_Ratio": float(registro["Case_Fatality_Ratio"]),
                "peak_case_share_pct": float(registro["peak_case_share_pct"]),
                "peak_incidence_per_100k": float(registro["peak_incidence_per_100k"]),
                "window_fatality_ratio": float(registro["window_fatality_ratio"]),
            },
            "peakWeekDate": pd.to_datetime(registro["peak_week_date"]).strftime(
                "%Y-%m-%d"
            ),
            "peakWeekCases": float(registro["peak_week_cases"]),
            "windowCases5w": float(registro["window_cases_5w"]),
            "windowDeaths5w": float(registro["window_deaths_5w"]),
            "testingImputed": bool(registro["Testing_Rate_Imputado"]),
            "testingOriginal": (
                None
                if pd.isna(registro["Testing_Rate_Original"])
                else float(registro["Testing_Rate_Original"])
            ),
            "rawRows": filas_estado[estado],
            "timeline": timeline_estado[estado],
        }

    resumen_cluster = (
        datos.groupby("Perfil_Label")[FEATURES_REPORTE].mean().round(4).to_dict(orient="index")
    )
    rangos = {}
    for variable in FEATURES_REPORTE:
        rangos[variable] = {
            "min": float(datos[variable].min()),
            "max": float(datos[variable].max()),
        }

    payload = {
        "states": datos_estado,
        "clusters": resumen_cluster,
        "ranges": rangos,
        "features": FEATURES_REPORTE,
        "featureLabels": {
            "Incident_Rate": "Incidencia acumulada",
            "Testing_Rate": "Testing rate",
            "Case_Fatality_Ratio": "Letalidad acumulada",
            "peak_case_share_pct": "% peor semana",
            "peak_incidence_per_100k": "Pico semanal / 100k",
            "window_fatality_ratio": "Letalidad ventana",
        },
        "profileColors": PROFILE_COLORS,
        "geojson": geojson_estados,
    }
    payload = limpiar_para_json(payload)

    template = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dashboard Interactivo de Resiliencia Sanitaria</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #e9eef5;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --line: #d7e1ed;
      --soft: #f8fafc;
      --danger: #dc2626;
      --shadow: 0 14px 34px rgba(15, 23, 42, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      overflow: hidden;
    }

    .app {
      height: 100vh;
      padding: 12px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 10px;
    }

    .topbar {
      min-height: 72px;
      display: grid;
      grid-template-columns: 1.2fr 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(135deg, #0f172a, #1e40af 58%, #0f766e);
      color: white;
      box-shadow: var(--shadow);
    }

    h1 {
      margin: 0 0 4px;
      font-size: clamp(18px, 2vw, 28px);
      line-height: 1.08;
      letter-spacing: 0;
    }

    .topbar p {
      margin: 0;
      color: #dbeafe;
      font-size: 13px;
    }

    .quick-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }

    .stat {
      padding: 8px 10px;
      border: 1px solid rgba(255,255,255,0.24);
      border-radius: 8px;
      background: rgba(255,255,255,0.11);
    }

    .stat strong {
      display: block;
      font-size: 18px;
      line-height: 1;
    }

    .stat span {
      color: #dbeafe;
      font-size: 11px;
      font-weight: 700;
    }

    .selected-pill {
      max-width: 300px;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(255,255,255,0.16);
      border: 1px solid rgba(255,255,255,0.26);
      font-weight: 800;
      font-size: 13px;
    }

    .grid {
      min-height: 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 10px;
    }

    .panel {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
    }

    .panel-title {
      margin: 0;
      font-size: 15px;
      line-height: 1.15;
    }

    .panel-subtitle {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 12px;
    }

    .tag {
      flex: 0 0 auto;
      padding: 5px 8px;
      border-radius: 999px;
      color: #1d4ed8;
      background: #dbeafe;
      font-size: 11px;
      font-weight: 900;
      white-space: nowrap;
    }

    #map {
      height: 100%;
      min-height: 0;
    }

    .map-legend {
      padding: 8px 10px;
      background: rgba(255,255,255,0.94);
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(15,23,42,0.16);
      font-size: 12px;
      line-height: 1.5;
    }

    .map-legend b { display: block; margin-bottom: 4px; }
    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 2px;
      vertical-align: middle;
    }

    .data-layout {
      min-height: 0;
      display: grid;
      grid-template-columns: 170px 1fr;
      gap: 0;
    }

    .state-list {
      min-height: 0;
      border-right: 1px solid var(--line);
      background: var(--soft);
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .state-list input {
      width: calc(100% - 16px);
      margin: 8px;
      padding: 8px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font: inherit;
      font-size: 12px;
    }

    .state-buttons {
      min-height: 0;
      overflow: auto;
      padding: 0 8px 8px;
    }

    .state-button {
      width: 100%;
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      padding: 7px 6px;
      border-radius: 7px;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      text-align: left;
    }

    .state-button:hover,
    .state-button.active {
      border-color: #bfdbfe;
      background: #eff6ff;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      flex: 0 0 auto;
    }

    .raw-area {
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .metric-strip {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
    }

    .mini-metric {
      padding: 8px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
    }

    .mini-metric strong {
      display: block;
      font-size: 14px;
      line-height: 1.1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .mini-metric span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }

    .table-wrap {
      min-height: 0;
      overflow: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid #e2e8f0;
      text-align: right;
      white-space: nowrap;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 1;
      color: #334155;
      background: #f8fafc;
      font-size: 11px;
    }

    th:first-child, td:first-child { text-align: left; }

    .chart {
      min-height: 0;
      width: 100%;
      height: 100%;
    }

    @media (max-width: 1100px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; }
      .topbar,
      .grid,
      .data-layout {
        grid-template-columns: 1fr;
      }
      .grid { grid-template-rows: repeat(4, 460px); }
      .state-list { border-right: 0; border-bottom: 1px solid var(--line); }
      .state-buttons { max-height: 160px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div>
        <h1>Dashboard interactivo de resiliencia sanitaria</h1>
        <p>Haz clic en un estado del mapa o de la lista para activar su radiografia epidemiologica.</p>
      </div>
      <div class="quick-stats">
        <div class="stat"><strong>50</strong><span>estados analizados</span></div>
        <div class="stat"><strong>4</strong><span>clusters base</span></div>
        <div class="stat"><strong>5 sem.</strong><span>ventana critica</span></div>
      </div>
      <div class="selected-pill" id="selectedPill">Seleccion: --</div>
    </header>

    <main class="grid">
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">Cuadrante 1: mapa del clustering</h2>
            <p class="panel-subtitle">El color del estado representa su cluster epidemiologico base.</p>
          </div>
          <span class="tag">Detonante de clics</span>
        </div>
        <div id="map"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">Cuadrante 2: tabla de datos crudos</h2>
            <p class="panel-subtitle">Ultimas filas historicas del estado seleccionado.</p>
          </div>
          <span class="tag">Selector</span>
        </div>
        <div class="data-layout">
          <aside class="state-list">
            <input id="stateSearch" type="search" placeholder="Buscar estado..." />
            <div class="state-buttons" id="stateButtons"></div>
          </aside>
          <div class="raw-area">
            <div class="metric-strip" id="metricStrip"></div>
            <div class="table-wrap">
              <table>
                <thead id="rawHead"></thead>
                <tbody id="rawBody"></tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">Cuadrante 3: grafico de comparacion</h2>
            <p class="panel-subtitle">Radar normalizado: estado seleccionado vs promedio de su cluster.</p>
          </div>
          <span class="tag">Radiografia</span>
        </div>
        <div id="radarChart" class="chart"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">Cuadrante 4: linea de tiempo historica</h2>
            <p class="panel-subtitle">La franja roja marca la semana explosiva de contagios.</p>
          </div>
          <span class="tag">Semana explosiva</span>
        </div>
        <div id="timelineChart" class="chart"></div>
      </section>
    </main>
  </div>

  <script>
    const appData = __APP_DATA__;
    const states = appData.states;
    const stateNames = Object.keys(states).sort();
    let selectedState = stateNames.includes("California") ? "California" : stateNames[0];
    let selectedLayer = null;
    const layersByState = {};
    const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

    function formatValue(value, suffix = "") {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return numberFmt.format(Number(value)) + suffix;
    }

    function normalized(feature, value) {
      const range = appData.ranges[feature];
      if (!range || range.max === range.min) return 50;
      return ((value - range.min) / (range.max - range.min)) * 100;
    }

    function profileColor(profileId) {
      return appData.profileColors[String(profileId)] || "#94a3b8";
    }

    const map = L.map("map", { zoomControl: false, minZoom: 3 });
    L.control.zoom({ position: "topright" }).addTo(map);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap &copy; CARTO",
      subdomains: "abcd",
      maxZoom: 19
    }).addTo(map);

    const geoLayer = L.geoJSON(appData.geojson, {
      style(feature) {
        const state = states[feature.properties.name];
        return {
          color: "#ffffff",
          weight: 1,
          fillColor: state ? state.color : "#cbd5e1",
          fillOpacity: state ? 0.86 : 0.28
        };
      },
      onEachFeature(feature, layer) {
        const name = feature.properties.name;
        const state = states[name];
        if (state) {
          layersByState[name] = layer;
          layer.bindTooltip(`<b>${name}</b><br>${state.profile}<br>Peor semana: ${state.peakWeekDate}`, {
            sticky: true,
            direction: "auto"
          });
          layer.on("click", () => selectState(name));
          layer.on("mouseover", () => layer.setStyle({ weight: 2.6, color: "#111827", fillOpacity: 0.96 }));
          layer.on("mouseout", () => {
            if (name !== selectedState) layer.setStyle({ weight: 1, color: "#ffffff", fillOpacity: 0.86 });
          });
        }
      }
    }).addTo(map);
    map.fitBounds(geoLayer.getBounds(), { padding: [12, 12] });

    const legend = L.control({ position: "bottomright" });
    legend.onAdd = function () {
      const div = L.DomUtil.create("div", "map-legend");
      div.innerHTML = `
        <b>Clusters base</b>
        <div><span class="swatch" style="background:#2563eb"></span>Alta incidencia / testing medio-bajo</div>
        <div><span class="swatch" style="background:#16a34a"></span>Testing muy alto / letalidad media-baja</div>
        <div><span class="swatch" style="background:#dc2626"></span>Letalidad alta / testing medio</div>
        <div><span class="swatch" style="background:#9333ea"></span>Incidencia menor / letalidad baja</div>
      `;
      return div;
    };
    legend.addTo(map);

    function renderStateButtons(filter = "") {
      const container = document.getElementById("stateButtons");
      const query = filter.trim().toLowerCase();
      container.innerHTML = "";
      stateNames
        .filter(name => name.toLowerCase().includes(query))
        .forEach(name => {
          const state = states[name];
          const button = document.createElement("button");
          button.className = `state-button${name === selectedState ? " active" : ""}`;
          button.innerHTML = `<span>${name}</span><span class="dot" style="background:${state.color}"></span>`;
          button.addEventListener("click", () => selectState(name));
          container.appendChild(button);
        });
    }

    document.getElementById("stateSearch").addEventListener("input", event => {
      renderStateButtons(event.target.value);
    });

    function renderMetricStrip(state) {
      const original = state.testingImputed
        ? `<span>Testing imputado por dato faltante/invalido</span>`
        : `<span>Testing del dataset</span>`;
      document.getElementById("metricStrip").innerHTML = `
        <div class="mini-metric"><strong>${state.profileName}</strong><span>Perfil base</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.Incident_Rate)}</strong><span>Incidencia acumulada</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.Testing_Rate)}</strong>${original}</div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.Case_Fatality_Ratio, "%")}</strong><span>Letalidad acumulada</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.peak_case_share_pct, "%")}</strong><span>% casos en peor semana</span></div>
        <div class="mini-metric"><strong>${state.peakWeekDate}</strong><span>Semana explosiva</span></div>
      `;
    }

    function renderRawTable(state) {
      const columns = [
        ["Fecha", "Fecha"],
        ["Confirmed", "Confirmados"],
        ["Deaths", "Muertes"],
        ["Incident_Rate", "Incidencia"],
        ["Testing_Rate", "Testing"],
        ["Case_Fatality_Ratio", "Letalidad %"],
        ["Casos_Nuevos", "Casos nuevos"],
        ["Muertes_Nuevas", "Muertes nuevas"],
      ];
      document.getElementById("rawHead").innerHTML = `<tr>${columns.map(([, label]) => `<th>${label}</th>`).join("")}</tr>`;
      document.getElementById("rawBody").innerHTML = state.rawRows.map(row => {
        return `<tr>${columns.map(([key]) => {
          const value = key === "Fecha" ? row[key] : formatValue(row[key]);
          return `<td>${value}</td>`;
        }).join("")}</tr>`;
      }).join("");
    }

    function renderRadar(state) {
      const features = appData.features;
      const labels = features.map(feature => appData.featureLabels[feature]);
      const cluster = appData.clusters[state.profile];
      const stateValues = features.map(feature => normalized(feature, state.metrics[feature]));
      const clusterValues = features.map(feature => normalized(feature, cluster[feature]));
      const closedLabels = [...labels, labels[0]];
      const closedState = [...stateValues, stateValues[0]];
      const closedCluster = [...clusterValues, clusterValues[0]];

      Plotly.react("radarChart", [
        {
          type: "scatterpolar",
          r: closedCluster,
          theta: closedLabels,
          fill: "toself",
          name: "Promedio del cluster",
          line: { color: "#94a3b8", width: 2 },
          fillcolor: "rgba(148, 163, 184, 0.22)"
        },
        {
          type: "scatterpolar",
          r: closedState,
          theta: closedLabels,
          fill: "toself",
          name: state.state,
          line: { color: state.color, width: 3 },
          fillcolor: `${state.color}33`
        }
      ], {
        margin: { l: 56, r: 24, t: 38, b: 34 },
        paper_bgcolor: "rgba(0,0,0,0)",
        polar: {
          radialaxis: { visible: true, range: [0, 100], tickfont: { size: 10 }, gridcolor: "#dbe3ee" },
          angularaxis: { tickfont: { size: 11 } }
        },
        legend: { orientation: "h", y: -0.08 },
        title: { text: `${state.state} vs ${state.profileName}`, font: { size: 15 } }
      }, { responsive: true, displayModeBar: false });
    }

    function renderTimeline(state) {
      const x = state.timeline.map(row => row.Fecha);
      const cases = state.timeline.map(row => row.cases_week);
      const deaths = state.timeline.map(row => row.deaths_week);
      const peakStart = state.peakWeekDate;
      const peakEndDate = new Date(`${state.peakWeekDate}T00:00:00`);
      peakEndDate.setDate(peakEndDate.getDate() + 7);
      const peakEnd = peakEndDate.toISOString().slice(0, 10);

      Plotly.react("timelineChart", [
        {
          type: "scatter",
          mode: "lines",
          x,
          y: cases,
          name: "Casos semanales",
          line: { color: state.color, width: 3 },
          fill: "tozeroy",
          fillcolor: `${state.color}1f`
        },
        {
          type: "scatter",
          mode: "lines",
          x,
          y: deaths,
          name: "Muertes semanales",
          line: { color: "#111827", width: 1.7, dash: "dot" },
          yaxis: "y2"
        }
      ], {
        margin: { l: 62, r: 58, t: 38, b: 44 },
        paper_bgcolor: "rgba(0,0,0,0)",
        title: { text: `Historia semanal de ${state.state}`, font: { size: 15 } },
        xaxis: { gridcolor: "#e2e8f0" },
        yaxis: { title: "Casos", gridcolor: "#e2e8f0" },
        yaxis2: { title: "Muertes", overlaying: "y", side: "right", showgrid: false },
        legend: { orientation: "h", y: -0.22 },
        shapes: [{
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: peakStart,
          x1: peakEnd,
          y0: 0,
          y1: 1,
          fillcolor: "rgba(220, 38, 38, 0.24)",
          line: { width: 0 },
          layer: "below"
        }],
        annotations: [{
          x: peakStart,
          y: 1,
          yref: "paper",
          text: "Semana explosiva",
          showarrow: true,
          arrowcolor: "#dc2626",
          font: { color: "#991b1b", size: 12 },
          bgcolor: "rgba(255,255,255,0.86)",
          bordercolor: "#fecaca"
        }]
      }, { responsive: true, displayModeBar: false });
    }

    function highlightMapState(name) {
      if (selectedLayer) {
        selectedLayer.setStyle({ weight: 1, color: "#ffffff", fillOpacity: 0.86 });
      }
      selectedLayer = layersByState[name];
      if (selectedLayer) {
        selectedLayer.setStyle({ weight: 3.2, color: "#111827", fillOpacity: 0.98 });
        selectedLayer.bringToFront();
      }
    }

    function selectState(name) {
      selectedState = name;
      const state = states[name];
      document.getElementById("selectedPill").textContent = `Seleccion: ${state.state} · ${state.profileName}`;
      renderStateButtons(document.getElementById("stateSearch").value);
      renderMetricStrip(state);
      renderRawTable(state);
      renderRadar(state);
      renderTimeline(state);
      highlightMapState(name);
    }

    renderStateButtons();
    selectState(selectedState);
    window.addEventListener("resize", () => {
      map.invalidateSize();
      Plotly.Plots.resize("radarChart");
      Plotly.Plots.resize("timelineChart");
    });
  </script>
</body>
</html>
"""
    html = template.replace(
        "__APP_DATA__",
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
    )
    Path("dashboard_resiliencia.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    metricas = construir_metricas_resiliencia()
    datos = asignar_clusters(metricas)

    print("\nPROBLEMA GENERAL")
    print(
        "Identificacion de perfiles de resiliencia sanitaria mediante clustering "
        "base y prueba de estres por semana explosiva."
    )
    print("\nVariables usadas para el clustering base:")
    for variable in FEATURES_CLUSTER_BASE:
        print(f"- {variable}")

    print("\nVariables usadas para evaluar la semana explosiva:")
    for variable in FEATURES_STRESS:
        print(f"- {variable}")

    resumen = (
        datos.groupby("Perfil_Label")[FEATURES_REPORTE]
        .mean()
        .round(2)
    )
    print("\nResumen de clusters:")
    print(resumen.to_string())

    print("\nEstados por perfil:")
    for cluster, grupo in datos.sort_values(["Perfil_ID", "Province_State"]).groupby(
        "Perfil_Label"
    ):
        estados = ", ".join(grupo["Province_State"].tolist())
        print(f"{cluster}: {estados}")

    crear_scatter_resiliencia(datos)
    crear_boxplot_resiliencia(datos)
    crear_heatmap_perfiles(datos)
    crear_mapa_resiliencia(datos)
    crear_dashboard_interactivo(datos)


if __name__ == "__main__":
    main()
