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
        )
        .reset_index()
    )
    series_semanales["Fecha"] = series_semanales["Fecha"].dt.strftime("%Y-%m-%d")

    timeline_estado = {}
    for estado in sorted(estados_validos):
        timeline_estado[estado] = (
            series_semanales[series_semanales["Province_State"] == estado]
            .sort_values("Fecha")[["Fecha", "cases_week"]]
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
      --bg: #edf2f7;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --line: #d8e2ee;
      --soft: #f8fafc;
      --danger: #dc2626;
      --shadow: 0 18px 44px rgba(15, 23, 42, 0.13);
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
      grid-template-columns: minmax(620px, 1fr) minmax(390px, 460px);
      gap: 12px;
    }

    .map-shell,
    .detail-shell {
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .map-shell {
      position: relative;
      display: block;
    }

    .map-head {
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 700;
      width: min(620px, calc(100% - 32px));
      padding: 14px 16px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(10px);
      color: white;
      box-shadow: 0 14px 32px rgba(15, 23, 42, 0.22);
    }

    h1 {
      margin: 0 0 4px;
      font-size: clamp(22px, 2.4vw, 34px);
      line-height: 1.05;
      letter-spacing: 0;
    }

    .map-head p {
      margin: 0;
      color: #dbeafe;
      font-size: 13.5px;
    }

    #map {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      min-height: 0;
      z-index: 1;
    }

    .map-tools {
      position: absolute;
      left: 16px;
      bottom: 16px;
      z-index: 700;
      width: min(380px, calc(100% - 32px));
      display: grid;
      gap: 8px;
      pointer-events: none;
    }

    .search-toggle {
      width: 46px;
      height: 46px;
      display: inline-grid;
      place-items: center;
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 999px;
      background: rgba(255,255,255,0.96);
      color: #0f172a;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
      cursor: pointer;
      font-size: 21px;
      pointer-events: auto;
    }

    .search-box {
      display: none;
      padding: 10px;
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 10px;
      background: rgba(255,255,255,0.94);
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
      pointer-events: auto;
    }

    .search-box.open {
      display: block;
    }

    .search-box input {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font: inherit;
      font-size: 13px;
    }

    .state-results {
      max-height: 130px;
      overflow: auto;
      margin-top: 8px;
      display: grid;
      gap: 4px;
    }

    .state-button {
      width: 100%;
      border: 1px solid transparent;
      background: #f8fafc;
      color: var(--ink);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 9px;
      border-radius: 7px;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      text-align: left;
    }

    .state-button:hover,
    .state-button.active {
      border-color: #93c5fd;
      background: #eff6ff;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex: 0 0 auto;
    }

    .map-legend {
      padding: 10px 12px;
      background: rgba(255,255,255,0.94);
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(15,23,42,0.16);
      font-size: 12px;
      line-height: 1.5;
      pointer-events: auto;
    }

    .map-legend b { display: block; margin-bottom: 4px; }

    .legend-row,
    .legend-clear {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      display: block;
      font: inherit;
      margin: 2px 0;
      padding: 5px 6px;
      text-align: left;
    }

    .legend-row:hover,
    .legend-row.active {
      background: #eff6ff;
      border-color: #bfdbfe;
    }

    .legend-clear {
      margin-top: 6px;
      color: #2563eb;
      font-weight: 800;
      text-align: center;
    }

    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 2px;
      vertical-align: middle;
    }

    .detail-shell {
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .detail-head {
      padding: 18px 18px 14px;
      color: white;
      background: linear-gradient(135deg, #0f172a, #1e40af 64%, #0f766e);
    }

    .eyebrow {
      margin: 0 0 6px;
      color: #bfdbfe;
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .detail-head h2 {
      margin: 0;
      font-size: 28px;
      line-height: 1.05;
      letter-spacing: 0;
    }

    .profile-pill {
      display: inline-block;
      margin-top: 10px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.16);
      border: 1px solid rgba(255,255,255,0.26);
      font-size: 12px;
      font-weight: 800;
    }

    .detail-body {
      min-height: 0;
      overflow: auto;
      padding: 14px;
    }

    .metric-strip {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }

    .mini-metric {
      padding: 11px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
    }

    .mini-metric strong {
      display: block;
      font-size: 18px;
      line-height: 1.08;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .mini-metric span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }

    .note {
      margin: 0 0 12px;
      padding: 10px 12px;
      border-left: 4px solid var(--danger);
      border-radius: 8px;
      background: #fff7ed;
      color: #7c2d12;
      font-size: 12px;
    }

    .section-card {
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: white;
      overflow: hidden;
    }

    .section-card h3 {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
      font-size: 14px;
    }

    .chart {
      width: 100%;
      height: 330px;
    }

    #timelineChart {
      height: 285px;
    }

    .leaflet-interactive {
      transition: fill-opacity 140ms ease, stroke-width 140ms ease;
    }

    @media (max-width: 980px) {
      body { overflow: auto; }
      .app {
        height: auto;
        min-height: 100vh;
        grid-template-columns: 1fr;
      }
      .map-shell { min-height: 620px; }
      #map { min-height: 620px; }
      .map-head,
      .map-tools {
        position: absolute;
        width: auto;
      }
      .detail-shell { min-height: 780px; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="map-shell">
      <div class="map-head">
        <h1>Mapa interactivo de clusters epidemiologicos</h1>
        <p>Haz clic en un estado coloreado para abrir su radiografia: resumen, radar y semana explosiva.</p>
      </div>
      <div id="map"></div>
      <div class="map-tools">
        <button class="search-toggle" id="searchToggle" aria-label="Buscar estado" title="Buscar estado">&#128269;</button>
        <div class="search-box" id="searchBox">
          <input id="stateSearch" type="search" placeholder="Buscar estado..." />
          <div class="state-results" id="stateButtons"></div>
        </div>
        <div class="map-legend">
          <b>Clusters base</b>
          <button class="legend-row" data-cluster="1"><span class="swatch" style="background:#2563eb"></span>Alta incidencia / testing medio-bajo</button>
          <button class="legend-row" data-cluster="2"><span class="swatch" style="background:#16a34a"></span>Testing muy alto / letalidad media-baja</button>
          <button class="legend-row" data-cluster="3"><span class="swatch" style="background:#dc2626"></span>Letalidad alta / testing medio</button>
          <button class="legend-row" data-cluster="4"><span class="swatch" style="background:#9333ea"></span>Incidencia menor / letalidad baja</button>
          <button class="legend-clear" id="clearClusterFilter">Ver todos</button>
        </div>
      </div>
    </section>

    <aside class="detail-shell">
      <header class="detail-head">
        <p class="eyebrow">Estado seleccionado</p>
        <h2 id="stateTitle">Selecciona un estado</h2>
        <div class="profile-pill" id="profilePill">Cluster pendiente</div>
      </header>

      <div class="detail-body">
        <div class="metric-strip" id="metricStrip"></div>
        <p class="note" id="stressNote"></p>

        <section class="section-card">
          <h3>Radar: estado vs promedio de su cluster</h3>
          <div id="radarChart" class="chart"></div>
        </section>

        <section class="section-card">
          <h3>Linea de tiempo: semana explosiva resaltada</h3>
          <div id="timelineChart" class="chart"></div>
        </section>

      </div>
    </aside>
  </main>

  <script>
    const appData = __APP_DATA__;
    const states = appData.states;
    const stateNames = Object.keys(states).sort();
    let selectedState = stateNames.includes("California") ? "California" : stateNames[0];
    let selectedLayer = null;
    let activeClusterFilter = null;
    const layersByState = {};

    function formatValue(value, suffix = "", digits = 0) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return new Intl.NumberFormat("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      }).format(Number(value)) + suffix;
    }

    function normalized(feature, value) {
      const range = appData.ranges[feature];
      if (!range || range.max === range.min) return 50;
      return ((value - range.min) / (range.max - range.min)) * 100;
    }

    const map = L.map("map", {
      zoomControl: false,
      minZoom: 3,
      maxBoundsViscosity: 0.65
    }).setView([39.5, -98.35], 4);
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
          color: state ? "#ffffff" : "#94a3b8",
          weight: state ? 1.3 : 0.8,
          fillColor: state ? state.color : "#dbe2ea",
          fillOpacity: state ? 0.82 : 0.18,
          dashArray: state ? "" : "3"
        };
      },
      onEachFeature(feature, layer) {
        const name = feature.properties.name;
        const state = states[name];
        if (state) {
          layer.stateName = name;
          layersByState[name] = layer;
          layer.bindTooltip(`<b>${name}</b><br>${state.profile}<br>Peor semana: ${state.peakWeekDate}`, {
            sticky: true,
            direction: "auto"
          });
          layer.on("click", () => selectState(name));
          layer.on("mouseover", () => layer.setStyle({ weight: 2.8, color: "#111827", fillOpacity: 0.96 }));
          layer.on("mouseout", () => {
            if (name !== selectedState) layer.setStyle({ weight: 1.3, color: "#ffffff", fillOpacity: 0.82 });
          });
        }
      }
    }).addTo(map);
    setTimeout(() => {
      map.invalidateSize();
      map.fitBounds(geoLayer.getBounds(), { padding: [28, 28], maxZoom: 5 });
      map.setMaxBounds(geoLayer.getBounds().pad(0.55));
    }, 250);

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
          button.addEventListener("click", () => {
            selectState(name);
            document.getElementById("searchBox").classList.remove("open");
          });
          container.appendChild(button);
        });
    }

    document.getElementById("searchToggle").addEventListener("click", () => {
      const box = document.getElementById("searchBox");
      box.classList.toggle("open");
      if (box.classList.contains("open")) {
        document.getElementById("stateSearch").focus();
      }
    });

    document.getElementById("stateSearch").addEventListener("input", event => {
      renderStateButtons(event.target.value);
    });

    document.querySelectorAll(".legend-row").forEach(button => {
      button.addEventListener("click", () => {
        const cluster = Number(button.dataset.cluster);
        activeClusterFilter = activeClusterFilter === cluster ? null : cluster;
        applyClusterFilter();
      });
    });

    document.getElementById("clearClusterFilter").addEventListener("click", () => {
      activeClusterFilter = null;
      applyClusterFilter();
    });

    function renderMetricStrip(state) {
      const original = state.testingImputed
        ? "Testing imputado"
        : "Testing del dataset";
      document.getElementById("metricStrip").innerHTML = `
        <div class="mini-metric"><strong>${formatValue(state.metrics.Incident_Rate)}</strong><span>Incidencia acumulada</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.Testing_Rate)}</strong><span>${original}</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.Case_Fatality_Ratio, "%", 1)}</strong><span>Letalidad acumulada</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.peak_case_share_pct, "%", 1)}</strong><span>% casos en peor semana</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.peak_incidence_per_100k)}</strong><span>Pico semanal / 100k</span></div>
        <div class="mini-metric"><strong>${formatValue(state.metrics.window_fatality_ratio, "%", 1)}</strong><span>Letalidad ventana</span></div>
      `;
      document.getElementById("stressNote").textContent =
        `${state.state} tuvo su semana explosiva el ${state.peakWeekDate}: ${formatValue(state.peakWeekCases)} casos en una semana, equivalentes al ${formatValue(state.metrics.peak_case_share_pct, "%", 1)} de sus casos historicos.`;
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
        margin: { l: 48, r: 28, t: 18, b: 42 },
        paper_bgcolor: "rgba(0,0,0,0)",
        polar: {
          radialaxis: { visible: true, range: [0, 100], tickfont: { size: 10 }, gridcolor: "#dbe3ee" },
          angularaxis: { tickfont: { size: 11 } }
        },
        legend: { orientation: "h", y: -0.08 }
      }, { responsive: true, displayModeBar: false });
    }

    function renderTimeline(state) {
      const x = state.timeline.map(row => row.Fecha);
      const cases = state.timeline.map(row => row.cases_week);
      const peakStart = state.peakWeekDate;
      const peakEndDate = new Date(`${state.peakWeekDate}T00:00:00`);
      peakEndDate.setDate(peakEndDate.getDate() + 7);
      const peakEnd = peakEndDate.toISOString().slice(0, 10);
      const peakIndex = x.indexOf(peakStart);
      const peakY = peakIndex >= 0 ? cases[peakIndex] : Math.max(...cases);

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
          mode: "markers",
          x: [peakStart],
          y: [peakY],
          name: "Pico explosivo",
          marker: { color: "#dc2626", size: 11, line: { color: "white", width: 2 } }
        }
      ], {
        margin: { l: 56, r: 24, t: 18, b: 46 },
        paper_bgcolor: "rgba(0,0,0,0)",
        xaxis: { gridcolor: "#e2e8f0" },
        yaxis: { title: "Casos", gridcolor: "#e2e8f0" },
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

    function applyClusterFilter() {
      document.querySelectorAll(".legend-row").forEach(button => {
        button.classList.toggle("active", Number(button.dataset.cluster) === activeClusterFilter);
      });
      Object.entries(layersByState).forEach(([name, layer]) => {
        const state = states[name];
        const visible = activeClusterFilter === null || state.profileId === activeClusterFilter;
        layer.setStyle({
          fillOpacity: name === selectedState ? 0.98 : (visible ? 0.82 : 0.13),
          weight: name === selectedState ? 3.4 : (visible ? 1.3 : 0.8),
          color: name === selectedState ? "#111827" : "#ffffff"
        });
      });
    }

    function highlightMapState(name) {
      if (selectedLayer) {
        selectedLayer.setStyle({ weight: 1.3, color: "#ffffff", fillOpacity: 0.82 });
      }
      selectedLayer = layersByState[name];
      if (selectedLayer) {
        selectedLayer.setStyle({ weight: 3.4, color: "#111827", fillOpacity: 0.98 });
        selectedLayer.bringToFront();
        map.panTo(selectedLayer.getBounds().getCenter(), { animate: true, duration: 0.4 });
      }
    }

    function selectState(name) {
      selectedState = name;
      const state = states[name];
      document.getElementById("stateTitle").textContent = state.state;
      const pill = document.getElementById("profilePill");
      pill.textContent = state.profile;
      pill.style.borderColor = `${state.color}66`;
      pill.style.background = `${state.color}44`;
      renderStateButtons(document.getElementById("stateSearch").value);
      renderMetricStrip(state);
      renderRadar(state);
      renderTimeline(state);
      highlightMapState(name);
      applyClusterFilter();
    }

    renderStateButtons();
    selectState(selectedState);
    window.addEventListener("resize", () => {
      map.invalidateSize();
      map.fitBounds(geoLayer.getBounds(), { padding: [28, 28], maxZoom: 5 });
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
