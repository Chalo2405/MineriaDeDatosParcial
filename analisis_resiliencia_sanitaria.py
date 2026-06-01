import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

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
    cargar_geojson_estados,
    cargar_y_limpiar_datos,
    ultimo_registro_con_columnas,
    ultimo_registro_por_estado,
)


FEATURES_RESILIENCIA = [
    "peak_case_share_pct",
    "peak_incidence_per_100k",
    "Testing_Rate",
    "window_fatality_ratio",
]

CLUSTER_COLORS = {
    0: "#f59e0b",
    1: "#2563eb",
    2: "#dc2626",
    3: "#16a34a",
}

CLUSTER_NAMES = {
    0: "Brote explosivo / control intermedio",
    1: "Baja explosividad / resiliencia intermedia",
    2: "Perfil critico / letalidad alta",
    3: "Alta capacidad de testeo / letalidad baja",
}


def construir_metricas_resiliencia() -> pd.DataFrame:
    df = cargar_y_limpiar_datos()

    ultimo = ultimo_registro_por_estado(df).dropna(
        subset=["Codigo_Estado", "Incident_Rate", "Case_Fatality_Ratio"]
    )
    testing = ultimo_registro_con_columnas(df, ["Testing_Rate"])[
        ["Province_State", "Testing_Rate"]
    ]
    ultimo = (
        ultimo.drop(columns=["Testing_Rate"], errors="ignore")
        .merge(testing, on="Province_State", how="left")
        .query("Testing_Rate > 1000")
        .copy()
    )

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

    metricas = pd.DataFrame(filas).dropna(subset=FEATURES_RESILIENCIA)
    return metricas


def asignar_clusters(metricas: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    scaler = StandardScaler()
    matriz = scaler.fit_transform(metricas[FEATURES_RESILIENCIA])
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)

    datos = metricas.copy()
    datos["Cluster"] = modelo.fit_predict(matriz)
    datos["Cluster_Label"] = datos["Cluster"].map(
        lambda cluster: f"Cluster {cluster + 1}: {CLUSTER_NAMES[cluster]}"
    )
    datos.to_csv(OUTPUT_DIR / "metricas_resiliencia_sanitaria.csv", index=False)

    resumen = (
        datos.groupby("Cluster_Label")[FEATURES_RESILIENCIA + ["Case_Fatality_Ratio"]]
        .mean()
        .round(2)
        .reset_index()
    )
    resumen.to_csv(OUTPUT_DIR / "resumen_clusters_resiliencia.csv", index=False)
    return datos


def crear_scatter_resiliencia(datos: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))

    min_testing = datos["Testing_Rate"].min()
    max_testing = datos["Testing_Rate"].max()

    def escalar_tamanio(serie: pd.Series) -> pd.Series:
        normalizado = (serie - min_testing) / (max_testing - min_testing)
        return 70 + normalizado * 560

    for cluster, grupo in datos.groupby("Cluster"):
        ax.scatter(
            grupo["peak_incidence_per_100k"],
            grupo["window_fatality_ratio"],
            s=escalar_tamanio(grupo["Testing_Rate"]),
            color=CLUSTER_COLORS[cluster],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.8,
            label=f"Cluster {cluster + 1}: {CLUSTER_NAMES[cluster]}",
        )

    handles_cluster, labels_cluster = ax.get_legend_handles_labels()
    legend_cluster = ax.legend(
        handles_cluster,
        labels_cluster,
        title="Perfil de resiliencia sanitaria",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(legend_cluster)

    valores_testing = [
        datos["Testing_Rate"].quantile(0.25),
        datos["Testing_Rate"].quantile(0.50),
        datos["Testing_Rate"].quantile(0.75),
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#94a3b8",
            markeredgecolor="white",
            markersize=(escalar_tamanio(pd.Series([valor])).iloc[0] ** 0.5),
            label=f"{valor:,.0f}",
        )
        for valor in valores_testing
    ]
    ax.legend(
        handles=size_handles,
        title="Testing rate",
        loc="lower right",
        frameon=True,
    )

    ax.set_title("Perfiles de resiliencia ante brotes explosivos")
    ax.set_xlabel("Intensidad del pico semanal (casos por 100,000 habitantes)")
    ax.set_ylabel("Letalidad en ventana critica de 5 semanas (%)")
    ax.text(
        0.01,
        0.02,
        "Nota: puntos mas grandes = mayor testing rate",
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#cbd5e1",
            alpha=0.9,
        ),
    )
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resiliencia_scatter_clusters.png", dpi=170)
    plt.close(fig)


def crear_heatmap_perfiles(datos: pd.DataFrame) -> None:
    resumen = datos.groupby("Cluster_Label")[FEATURES_RESILIENCIA].mean()
    normalizado = (resumen - resumen.min()) / (resumen.max() - resumen.min())

    fig, ax = plt.subplots(figsize=(13, 5.6))
    im = ax.imshow(normalizado.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(FEATURES_RESILIENCIA)))
    ax.set_xticklabels(
        [
            "% casos en peor semana",
            "Pico semanal por 100k",
            "Testing rate",
            "Letalidad ventana 5 sem.",
        ],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(normalizado.index)))
    ax.set_yticklabels(normalizado.index)
    ax.set_title("Perfil normalizado de cada cluster de resiliencia")

    for i in range(normalizado.shape[0]):
        for j in range(normalizado.shape[1]):
            ax.text(
                j,
                i,
                f"{resumen.iloc[i, j]:,.2f}",
                ha="center",
                va="center",
                color="#0f172a",
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
                    "peak_share": "Sin datos",
                    "peak_incidence": "Sin datos",
                    "fatality_window": "Sin datos",
                    "testing": "Sin datos",
                }
            )
            continue

        props.update(
            {
                "cluster_id": int(registro["Cluster"]),
                "cluster": registro["Cluster_Label"],
                "region": registro["Region"],
                "peak_share": f"{registro['peak_case_share_pct']:.2f}%",
                "peak_incidence": f"{registro['peak_incidence_per_100k']:,.0f}",
                "fatality_window": f"{registro['window_fatality_ratio']:.2f}%",
                "testing": f"{registro['Testing_Rate']:,.0f}",
            }
        )

    def estilo_estado(feature: dict) -> dict:
        cluster = feature["properties"].get("cluster_id")
        return {
            "fillColor": "#d9d9d9" if cluster is None else CLUSTER_COLORS[cluster],
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
                "peak_share",
                "peak_incidence",
                "fatality_window",
                "testing",
            ],
            aliases=[
                "Estado:",
                "Perfil:",
                "Region:",
                "% casos en peor semana:",
                "Pico semanal por 100k:",
                "Letalidad ventana 5 semanas:",
                "Testing rate:",
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
        <b>Perfiles de resiliencia</b><br>
        <span style="color:#f59e0b;">■</span> Brote explosivo / control intermedio<br>
        <span style="color:#2563eb;">■</span> Baja explosividad / resiliencia intermedia<br>
        <span style="color:#dc2626;">■</span> Perfil critico / letalidad alta<br>
        <span style="color:#16a34a;">■</span> Alta capacidad de testeo / letalidad baja<br>
    </div>
    """
    titulo = """
    <div style="
        position: fixed; top: 18px; left: 60px; z-index: 9999;
        background: rgba(255,255,255,0.94); padding: 14px 18px;
        border-radius: 10px; box-shadow: 0 6px 24px rgba(15,23,42,0.18);
        font-family: Arial, sans-serif; max-width: 520px;">
        <div style="font-size: 18px; font-weight: 700; color: #111827;">
            Clusters de resiliencia sanitaria
        </div>
        <div style="font-size: 13px; color: #374151; margin-top: 4px;">
            Agrupacion de estados segun brote explosivo, testing y letalidad durante la ventana critica.
        </div>
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(titulo))
    mapa.get_root().html.add_child(folium.Element(leyenda))
    plugins.Fullscreen(position="topright").add_to(mapa)
    plugins.MiniMap(toggle_display=True, position="bottomleft").add_to(mapa)
    folium.LayerControl(collapsed=True).add_to(mapa)
    mapa.save(OUTPUT_DIR / "mapa_resiliencia_clusters.html")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    metricas = construir_metricas_resiliencia()
    datos = asignar_clusters(metricas)

    print("\nPROBLEMA GENERAL")
    print(
        "Identificacion de perfiles de resiliencia sanitaria mediante clustering "
        "durante brotes epidemiologicos explosivos."
    )
    print("\nVariables usadas en clustering:")
    for variable in FEATURES_RESILIENCIA:
        print(f"- {variable}")

    resumen = (
        datos.groupby("Cluster_Label")[FEATURES_RESILIENCIA + ["Case_Fatality_Ratio"]]
        .mean()
        .round(2)
    )
    print("\nResumen de clusters:")
    print(resumen.to_string())

    print("\nEstados por perfil:")
    for cluster, grupo in datos.sort_values(["Cluster", "Province_State"]).groupby(
        "Cluster_Label"
    ):
        estados = ", ".join(grupo["Province_State"].tolist())
        print(f"{cluster}: {estados}")

    crear_scatter_resiliencia(datos)
    crear_heatmap_perfiles(datos)
    crear_mapa_resiliencia(datos)


if __name__ == "__main__":
    main()
