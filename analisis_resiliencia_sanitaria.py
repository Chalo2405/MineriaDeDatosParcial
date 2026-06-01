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


if __name__ == "__main__":
    main()
