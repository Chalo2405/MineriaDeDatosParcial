import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import folium
from folium import plugins
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from covid_utils import (
    OUTPUT_DIR,
    cargar_geojson_estados,
    cargar_y_limpiar_datos,
    ultimo_registro_con_columnas,
)


FEATURES_CLUSTER = ["Incident_Rate", "Testing_Rate", "Case_Fatality_Ratio"]
CLUSTER_COLORS = {
    0: "#2563eb",
    1: "#dc2626",
    2: "#16a34a",
    3: "#9333ea",
}
CLUSTER_NAMES = {
    0: "Alta incidencia / testing medio-bajo",
    1: "Testing muy alto / letalidad media-baja",
    2: "Letalidad alta / testing medio",
    3: "Incidencia menor / letalidad baja",
}


def preparar_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    datos = ultimo_registro_con_columnas(df, FEATURES_CLUSTER).copy()
    datos = datos.dropna(subset=["Codigo_Estado"]).copy()

    scaler = StandardScaler()
    matriz = scaler.fit_transform(datos[FEATURES_CLUSTER])

    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    datos["Cluster"] = modelo.fit_predict(matriz)
    datos["Cluster_Label"] = datos["Cluster"].map(
        lambda x: f"Cluster {x + 1}: {CLUSTER_NAMES[x]}"
    )

    datos.to_csv(OUTPUT_DIR / "clusters_hipotesis_2.csv", index=False)
    return datos


def crear_scatter_clusters(datos: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))

    for cluster, grupo in datos.groupby("Cluster"):
        tamanio = 45 + (grupo["Incident_Rate"] / datos["Incident_Rate"].max()) * 240
        plt.scatter(
            grupo["Testing_Rate"],
            grupo["Case_Fatality_Ratio"],
            s=tamanio,
            color=CLUSTER_COLORS[cluster],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.7,
            label=f"Cluster {cluster + 1}: {CLUSTER_NAMES[cluster]}",
        )

    for _, row in datos.nlargest(5, "Case_Fatality_Ratio").iterrows():
        plt.annotate(
            row["Province_State"],
            (row["Testing_Rate"], row["Case_Fatality_Ratio"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title("H2: Perfiles epidemiologicos por clustering")
    plt.xlabel("Testing rate")
    plt.ylabel("Letalidad aparente (%)")
    plt.grid(alpha=0.25)
    plt.legend(title="Perfil")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hipotesis_2_clustering_perfiles.png", dpi=160)
    plt.close()


def crear_mapa_clusters(datos: pd.DataFrame) -> None:
    mapa = datos.dropna(subset=["Codigo_Estado"]).copy()
    geojson_estados = cargar_geojson_estados()
    datos_por_estado = mapa.set_index("Codigo_Estado").to_dict(orient="index")

    for feature in geojson_estados["features"]:
        codigo = feature.get("id")
        registro = datos_por_estado.get(codigo)
        props = feature["properties"]

        if registro is None:
            props.update(
                {
                    "cluster": "Sin datos",
                    "region": "Sin datos",
                    "incidencia": "Sin datos",
                    "testing": "Sin datos",
                    "letalidad": "Sin datos",
                }
            )
            continue

        props.update(
            {
                "cluster_id": int(registro["Cluster"]),
                "cluster": registro["Cluster_Label"],
                "region": registro["Region"],
                "incidencia": f"{registro['Incident_Rate']:,.0f}",
                "testing": f"{registro['Testing_Rate']:,.0f}",
                "letalidad": f"{registro['Case_Fatality_Ratio']:.2f}%",
            }
        )

    def estilo_estado(feature: dict) -> dict:
        cluster = feature["properties"].get("cluster_id")
        return {
            "fillColor": "#d9d9d9" if cluster is None else CLUSTER_COLORS[cluster],
            "color": "#ffffff",
            "weight": 1.1,
            "fillOpacity": 0.82,
        }

    def estilo_resaltado(_: dict) -> dict:
        return {"fillOpacity": 0.96, "weight": 2.6, "color": "#111827"}

    mapa_folium = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("CartoDB Voyager", name="Mapa claro").add_to(mapa_folium)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(mapa_folium)

    folium.GeoJson(
        geojson_estados,
        name="Clusters epidemiologicos",
        style_function=estilo_estado,
        highlight_function=estilo_resaltado,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "cluster", "region", "incidencia", "testing", "letalidad"],
            aliases=[
                "Estado:",
                "Cluster:",
                "Region:",
                "Incidencia acumulada:",
                "Testing rate:",
                "Letalidad aparente:",
            ],
            sticky=True,
            labels=True,
            style=(
                "background-color: white; color: #111827; font-family: Arial; "
                "font-size: 13px; padding: 10px; border-radius: 6px; "
                "box-shadow: 0 2px 12px rgba(0,0,0,0.18);"
            ),
        ),
    ).add_to(mapa_folium)

    leyenda = """
    <div style="
        position: fixed; bottom: 32px; right: 24px; z-index: 9999;
        background: rgba(255,255,255,0.94); padding: 12px 14px;
        border-radius: 10px; box-shadow: 0 6px 24px rgba(15,23,42,0.18);
        font-family: Arial, sans-serif; font-size: 13px; color: #111827;">
        <b>Clusters epidemiologicos</b><br>
        <span style="color:#2563eb;">■</span> C1: Alta incidencia / testing medio-bajo<br>
        <span style="color:#dc2626;">■</span> C2: Testing muy alto / letalidad media-baja<br>
        <span style="color:#16a34a;">■</span> C3: Letalidad alta / testing medio<br>
        <span style="color:#9333ea;">■</span> C4: Incidencia menor / letalidad baja<br>
    </div>
    """
    titulo = """
    <div style="
        position: fixed; top: 18px; left: 60px; z-index: 9999;
        background: rgba(255,255,255,0.94); padding: 14px 18px;
        border-radius: 10px; box-shadow: 0 6px 24px rgba(15,23,42,0.18);
        font-family: Arial, sans-serif; max-width: 480px;">
        <div style="font-size: 18px; font-weight: 700; color: #111827;">
            H2: Clusters epidemiologicos por estado
        </div>
        <div style="font-size: 13px; color: #374151; margin-top: 4px;">
            K-Means agrupa estados segun incidencia, testing y letalidad aparente.
            El color no representa geografia: representa perfil estadistico.
        </div>
    </div>
    """
    mapa_folium.get_root().html.add_child(folium.Element(titulo))
    mapa_folium.get_root().html.add_child(folium.Element(leyenda))
    plugins.Fullscreen(position="topright").add_to(mapa_folium)
    plugins.MiniMap(toggle_display=True, position="bottomleft").add_to(mapa_folium)
    folium.LayerControl(collapsed=True).add_to(mapa_folium)
    mapa_folium.save(OUTPUT_DIR / "mapa_interactivo_hipotesis_2_clusters.html")


def resolver_hipotesis_2() -> None:
    df = cargar_y_limpiar_datos()
    datos = preparar_clusters(df)

    resumen = (
        datos.groupby("Cluster_Label")[FEATURES_CLUSTER]
        .mean()
        .round(2)
        .sort_index()
    )

    print("\nHIPOTESIS 2: Clustering de perfiles epidemiologicos")
    print("Variables usadas:", ", ".join(FEATURES_CLUSTER))
    print("\nPromedio de variables por cluster:")
    print(resumen.to_string())
    print("\nEstados por cluster:")
    for cluster, grupo in datos.sort_values(["Cluster", "Province_State"]).groupby("Cluster_Label"):
        estados = ", ".join(grupo["Province_State"].tolist())
        print(f"{cluster}: {estados}")

    crear_scatter_clusters(datos)
    crear_mapa_clusters(datos)


if __name__ == "__main__":
    resolver_hipotesis_2()
