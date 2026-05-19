import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import branca.colormap as cm
import folium
from folium import plugins
import matplotlib.pyplot as plt
import pandas as pd

from covid_utils import (
    OUTPUT_DIR,
    cargar_geojson_estados,
    cargar_y_limpiar_datos,
    ultimo_registro_por_estado,
)


def crear_mapa_letalidad(ultimo: pd.DataFrame) -> None:
    mapa = ultimo.dropna(
        subset=["Codigo_Estado", "Incident_Rate", "Case_Fatality_Ratio"]
    ).copy()
    geojson_estados = cargar_geojson_estados()
    datos_por_estado = mapa.set_index("Codigo_Estado").to_dict(orient="index")
    valores_letalidad = mapa["Case_Fatality_Ratio"].dropna()
    escala_color = cm.LinearColormap(
        colors=["#1a9850", "#fee08b", "#d73027"],
        vmin=float(valores_letalidad.min()),
        vmax=float(valores_letalidad.max()),
        caption="Letalidad aparente (%)",
    )

    for feature in geojson_estados["features"]:
        codigo = feature.get("id")
        datos = datos_por_estado.get(codigo)
        props = feature["properties"]

        if datos is None:
            props.update(
                {
                    "region": "Sin datos",
                    "incidencia": "Sin datos",
                    "letalidad": "Sin datos",
                    "confirmed": "Sin datos",
                    "deaths": "Sin datos",
                }
            )
            continue

        props.update(
            {
                "region": datos["Region"],
                "incidencia": f"{datos['Incident_Rate']:,.0f}",
                "letalidad": f"{datos['Case_Fatality_Ratio']:.2f}%",
                "confirmed": f"{datos['Confirmed']:,.0f}",
                "deaths": f"{datos['Deaths']:,.0f}",
                "letalidad_valor": float(datos["Case_Fatality_Ratio"]),
            }
        )

    def estilo_estado(feature: dict) -> dict:
        valor = feature["properties"].get("letalidad_valor")
        return {
            "fillColor": "#d9d9d9" if valor is None else escala_color(valor),
            "color": "#ffffff",
            "weight": 1.1,
            "fillOpacity": 0.82,
        }

    def estilo_resaltado(_: dict) -> dict:
        return {"fillOpacity": 0.95, "weight": 2.6, "color": "#1f2937"}

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
        name="Letalidad aparente por estado",
        style_function=estilo_estado,
        highlight_function=estilo_resaltado,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "region", "incidencia", "letalidad", "confirmed", "deaths"],
            aliases=[
                "Estado:",
                "Region:",
                "Incidencia acumulada:",
                "Letalidad aparente:",
                "Casos confirmados:",
                "Muertes:",
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

    escala_color.add_to(mapa_folium)
    plugins.Fullscreen(position="topright").add_to(mapa_folium)
    plugins.MiniMap(toggle_display=True, position="bottomleft").add_to(mapa_folium)

    titulo_html = """
    <div style="
        position: fixed; top: 18px; left: 60px; z-index: 9999;
        background: rgba(255,255,255,0.94); padding: 14px 18px;
        border-radius: 10px; box-shadow: 0 6px 24px rgba(15,23,42,0.18);
        font-family: Arial, sans-serif; max-width: 460px;">
        <div style="font-size: 18px; font-weight: 700; color: #111827;">
            H1: Letalidad aparente por estado
        </div>
        <div style="font-size: 13px; color: #374151; margin-top: 4px;">
            Verde = menor letalidad | Amarillo = media | Rojo = mayor letalidad.
            Pasa el mouse para ver incidencia, casos y muertes.
        </div>
    </div>
    """
    mapa_folium.get_root().html.add_child(folium.Element(titulo_html))
    folium.LayerControl(collapsed=True).add_to(mapa_folium)
    mapa_folium.save(OUTPUT_DIR / "mapa_interactivo_hipotesis_1.html")


def resolver_hipotesis_1() -> None:
    df = cargar_y_limpiar_datos()
    ultimo = ultimo_registro_por_estado(df).dropna(
        subset=["Incident_Rate", "Case_Fatality_Ratio"]
    )

    correlacion = ultimo["Incident_Rate"].corr(ultimo["Case_Fatality_Ratio"])
    q_inc_alta = ultimo["Incident_Rate"].quantile(0.75)
    q_inc_baja = ultimo["Incident_Rate"].quantile(0.25)
    q_let_alta = ultimo["Case_Fatality_Ratio"].quantile(0.75)
    q_let_baja = ultimo["Case_Fatality_Ratio"].quantile(0.25)

    alta_inc_baja_let = ultimo[
        (ultimo["Incident_Rate"] >= q_inc_alta)
        & (ultimo["Case_Fatality_Ratio"] <= q_let_baja)
    ][["Province_State", "Incident_Rate", "Case_Fatality_Ratio"]]

    baja_inc_alta_let = ultimo[
        (ultimo["Incident_Rate"] <= q_inc_baja)
        & (ultimo["Case_Fatality_Ratio"] >= q_let_alta)
    ][["Province_State", "Incident_Rate", "Case_Fatality_Ratio"]]

    print("\nHIPOTESIS 1: Alta incidencia no implica alta letalidad")
    print(f"Correlacion Incident_Rate vs Case_Fatality_Ratio: {correlacion:.3f}")
    print("\nAlta incidencia y baja letalidad:")
    print(alta_inc_baja_let.sort_values("Incident_Rate", ascending=False).to_string(index=False))
    print("\nBaja incidencia y alta letalidad:")
    print(baja_inc_alta_let.sort_values("Case_Fatality_Ratio", ascending=False).to_string(index=False))

    colores = {
        "Northeast": "#003f5c",
        "Midwest": "#58508d",
        "South": "#bc5090",
        "West": "#ff6361",
        "Territory": "#ffa600",
        "Other": "#444444",
    }

    plt.figure(figsize=(11, 7))
    for region, grupo in ultimo.groupby("Region"):
        plt.scatter(
            grupo["Incident_Rate"],
            grupo["Case_Fatality_Ratio"],
            label=region,
            color=colores.get(region, "#444444"),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
        )

    plt.axvline(q_inc_alta, color="#555555", linestyle="--", alpha=0.55)
    plt.axhline(q_let_baja, color="#555555", linestyle="--", alpha=0.55)

    plt.title(f"H1: Alta incidencia no implica alta letalidad (r = {correlacion:.2f})")
    plt.xlabel("Tasa de incidencia acumulada (casos por 100,000 habitantes aprox.)")
    plt.ylabel("Letalidad aparente: muertes / casos confirmados (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hipotesis_1_incidencia_letalidad.png", dpi=160)
    plt.close()

    crear_mapa_letalidad(ultimo)


if __name__ == "__main__":
    resolver_hipotesis_1()
