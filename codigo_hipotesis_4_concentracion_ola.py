import matplotlib.pyplot as plt
import pandas as pd

from covid_utils import OUTPUT_DIR, REGIONS, cargar_y_limpiar_datos


def resolver_hipotesis_4() -> None:
    df = cargar_y_limpiar_datos()

    semanal = (
        df.groupby(["Province_State", pd.Grouper(key="Fecha", freq="W")])
        .agg(casos_semana=("Casos_Nuevos", "sum"))
        .reset_index()
    )

    concentracion = (
        semanal.groupby("Province_State")
        .agg(casos_totales=("casos_semana", "sum"), peor_semana=("casos_semana", "max"))
        .reset_index()
    )
    concentracion = concentracion[concentracion["casos_totales"] > 1000].copy()
    concentracion["porcentaje_peor_semana"] = (
        concentracion["peor_semana"] / concentracion["casos_totales"] * 100
    )
    concentracion["Region"] = concentracion["Province_State"].map(REGIONS).fillna("Other")
    concentracion.to_csv(OUTPUT_DIR / "concentracion_hipotesis_4.csv", index=False)

    top = concentracion.nlargest(12, "porcentaje_peor_semana")

    print("\nHIPOTESIS 4: Una sola ola concentro muchos casos")
    print(
        top[
            [
                "Province_State",
                "casos_totales",
                "peor_semana",
                "porcentaje_peor_semana",
                "Region",
            ]
        ].to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    axes[0].barh(
        top["Province_State"][::-1],
        top["porcentaje_peor_semana"][::-1],
        color="#ffa600",
    )
    axes[0].set_title("Top lugares por concentracion en su peor semana")
    axes[0].set_xlabel("% de casos historicos ocurridos en la peor semana")
    axes[0].grid(axis="x", alpha=0.25)

    regiones = ["Northeast", "Midwest", "South", "West", "Territory"]
    datos_boxplot = [
        concentracion.loc[
            concentracion["Region"] == region, "porcentaje_peor_semana"
        ].dropna()
        for region in regiones
    ]
    axes[1].boxplot(datos_boxplot, tick_labels=regiones, patch_artist=True)
    axes[1].set_title("Concentracion de la peor semana por region")
    axes[1].set_ylabel("% de casos historicos en la peor semana")
    axes[1].grid(axis="y", alpha=0.25)

    plt.suptitle("H4: Algunos brotes fueron mucho mas explosivos que otros", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hipotesis_4_concentracion_ola.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    resolver_hipotesis_4()
