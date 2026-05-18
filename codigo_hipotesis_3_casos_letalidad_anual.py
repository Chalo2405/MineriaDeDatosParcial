import matplotlib.pyplot as plt

from covid_utils import OUTPUT_DIR, cargar_y_limpiar_datos


def resolver_hipotesis_3() -> None:
    df = cargar_y_limpiar_datos()

    anual = (
        df.groupby(df["Fecha"].dt.year)
        .agg(casos=("Casos_Nuevos", "sum"), muertes=("Muertes_Nuevas", "sum"))
        .reset_index()
        .rename(columns={"Fecha": "Anio"})
    )
    anual = anual[anual["Anio"].between(2020, 2022)].copy()
    anual["Letalidad_Anual"] = anual["muertes"] / anual["casos"] * 100
    anual["Casos_Millones"] = anual["casos"] / 1_000_000

    print("\nHIPOTESIS 3: Mas casos en 2022, pero menor letalidad anual")
    print(anual[["Anio", "casos", "muertes", "Letalidad_Anual"]].to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(11, 7))
    barras = ax1.bar(
        anual["Anio"].astype(str),
        anual["Casos_Millones"],
        color="#ef5675",
        alpha=0.85,
    )
    ax1.set_title("H3: 2022 fue mas masivo, pero menos letal proporcionalmente")
    ax1.set_xlabel("Anio")
    ax1.set_ylabel("Casos nuevos anuales (millones)")
    ax1.grid(axis="y", alpha=0.25)

    for barra, casos in zip(barras, anual["casos"]):
        ax1.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.7,
            f"{casos / 1_000_000:.1f} M",
            ha="center",
            color="#7a1f3d",
            fontweight="bold",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        anual["Anio"].astype(str),
        anual["Letalidad_Anual"],
        color="#003f5c",
        marker="o",
        linewidth=3,
        label="Letalidad anual",
    )
    ax2.set_ylabel("Letalidad anual aparente (%)")

    for _, row in anual.iterrows():
        ax2.text(
            str(int(row["Anio"])),
            row["Letalidad_Anual"] + 0.05,
            f"{row['Letalidad_Anual']:.2f}%",
            ha="center",
            color="#003f5c",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hipotesis_3_casos_letalidad_anual.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    resolver_hipotesis_3()
