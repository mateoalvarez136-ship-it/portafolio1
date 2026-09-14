"""
Módulo 5 — Series de tiempo reales: nivel de ríos y quebradas (CORNARE)
ERA 1: Fundamentos de Computación Científica

Estación seleccionada:
    11 - Abejorral (Aguas)

La aplicación:
1. Consulta datos reales de la API MARCO de CORNARE.
2. Convierte fechas y niveles a tipos numéricos/temporales.
3. Ordena cronológicamente la serie.
4. Detecta huecos reales mediante una frecuencia temporal regular.
5. Rellena huecos con interpolación temporal.
6. Detecta outliers con IQR y la restricción física nivel < 0.
7. Calcula Min-Max y Z-Score.
8. Divide cronológicamente en Train/Validation/Test (70/15/15).
9. Presenta estadísticas descriptivas.
10. Muestra el máximo y su fecha.
11. Calcula el nivel promedio por hora del día.
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------------
# Parámetros del trabajo
# -----------------------------------------------------------------------------
NOMBRE_ESTUDIANTE = "Mateo Álvarez Taborda"
CODIGO_ESTACION = "11"
NOMBRE_ESTACION = "Abejorral (Aguas)"
CALIDAD = 1

FECHA_DESDE_DEFECTO = "2026-08-30"
FECHA_HASTA_DEFECTO = "2026-09-02"

API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"

st.set_page_config(
    page_title="Módulo 5 — CORNARE estación 11",
    page_icon="🌊",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Consulta de API
# -----------------------------------------------------------------------------
def obtener_serie_nivel(codigo_estacion, desde, hasta, calidad=1, timeout=30):
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"
    params = {"desde": desde, "hasta": hasta, "calidad": calidad}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }

    try:
        respuesta = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=False,
        )

        if respuesta.status_code == 200:
            return respuesta.json(), None

        return None, f"HTTP {respuesta.status_code}: {respuesta.text[:200]}"

    except requests.exceptions.RequestException as error:
        return None, f"Error de red: {error}"


def obtener_todas_las_paginas(datos_json, timeout=30):
    """Recorre todas las páginas indicadas por el campo 'next'."""
    if not isinstance(datos_json, dict):
        return []

    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")

    while siguiente_url:
        try:
            respuesta = requests.get(
                siguiente_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=timeout,
                verify=False,
            )
        except requests.exceptions.RequestException:
            break

        if respuesta.status_code != 200:
            break

        pagina = respuesta.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")

    return registros


# -----------------------------------------------------------------------------
# Preparación de datos
# -----------------------------------------------------------------------------
def construir_dataframe(registros):
    """Adapta tanto el esquema real de CORNARE como el esquema anterior."""
    df = pd.DataFrame(registros)

    # En la respuesta observada de CORNARE las columnas son fecha y nivel.
    # También se aceptan level_date/level por compatibilidad con el código base.
    posibles_fecha = ["fecha", "level_date", "date", "datetime"]
    posibles_nivel = ["nivel", "level"]

    columna_fecha = next((c for c in posibles_fecha if c in df.columns), None)
    columna_nivel = next((c for c in posibles_nivel if c in df.columns), None)

    if columna_fecha is None or columna_nivel is None:
        raise ValueError(
            "La API no contiene columnas reconocibles de fecha y nivel. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            columna_fecha: "fecha",
            columna_nivel: "nivel",
        }
    )

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce")

    df = df.dropna(subset=["fecha", "nivel"])
    df = df.sort_values("fecha")

    # Una marca temporal duplicada no representa un hueco; se conserva la
    # primera lectura para mantener una serie temporal con índice único.
    df = df.drop_duplicates(subset="fecha", keep="first")
    df = df.reset_index(drop=True)

    return df


# -----------------------------------------------------------------------------
# Missing values / huecos
# -----------------------------------------------------------------------------
def detectar_huecos(df):
    """Reindexa la serie a su frecuencia típica y cuenta huecos reales."""
    if len(df) < 2:
        indice = df.set_index("fecha")
        return indice, pd.Timedelta(minutes=1), 0

    diferencias = df["fecha"].diff().dropna()
    frecuencia_tipica = diferencias.mode().iloc[0]

    if frecuencia_tipica <= pd.Timedelta(0):
        frecuencia_tipica = pd.Timedelta(minutes=1)

    indice = df.set_index("fecha")
    rango_completo = pd.date_range(
        start=indice.index.min(),
        end=indice.index.max(),
        freq=frecuencia_tipica,
    )

    df_regular = indice.reindex(rango_completo)
    huecos = int(df_regular["nivel"].isna().sum())

    return df_regular, frecuencia_tipica, huecos


def interpolar_huecos(df_regular):
    """Rellena huecos usando interpolación basada en el tiempo."""
    df_limpio = df_regular.copy()

    df_limpio["nivel"] = (
        df_limpio["nivel"]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    return (
        df_limpio
        .reset_index()
        .rename(columns={"index": "fecha"})
    )


# -----------------------------------------------------------------------------
# Outliers
# -----------------------------------------------------------------------------
def detectar_outliers_iqr(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    mascara_iqr = (
        (serie < limite_inferior)
        | (serie > limite_superior)
    )

    return mascara_iqr, limite_inferior, limite_superior


# -----------------------------------------------------------------------------
# Transformaciones
# -----------------------------------------------------------------------------
def normalizar_min_max(serie):
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return pd.Series(0.0, index=serie.index)

    return (serie - minimo) / (maximo - minimo)


def estandarizar_zscore(serie):
    media = serie.mean()
    desviacion = serie.std()

    if desviacion == 0 or pd.isna(desviacion):
        return pd.Series(0.0, index=serie.index)

    return (serie - media) / desviacion


# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------
st.sidebar.header("Parámetros de tu consulta")
st.sidebar.write(f"**Estudiante:** {NOMBRE_ESTUDIANTE}")
st.sidebar.write(f"**Estación:** {CODIGO_ESTACION} - {NOMBRE_ESTACION}")
st.sidebar.write("**Calidad:** 1 — datos validados")

fecha_desde = st.sidebar.date_input(
    "Desde",
    pd.to_datetime(FECHA_DESDE_DEFECTO),
).strftime("%Y-%m-%d")

fecha_hasta = st.sidebar.date_input(
    "Hasta",
    pd.to_datetime(FECHA_HASTA_DEFECTO),
).strftime("%Y-%m-%d")

consultar = st.sidebar.button(
    "🔍 Consultar estación 11",
    type="primary",
)

st.title("🌊 Módulo 5 — Series de tiempo reales")
st.caption(
    f"Estudiante: **{NOMBRE_ESTUDIANTE}** · "
    f"Estación: **{CODIGO_ESTACION} - {NOMBRE_ESTACION}**"
)

if not consultar:
    st.info("Selecciona el rango de fechas y presiona **Consultar estación 11**.")
    st.stop()

if pd.to_datetime(fecha_desde) > pd.to_datetime(fecha_hasta):
    st.error("❌ La fecha inicial no puede ser posterior a la fecha final.")
    st.stop()

# -----------------------------------------------------------------------------
# Consulta
# -----------------------------------------------------------------------------
with st.spinner("Consultando datos reales de CORNARE..."):
    datos_crudos, error = obtener_serie_nivel(
        CODIGO_ESTACION,
        fecha_desde,
        fecha_hasta,
        CALIDAD,
    )

if error:
    st.error(f"❌ No fue posible consultar CORNARE: {error}")
    st.stop()

registros = obtener_todas_las_paginas(datos_crudos)

if not registros:
    st.warning("No se encontraron registros para la estación 11 en el rango seleccionado.")
    st.stop()

try:
    df = construir_dataframe(registros)
except ValueError as error:
    st.error(f"❌ {error}")
    st.stop()

if df.empty:
    st.warning("Los registros recibidos no contienen datos válidos de fecha y nivel.")
    st.stop()

# -----------------------------------------------------------------------------
# 1. Serie temporal
# -----------------------------------------------------------------------------
df_regular, frecuencia, huecos = detectar_huecos(df)
df_limpio = interpolar_huecos(df_regular)

st.subheader("1. Serie temporal original")
st.write(
    f"Periodo recibido: **{df['fecha'].min()} → {df['fecha'].max()}** · "
    f"Frecuencia típica: **{frecuencia}**"
)
st.line_chart(df.set_index("fecha")["nivel"])

# -----------------------------------------------------------------------------
# 2. Missing values / huecos
# -----------------------------------------------------------------------------
st.subheader("2. Detección de missing values (huecos reales)")

c1, c2, c3 = st.columns(3)
c1.metric("Lecturas recibidas", len(df))
c2.metric("Lecturas esperadas", len(df_regular))
c3.metric("Huecos detectados", huecos)

if huecos == 0:
    st.success("No se encontraron huecos en la frecuencia temporal típica.")
else:
    st.warning(
        f"Se detectaron {huecos} posiciones faltantes al reindexar la serie "
        "a su frecuencia temporal típica."
    )

# -----------------------------------------------------------------------------
# 3. Interpolación temporal
# -----------------------------------------------------------------------------
st.subheader("3. Tratamiento de missing values: interpolación temporal")
st.write(
    "Los huecos se rellenan mediante `interpolate(method='time')`; "
    "`ffill` y `bfill` solo se aplican como respaldo en extremos."
)
st.line_chart(df_limpio.set_index("fecha")["nivel"])

# -----------------------------------------------------------------------------
# 4. Outliers
# -----------------------------------------------------------------------------
st.subheader("4. Detección de outliers: IQR + restricción física")

mascara_iqr, lim_inf, lim_sup = detectar_outliers_iqr(df_limpio["nivel"])
mascara_fisica = df_limpio["nivel"] < 0
mascara_outliers = mascara_iqr | mascara_fisica

n_outliers = int(mascara_outliers.sum())

c1, c2, c3 = st.columns(3)
c1.metric("Q1 - 1.5×IQR", f"{lim_inf:.3f}")
c2.metric("Q3 + 1.5×IQR", f"{lim_sup:.3f}")
c3.metric("Outliers detectados", n_outliers)

st.write(
    "Se considera outlier un valor fuera de los límites del IQR o un nivel "
    "negativo (`nivel < 0`), por ser físicamente inválido."
)

if n_outliers > 0:
    st.dataframe(
        df_limpio.loc[mascara_outliers, ["fecha", "nivel"]],
        use_container_width=True,
    )
else:
    st.success("No se detectaron outliers con los criterios definidos.")

df_sin_outliers = df_limpio.loc[~mascara_outliers].copy()

st.write("**Serie después del tratamiento de outliers**")
st.line_chart(df_sin_outliers.set_index("fecha")["nivel"])

# -----------------------------------------------------------------------------
# 5. Normalización y estandarización
# -----------------------------------------------------------------------------
st.subheader("5. Normalización Min-Max y estandarización Z-Score")

df_sin_outliers["nivel_normalizado"] = normalizar_min_max(
    df_sin_outliers["nivel"]
)
df_sin_outliers["nivel_zscore"] = estandarizar_zscore(
    df_sin_outliers["nivel"]
)

c1, c2 = st.columns(2)
with c1:
    st.write("**Min-Max [0, 1]**")
    st.line_chart(
        df_sin_outliers.set_index("fecha")["nivel_normalizado"]
    )

with c2:
    st.write("**Z-Score**")
    st.line_chart(
        df_sin_outliers.set_index("fecha")["nivel_zscore"]
    )

# -----------------------------------------------------------------------------
# 6. Train / Validation / Test
# -----------------------------------------------------------------------------
st.subheader("6. División cronológica Train / Validation / Test")

df_ordenado = df_sin_outliers.sort_values("fecha").reset_index(drop=True)
n_total = len(df_ordenado)

corte_train = int(n_total * 0.70)
corte_val = int(n_total * 0.85)

train = df_ordenado.iloc[:corte_train]
validation = df_ordenado.iloc[corte_train:corte_val]
test = df_ordenado.iloc[corte_val:]

c1, c2, c3 = st.columns(3)
c1.metric("Train (70%)", len(train))
c2.metric("Validation (15%)", len(validation))
c3.metric("Test (15%)", len(test))

for nombre, subconjunto in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    if not subconjunto.empty:
        st.write(
            f"**{nombre}:** {subconjunto['fecha'].min()} → "
            f"{subconjunto['fecha'].max()}"
        )

st.caption(
    "La división conserva el orden temporal: los datos más antiguos se "
    "usan para entrenamiento y los más recientes para validación y prueba."
)

# -----------------------------------------------------------------------------
# 7. Estadística descriptiva
# -----------------------------------------------------------------------------
st.subheader("7. Estadística descriptiva")

estadisticas = (
    df_sin_outliers["nivel"]
    .agg(["mean", "var", "std", "min", "max"])
    .rename({
        "mean": "Media",
        "var": "Varianza",
        "std": "Desviación estándar",
        "min": "Mínimo",
        "max": "Máximo",
    })
    .round(4)
)

st.dataframe(estadisticas, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. Máximo y fecha
# -----------------------------------------------------------------------------
st.subheader("8. Máximo y fecha de ocurrencia")

indice_maximo_raw = df["nivel"].idxmax()
maximo_raw = df.loc[indice_maximo_raw]

c1, c2 = st.columns(2)
c1.metric("Máximo de la serie recibida", f"{maximo_raw['nivel']:.4f}")
c2.write(f"**Fecha:** {maximo_raw['fecha']}")

if not df_sin_outliers.empty:
    indice_maximo_limpio = df_sin_outliers["nivel"].idxmax()
    maximo_limpio = df_sin_outliers.loc[indice_maximo_limpio]
    st.write(
        f"Máximo después del tratamiento de outliers: **{maximo_limpio['nivel']:.4f}** "
        f"el **{maximo_limpio['fecha']}**."
    )

# -----------------------------------------------------------------------------
# 9. Promedio por hora
# -----------------------------------------------------------------------------
st.subheader("9. Nivel promedio por hora del día")

df_sin_outliers["hora"] = df_sin_outliers["fecha"].dt.hour
promedio_por_hora = df_sin_outliers.groupby("hora")["nivel"].mean()

st.line_chart(promedio_por_hora)
st.dataframe(
    promedio_por_hora.rename("nivel_promedio").round(4),
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# 10. Resumen de resultados
# -----------------------------------------------------------------------------
st.subheader("10. Resumen de resultados")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lecturas originales", len(df))
c2.metric("Huecos", huecos)
c3.metric("Outliers", n_outliers)
c4.metric("Lecturas finales", len(df_sin_outliers))

st.success(
    "✅ Módulo 5 completado para la estación 11 — Abejorral (Aguas)."
)

# -----------------------------------------------------------------------------
# Datos descargables
# -----------------------------------------------------------------------------
st.subheader("📥 Descargar resultados")

original_descarga = df.copy()
limpio_descarga = df_sin_outliers.copy()

csv_original = original_descarga.to_csv(index=False).encode("utf-8")
csv_limpio = limpio_descarga.to_csv(index=False).encode("utf-8")

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Datos originales",
        csv_original,
        file_name="nivel_estacion_11_abejorral_original.csv",
        mime="text/csv",
    )

with c2:
    st.download_button(
        "⬇️ Datos procesados",
        csv_limpio,
        file_name="nivel_estacion_11_abejorral_procesado.csv",
        mime="text/csv",
    )

with st.expander("📋 Ver datos originales"):
    st.dataframe(original_descarga, use_container_width=True)

with st.expander("🧹 Ver datos procesados"):
    st.dataframe(limpio_descarga, use_container_width=True)
