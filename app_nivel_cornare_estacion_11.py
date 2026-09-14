"""
App de Streamlit — Serie de tiempo de nivel de ríos/quebradas (CORNARE / MARCO)
-------------------------------------------------------------------------------
Módulo 5: Series de tiempo reales.

Estación del estudiante:
    11 - Abejorral (Aguas)

La aplicación integra:
- Consulta de la API MARCO de CORNARE.
- Serie de tiempo y orden cronológico.
- Detección de missing values como huecos reales.
- Interpolación temporal.
- Detección de outliers mediante IQR + límite físico.
- Normalización Min-Max y estandarización Z-Score.
- Split cronológico Train / Validation / Test.
- Estadística descriptiva.
- Promedio del nivel por hora.
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------------------------------------------------
# Parámetros fijos del trabajo
# ------------------------------------------------------------------
NOMBRE_ESTUDIANTE = "Tu Nombre Aquí"
CODIGO_ESTACION = "11"
NOMBRE_ESTACION = "Abejorral (Aguas)"
CALIDAD = 1

FECHA_DESDE_DEFECTO = "2026-08-30"
FECHA_HASTA_DEFECTO = "2026-09-02"

# Coordenadas por defecto si la API no entrega latitud/longitud.
LAT_DEFECTO = 6.2766
LON_DEFECTO = -75.5901

API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"

LLAVE_FECHA = "level_date"
LLAVE_VALOR = "level"

CANDIDATOS_LAT = ["lat", "latitude", "latitud"]
CANDIDATOS_LON = ["lng", "lon", "longitude", "longitud"]

st.set_page_config(
    page_title="Nivel estación 11 — CORNARE",
    page_icon="🌊",
    layout="wide"
)


# ------------------------------------------------------------------
# Funciones de consulta
# ------------------------------------------------------------------
def obtener_serie_nivel(codigo_estacion, desde, hasta, calidad=1, timeout=30):
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"
    params = {"desde": desde, "hasta": hasta, "calidad": calidad}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=False
        )

        if resp.status_code == 200:
            return resp.json(), None

        return None, f"HTTP {resp.status_code}"

    except requests.exceptions.RequestException as e:
        return None, f"Error de red: {e}"


def obtener_todas_las_paginas(datos_json, timeout=30):
    """Obtiene todos los registros siguiendo el campo 'next'."""
    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")

    while siguiente_url:
        try:
            resp = requests.get(
                siguiente_url,
                timeout=timeout,
                verify=False
            )
        except requests.exceptions.RequestException as e:
            st.warning(f"No se pudo seguir la paginación: {e}")
            break

        if resp.status_code != 200:
            st.warning(
                f"La siguiente página respondió HTTP {resp.status_code}."
            )
            break

        pagina = resp.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")

    return registros


def detectar_coordenadas(datos_json):
    """Busca latitud/longitud en la respuesta de la API."""
    if not isinstance(datos_json, dict):
        return LAT_DEFECTO, LON_DEFECTO, False

    lat = next(
        (datos_json[k] for k in CANDIDATOS_LAT if k in datos_json),
        None
    )
    lon = next(
        (datos_json[k] for k in CANDIDATOS_LON if k in datos_json),
        None
    )

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), True
        except (TypeError, ValueError):
            pass

    return LAT_DEFECTO, LON_DEFECTO, False


# ------------------------------------------------------------------
# Funciones del análisis del Módulo 5
# ------------------------------------------------------------------
def detectar_huecos(df):
    """
    Detecta missing values reales de una serie minuto a minuto
    reindexando la serie a su frecuencia típica.
    """
    df_indexado = df.set_index("fecha")

    diferencias = df["fecha"].diff().dropna()

    if diferencias.empty:
        return df_indexado.copy(), pd.Timedelta(minutes=1), 0

    frecuencia_tipica = diferencias.mode()[0]

    rango_completo = pd.date_range(
        start=df_indexado.index.min(),
        end=df_indexado.index.max(),
        freq=frecuencia_tipica
    )

    df_regular = df_indexado.reindex(rango_completo)

    huecos = int(df_regular["nivel"].isna().sum())

    return df_regular, frecuencia_tipica, huecos


def interpolar_huecos(df_regular):
    """Rellena huecos respetando el orden temporal."""
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


def detectar_outliers_iqr(serie):
    """Detecta outliers con IQR."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    mascara = (
        (serie < limite_inferior)
        | (serie > limite_superior)
    )

    return mascara, limite_inferior, limite_superior


def normalizar_min_max(serie):
    """Normalización al intervalo [0, 1]."""
    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return pd.Series(
            np.zeros(len(serie)),
            index=serie.index
        )

    return (serie - minimo) / (maximo - minimo)


def estandarizar_zscore(serie):
    """Estandarización mediante Z-Score."""
    media = serie.mean()
    desviacion = serie.std()

    if desviacion == 0:
        return pd.Series(
            np.zeros(len(serie)),
            index=serie.index
        )

    return (serie - media) / desviacion


def calcular_indice_calidad(df, huecos, mascara_outliers):
    """
    Índice simple (0-100) basado en:
    - 70% completitud de la serie.
    - 30% proporción de datos sin outliers.
    """
    if df.empty:
        return 0.0

    if huecos == 0:
        completitud = 1.0
    else:
        diferencias = df["fecha"].diff().dropna()

        if diferencias.empty:
            completitud = 0.0
        else:
            frecuencia = diferencias.mode()[0]
            esperados = len(
                pd.date_range(
                    start=df["fecha"].min(),
                    end=df["fecha"].max(),
                    freq=frecuencia
                )
            )

            completitud = (
                max(0.0, 1 - (huecos / esperados))
                if esperados > 0
                else 0.0
            )

    proporcion_outliers = (
        mascara_outliers.mean()
        if len(mascara_outliers) > 0
        else 0
    )

    indice = (
        completitud * 0.7
        + (1 - proporcion_outliers) * 0.3
    ) * 100

    return round(indice, 1)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Parámetros de tu consulta")

st.sidebar.write(f"**Estudiante:** {NOMBRE_ESTUDIANTE}")
st.sidebar.write(
    f"**Estación:** {CODIGO_ESTACION} - {NOMBRE_ESTACION}"
)
st.sidebar.write("**Calidad:** 1 — datos validados")

fecha_desde = st.sidebar.date_input(
    "Desde",
    pd.to_datetime(FECHA_DESDE_DEFECTO)
).strftime("%Y-%m-%d")

fecha_hasta = st.sidebar.date_input(
    "Hasta",
    pd.to_datetime(FECHA_HASTA_DEFECTO)
).strftime("%Y-%m-%d")

consultar = st.sidebar.button(
    "🔍 Consultar estación 11",
    type="primary"
)

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("🌊 Serie de tiempo — CORNARE")
st.caption(
    f"Estudiante: **{NOMBRE_ESTUDIANTE}** · "
    f"Estación: **11 - {NOMBRE_ESTACION}**"
)

st.info(
    "Esta aplicación corresponde al Módulo 5 y trabaja "
    "exclusivamente con la estación 11 - Abejorral (Aguas)."
)


# ------------------------------------------------------------------
# Consulta y procesamiento
# ------------------------------------------------------------------
if consultar:

    if pd.to_datetime(fecha_desde) > pd.to_datetime(fecha_hasta):
        st.error("❌ La fecha inicial no puede ser posterior a la fecha final.")
        st.stop()

    with st.spinner("Consultando la API de CORNARE..."):
        datos_crudos, error = obtener_serie_nivel(
            CODIGO_ESTACION,
            fecha_desde,
            fecha_hasta,
            CALIDAD
        )

    if error:
        st.error(f"❌ {error}")
        st.stop()

    registros = obtener_todas_las_paginas(datos_crudos)

    if not registros:
        st.warning(
            "No hay registros para la estación 11 "
            "en el rango seleccionado."
        )
        st.stop()

    # --------------------------------------------------------------
    # 1. Construcción del DataFrame
    # --------------------------------------------------------------
    df = pd.DataFrame(registros)

    df = df.rename(
        columns={
            LLAVE_FECHA: "fecha",
            LLAVE_VALOR: "nivel"
        }
    )

    if "fecha" not in df.columns or "nivel" not in df.columns:
        st.error(
            "La respuesta de la API no contiene las columnas "
            "esperadas para fecha y nivel."
        )
        st.stop()

    df["fecha"] = pd.to_datetime(
        df["fecha"],
        errors="coerce"
    )

    df["nivel"] = pd.to_numeric(
        df["nivel"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["fecha", "nivel"])
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    # Metadatos
    df["codigo_estacion"] = CODIGO_ESTACION
    df["estacion"] = NOMBRE_ESTACION
    df["estudiante"] = NOMBRE_ESTUDIANTE

    # --------------------------------------------------------------
    # 2. Missing values reales
    # --------------------------------------------------------------
    df_regular, frecuencia, huecos = detectar_huecos(df)

    # --------------------------------------------------------------
    # 3. Interpolación temporal
    # --------------------------------------------------------------
    df_limpio = interpolar_huecos(df_regular)

    # --------------------------------------------------------------
    # 4. Outliers IQR + límite físico
    # --------------------------------------------------------------
    outliers_estadisticos, lim_inf, lim_sup = detectar_outliers_iqr(
        df_limpio["nivel"]
    )

    outliers_fisicos = df_limpio["nivel"] < 0

    outliers_mask = (
        outliers_estadisticos
        | outliers_fisicos
    )

    n_outliers = int(outliers_mask.sum())

    df_sin_outliers = (
        df_limpio.loc[~outliers_mask]
        .copy()
    )

    # --------------------------------------------------------------
    # 5. Índice de calidad
    # --------------------------------------------------------------
    indice_calidad = calcular_indice_calidad(
        df,
        huecos,
        outliers_mask
    )

    # --------------------------------------------------------------
    # Métricas principales
    # --------------------------------------------------------------
    st.subheader("Resumen de la consulta")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Lecturas crudas",
        len(df)
    )

    col2.metric(
        "Huecos reales",
        huecos
    )

    col3.metric(
        "Outliers",
        n_outliers
    )

    col4.metric(
        "Índice de calidad",
        f"{indice_calidad} / 100"
    )

    # --------------------------------------------------------------
    # Información temporal
    # --------------------------------------------------------------
    st.write(
        f"**Periodo real recibido:** "
        f"{df['fecha'].min()} → {df['fecha'].max()}"
    )

    st.write(
        f"**Frecuencia típica:** {frecuencia}"
    )

    # --------------------------------------------------------------
    # Serie original
    # --------------------------------------------------------------
    st.subheader("1. Serie de nivel original")

    st.line_chart(
        df.set_index("fecha")["nivel"]
    )

    # --------------------------------------------------------------
    # Missing values
    # --------------------------------------------------------------
    st.subheader("2. Missing values — huecos reales")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Lecturas recibidas",
        len(df)
    )

    col2.metric(
        "Lecturas esperadas",
        len(df_regular)
    )

    col3.metric(
        "Huecos detectados",
        huecos
    )

    if huecos > 0:
        st.warning(
            f"Se encontraron {huecos} huecos en la serie. "
            "Estos fueron tratados mediante interpolación temporal."
        )
    else:
        st.success(
            "No se encontraron huecos en la serie."
        )

    # --------------------------------------------------------------
    # Serie limpia
    # --------------------------------------------------------------
    st.subheader("3. Serie después de interpolación")

    st.line_chart(
        df_limpio.set_index("fecha")["nivel"]
    )

    # --------------------------------------------------------------
    # Outliers
    # --------------------------------------------------------------
    st.subheader("4. Outliers — IQR + límite físico")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Límite inferior IQR",
        f"{lim_inf:.2f}"
    )

    col2.metric(
        "Límite superior IQR",
        f"{lim_sup:.2f}"
    )

    col3.metric(
        "Outliers detectados",
        n_outliers
    )

    st.write(
        "Además del método IQR, se considera físicamente inválido "
        "un nivel negativo."
    )

    if n_outliers > 0:
        st.dataframe(
            df_limpio.loc[
                outliers_mask,
                ["fecha", "nivel"]
            ],
            use_container_width=True
        )

    st.write("### Serie sin outliers")

    st.line_chart(
        df_sin_outliers.set_index("fecha")["nivel"]
    )

    # --------------------------------------------------------------
    # Normalización y estandarización
    # --------------------------------------------------------------
    st.subheader("5. Normalización y estandarización")

    df_sin_outliers["nivel_normalizado"] = normalizar_min_max(
        df_sin_outliers["nivel"]
    )

    df_sin_outliers["nivel_zscore"] = estandarizar_zscore(
        df_sin_outliers["nivel"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Nivel normalizado [0, 1]**")
        st.line_chart(
            df_sin_outliers.set_index("fecha")[
                "nivel_normalizado"
            ]
        )

    with col2:
        st.write("**Nivel estandarizado (Z-Score)**")
        st.line_chart(
            df_sin_outliers.set_index("fecha")[
                "nivel_zscore"
            ]
        )

    # --------------------------------------------------------------
    # Train / Validation / Test
    # --------------------------------------------------------------
    st.subheader("6. Train / Validation / Test")

    df_ordenado = (
        df_sin_outliers
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    n_total = len(df_ordenado)

    corte_train = int(n_total * 0.70)
    corte_val = int(n_total * 0.85)

    train_temporal = df_ordenado.iloc[:corte_train]
    val_temporal = df_ordenado.iloc[corte_train:corte_val]
    test_temporal = df_ordenado.iloc[corte_val:]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Train (70%)",
        len(train_temporal)
    )

    col2.metric(
        "Validation (15%)",
        len(val_temporal)
    )

    col3.metric(
        "Test (15%)",
        len(test_temporal)
    )

    if not train_temporal.empty:
        st.write(
            f"**Train:** {train_temporal['fecha'].min()} "
            f"→ {train_temporal['fecha'].max()}"
        )

    if not val_temporal.empty:
        st.write(
            f"**Validation:** {val_temporal['fecha'].min()} "
            f"→ {val_temporal['fecha'].max()}"
        )

    if not test_temporal.empty:
        st.write(
            f"**Test:** {test_temporal['fecha'].min()} "
            f"→ {test_temporal['fecha'].max()}"
        )

    st.caption(
        "El split es cronológico: los datos más antiguos se usan "
        "para entrenamiento y los más recientes para evaluación."
    )

    # --------------------------------------------------------------
    # Estadística descriptiva
    # --------------------------------------------------------------
    st.subheader("7. Estadística descriptiva")

    resumen = (
        df_sin_outliers["nivel"]
        .agg(["mean", "var", "std", "min", "max"])
        .round(3)
    )

    st.dataframe(
        resumen,
        use_container_width=True
    )

    # --------------------------------------------------------------
    # Máximo y fecha de ocurrencia
    # --------------------------------------------------------------
    st.subheader("8. Evento máximo registrado")

    if not df_sin_outliers.empty:
        indice_maximo = df_sin_outliers["nivel"].idxmax()
        evento_maximo = df_sin_outliers.loc[indice_maximo]

        col1, col2 = st.columns(2)

        col1.metric(
            "Nivel máximo",
            f"{evento_maximo['nivel']:.3f}"
        )

        col2.write(
            f"**Fecha de ocurrencia:** "
            f"{evento_maximo['fecha']}"
        )

    # --------------------------------------------------------------
    # Promedio por hora
    # --------------------------------------------------------------
    st.subheader("9. Nivel promedio por hora del día")

    df_sin_outliers["hora"] = (
        df_sin_outliers["fecha"].dt.hour
    )

    promedio_por_hora = (
        df_sin_outliers
        .groupby("hora")["nivel"]
        .mean()
    )

    st.line_chart(promedio_por_hora)

    # --------------------------------------------------------------
    # Mapa
    # --------------------------------------------------------------
    st.subheader("10. Ubicación de la estación")

    lat, lon, coords_reales = detectar_coordenadas(
        datos_crudos
    )

    if not coords_reales:
        st.caption(
            "La API no entregó coordenadas en las llaves esperadas; "
            "se muestra el punto de referencia configurado "
            "en la aplicación."
        )

    st.map(
        pd.DataFrame(
            {"lat": [lat], "lon": [lon]}
        ),
        zoom=10
    )

    # --------------------------------------------------------------
    # Detalle de calidad
    # --------------------------------------------------------------
    with st.expander("📊 Detalle del índice de calidad"):
        st.write(
            f"- Huecos de reporte detectados: **{huecos}**"
        )
        st.write(
            f"- Outliers detectados: **{n_outliers}** "
            f"de {len(df_limpio)} lecturas"
        )
        st.write(
            "- Completitud de la serie: **70%** del índice"
        )
        st.write(
            "- Ausencia de outliers: **30%** del índice"
        )

    # --------------------------------------------------------------
    # Datos
    # --------------------------------------------------------------
    with st.expander("📋 Ver datos originales"):
        st.dataframe(
            df,
            use_container_width=True
        )

    with st.expander("🧹 Ver datos limpios"):
        st.dataframe(
            df_sin_outliers,
            use_container_width=True
        )

    # --------------------------------------------------------------
    # Descargas
    # --------------------------------------------------------------
    st.subheader("11. Descargar resultados")

    csv_original = df.to_csv(
        index=False
    ).encode("utf-8")

    csv_limpio = df_sin_outliers.to_csv(
        index=False
    ).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇️ Descargar datos originales",
            csv_original,
            file_name="nivel_estacion_11_abejorral_original.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            "⬇️ Descargar datos limpios",
            csv_limpio,
            file_name="nivel_estacion_11_abejorral_limpio.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------------
    # Resumen final del módulo
    # --------------------------------------------------------------
    st.divider()

    st.header("📌 Resumen de la consulta")

    st.write(
        f"**Estudiante:** {NOMBRE_ESTUDIANTE}"
    )
    st.write(
        f"**Estación:** {CODIGO_ESTACION} - {NOMBRE_ESTACION}"
    )
    st.write(
        f"**Rango solicitado:** {fecha_desde} → {fecha_hasta}"
    )
    st.write(
        f"**Calidad:** {CALIDAD}"
    )
    st.write(
        f"**Lecturas crudas:** {len(df)}"
    )
    st.write(
        f"**Lecturas limpias:** {len(df_sin_outliers)}"
    )
    st.write(
        f"**Huecos detectados:** {huecos}"
    )
    st.write(
        f"**Outliers removidos:** {n_outliers}"
    )

    st.success(
        "✅ Análisis del Módulo 5 completado para la estación "
        "11 - Abejorral (Aguas)."
    )

else:
    st.info(
        "Ajusta las fechas en el sidebar y presiona "
        "**Consultar estación 11**."
    )
