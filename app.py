"""
Dashboard de Supervivencia de Llantas — SOPORTE COMPLETO GBSA / RSF
Genera curvas dinámicas y predicción de kilometraje para todos los modelos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# ── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tire Survival · Perfiles",
    page_icon="🛞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS PERSONALIZADO (Estética Dark Premium) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0d0f14;
    color: #e8e6df;
}
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] * {
    color: #c8c5bc !important;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
.metric-container {
    background: #13161e;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1e2330;
    margin-bottom: 15px;
}
.metric-title {
    font-size: 0.85rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-pred {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    font-weight: bold;
    color: #34d399;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ── CARGA DE ESTRUCTURAS PRE-ENTRENADAS ──────────────────────────────────
@st.cache_resource
def cargar_modelos():
    ruta_pkl = "models_trained.pkl"
    if not os.path.exists(ruta_pkl):
        st.error(f"No se encontró el archivo {ruta_pkl}. Confirma su ubicación.")
        return None
    with open(ruta_pkl, "rb") as f:
        return pickle.load(f)

def cargar_lista_desde_txt(ruta_archivo):
    """
    Lee un archivo txt y devuelve una lista limpia de opciones.
    Asume que cada opción está en una línea diferente del archivo.
    """
    if not os.path.exists(ruta_archivo):
        return ["Error: Archivo no encontrado"] # Valor por defecto si falta el txt
        
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        # .strip() elimina saltos de línea (\n) y espacios sobrantes
        opciones = [linea.strip() for linea in f if linea.strip()]
        
    return opciones

estructuras_modelos = cargar_modelos()
def obtener_indice(lista, valor_buscar):
    try:
        return lista.index(valor_buscar)
    except ValueError:
        return 0 # Si no lo encuentra, selecciona el primero de la lista

# ── INTERFAZ PRINCIPAL ───────────────────────────────────────────────────
st.title("🛞 Simulador de Perfiles y Vida Útil de Llantas")
st.markdown("Evaluación predictiva de la estabilidad operativa por perfiles lógicos de desgaste y configuraciones de ruta.")

if estructuras_modelos:
    
    # ── PANEL LATERAL: SELECCIÓN DE PERFILES Y CRITERIOS ─────────────────
    st.sidebar.header("📋 Configuración del Perfil")
    
    transicion_seleccionada = "models"
    
    st.sidebar.markdown("---")
    
    # Lógica de Perfiles Agrupadores Predefinidos
    perfil_operativo = st.sidebar.selectbox(
        "Selecciona un Perfil de Llanta:",
        [
            "Personalizado (Ajuste libre)",
            "Eje de Tracción - Carga Pesada (Ruta Norte)",
            "Eje de Dirección - Operación Local",
            "Eje de Arrastre - Alta Severidad (Pacífico)"
        ]
    )
    
    # Valores dinámicos según el perfil seleccionado
    if "Carga Pesada" in perfil_operativo:
        init_kms, init_peso, init_prof, init_peso_max, init_n_viajes = 55000, 42.5, 7.5, 130.0, 10
    elif "Operación Local" in perfil_operativo:
        init_kms, init_peso, init_prof, init_peso_max, init_n_viajes = 22000, 15.0, 14.0, 50.0, 5
    elif "Alta Severidad" in perfil_operativo:
        init_kms, init_peso, init_prof, init_peso_max, init_n_viajes = 70000, 38.0, 5.0, 90.0, 15
    else:
        init_kms, init_peso, init_prof, init_peso_max, init_n_viajes = 40000, 24.0, 11.0, 90.0, 8

    st.sidebar.subheader("⚙️ Parámetros del Escenario")
    kms_acumulados = st.sidebar.number_input("Kilómetros Acumulados:", min_value=0, value=init_kms, step=5000)
    peso_carga_promedio = st.sidebar.slider("Peso de Carga Promedio (Tons):", min_value=1.0, max_value=33000.0, value=init_peso, step=0.5)
    profundidad_actual = st.sidebar.slider("Profundidad de Piso Actual (mm):", min_value=0.0, max_value=25.0, value=init_prof, step=0.5)
    peso_max = st.sidebar.slider("Peso máximo registrado (kg):", min_value=0.0, max_value=13000000.0, value=init_peso_max, step=0.5)
    n_viajes = st.sidebar.slider("Número de viajes:", min_value=1, max_value=4000, value=init_n_viajes, step=1)

    lista_rutas = cargar_lista_desde_txt("nombres_rutas.txt")
    lista_marcas = cargar_lista_desde_txt("marcas.txt")
    lista_rutas = [str(r) for r in lista_rutas]
    lista_marcas = [str(m) for m in lista_marcas]
    init_ruta, init_marca, init_pos, init_eje = "", "", 0, 1
    idx_ruta = obtener_indice(lista_rutas, init_ruta)
    idx_marca = obtener_indice(lista_marcas, init_marca)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏷️ Variables de Entorno")
    marca_llanta = st.sidebar.selectbox("Marca:", options=lista_marcas, index=idx_marca)
    posicion = st.sidebar.selectbox("Posición:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].index(init_pos))
    eje_unidad = st.sidebar.selectbox("Eje:", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(init_eje))
    ruta_frecuente = st.sidebar.selectbox("Ruta:", options=lista_rutas, index=idx_ruta)
    
    datos_usuario = {
        'kms_totales': kms_acumulados,
        'peso_promedio': peso_carga_promedio,
        'profundidad_actual': profundidad_actual,
        'peso_acumulado': float(peso_carga_promedio * n_viajes),
        'peso_max': peso_max,
        'n_viajes': n_viajes,
        'posicion': str(posicion),
        'eje': str(eje_unidad),
    }
    input_data = pd.DataFrame([datos_usuario])
    input_data['marca'] = str(marca_llanta)
    input_data['ruta_frecuente'] = str(ruta_frecuente)

    # for marca in lista_marcas:
    #     # Si el usuario seleccionó esta marca en el sidebar, le asigna 1.0, de lo contrario 0.0
    #     datos_usuario[f"nombre_{marca}"] = 1.0 if marca_llanta == marca else 0.0
    # for ruta in lista_rutas:
    #     # Si el usuario seleccionó esta marca en el sidebar, le asigna 1.0, de lo contrario 0.0
    #     datos_usuario[f"nombre_{ruta}"] = 1.0 if ruta_frecuente == ruta else 0.0

    # ── MÓDULO PREDICTIVO Y DESPLIEGUE (ACTUALIZADO PARA GBSA) ───────────────
    datos_transicion = estructuras_modelos[transicion_seleccionada]
    preprocesador = estructuras_modelos['processors']['std'] # Usamos el StandardScaler
    modelo_desgaste = datos_transicion['Normal → Desgaste Operativo']['model']
    modelo_estructural = datos_transicion['Normal → Daño Estructural']['model']
    modelo_catastrofica = datos_transicion['Normal → Falla Catastrófica']['model']
    modelo_final = modelo_desgaste
    c_index = datos_transicion['Normal → Desgaste Operativo']['c_index']

    x_plot = [] 
    y_plot = []
    vida_util_texto = "Calculando..."
    km_adicionales_estimados = 0
    hr = 1.0 # Hazard Ratio default

    try:
        X_procesado = preprocesador.transform(input_data)
        if hasattr(modelo_final, "predict_survival_function"):
            resultado = modelo_final.predict_survival_function(X_procesado)
            funciones_surv = modelo_desgaste.predict_survival_function(X_procesado, return_array = False)[0]
            fn = resultado[0] if isinstance(resultado, (list, np.ndarray)) else resultado
            x_plot = fn.x
            y_plot = fn.y
        
        
        elif hasattr(modelo_final, "predict"):
            resultado = modelo_final.predict(X_procesado)
            score_riesgo = resultado[0] if isinstance(resultado, (list, np.ndarray)) else resultado
            
            hr = np.exp(np.clip(score_riesgo, -2.0, 2.0))
            x_plot = np.linspace(0, 120000, 100)
            y_plot = np.exp(- (x_plot / 60000) ** 2.2 * hr)

        # CÁLCULO DE VIDA ÚTIL RESTANTE (Punto de 50% de probabilidad)
        if len(x_plot) > 0 and len(y_plot) > 0:
            bajo_umbral = np.where(y_plot <= 0.50)[0]
            if len(bajo_umbral) > 0:
                km_adicionales_estimados = x_plot[bajo_umbral[0]]
                vida_util_texto = f"{km_adicionales_estimados:,.0f} KM"
            else:
                vida_util_texto = f"> {x_plot[-1]:,.0f} KM"
                
    except Exception as e:
        st.error(f"Error procesando los datos: Revisa las variables en el Pipeline. Detalle: {str(e)}")

    # ── RENDERIZADO VISUAL EN PANTALLA ──
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔮 Predicción Numérica")
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Vida Útil Adicional Estimada</div>
            <div class="metric-pred">{vida_util_texto}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container" style="background: #0f111a;">
            <div class="metric-title">Perfil Evaluado Activo</div>
            <div style="font-size:1.05rem; font-weight:600; color:#60a5fa; margin-top:5px;">{perfil_operativo}</div>
            <div style="font-size:0.8rem; color:#6b7280; margin-top:2px;">Fiabilidad del Modelo (C-Index): {c_index:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Diagnóstico")
        if km_adicionales_estimados > 0 and km_adicionales_estimados < 15000:
            st.error("🚨 **Retiro Inminente:** El desgaste y las condiciones actuales provocan un decaimiento acelerado de la vida útil de la llanta.")
        elif hr > 1.5:
            st.warning("⚠️ **Alta Severidad Detectada:** El modelo (GBSA) detecta una tasa de riesgo superior al promedio por las cargas asociadas al perfil.")
        else:
            st.success("✅ **Operación Segura:** La curva de desgaste indica una estabilidad adecuada prolongando el rendimiento de la llanta.")

    with col2:
        st.subheader("📈 Curva Dinámica del Ciclo de Vida")
        
        if len(x_plot) > 0:
            fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0d0f14")
            ax.set_facecolor("#13161e")
            
            # Trazar la curva generada
            ax.step(x_plot, y_plot, where="post", color="#34d399", linewidth=2.5, label="Trayectoria de Supervivencia")
            
            # Trazar la línea de retiro
            if km_adicionales_estimados > 0:
                ax.axvline(x=km_adicionales_estimados, color="#ef4444", linestyle=":", alpha=0.8, 
                           label=f"Umbral de Retiro (50% Prob.)")
            
            ax.set_title(f"Decaimiento de la Probabilidad (Modelo Activo)\nTransición: {transicion_seleccionada}", 
                         color="#ffffff", fontsize=11, pad=15, weight="bold")
            ax.set_xlabel("Kilómetros Adicionales", color="#c8c5bc", fontsize=9)
            ax.set_ylabel("Probabilidad (S(t))", color="#c8c5bc", fontsize=9)
            ax.tick_params(colors="#c8c5bc", labelsize=8)
            ax.grid(True, color="#1e2330", linestyle="--", linewidth=0.5)
            ax.set_ylim(-0.05, 1.05)
            
            for spine in ax.spines.values():
                spine.set_color("#1e2330")
                
            ax.legend(facecolor="#13161e", edgecolor="#1e2330", labelcolor="#c8c5bc")
            st.pyplot(fig)

st.markdown("""
<div style="margin-top:40px; padding-top:16px; border-top:1px solid #1e2330; text-align: center;">
    <span style="font-size:0.75rem; color:#4b5563; font-family: 'DM Mono', monospace;">
        Motor de Riesgos Proporcionales GBSA · Predicción de Mantenimiento (v4)
    </span>
</div>
""", unsafe_allow_html=True)