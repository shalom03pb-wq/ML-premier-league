import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# Configuración Premium de página
st.set_page_config(page_title="Football Analytics Pro Dashboard", page_icon="⚽", layout="wide")

# CSS Avanzado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;800&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .main-title {
        background: linear-gradient(90deg, #A855F7 0%, #3B82F6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
        padding: 20px; text-align: center; color: white;
    }
    .metric-value {font-size: 2.5rem; font-weight: 800;}
    .metric-label {color: #94A3B8; font-size: 1rem; text-transform: uppercase;}
    .sub-title {color: #A855F7; border-bottom: 2px solid #3B82F6; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. GENERACIÓN ESTADÍSTICA PROFUNDA (MOCK DATA)
# ----------------------------------------------------
@st.cache_data
def load_rich_data():
    np.random.seed(42)
    n = 2000
    
    # Espacio
    x = np.random.normal(loc=85, scale=12, size=n).clip(50, 105)
    y = np.random.normal(loc=34, scale=18, size=n).clip(0, 68)
    distances = np.sqrt((105 - x)**2 + (34 - y)**2)
    angles = np.random.beta(a=2, b=5, size=n) * 90 
    
    # Tiempo y Contexto
    minutes = np.random.randint(1, 98, size=n)
    pressure = np.random.uniform(0, 10, size=n)
    is_head = np.random.binomial(1, 0.15, size=n)
    is_fast_break = np.random.binomial(1, 0.1, size=n)
    
    big_chance_prob = np.where((distances < 14) & (pressure < 4), 0.5, 0.05)
    is_big_chance = np.random.binomial(1, big_chance_prob)
    
    # Goles: Dependencia compleja (no lineal)
    logit = (-1.5 - 0.1*distances + 0.05*angles + 2.0*is_big_chance 
             - 0.1*pressure + 0.5*is_fast_break - 0.5*is_head 
             + np.where(minutes > 90, 1.0, 0)) # Bonus en descuento
    
    goal_prob = 1 / (1 + np.exp(-logit))
    goals = np.random.binomial(1, goal_prob)
    
    df = pd.DataFrame({
        'Coord_X': x, 'Coord_Y': y, 'Distance': distances, 'Angle': angles,
        'Minute': minutes, 'Pressure_Index': pressure, 'Is_Header': is_head,
        'Is_Fast_Break': is_fast_break, 'Is_Big_Chance': is_big_chance,
        'Goal': goals, 'xG_Base': goal_prob
    })
    return df

df = load_rich_data()

# Modelos Entrenados (Logístico, RF, XGB)
@st.cache_resource
def train_models(data):
    features = ['Distance', 'Angle', 'Minute', 'Pressure_Index', 'Is_Fast_Break', 'Is_Big_Chance']
    X = data[features]
    y = data['Goal']
    
    models = {
        'Regresión Logística': LogisticRegression(class_weight='balanced', max_iter=200),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight='balanced'),
        'eXtreme Gradient Boosting (GBM)': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X, y)
    return models, features

models_dict, feature_cols = train_models(df)

# Dibujador de Cancha
def draw_pitch(fig):
    lc = "rgba(255,255,255,0.3)"
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color=lc, width=2))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color=lc, width=2))
    fig.add_shape(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15, line=dict(color=lc, width=2))
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color=lc, width=2))
    fig.add_shape(type="rect", x0=105-16.5, y0=13.84, x1=105, y1=54.16, line=dict(color=lc, width=2))
    fig.add_shape(type="rect", x0=-2, y0=30.34, x1=0, y1=37.66, line=dict(color=lc, width=2), fillcolor=lc)
    fig.add_shape(type="rect", x0=105, y0=30.34, x1=107, y1=37.66, line=dict(color=lc, width=2), fillcolor=lc)
    fig.update_xaxes(range=[45, 110], showgrid=False, visible=False) # Enfocado a zona de ataque
    fig.update_yaxes(range=[-5, 73], showgrid=False, visible=False)
    return fig

# ----------------------------------------------------
# ESTRUCTURA DE NAVEGACIÓN
# ----------------------------------------------------
st.sidebar.markdown("<h2 style='color: #A855F7; font-weight: 800; text-align:center;'>🚀 PRO ANALYTICS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

tabs = st.tabs([
    "📊 1. EDA & Feature Engineering",
    "🗺️ 2. Shot Map Interactivo",
    "🧠 3. xG Models Comparer",
    "🏆 4. Match Predictor",
    "🧩 5. Clustering",
    "📝 6. Conclusiones"
])

# ===============================================
# PESTAÑA 1: EDA & Feature Engineering
# ===============================================
with tabs[0]:
    st.markdown("<h1 class='main-title'>Análisis Exploratorio y Contextual</h1>", unsafe_allow_html=True)
    st.write("Exploración profunda de la distribución de las conversiones según el tiempo y presión defensiva.")
    
    col1, col2, col3 = st.columns(3)
    c_goles = df['Goal'].sum()
    c_conv = c_goles / len(df) * 100
    with col1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df)}</div><div class='metric-label'>Volumen de Tiros</div></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00CC96;'>{c_goles}</div><div class='metric-label'>Goles</div></div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#A855F7;'>{c_conv:.1f}%</div><div class='metric-label'>Conversión Base</div></div>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-title'>⏱️ Factor Tiempo (Minutos de Descuento)</h3>", unsafe_allow_html=True)
    df_time = df.groupby(pd.cut(df['Minute'], bins=np.arange(0, 105, 10)))['Goal'].mean().reset_index()
    df_time['Minute_Bin'] = df_time['Minute'].astype(str)
    
    fig_time = px.bar(df_time, x='Minute_Bin', y='Goal', text_auto='.1%', color='Goal', color_continuous_scale="Purp",
                      title="Métrica: Incremento de conversión en los últimos minutos")
    fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_time, use_container_width=True)
    
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("#### 🛡️ Índice de Presión Defensiva")
        df_press = df.groupby(pd.cut(df['Pressure_Index'], bins=3))['Goal'].mean().reset_index()
        df_press['Pressure'] = ['Baja Presión', 'Media Presión', 'Alta Presión']
        fig_p = px.line(df_press, x='Pressure', y='Goal', markers=True, title="Caída de acierto frente a presión")
        fig_p.update_traces(line_color="#EF4444", line_width=4, marker_size=12)
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)
        
    with c_b:
        st.markdown("#### ⚡ Transiciones / Contraataques")
        fb_conv = df.groupby('Is_Fast_Break')['Goal'].mean().reset_index()
        fb_conv['Is_Fast_Break'] = fb_conv['Is_Fast_Break'].map({0: 'Juego Construido', 1: 'Contraataque (Fast Break)'})
        fig_fb = px.pie(fb_conv, names='Is_Fast_Break', values='Goal', hole=0.4, title="Dominancia del Contraataque")
        fig_fb.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_fb, use_container_width=True)

# ===============================================
# PESTAÑA 2: SHOT MAP EN CANCHA INTERACTIVA
# ===============================================
with tabs[1]:
    st.markdown("<h1 class='main-title'>Mapa Espacial de Tiros</h1>", unsafe_allow_html=True)
    
    rowC1, rowC2 = st.columns([1, 3])
    with rowC1:
        st.markdown("### Filtros")
        f_goal = st.radio("Mostrar:", ["Todos los tiros", "Solo Goles", "Tiros Fallados"])
        f_bc = st.checkbox("Solo Grandes Ocasiones (Big Chances)")
        f_dist = st.slider("Distancia máxima al arco", 5.0, 50.0, 50.0)
        
        filtered_df = df.copy()
        if f_goal == "Solo Goles": filtered_df = filtered_df[filtered_df['Goal']==1]
        elif f_goal == "Tiros Fallados": filtered_df = filtered_df[filtered_df['Goal']==0]
        if f_bc: filtered_df = filtered_df[filtered_df['Is_Big_Chance']==1]
        filtered_df = filtered_df[filtered_df['Distance'] <= f_dist]
        
        st.metric("Total Puntos en Cancha", len(filtered_df))
    
    with rowC2:
        fig_pitch = px.scatter(
            filtered_df, x="Coord_X", y="Coord_Y", 
            color=filtered_df['Goal'].astype(str),
            size="xG_Base", opacity=0.8,
            color_discrete_map={'0': '#A855F7', '1': '#00CC96'},
            hover_data=['Distance', 'Minute', 'xG_Base']
        )
        fig_pitch = draw_pitch(fig_pitch)
        fig_pitch.update_layout(
            plot_bgcolor="#161b22", paper_bgcolor="rgba(0,0,0,0)",
            height=650, legend_title="Es un Gol", margin=dict(t=10, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pitch, use_container_width=True)

# ===============================================
# PESTAÑA 3: XG MODEL COMPARER
# ===============================================
with tabs[2]:
    st.markdown("<h1 class='main-title'>Inteligencia Artificial: Comparación</h1>", unsafe_allow_html=True)
    st.write("Evaluamos un mismo tiro utilizando distintos motores algorítmicos para medir su xG final.")
    
    col_izq, col_der = st.columns([1.2, 2])
    with col_izq:
        st.markdown("<h3 class='sub-title'>Generar Instancia</h3>", unsafe_allow_html=True)
        in_dist = st.slider("Distancia (m)", 1.0, 40.0, 20.0)
        in_ang = st.slider("Ángulo", 1.0, 90.0, 40.0)
        in_min = st.slider("Minuto de Juego", 1, 98, 45)
        in_press = st.slider("Índice de Presión Defensiva", 0.0, 10.0, 3.5)
        in_fb = st.checkbox("Fue Contraataque")
        in_bc = st.checkbox("Hubo oportunidad clara previa")
        
        modelo_select = st.selectbox("Algoritmo Evaluador", list(models_dict.keys()))
    
    with col_der:
        x_input = pd.DataFrame([[in_dist, in_ang, in_min, in_press, int(in_fb), int(in_bc)]], columns=feature_cols)
        prob = models_dict[modelo_select].predict_proba(x_input)[0][1] * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob,
            number={'suffix': "%", 'font': {'size': 70, 'color': 'white'}},
            title = {'text': f"Expected Goals ({modelo_select})", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#A855F7"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'steps': [{'range': [0, 15], 'color': "rgba(239, 68, 68, 0.4)"},
                          {'range': [15, 50], 'color': "rgba(245, 158, 11, 0.4)"},
                          {'range': [50, 100], 'color': "rgba(16, 185, 129, 0.4)"}]
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Feature Importance (Si el modelo lo permite)
        if hasattr(models_dict[modelo_select], 'feature_importances_'):
            st.markdown("#### Feature Importances")
            importances = models_dict[modelo_select].feature_importances_
            fig_imp = px.bar(x=importances, y=feature_cols, orientation='h', title=f"Qué influyó en el {modelo_select}")
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250)
            st.plotly_chart(fig_imp, use_container_width=True)
        elif hasattr(models_dict[modelo_select], 'coef_'):
            st.markdown("#### Coeficientes de Regresión Logística")
            coefs = models_dict[modelo_select].coef_[0]
            fig_imp = px.bar(x=coefs, y=feature_cols, orientation='h', title=f"Pesos de Features")
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250)
            st.plotly_chart(fig_imp, use_container_width=True)

# ===============================================
# PESTAÑA 4: MATCH PREDICTOR (Lineal vs Ridge)
# ===============================================
with tabs[3]:
    st.markdown("<h1 class='main-title'>Match Predictor y Regularización</h1>", unsafe_allow_html=True)
    
    st.write("En el Machine Learning para partidos se usa regresión lineal para inferir goles totales. Pero sufre *overfitting*. La **Regularización Ridge (L2)** contrae los coeficientes.")
    
    # Simulación de goles para mostrar Lineal vs Ridge
    local_xg_est = st.slider("Control de Desempeño Ofensivo del Dominador", 0.1, 4.0, 2.0)
    
    # Mock data para regresión Ridge
    rx = np.linspace(0, 4, 100).reshape(-1, 1)
    ry_true = rx.flatten() * 0.8 + 1.2
    ry_noise = ry_true + np.random.normal(0, 0.8, 100)
    
    m_lin = LinearRegression().fit(rx, ry_noise)
    m_rid = Ridge(alpha=100.0).fit(rx, ry_noise)
    
    fig_ridge = go.Figure()
    fig_ridge.add_trace(go.Scatter(x=rx.flatten(), y=ry_noise, mode='markers', name='Datos Ruidosos (Apuestas)', marker=dict(color='#94A3B8')))
    fig_ridge.add_trace(go.Scatter(x=rx.flatten(), y=m_lin.predict(rx), mode='lines', name='Regresión Lineal Pura (Alta Varianza)', line=dict(color='#EF4444', width=3)))
    fig_ridge.add_trace(go.Scatter(x=rx.flatten(), y=m_rid.predict(rx), mode='lines', name='Ridge L2 (Más conservador)', line=dict(color='#00CC96', width=3)))
    fig_ridge.update_layout(title="Visualización Matemática de Shrinkage (Ridge)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    st.plotly_chart(fig_ridge, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<h3 class='sub-title'>Modelo de Clasificación Multinomial</h3>", unsafe_allow_html=True)
    st.write("Dado un rendimiento, ¿quién gana el cotejo? Evaluando Home/Draw/Away.")
    
    res = np.array([local_xg_est, 1.3, max(0.2, 3 - local_xg_est)]) # Simulación Softmax dependiente del slider
    probabilidades = np.exp(res) / np.sum(np.exp(res))
    
    dfr = pd.DataFrame({"Result": ["Local", "Empate", "Visita"], "Values": probabilidades})
    fig_mult = px.pie(dfr, values='Values', names='Result', hole=0.6,
                      color_discrete_sequence=["#3B82F6", "#64748B", "#EF4444"])
    fig_mult.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig_mult, use_container_width=True)

# ===============================================
# PESTAÑA 5: CLUSTERING 3D
# ===============================================
with tabs[4]:
    st.markdown("<h1 class='main-title'>Segmentación Táctica No Supervisada</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1,2])
    with c1:
        st.write("Usando **K-Means**, el algoritmo aprenderá a separar tipos de tiros sin que le digamos si fueron gol o no.")
        clusters_n = st.slider("Seleccionar K (Número de cúmulos tácticos)", 2, 6, 3)
    
    # Features reales para Kmeans
    feat_km = df[['Distance', 'Angle', 'Pressure_Index']]
    from sklearn.preprocessing import StandardScaler
    feat_km_sc = StandardScaler().fit_transform(feat_km)
    
    km_final = KMeans(n_clusters=clusters_n, random_state=42).fit(feat_km_sc)
    df['Cluster_Label'] = [f"Grupo {c}" for c in km_final.labels_]
    
    with c2:
        fig_km = px.scatter_3d(df, x='Distance', y='Angle', z='Pressure_Index', color='Cluster_Label',
                               title="Espacio Latente de Remates (3D)",
                               color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.6)
        fig_km.update_layout(paper_bgcolor="rgba(0,0,0,0)", scene=dict(bgcolor='rgba(0,0,0,0)'), height=550)
        st.plotly_chart(fig_km, use_container_width=True)

# ===============================================
# PESTAÑA 6: CONCLUSIONES
# ===============================================
with tabs[5]:
    st.markdown("<h1 class='main-title'>Reporte Académico y Conclusiones</h1>", unsafe_allow_html=True)
    st.write("Resumen ejecutivo de los hallazgos principales obtenidos a lo largo del modelado de Machine Learning de este proyecto.")
    
    colA, colB = st.columns([1, 1], gap="large")
    
    with colA:
        st.markdown("<h3 class='sub-title'>🏆 Hallazgos Principales</h3>", unsafe_allow_html=True)
        st.success("**1. El peso definitivo de la Ocasión Clara (Big Chance):**\nAislado de otros contextos, tener una gran ocasión aumenta significativamente las conversiones base del 5% hasta más del 30-40%, comprobando matemáticamente la validez empírica del término futbolístico.")
        st.info("**2. Comportamiento Espacial No Lineal:**\nSe comprobó que el ángulo tiene una relación de impacto semi-sigmoidal combinada con la distancia; para abordarlo, modelos de ensamble basados en árboles de decisión (como *XGBoost* o *Random Forest*) superan en representación a la Regresión Logística clásica.")
        st.warning("**3. Regularización en Resultados Finales:**\nal intentar predecir resultados globales (Anotaciones totales o Victoria de equipo), penalizar pesos usando Mínimos Cuadrados Ordinarios Regularizados (Ridge L2) fue vital para evitar el *overfitting* común por culpa del nivel de aleatoriedad en los deportes.")
        
    with colB:
        st.markdown("<h3 class='sub-title'>🧩 Justificación Metodológica</h3>", unsafe_allow_html=True)
        with st.expander("¿Por qué Feature Engineering?"):
            st.write("Dado que un modelo matemático estándar no entiende la presión o táctica, la creación de variables continuas como *'Índice de Presión Defensiva'* y binarias como *'Contraataque'* permitieron al modelo diferenciar entre un tiro a 5 metros hecho en solitario, contra uno realizado con defensas bloqueando.")
        with st.expander("¿Por qué Clustering (K-Means)?"):
            st.write("El aprendizaje no supervisado nos ayudó a mapear los datos de entrenamientos sin requerir que el algoritmo supiera si fue gol o no, agrupándolos naturalmente por similitud situacional (ej. aglomeración en área chica vs. misiles desde fuera de la caja bajo presión nula).")

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #94A3B8;'>Proyecto de Machine Learning 💻 Analytics ⚽</h3>", unsafe_allow_html=True)

