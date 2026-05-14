import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG 
st.set_page_config(
    page_title="Weather & Air Quality Analysis",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# CONSTANTS 
FEATURES = [
    'temperature', 'humidity', 'pressure', 'wind_speed',
    'cloudcover', 'precip', 'feelslike', 'uv_index',
    'co', 'no2', 'o3', 'so2', 'pm10',
    'wind_dir_enc', 'pm_ratio', 'nox', 'heat_index'
]
TARGET = 'AQI'

AQI_COLORS = {
    "Good":      "#22c55e",
    "Moderate":  "#eab308",
    "Poor":      "#f97316",
    "Very Poor": "#ef4444",
}

def aqi_category(val):
    if val <= 50:   return "Good",      "#22c55e"
    if val <= 100:  return "Moderate",  "#eab308"
    if val <= 200:  return "Poor",      "#f97316"
    return "Very Poor", "#ef4444"

PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"

#  DATA LOADING & PROCESSING 
@st.cache_data
def load_and_process():
    df = pd.read_csv("indian_weather_data.csv")
    df.drop(["visibility", "lat", "lon", "wind_degree"], axis=1, errors='ignore', inplace=True)

    df['AQI'] = (df['pm2_5'] + df['pm10'] + df['no2'] + df['so2'] + df['o3']) / 5

    df['temp_category'] = pd.cut(df['temperature'], bins=[0,15,30,50],
                                  labels=['Cold','Warm','Hot'])
    df['aqi_category']  = pd.cut(df['AQI'], bins=[0,50,100,200,500],
                                  labels=['Good','Moderate','Poor','Very Poor'])

    le = LabelEncoder()
    df['wind_dir_enc'] = le.fit_transform(df['wind_dir'])
    df['pm_ratio']     = df['pm2_5'] / (df['pm10'] + 1e-5)
    df['nox']          = df['no2'] + df['so2']
    df['heat_index']   = df['temperature'] * df['humidity'] / 100

    return df

@st.cache_resource
def train_models(_df):
    X = _df[FEATURES]
    y = _df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":  LinearRegression(),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cv    = cross_val_score(model, X, y, cv=5, scoring='r2')
        results[name] = {
            "model": model, "preds": preds,
            "MAE":   mean_absolute_error(y_test, preds),
            "RMSE":  np.sqrt(mean_squared_error(y_test, preds)),
            "R2":    r2_score(y_test, preds),
            "CV_R2": cv.mean(),
        }
    best_name  = max(results, key=lambda n: results[n]['R2'])
    rf_model   = results["Random Forest"]["model"]
    all_preds  = rf_model.predict(X)          # predictions for every row
    return results, best_name, rf_model, X_test, y_test, all_preds

# LOAD DATA 
try:
    df = load_and_process()
    results, best_name, rf_model, X_test, y_test, all_preds = train_models(df)
    df = df.copy()
    df['AQI_predicted'] = all_preds
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.error(f" Could not load `indian_weather_data.csv`. Make sure it's in the same folder as app.py.\n\n`{e}`")
    st.stop()

city_summary = (
    df.groupby('city')[['AQI','AQI_predicted','pm2_5','temperature','humidity']]
    .mean().round(2).sort_values('AQI', ascending=False)
)

#  SIDEBAR 
with st.sidebar:
    st.markdown("## 🌫️ **AQI Dashboard**")
    st.markdown("*Weather & Air Quality Analysis*")
    st.divider()

    page = st.radio(
        "Navigate",
        [" Overview", "EDA", " ML Models", "AQI Predictor", " City Explorer"],
        label_visibility="collapsed"
    )
    st.divider()

    cities = sorted(df['city'].unique())
    selected_city = st.selectbox(" Focus City", ["All Cities"] + cities,
                                  index=cities.index("Patna") + 1 if "Patna" in cities else 0)
    st.divider()
    st.markdown("**Dataset Info**")
    st.markdown(f"-  `{df.shape[0]:,}` records")
    st.markdown(f"- `{df['city'].nunique()}` cities")
    st.markdown(f"-  `{df.shape[1]}` features")
    st.markdown(f"-  Best model: `{best_name}`")

# FILTER BY CITY 
view_df = df if selected_city == "All Cities" else df[df['city'] == selected_city]


#  PAGE 1 — OVERVIEW

if page == " Overview":
    st.title(" Weather & Air Quality — Overview")
    st.caption(f"Showing: **{selected_city}** · {len(view_df):,} records")
    st.divider()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg AQI",        f"{view_df['AQI'].mean():.1f}")
    col2.metric("Avg PM2.5",      f"{view_df['pm2_5'].mean():.1f} µg/m³")
    col3.metric("Avg Temperature",f"{view_df['temperature'].mean():.1f} °C")
    col4.metric("Avg Humidity",   f"{view_df['humidity'].mean():.1f}%")
    col5.metric("Avg Wind Speed", f"{view_df['wind_speed'].mean():.1f} km/h")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">AQI Category Breakdown</div>', unsafe_allow_html=True)
        cat_counts = view_df['aqi_category'].value_counts().reset_index()
        cat_counts.columns = ['Category','Count']
        fig = px.pie(cat_counts, values='Count', names='Category',
                     color='Category',
                     color_discrete_map=AQI_COLORS,
                     hole=0.45, template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(paper_bgcolor=CHART_BG, showlegend=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Top 15 Cities by Avg AQI</div>', unsafe_allow_html=True)
        top15 = city_summary.head(15).reset_index()
        fig = px.bar(top15, x='AQI', y='city', orientation='h',
                     color='AQI', color_continuous_scale='RdYlGn_r',
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                          yaxis_title="", coloraxis_showscale=False,
                          margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Pollutant Averages</div>', unsafe_allow_html=True)
    pollutants = ['pm2_5','pm10','no2','so2','o3','co']
    poll_means = view_df[pollutants].mean().reset_index()
    poll_means.columns = ['Pollutant','Avg Concentration']
    fig = px.bar(poll_means, x='Pollutant', y='Avg Concentration',
                 color='Avg Concentration', color_continuous_scale='Reds',
                 template=PLOTLY_TEMPLATE)
    fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                      coloraxis_showscale=False, margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)


#  PAGE 2 — EDA

elif page == " EDA":
    st.title(" Exploratory Data Analysis")
    st.caption(f"Showing: **{selected_city}** · {len(view_df):,} records")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Scatter Plots", "Correlations", "Raw Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">PM2.5 Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(view_df, x='pm2_5', nbins=50, marginal='box',
                               color_discrete_sequence=['#7c8cde'], template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">AQI Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(view_df, x='AQI', nbins=50, marginal='box',
                               color_discrete_sequence=['#f97316'], template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-header">Temperature Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(view_df, x='temperature', nbins=40, marginal='box',
                               color_discrete_sequence=['#22c55e'], template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="section-header">AQI by Temperature Category</div>', unsafe_allow_html=True)
            fig = px.box(view_df, x='temp_category', y='AQI',
                         color='temp_category',
                         color_discrete_map={'Cold':'#60a5fa','Warm':'#fbbf24','Hot':'#f87171'},
                         template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Temperature vs Humidity</div>', unsafe_allow_html=True)
            fig = px.scatter(view_df, x='temperature', y='humidity',
                             color='AQI', color_continuous_scale='RdYlGn_r',
                             opacity=0.6, template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Wind Speed vs AQI</div>', unsafe_allow_html=True)
            fig = px.scatter(view_df, x='wind_speed', y='AQI',
                             color='AQI', color_continuous_scale='RdYlGn_r',
                             opacity=0.6, trendline='ols', template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-header">PM2.5 vs AQI</div>', unsafe_allow_html=True)
            fig = px.scatter(view_df, x='pm2_5', y='AQI', opacity=0.5,
                             color='aqi_category', color_discrete_map=AQI_COLORS,
                             template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="section-header">CO vs AQI</div>', unsafe_allow_html=True)
            fig = px.scatter(view_df, x='co', y='AQI', opacity=0.5,
                             color='AQI', color_continuous_scale='RdYlGn_r',
                             template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        num_cols = ['temperature','humidity','pressure','wind_speed','co','no2',
                    'o3','so2','pm2_5','pm10','AQI','uv_index']
        corr = view_df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale='RdBu_r', template=PLOTLY_TEMPLATE)
        fig.update_layout(paper_bgcolor=CHART_BG, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(view_df.head(100), use_container_width=True, height=400)
        st.caption(f"Showing first 100 of {len(view_df):,} rows")


#  PAGE 3 — ML 
elif page == " ML Models":
    st.title(" Machine Learning Models")
    st.divider()

    # Model comparison KPIs
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            is_best = (name == best_name)
            label = f"{' ' if is_best else ''}{name}"
            st.markdown(f"**{label}**")
            st.metric("R² Score", f"{res['R2']:.3f}")
            st.metric("MAE",      f"{res['MAE']:.2f}")
            st.metric("RMSE",     f"{res['RMSE']:.2f}")
            st.metric("CV R²",    f"{res['CV_R2']:.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">R² Score by Model</div>', unsafe_allow_html=True)
        names  = list(results.keys())
        r2vals = [results[n]['R2'] for n in names]
        colors = ['#7c8cde' if n != best_name else '#22c55e' for n in names]
        fig = go.Figure(go.Bar(x=names, y=r2vals, marker_color=colors,
                               text=[f"{v:.3f}" for v in r2vals], textposition='outside'))
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                          plot_bgcolor=CHART_BG, yaxis_range=[0,1.1],
                          margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Actual vs Predicted AQI (Best Model)</div>', unsafe_allow_html=True)
        best_res = results[best_name]
        mn, mx = y_test.min(), y_test.max()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=best_res['preds'], mode='markers',
                                  marker=dict(color='#7c8cde', opacity=0.6, size=5),
                                  name='Predictions'))
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                  line=dict(color='red', dash='dash'), name='Perfect'))
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                          plot_bgcolor=CHART_BG, xaxis_title="Actual AQI",
                          yaxis_title="Predicted AQI", margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Residuals Plot</div>', unsafe_allow_html=True)
        residuals = y_test - best_res['preds']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=best_res['preds'], y=residuals, mode='markers',
                                  marker=dict(color='#f97316', opacity=0.6, size=5)))
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                          plot_bgcolor=CHART_BG, xaxis_title="Predicted AQI",
                          yaxis_title="Residual", margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Top 10 Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
        rf_model_res = results["Random Forest"]["model"]
        imp = pd.Series(rf_model_res.feature_importances_, index=FEATURES).sort_values().tail(10)
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h',
                               marker_color='#ef4444'))
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                          plot_bgcolor=CHART_BG, xaxis_title="Importance",
                          margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)


#  PAGE 4 — AQI PREDICTOR

elif page == " AQI Predictor":
    st.title(" AQI Predictor")
    st.caption("Enter weather & pollutant readings to predict Air Quality Index in real time.")
    st.divider()

    # Presets
    presets = {
        "Patna (typical)"   : dict(temperature=35, humidity=65, pressure=1005, wind_speed=10,
                                    cloudcover=30, precip=0, feelslike=38, uv_index=6,
                                    co=800, no2=12, o3=160, so2=30, pm10=90),
        "Mumbai (coastal)"  : dict(temperature=30, humidity=80, pressure=1010, wind_speed=18,
                                    cloudcover=50, precip=2, feelslike=34, uv_index=7,
                                    co=500, no2=8, o3=100, so2=15, pm10=55),
        "Delhi (polluted)"  : dict(temperature=22, humidity=55, pressure=1008, wind_speed=6,
                                    cloudcover=20, precip=0, feelslike=22, uv_index=4,
                                    co=1200, no2=40, o3=180, so2=50, pm10=180),
        "Clean day"         : dict(temperature=25, humidity=45, pressure=1015, wind_speed=25,
                                    cloudcover=10, precip=0, feelslike=24, uv_index=5,
                                    co=200, no2=4, o3=60, so2=5, pm10=20),
    }

    preset_choice = st.selectbox("Load a preset city scenario", ["— Custom —"] + list(presets.keys()))
    preset = presets.get(preset_choice, {})

    def pv(key, default): return preset.get(key, default)

    st.markdown('<div class="section-header">Weather Parameters</div>', unsafe_allow_html=True)
    wc1, wc2, wc3, wc4 = st.columns(4)
    temperature = wc1.slider("Temperature (°C)",  0, 50,  pv('temperature', 30))
    humidity    = wc2.slider("Humidity (%)",       0, 100, pv('humidity', 60))
    pressure    = wc3.slider("Pressure (hPa)", 980, 1040, pv('pressure', 1010))
    wind_speed  = wc4.slider("Wind Speed (km/h)",  0, 60,  pv('wind_speed', 12))

    wc5, wc6, wc7, wc8 = st.columns(4)
    cloudcover = wc5.slider("Cloud Cover (%)",  0, 100, pv('cloudcover', 30))
    precip     = wc6.slider("Precipitation (mm)", 0, 50, pv('precip', 0))
    feelslike  = wc7.slider("Feels Like (°C)",  0, 55, pv('feelslike', 33))
    uv_index   = wc8.slider("UV Index",         0, 12,  pv('uv_index', 5))

    st.markdown('<div class="section-header">Pollutant Levels</div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    co   = pc1.number_input("CO (µg/m³)",  0, 5000, pv('co', 600))
    no2  = pc2.number_input("NO₂ (µg/m³)", 0, 200,  pv('no2', 15))
    o3   = pc3.number_input("O₃ (µg/m³)",  0, 400,  pv('o3', 140))
    so2  = pc4.number_input("SO₂ (µg/m³)", 0, 200,  pv('so2', 25))
    pm10 = pc5.number_input("PM10 (µg/m³)",0, 500,  pv('pm10', 80))

    if st.button(" Predict AQI", type="primary", use_container_width=True):
        wind_dir_enc = 4
        pm_ratio   = 0.6
        nox        = no2 + so2
        heat_index = temperature * humidity / 100
        row = pd.DataFrame([[
            temperature, humidity, pressure, wind_speed, cloudcover,
            precip, feelslike, uv_index, co, no2, o3, so2, pm10,
            wind_dir_enc, pm_ratio, nox, heat_index
        ]], columns=FEATURES)
        aqi_val = round(rf_model.predict(row)[0], 1)
        cat, color = aqi_category(aqi_val)

        st.markdown(f"""
        <div class="pred-box">
            <p>Predicted Air Quality Index</p>
            <h2 style="color:{color};">{aqi_val}</h2>
            <span class="aqi-badge" style="background:{color}22; color:{color}; border:1px solid {color}44;">
                {cat}
            </span>
            <p style="margin-top:12px;">Using Random Forest · {best_name} (Best: R²={results[best_name]['R2']:.3f})</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            title={'text': "AQI", 'font': {'color':'#e2e8f0'}},
            gauge={
                'axis': {'range':[0,500], 'tickcolor':'#8892a4'},
                'bar': {'color': color},
                'steps': [
                    {'range':[0,50],    'color':'#166534'},
                    {'range':[50,100],  'color':'#854d0e'},
                    {'range':[100,200], 'color':'#9a3412'},
                    {'range':[200,500], 'color':'#7f1d1d'},
                ],
                'threshold': {'line':{'color':'white','width':3}, 'value': aqi_val}
            }
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG,
                          height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # AQI reference table
        st.markdown('<div class="section-header">AQI Reference Scale</div>', unsafe_allow_html=True)
        ref = pd.DataFrame({
            "Range":   ["0–50","51–100","101–200","201–500"],
            "Category":["Good","Moderate","Poor","Very Poor"],
            "Health Impact":["Minimal","Acceptable, sensitive groups at risk",
                             "Unhealthy for sensitive groups","Hazardous"]
        })
        st.dataframe(ref, hide_index=True, use_container_width=True)


#  PAGE 5—CITY EXPLORER

elif page == " City Explorer":
    st.title(" City Explorer")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">City AQI Ranking</div>', unsafe_allow_html=True)
        n_cities = st.slider("Show top N cities", 5, len(city_summary), 20)
        top_n = city_summary.head(n_cities).reset_index()
        fig = px.bar(top_n, x='city', y='AQI',
                     color='AQI', color_continuous_scale='RdYlGn_r',
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                          xaxis_tickangle=-45, coloraxis_showscale=False,
                          margin=dict(t=20,b=80))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Actual vs Predicted AQI (by City)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        top_n_data = city_summary.head(n_cities).reset_index()
        fig.add_trace(go.Bar(name='Actual AQI',    x=top_n_data['city'], y=top_n_data['AQI'],
                             marker_color='#7c8cde'))
        fig.add_trace(go.Bar(name='Predicted AQI', x=top_n_data['city'], y=top_n_data['AQI_predicted'],
                             marker_color='#f97316', opacity=0.8))
        fig.update_layout(barmode='group', template=PLOTLY_TEMPLATE,
                          paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                          xaxis_tickangle=-45, margin=dict(t=20,b=80))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">City Comparison Table</div>', unsafe_allow_html=True)
    display_df = city_summary.reset_index().copy()
    display_df.columns = ['City','Avg AQI','Predicted AQI','Avg PM2.5','Avg Temp (°C)','Avg Humidity (%)']
    display_df['AQI Status'] = display_df['Avg AQI'].apply(lambda x: aqi_category(x)[0])
    st.dataframe(
        display_df.style.background_gradient(subset=['Avg AQI'], cmap='RdYlGn_r'),
        use_container_width=True, height=450, hide_index=True
    )

    st.markdown('<div class="section-header">PM2.5 vs AQI by City (Bubble Chart)</div>', unsafe_allow_html=True)
    fig = px.scatter(city_summary.reset_index(), x='temperature', y='AQI',
                     size='pm2_5', color='AQI', text='city',
                     color_continuous_scale='RdYlGn_r', size_max=40,
                     template=PLOTLY_TEMPLATE)
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                      xaxis_title="Avg Temperature (°C)", margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)
