import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np

# AI/ML Imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# --- Page Configuration ---
st.set_page_config(
    page_title="Aqua-Analytics Advanced AI",
    page_icon="💧",
    layout="wide"
)

# --- ORIGINAL CALCULATION LOGIC & STANDARDS ---
STANDARDS = {
    'Iron (Fe)': 1.0, 'Manganese (Mn)': 0.3, 'Copper (Cu)': 1.5,
    'Zinc (Zn)': 15.0, 'Cadmium (Cd)': 0.003, 'Lead (Pb)': 0.01,
    'Chromium (Cr)': 0.05
}

def calculate_hpi(df, metals_present):
    Wi = {metal: 1 / STANDARDS[metal] for metal in metals_present}
    sum_Wi = sum(Wi.values())
    hpi_values = []
    for _, row in df.iterrows():
        numerator = 0
        for metal in metals_present:
            if pd.notna(row[metal]):
                Mi = row[metal]
                Si = STANDARDS[metal]
                Qi = (Mi / Si) * 100
                numerator += Wi[metal] * Qi
        hpi = numerator / sum_Wi if sum_Wi != 0 else 0
        hpi_values.append(hpi)
    return hpi_values

def calculate_hei_cd(df, metals_present):
    hei_values = []
    for _, row in df.iterrows():
        hei_sum = 0
        for metal in metals_present:
            if pd.notna(row[metal]):
                hei_sum += row[metal] / STANDARDS[metal]
        hei_values.append(hei_sum)
    return hei_values

def calculate_cf_pli(df, metals_present):
    cf_df = pd.DataFrame()
    for metal in metals_present:
        cf_df[f'CF_{metal}'] = df[metal] / STANDARDS[metal]
    cf_df[cf_df < 0] = 0 # Ensure no negative CF values
    if len(metals_present) > 0:
        pli_values = (cf_df + 1e-9).prod(axis=1) ** (1 / len(metals_present)) # Add small epsilon to avoid log(0) type errors
    else:
        pli_values = pd.Series([0] * len(df)) # Default to 0 if no metals
    return cf_df, pli_values.tolist()

def calculate_all_indices(df):
    result_df = df.copy()
    metals_present = [metal for metal in STANDARDS.keys() if metal in df.columns]
    
    if not metals_present:
        st.warning("Warning: None of the standard metals for HPI, HEI, PLI calculation were found.")
        result_df['HPI'] = np.nan
        result_df['HEI'] = np.nan
        result_df['PLI'] = np.nan
        return result_df
    
    result_df['HPI'] = calculate_hpi(result_df, metals_present)
    result_df['HEI'] = calculate_hei_cd(result_df, metals_present)
    _, pli_values = calculate_cf_pli(result_df, metals_present)
    result_df['PLI'] = pli_values
    
    return result_df

# --- METAL EXCEEDANCE AND TREATMENT RECOMMENDATION ---
def detect_exceedances(row, metals_present):
    exceeded_metals = []
    for metal in metals_present:
        if pd.notna(row[metal]) and row[metal] > STANDARDS[metal]:
            exceeded_metals.append(f"{metal} ({row[metal]:.3f})")
    return ", ".join(exceeded_metals) if exceeded_metals else "None"

def recommend_treatment(row, features):
    recs = []
    for feature in features:
        if feature in STANDARDS and pd.notna(row[feature]) and row[feature] > STANDARDS[feature]:
            if 'Pb' in feature: recs.append("Ion Exchange for Lead")
            elif 'Fe' in feature: recs.append("Aeration + Filtration for Iron")
            elif 'Cd' in feature: recs.append("Activated Carbon for Cadmium")
            elif 'Cr' in feature: recs.append("Chemical Precipitation for Chromium")
            elif 'Cu' in feature: recs.append("Lime Softening for Copper")
            elif 'Zn' in feature: recs.append("Reverse Osmosis for Zinc")
            elif 'Mn' in feature: recs.append("Oxidation + Filtration for Manganese")
    
    return ", ".join(list(set(recs))) if recs else "No specific treatment needed"

# --- AI/ML FEATURES ---

@st.cache_data
def perform_clustering(_df, features, n_clusters=4):
    df_cluster = _df.copy()
    df_clean = df_cluster[features].dropna()
    if len(df_clean) < n_clusters:
        st.warning("Not enough data to perform clustering. Setting all to 'N/A'.")
        df_cluster['Cluster'] = 'N/A'
        return df_cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_scaled)
    # Merge back to original DataFrame, filling NaNs in new columns with default values
    df_cluster = df_cluster.merge(df_clean[['Cluster']], how='left', left_index=True, right_index=True)
    df_cluster['Cluster'] = df_cluster['Cluster'].fillna('N/A')
    return df_cluster

@st.cache_resource
def build_and_train_autoencoder(_df_scaled):
    input_dim = _df_scaled.shape[1]
    inp = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(inp)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    out = Dense(input_dim, activation='linear')(decoded)
    model = Model(inp, out)
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(_df_scaled, _df_scaled, epochs=50, batch_size=32, verbose=0)
    return model

def detect_anomalies_autoencoder(_df, features):
    df_anomaly = _df.copy()
    df_clean = df_anomaly[features].dropna()
    if len(df_clean) < 20: # Require a minimum number of samples for meaningful anomaly detection
        st.warning("Not enough clean data for adaptive anomaly detection. Setting all as non-anomalous.")
        df_anomaly['Is_Anomaly_AE'] = 0
        df_anomaly['Anomaly_Score_AE'] = 0.0
        return df_anomaly
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    autoencoder = build_and_train_autoencoder(pd.DataFrame(X_scaled)) # Pass DataFrame to autoencoder
    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95) # Top 5% as anomalies
    df_clean['Is_Anomaly_AE'] = (mse > threshold).astype(int)
    df_clean['Anomaly_Score_AE'] = mse
    
    # Merge back to original DataFrame, filling NaNs in new columns with default values
    df_anomaly = df_anomaly.merge(df_clean[['Is_Anomaly_AE', 'Anomaly_Score_AE']], how='left', left_index=True, right_index=True)
    df_anomaly['Is_Anomaly_AE'] = df_anomaly['Is_Anomaly_AE'].fillna(0).astype(int)
    df_anomaly['Anomaly_Score_AE'] = df_anomaly['Anomaly_Score_AE'].fillna(0.0) # Ensure numeric type
    return df_anomaly

def compute_hybrid_index(_df):
    df_hybrid = _df.copy()
    if 'HPI' not in df_hybrid or df_hybrid['HPI'].isnull().all() or df_hybrid['HPI'].max() == 0:
        st.warning("HPI not available or all zero, Hybrid Health Index cannot be computed.")
        df_hybrid['Hybrid_Health_Index'] = np.nan
        return df_hybrid
    
    # Normalize HPI to a 0-40 range (lower HPI is better, so it contributes positively to health index)
    min_hpi = df_hybrid['HPI'].min()
    max_hpi = df_hybrid['HPI'].max()
    if (max_hpi - min_hpi) == 0:
        hpi_component = 20 # Neutral if all HPI are the same
    else:
        hpi_component = 40 * (1 - (df_hybrid['HPI'] - min_hpi) / (max_hpi - min_hpi))
    
    # Anomaly component (0 or -30)
    anomaly_component = df_hybrid.get('Is_Anomaly_AE', 0) * (-30) # Penalize anomalies

    # Cluster component (scale cluster number to -30 to 0)
    if 'Cluster' in df_hybrid and df_hybrid['Cluster'].notna().any() and df_hybrid['Cluster'].astype(str).str.isnumeric().any(): # Check if any cluster values are numeric
        numeric_clusters = pd.to_numeric(df_hybrid['Cluster'], errors='coerce').dropna()
        if not numeric_clusters.empty and numeric_clusters.max() > 0:
            cluster_component = 0 - (numeric_clusters / numeric_clusters.max()) * 30
            # Need to align this component back to the original index
            df_hybrid_temp = df_hybrid.copy()
            df_hybrid_temp['Cluster_Numeric'] = pd.to_numeric(df_hybrid_temp['Cluster'], errors='coerce')
            cluster_component_series = pd.Series(0, index=df_hybrid_temp.index)
            cluster_component_series.loc[numeric_clusters.index] = 0 - (numeric_clusters / numeric_clusters.max()) * 30
            cluster_component = cluster_component_series
        else:
            cluster_component = 0
    else:
        cluster_component = 0

    # Base score is 100. Add positive HPI component, subtract penalties.
    df_hybrid['Hybrid_Health_Index'] = 100 + hpi_component + anomaly_component + cluster_component
    df_hybrid['Hybrid_Health_Index'] = df_hybrid['Hybrid_Health_Index'].clip(0, 100)
    return df_hybrid


@st.cache_data
def train_and_predict_quality(df_input):
    df = df_input.copy()
    
    # Define Quality_Class based on HPI (if HPI exists)
    if 'HPI' in df.columns and not df['HPI'].isnull().all():
        conditions = [
            (df['HPI'] > 150),
            (df['HPI'] >= 100) & (df['HPI'] <= 150),
            (df['HPI'] < 100)
        ]
        classes = ['Poor', 'Moderate', 'Good']
        df['Quality_Class'] = np.select(conditions, classes, default='Unknown')
    else:
        df['Quality_Class'] = 'Unknown'
        st.info("HPI not available for rule-based quality classification.")

    features = [metal for metal in STANDARDS.keys() if metal in df.columns]
    
    # Handle cases where not enough features or no numeric data
    if not features or not df[features].select_dtypes(include=np.number).columns.tolist():
        st.info("No numeric metal features available for AI quality prediction. Displaying rule-based classification based on HPI (if available).")
        df['Predicted_Quality'] = df['Quality_Class']
        return df, None, None # Return model and features as None

    df_cleaned = df.dropna(subset=features + ['Quality_Class'])
    
    if df_cleaned.empty or len(df_cleaned) < 10:
        st.info("Not enough clean data for robust AI predictive model. Displaying rule-based classification.")
        df['Predicted_Quality'] = df['Quality_Class']
        return df, None, None

    X = df_cleaned[features]
    y = df_cleaned['Quality_Class']

    if len(y.unique()) < 2:
        st.info(f"Only one unique water quality class ({y.unique()[0]}) found. Predictive model not applicable.")
        df['Predicted_Quality'] = df['Quality_Class']
        return df, None, None

    test_size_fraction = 0.2
    min_class_count = y.value_counts().min()
    use_stratify = False
    if min_class_count >= 2: # Stratification needs at least 2 samples per class in both train/test
        use_stratify = True

    try:
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Impute missing values for prediction consistently
        df_for_prediction = df[features].copy()
        for col in features:
            if not pd.api.types.is_numeric_dtype(df_for_prediction[col]):
                df_for_prediction[col] = pd.to_numeric(df_for_prediction[col], errors='coerce')
            df_for_prediction[col] = df_for_prediction[col].fillna(X_train[col].mean() if col in X_train.columns else 0)

        predictions = model.predict(df_for_prediction)
        df['Predicted_Quality'] = predictions
        
        y_pred_test = model.predict(X_test)
        st.success(f"AI Model Trained for Quality Prediction. Test Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
        
        return df, model, features

    except Exception as e:
        st.error(f"Error during AI model training for quality prediction: {e}. Falling back to rule-based classification.")
        df['Predicted_Quality'] = df['Quality_Class']
        return df, None, None


# --- UI HELPER FOR CARDS ---
def feature_card(icon, title, text):
    st.markdown(f"""
    <div style="border: 1px solid #DCDCDC; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 100%;">
        <div style="font-size: 40px;">{icon}</div>
        <h3 style="margin-top: 15px;">{title}</h3>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)

# --- UI LAYOUT ---
# st.sidebar.image("logo.png", width=100) # Assuming 'logo.png' exists
# st.sidebar.header("About Aqua-Analytics")
# st.sidebar.info("This application automates water quality index calculations and provides advanced AI insights, including clustering, anomaly detection, and treatment recommendations.")
# st.sidebar.header("Tech Stack")
# st.sidebar.code("- Streamlit, Pandas\n- Plotly, Folium\n- scikit-learn\n- TensorFlow/Keras")

st.image("banner.png") # Assuming 'banner.png' exists
st.title("Aqua-Analytics: Advanced Water Quality AI 💧")
st.markdown("A comprehensive AI tool for in-depth analysis of groundwater quality. Upload your data to unlock advanced insights.")
st.write("---")

st.header("Key Features")
col1, col2, col3, col4 = st.columns(4)
with col1:
    feature_card("⚙️", "Automated Index Calculation", "Computes standard pollution indices like HPI, HEI, and PLI for heavy metals.")
with col2:
    feature_card("🧠", "Data Clustering", "Groups similar water samples together using K-Means to identify distinct water quality profiles.")
with col3:
    feature_card("🚨", "Adaptive Anomaly Detection", "Uses a Deep Learning Autoencoder to find statistically unusual samples.")
with col4:
    feature_card("🌿", "Treatment Recommendations", "Provides rule-based suggestions for water treatment based on parameter exceedances.")

st.write("---")
uploaded_file = st.file_uploader("**Upload Your Groundwater Data to Get Started**", type=['csv', 'xlsx'])

if uploaded_file is not None:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df = df
            if 'processed_df' in st.session_state:
                del st.session_state.processed_df
            st.session_state['rf_model'] = None # Reset model on new upload
            st.session_state['rf_features'] = None
            st.session_state['rf_feature_importances'] = None
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            st.stop()

    if 'processed_df' not in st.session_state:
        st.header("Data Mapping")
        st.info("Select columns from your file that correspond to the heavy metals for analysis. Also map Latitude and Longitude for geographical analysis.")
        file_columns = st.session_state.df.columns.tolist()
        required_app_columns = ['Latitude', 'Longitude'] + list(STANDARDS.keys())
        
        mapped_columns = {}
        cols = st.columns(4)
        for i, req_col in enumerate(required_app_columns):
            with cols[i % 4]:
                options = [f"Select column for '{req_col}'"] + file_columns
                default_index = 0
                if req_col in file_columns:
                    default_index = options.index(req_col)
                mapped_columns[req_col] = st.selectbox(f"'{req_col}'", options, index=default_index, key=f"map_{req_col}")
        
        if st.button("Process Data and Run Full Analysis"):
            with st.spinner('Processing... This may take a moment.'):
                try:
                    final_mapped = {k: v for k, v in mapped_columns.items() if not v.startswith("Select column for")}
                    
                    if not final_mapped:
                        st.error("No valid columns were mapped. Please map at least some parameters.")
                        st.stop()
                    
                    processed_df = st.session_state.df[list(final_mapped.values())].copy()
                    processed_df.columns = list(final_mapped.keys())
                    
                    for col in processed_df.columns:
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    
                    metals_present = [p for p in STANDARDS.keys() if p in processed_df.columns]
                    
                    # --- Core Calculations ---
                    processed_df = calculate_all_indices(processed_df)
                    
                    # --- AI/ML Features ---
                    processed_df, rf_model, rf_features = train_and_predict_quality(processed_df)
                    st.session_state['rf_model'] = rf_model
                    st.session_state['rf_features'] = rf_features
                    if rf_model:
                        st.session_state['rf_feature_importances'] = rf_model.feature_importances_
                    else:
                        st.session_state['rf_feature_importances'] = None
                    
                    processed_df = perform_clustering(processed_df, metals_present)
                    processed_df = detect_anomalies_autoencoder(processed_df, metals_present)
                    
                    # --- Exceedances and Recommendations (ensured here) ---
                    processed_df['Exceeded_Metals'] = processed_df.apply(lambda row: detect_exceedances(row, metals_present), axis=1)
                    processed_df['Treatment_Recommendation'] = processed_df.apply(lambda row: recommend_treatment(row, metals_present), axis=1)
                    
                    # --- Hybrid Index ---
                    processed_df = compute_hybrid_index(processed_df)

                    st.session_state.processed_df = processed_df
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.exception(e) # Display full traceback for debugging

    else:
        processed_df = st.session_state.processed_df
        metals_present = [p for p in STANDARDS.keys() if p in processed_df.columns]
        
        st.header("Analysis Results")
        avg_hpi = processed_df['HPI'].mean() if 'HPI' in processed_df and not processed_df['HPI'].isnull().all() else np.nan
        anomalies_found = processed_df['Is_Anomaly_AE'].sum() if 'Is_Anomaly_AE' in processed_df else "N/A"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average HPI", f"{avg_hpi:.2f}" if pd.notna(avg_hpi) else "N/A", help="Heavy Metal Pollution Index. >150 is critical.")
        col2.metric("Anomalous Samples (AE)", f"{anomalies_found}", help="Samples flagged as unusual by the Autoencoder.")
        if 'Hybrid_Health_Index' in processed_df and not processed_df['Hybrid_Health_Index'].isnull().all():
            col3.metric("Avg. Hybrid Health Index", f"{processed_df['Hybrid_Health_Index'].mean():.2f}" if pd.notna(processed_df['Hybrid_Health_Index'].mean()) else "N/A", help="Score from 0-100. Higher is better.")
        else:
            col3.metric("Avg. Hybrid Health Index", "N/A", help="Score from 0-100. Higher is better.")
        
        st.download_button(label="📥 Download Full Results as CSV", data=processed_df.to_csv().encode('utf-8'), file_name="aqua_analytics_results.csv", mime="text/csv")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Full Data", "📈 Visualizations", "🗺️ Geospatial Map", "🧬 Advanced AI Insights", "🌿 Recommendations", "📘 Index Interpretations"])

        with tab1:
            st.subheader("Complete Analysis Data Table")
            st.dataframe(processed_df)
            st.markdown(f"""
                **Note on Metal Standards (BIS):**<br>
                Fe: {STANDARDS['Iron (Fe)']}, Mn: {STANDARDS['Manganese (Mn)']}, Cu: {STANDARDS['Copper (Cu)']}, Zn: {STANDARDS['Zinc (Zn)']},
                Cd: {STANDARDS['Cadmium (Cd)']}, Pb: {STANDARDS['Lead (Pb)']}, Cr: {STANDARDS['Chromium (Cr)']}<br>
                (All values in mg/L, except Cadmium and Lead which are typically in μg/L and converted to mg/L for consistency if input is mg/L)
            """)

        with tab2:
            st.subheader("Distribution of Key Indices")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if 'HPI' in processed_df and not processed_df['HPI'].isnull().all():
                    fig_hpi = px.histogram(processed_df, x='HPI', title='HPI Distribution', marginal="box", color_discrete_sequence=[st.config.get_option("theme.primaryColor")])
                    st.plotly_chart(fig_hpi, use_container_width=True)
                else:
                    st.info("HPI data not available for plotting.")
            with c2:
                if 'HEI' in processed_df and not processed_df['HEI'].isnull().all():
                    fig_hei = px.histogram(processed_df, x='HEI', title='HEI Distribution', marginal="box", color_discrete_sequence=[st.config.get_option("theme.primaryColor")])
                    st.plotly_chart(fig_hei, use_container_width=True)
                else:
                    st.info("HEI data not available for plotting.")
            with c3:
                if 'PLI' in processed_df and not processed_df['PLI'].isnull().all():
                    fig_pli = px.histogram(processed_df, x='PLI', title='PLI Distribution', marginal="box", color_discrete_sequence=[st.config.get_option("theme.primaryColor")])
                    st.plotly_chart(fig_pli, use_container_width=True)
                else:
                    st.info("PLI data not available for plotting.")
            
            if 'Predicted_Quality' in processed_df.columns and processed_df['Predicted_Quality'].nunique() > 1:
                st.subheader("AI Predicted Water Quality Distribution")
                quality_counts_df = processed_df['Predicted_Quality'].value_counts().reset_index()
                quality_counts_df.columns = ['Predicted_Quality_Category', 'count'] # Explicitly name columns
                fig_quality = px.pie(quality_counts_df, names='Predicted_Quality_Category', values='count', title='Distribution of AI Predicted Water Quality',
                                     color='Predicted_Quality_Category',
                                     color_discrete_map={'Poor': 'red', 'Moderate': 'orange', 'Good': 'green', 'Unknown': 'grey'})
                st.plotly_chart(fig_quality, use_container_width=True)
            else:
                st.info("AI Predicted Water Quality data not sufficient for pie chart.")
            
            if 'Is_Anomaly_AE' in processed_df and processed_df['Is_Anomaly_AE'].nunique() > 1 and 'Anomaly_Score_AE' in processed_df:
                st.subheader("Anomalous Samples Overview (Autoencoder)")
                fig_anomaly_hist = px.histogram(processed_df, x='Anomaly_Score_AE', color='Is_Anomaly_AE', title='Distribution of Anomaly Scores',
                                                color_discrete_map={0: 'blue', 1: 'red'})
                st.plotly_chart(fig_anomaly_hist, use_container_width=True)
            else:
                st.info("Anomaly detection data not sufficient for histogram.")


        with tab3:
            st.subheader("Pollution Hotspot Map")
            if 'Latitude' in processed_df.columns and 'Longitude' in processed_df.columns:
                map_df = processed_df.dropna(subset=['Latitude', 'Longitude'])
                
                if not map_df.empty:
                    map_center = [map_df['Latitude'].mean(), map_df['Longitude'].mean()]
                    m = folium.Map(location=map_center, zoom_start=10)
                    
                    def get_color(hpi_value, is_anomaly=0):
                        if is_anomaly == 1: return 'purple'
                        if hpi_value > 150: return 'red'
                        elif hpi_value >= 100: return 'orange'
                        else: return 'green'

                    for idx, row in map_df.iterrows():
                        popup_html = f"""
                        <b>Sample ID:</b> {idx}<br>
                        <b>HPI:</b> {row.get('HPI', np.nan):.2f}<br>
                        <b>PLI:</b> {row.get('PLI', np.nan):.2f}<br>
                        <b>AI Quality:</b> {row.get('Predicted_Quality', 'N/A')}<br>
                        <b>Anomaly:</b> {"Yes" if row.get('Is_Anomaly_AE', 0) == 1 else "No"}<br>
                        <b>Exceeded Metals:</b> {row.get('Exceeded_Metals', 'None')}<br>
                        <b>Hybrid Health Index:</b> {row.get('Hybrid_Health_Index', np.nan):.2f}<br>
                        """
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=5,
                            popup=folium.Popup(popup_html, max_width=300),
                            color=get_color(row.get('HPI', 0), row.get('Is_Anomaly_AE', 0)),
                            fill=True,
                            fill_color=get_color(row.get('HPI', 0), row.get('Is_Anomaly_AE', 0)),
                            fill_opacity=0.7
                        ).add_to(m)
                    st_folium(m, width=725, height=500, key="folium_map_1")
                else:
                    st.warning("Could not generate map. No valid Latitude/Longitude data found after removing missing values.")
            else:
                st.warning("Could not generate map. 'Latitude' and 'Longitude' columns not found or not mapped.")

        with tab4:
            st.subheader("Deep Dive into AI Insights & Hybrid Index")
            st.markdown("---")
            st.markdown("#### 🧬 Data Clustering (K-Means)")
            st.markdown("Samples are grouped into clusters based on their chemical profiles. This helps identify distinct types of water quality.")
            if 'Cluster' in processed_df and len(metals_present) > 1 and processed_df['Cluster'].dtype != 'object':
                fig_cluster = px.scatter_matrix(processed_df, dimensions=metals_present, color="Cluster", title="Water Quality Clusters", height=700)
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.info("Clustering plot requires at least two features and successful cluster calculation.")
            
            st.markdown("---")
            st.markdown("#### 🎯 Feature Importance for Overall Quality Prediction")
            st.markdown("Which metals are most influential in determining the overall water quality prediction?")
            if st.session_state.get('rf_feature_importances') is not None and st.session_state.get('rf_features') and len(st.session_state['rf_features']) > 0:
                feature_importance_df = pd.DataFrame({
                    'Metal': st.session_state['rf_features'],
                    'Importance': st.session_state['rf_feature_importances']
                }).sort_values(by='Importance', ascending=False)
                
                fig_importance = px.bar(feature_importance_df, x='Metal', y='Importance',
                                         title='Feature Importance for Water Quality Prediction',
                                         color_discrete_sequence=[st.config.get_option("theme.primaryColor")])
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Feature importances could not be calculated (AI model not trained or features missing).")

            st.markdown("---")
            st.markdown("#### 🕵️‍♂️ Adaptive Anomaly Detection (Autoencoder)")
            st.markdown("The deep learning model identifies samples that are statistically unusual compared to the majority of your data. These are flagged in the `Is_Anomaly_AE` column and have higher `Anomaly_Score_AE`.")
            if 'Is_Anomaly_AE' in processed_df and 'Anomaly_Score_AE' in processed_df and processed_df['Is_Anomaly_AE'].sum() > 0:
                anomalous_samples_count = processed_df['Is_Anomaly_AE'].sum()
                st.info(f"Detected **{anomalous_samples_count}** anomalous sample(s).")
                cols_to_display_anom = ['HPI', 'PLI', 'Predicted_Quality', 'Anomaly_Score_AE'] + [m for m in metals_present if m in processed_df.columns]
                st.dataframe(processed_df[processed_df['Is_Anomaly_AE'] == 1][cols_to_display_anom].fillna('N/A'))
            else:
                st.info("No anomalies detected or insufficient data for detection.")

            st.markdown("---")
            st.markdown("#### 🌍 Hybrid Health Index")
            st.markdown("This custom index (0-100) combines the traditional HPI, cluster assignment, and anomaly status into a single, comprehensive score of water health. **Higher values indicate healthier water.**")
            if 'Hybrid_Health_Index' in processed_df and not processed_df['Hybrid_Health_Index'].isnull().all():
                fig_hybrid_gauge = px.bar(processed_df, x=processed_df.index, y='Hybrid_Health_Index', color='Hybrid_Health_Index',
                                    title="Hybrid Health Index per Sample", range_y=[0, 100],
                                    color_continuous_scale=px.colors.diverging.RdYlGn)
                st.plotly_chart(fig_hybrid_gauge, use_container_width=True)
                
                st.write("Interpretation:")
                st.markdown("- **80-100**: Excellent Water Health")
                st.markdown("- **60-79**: Good Water Health")
                st.markdown("- **40-59**: Moderate Water Health")
                st.markdown("- **20-39**: Poor Water Health")
                st.markdown("- **0-19**: Critical Water Health")

            else:
                st.info("Hybrid Health Index could not be calculated.")

        with tab5: # Recommendations Tab
            st.subheader("🌿 Treatment Recommendations Based on Exceedances")
            st.markdown("This section provides specific treatment recommendations for samples where metal concentrations exceed the BIS standards.")
            df_recs = processed_df[processed_df['Treatment_Recommendation'] != "No specific treatment needed"].copy()
            if not df_recs.empty:
                st.dataframe(df_recs[['Exceeded_Metals', 'Treatment_Recommendation'] + [m for m in metals_present if m in processed_df.columns]].fillna('N/A'))
            else:
                st.info("No specific treatment recommendations needed for any samples based on current data and standards.")
        
        with tab6: # Index Interpretations Tab
            st.subheader("📘 Interpretation of Pollution Indices")
            st.markdown("Understanding the significance of the calculated indices is crucial for effective water quality management.")
            st.markdown("---")
            st.markdown("#### Heavy Metal Pollution Index (HPI):")
            st.markdown("""
            * **< 100:** Indicates **low pollution**. Water is generally considered safe for consumption and other uses concerning heavy metals.
            * **100 - 150:** Suggests **medium pollution**. Water quality is deteriorating, and caution is advised. Further investigation or monitoring may be necessary.
            * **> 150:** Represents **high pollution** or critically polluted water. This water is typically considered unsafe for drinking and may pose significant health risks. Urgent intervention and treatment are required.
            """)
            st.markdown("---")
            st.markdown("#### Heavy Metal Evaluation Index (HEI):")
            st.markdown("""
            The HEI is a numerical value that reflects the overall quality of water with respect to heavy metals. It is calculated as the sum of the ratio of each metal concentration to its respective standard.
            * **< 10:** Low pollution.
            * **10 - 20:** Medium pollution.
            * **> 20:** High pollution.
            Higher values of HEI signify a greater degree of heavy metal contamination and a higher potential risk.
            """)
            st.markdown("---")
            st.markdown("#### Pollution Load Index (PLI):")
            st.markdown("""
            The PLI is a comprehensive indicator of the overall metal pollution in a given site. It is derived from the product of contamination factors (CF) for individual metals.
            * **< 1:** Suggests **no pollution** by heavy metals. The water quality meets or is better than the standards.
            * **> 1:** Indicates that **pollution exists**, with higher values reflecting an increasing level of heavy metal contamination. A PLI significantly greater than 1 signifies severe pollution.
            """)
            st.markdown("---")
            st.markdown("#### AI Predicted Water Quality:")
            st.markdown("""
            This is a data-driven classification (Good, Moderate, Poor) based on the patterns learned by an AI model (Random Forest) from your metal concentration data. It provides an independent, statistical assessment of overall water quality.
            """)
            st.markdown("---")
            st.markdown("#### Hybrid Health Index:")
            st.markdown("""
            This is a custom, comprehensive index (scale 0-100) that integrates traditional pollution indices (like HPI), results from AI-driven clustering, and anomaly detection.
            * **Higher values indicate healthier water.**
            * It provides a more holistic view by penalizing high pollution, unusual (anomalous) samples, and samples belonging to potentially 'worse' clusters.
            """)

else:
    st.info("Upload your data using the button above to begin the full analysis.")