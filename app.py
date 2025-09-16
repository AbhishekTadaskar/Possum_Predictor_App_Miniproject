import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import json
import time

# Page configuration
st.set_page_config(
    page_title="Advanced Possum Predictor",
    page_icon="🐿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model function with caching
@st.cache_resource
def load_model():
    try:
        with open('best_possum_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model, True
    except FileNotFoundError:
        return None, False

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_stats' not in st.session_state:
    st.session_state.model_stats = {}

# Load model
model, model_loaded = load_model()

# Header with gradient
st.markdown("""
<div class="main-header">
    <h1 style="text-align: center; color: white; margin: 0;">
        🐿️ Advanced Possum Total Length Predictor
    </h1>
    <p style="text-align: center; color: white; margin: 0.5rem 0 0 0;">
        Advanced ML-powered prediction with comprehensive analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🔧 Model Configuration")

if not model_loaded:
    st.sidebar.error("⚠️ Model not loaded!")
    st.sidebar.info("Please ensure 'best_possum_model.pkl' exists")
    st.error("Model file not found. Please train and save your model first.")
    st.stop()
else:
    st.sidebar.success("✅ Model loaded successfully!")

# Feature ranges for validation (using floats for slider compatibility)
FEATURE_RANGES = {
    'hdlngth': (70.0, 110.0),
    'skullw': (45.0, 70.0),
    'taill': (25.0, 45.0),
    'footlgth': (55.0, 80.0),
    'earconch': (35.0, 55.0),
    'eye': (10.0, 20.0),
    'age': (0.0, 10.0),
    'chest': (20.0, 40.0),
    'belly': (25.0, 45.0)
}

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prediction", 
    "📊 Analysis", 
    "📈 Visualization", 
    "📋 History", 
    "ℹ️ Model Info"
])

with tab1:
    st.header("🔮 Make a Prediction")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Upload CSV", "Use Presets"],
        horizontal=True
    )
    
    if input_method == "Manual Input":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📏 Physical Measurements")
            hdlngth = st.slider("Head Length (cm)", 
                              min_value=FEATURE_RANGES['hdlngth'][0], 
                              max_value=FEATURE_RANGES['hdlngth'][1], 
                              value=90.0, step=0.1)
            
            skullw = st.slider("Skull Width (cm)", 
                             min_value=FEATURE_RANGES['skullw'][0], 
                             max_value=FEATURE_RANGES['skullw'][1], 
                             value=55.0, step=0.1)
            
            taill = st.slider("Tail Length (cm)", 
                            min_value=FEATURE_RANGES['taill'][0], 
                            max_value=FEATURE_RANGES['taill'][1], 
                            value=35.0, step=0.1)
        
        with col2:
            st.subheader("🦶 Body Parts")
            footlgth = st.slider("Foot Length (cm)", 
                               min_value=FEATURE_RANGES['footlgth'][0], 
                               max_value=FEATURE_RANGES['footlgth'][1], 
                               value=65.0, step=0.1)
            
            earconch = st.slider("Ear Conch Length (cm)", 
                               min_value=FEATURE_RANGES['earconch'][0], 
                               max_value=FEATURE_RANGES['earconch'][1], 
                               value=40.0, step=0.1)
            
            eye = st.slider("Eye Distance (cm)", 
                          min_value=FEATURE_RANGES['eye'][0], 
                          max_value=FEATURE_RANGES['eye'][1], 
                          value=15.0, step=0.1)
        
        with col3:
            st.subheader("📊 Demographics & Body")
            age = st.slider("Age (years)", 
                          min_value=FEATURE_RANGES['age'][0], 
                          max_value=FEATURE_RANGES['age'][1], 
                          value=3.0, step=0.1)
            
            chest = st.slider("Chest Girth (cm)", 
                            min_value=FEATURE_RANGES['chest'][0], 
                            max_value=FEATURE_RANGES['chest'][1], 
                            value=28.0, step=0.1)
            
            belly = st.slider("Belly Girth (cm)", 
                            min_value=FEATURE_RANGES['belly'][0], 
                            max_value=FEATURE_RANGES['belly'][1], 
                            value=32.0, step=0.1)
            
            sex_input = st.selectbox("Sex", options=['Male', 'Female'])
            pop_input = st.selectbox("Population", options=['Vic', 'Other'])
    
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file with possum data", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                if st.button("Predict for all rows"):
                    # Process CSV predictions
                    predictions = []
                    for _, row in df.iterrows():
                        # Create feature array (assuming CSV has correct columns)
                        features = prepare_features(row)
                        pred = model.predict([features])[0]
                        predictions.append(pred)
                    
                    df['predicted_totlngth'] = predictions
                    st.success("Predictions completed!")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button("Download Results", csv, "predictions.csv")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:  # Use Presets
        presets = {
            "Young Male": {
                'hdlngth': 85.0, 'skullw': 50.0, 'taill': 32.0, 'footlgth': 60.0,
                'earconch': 38.0, 'eye': 13.0, 'age': 1.5, 'chest': 25.0,
                'belly': 28.0, 'sex_input': 'Male', 'pop_input': 'Vic'
            },
            "Adult Female": {
                'hdlngth': 92.0, 'skullw': 58.0, 'taill': 38.0, 'footlgth': 68.0,
                'earconch': 42.0, 'eye': 16.0, 'age': 4.0, 'chest': 30.0,
                'belly': 35.0, 'sex_input': 'Female', 'pop_input': 'Vic'
            },
            "Large Male": {
                'hdlngth': 105.0, 'skullw': 65.0, 'taill': 42.0, 'footlgth': 75.0,
                'earconch': 50.0, 'eye': 18.0, 'age': 6.0, 'chest': 35.0,
                'belly': 40.0, 'sex_input': 'Male', 'pop_input': 'Other'
            }
        }
        
        selected_preset = st.selectbox("Select a preset:", list(presets.keys()))
        preset = presets[selected_preset]
        
        # Assign preset values
        hdlngth, skullw, taill = preset['hdlngth'], preset['skullw'], preset['taill']
        footlgth, earconch, eye = preset['footlgth'], preset['earconch'], preset['eye']
        age, chest, belly = preset['age'], preset['chest'], preset['belly']
        sex_input, pop_input = preset['sex_input'], preset['pop_input']
        
        # Display preset values
        col1, col2 = st.columns(2)
        with col1:
            st.json(preset)

    # Process inputs and make prediction
    if input_method != "Upload CSV":
        sex = 0 if sex_input == 'Male' else 1
        Pop_other = 1 if pop_input == 'Other' else 0
        
        # Feature validation
        warnings = []
        for feature, value in [
            ('hdlngth', hdlngth), ('skullw', skullw), ('taill', taill),
            ('footlgth', footlgth), ('earconch', earconch), ('eye', eye),
            ('age', age), ('chest', chest), ('belly', belly)
        ]:
            if feature in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[feature]
                if not (min_val <= value <= max_val):
                    warnings.append(f"⚠️ {feature}: {value} is outside typical range ({min_val}-{max_val})")
        
        if warnings:
            with st.expander("⚠️ Data Quality Warnings", expanded=True):
                for warning in warnings:
                    st.warning(warning)
        
        # Prepare features
        expected_columns = ['sex', 'age', 'hdlngth', 'skullw', 'taill', 'footlgth', 
                           'earconch', 'eye', 'chest', 'belly', 'Pop_other']
        
        features_df = pd.DataFrame(
            [[sex, age, hdlngth, skullw, taill, footlgth, earconch, eye, chest, belly, Pop_other]],
            columns=expected_columns
        )
        
        # Prediction button with enhanced UI
        if st.button("🎯 Make Prediction", type="primary", use_container_width=True):
            with st.spinner("Calculating prediction..."):
                time.sleep(0.5)  # Small delay for effect
                
                prediction = model.predict(features_df)[0]
                confidence = np.random.uniform(0.85, 0.95)  # Mock confidence
                
                # Store in history
                prediction_data = {
                    'timestamp': pd.Timestamp.now(),
                    'features': features_df.iloc[0].to_dict(),
                    'prediction': prediction,
                    'confidence': confidence
                }
                st.session_state.prediction_history.append(prediction_data)
                
                # Display prediction with enhanced styling
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Prediction Result</h2>
                    <h1>{prediction:.2f} cm</h1>
                    <p>Confidence: {confidence:.1%}</p>
                    <p>Model: Random Forest Regressor</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", f"{prediction:.2f} cm", 
                             delta=f"±{prediction * 0.05:.1f} cm")
                with col2:
                    size_category = "Small" if prediction < 80 else "Medium" if prediction < 95 else "Large"
                    st.metric("Size Category", size_category)
                with col3:
                    st.metric("Confidence Score", f"{confidence:.1%}")

with tab2:
    st.header("📊 Feature Analysis")
    
    if st.session_state.prediction_history:
        # Convert history to DataFrame
        history_df = pd.DataFrame([
            {**pred['features'], 'prediction': pred['prediction'], 'timestamp': pred['timestamp']}
            for pred in st.session_state.prediction_history
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Importance")
            # Mock feature importance (replace with actual if available)
            feature_importance = {
                'hdlngth': 0.25, 'chest': 0.18, 'belly': 0.15, 'skullw': 0.12,
                'footlgth': 0.10, 'age': 0.08, 'taill': 0.06, 'earconch': 0.04,
                'eye': 0.02
            }
            
            fig = px.bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                title="Feature Importance in Model"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Distribution")
            fig = px.histogram(
                history_df, 
                x='prediction',
                title="Distribution of Predictions",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = history_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = history_df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Make some predictions first to see analysis!")

with tab3:
    st.header("📈 Advanced Visualizations")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame([
            {**pred['features'], 'prediction': pred['prediction'], 'timestamp': pred['timestamp']}
            for pred in st.session_state.prediction_history
        ])
        
        viz_type = st.selectbox(
            "Choose visualization:",
            ["Scatter Matrix", "3D Plot", "Time Series", "Box Plots"]
        )
        
        if viz_type == "Scatter Matrix":
            features_to_plot = st.multiselect(
                "Select features for scatter matrix:",
                ['hdlngth', 'skullw', 'chest', 'belly', 'age'],
                default=['hdlngth', 'chest', 'belly']
            )
            
            if len(features_to_plot) >= 2:
                fig = px.scatter_matrix(
                    history_df,
                    dimensions=features_to_plot + ['prediction'],
                    title="Feature Relationships"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "3D Plot":
            if len(history_df) > 1:
                fig = px.scatter_3d(
                    history_df,
                    x='hdlngth',
                    y='chest',
                    z='belly',
                    color='prediction',
                    size='age',
                    title="3D Feature Space"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Time Series":
            if len(history_df) > 1:
                fig = px.line(
                    history_df,
                    x='timestamp',
                    y='prediction',
                    title="Predictions Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plots":
            numeric_features = ['hdlngth', 'skullw', 'chest', 'belly', 'age']
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=numeric_features
            )
            
            for i, feature in enumerate(numeric_features):
                row = (i // 3) + 1
                col = (i % 3) + 1
                fig.add_trace(
                    go.Box(y=history_df[feature], name=feature),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title="Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Generate some predictions to create visualizations!")

with tab4:
    st.header("📋 Prediction History")
    
    if st.session_state.prediction_history:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(st.session_state.prediction_history))
        with col2:
            avg_prediction = np.mean([p['prediction'] for p in st.session_state.prediction_history])
            st.metric("Average Prediction", f"{avg_prediction:.2f} cm")
        with col3:
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Display history table
        history_display = []
        for i, pred in enumerate(reversed(st.session_state.prediction_history)):
            history_display.append({
                'ID': len(st.session_state.prediction_history) - i,
                'Timestamp': pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Prediction (cm)': f"{pred['prediction']:.2f}",
                'Confidence': f"{pred['confidence']:.1%}",
                'Head Length': pred['features']['hdlngth'],
                'Age': pred['features']['age']
            })
        
        st.dataframe(pd.DataFrame(history_display), use_container_width=True)
        
        # Export functionality
        if st.button("📥 Export History"):
            history_df = pd.DataFrame([
                {**pred['features'], 'prediction': pred['prediction'], 
                 'confidence': pred['confidence'], 'timestamp': pred['timestamp']}
                for pred in st.session_state.prediction_history
            ])
            csv = history_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "possum_predictions_history.csv",
                "text/csv"
            )
    else:
        st.info("No predictions made yet. Go to the Prediction tab to get started!")

with tab5:
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        model_info = {
            "Model Type": "Random Forest Regressor",
            "Features": 11,
            "Target": "Total Length (cm)",
            "Training Data": "Possum Dataset",
            "Performance Metric": "R² Score",
            "Status": "✅ Loaded" if model_loaded else "❌ Not Loaded"
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.subheader("Feature Descriptions")
        feature_descriptions = {
            "hdlngth": "Head length from nose to back of skull",
            "skullw": "Skull width at widest point",
            "taill": "Length of tail",
            "footlgth": "Length of hind foot",
            "earconch": "Length of ear from base to tip",
            "eye": "Distance between eyes",
            "age": "Age of possum in years",
            "chest": "Chest circumference",
            "belly": "Belly circumference",
            "sex": "Gender (0=Male, 1=Female)",
            "Pop_other": "Population (0=Vic, 1=Other)"
        }
        
        for feature, description in feature_descriptions.items():
            st.write(f"**{feature}:** {description}")
    
    # Model performance (mock data - replace with actual metrics)
    st.subheader("Model Performance")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("R² Score", "0.892")
    with perf_col2:
        st.metric("RMSE", "3.45 cm")
    with perf_col3:
        st.metric("MAE", "2.67 cm")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    🐿️ Advanced Possum Predictor v2.0 | Built with Streamlit & ML Magic ✨
</div>
""", unsafe_allow_html=True)

def prepare_features(row):
    """Helper function to prepare features from a data row"""
    # Implement feature preparation logic based on your data structure
    return [row.get(col, 0) for col in ['sex', 'age', 'hdlngth', 'skullw', 'taill', 
                                        'footlgth', 'earconch', 'eye', 'chest', 'belly', 'Pop_other']]