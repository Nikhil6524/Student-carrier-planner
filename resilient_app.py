import streamlit as st
import os

# Set page configuration
st.set_page_config(page_title="Career Agent AI", layout="wide")
st.title("🚀 AI Career Agent: Your Dynamic Mission Control")
st.markdown("**From resume to readiness.** Your comprehensive AI-powered career advancement platform.")

# Initialize session state for required packages
if 'dependencies_loaded' not in st.session_state:
    st.session_state.dependencies_loaded = False
    st.session_state.missing_packages = []

# Try to import required packages with error handling
try:
    import pandas as pd
    import numpy as np
    import requests
    from bs4 import BeautifulSoup
    
    try:
        import google.generativeai as genai
    except ImportError:
        st.session_state.missing_packages.append("google-generativeai")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        st.session_state.missing_packages.append("scikit-learn")
    
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        st.session_state.missing_packages.append("PyPDF2")
    
    # If we've gotten this far, mark dependencies as loaded
    st.session_state.dependencies_loaded = True
    
except ImportError as e:
    st.error(f"Failed to import required packages: {str(e)}")
    st.session_state.missing_packages.append(str(e).split("'")[1])

# Check for missing packages and display warning
if st.session_state.missing_packages:
    st.warning(f"⚠️ Some packages are missing: {', '.join(st.session_state.missing_packages)}. Some features may not work.")

# --- API Configuration ---
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    if GEMINI_API_KEY and "google-generativeai" not in st.session_state.missing_packages:
        genai.configure(api_key=GEMINI_API_KEY)
        st.success("✅ Successfully configured Gemini API")
    else:
        st.warning("⚠️ Gemini API key not found in secrets or google-generativeai package is missing")
except Exception as e:
    st.error(f"Error configuring API: {str(e)}")

# --- Navigation Tabs ---
tabs = ["🎯 Career Analysis", "💰 Salary Insights", "📝 Cover Letter AI", "📊 Debug Info"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# === TAB 1: CAREER ANALYSIS ===
with tab1:
    st.header("Career Analysis")
    st.info("Upload your resume and explore job matches with AI-powered analysis.")
    
    # Simplified Resume Upload
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    
    if uploaded_file:
        st.success("Resume uploaded successfully!")
        st.info("This is a simplified version for debugging. Resume analysis functionality will be available soon.")

# === TAB 2: SALARY INSIGHTS ===
with tab2:
    st.header("Salary Insights")
    st.info("Explore salary trends and compensation insights.")
    st.write("This feature is under construction.")

# === TAB 3: COVER LETTER AI ===
with tab3:
    st.header("Cover Letter AI")
    st.info("Generate personalized cover letters with AI.")
    st.write("This feature is under construction.")

# === TAB 4: DEBUG INFO ===
with tab4:
    st.header("Debug Information")
    
    # Display environment info
    st.subheader("Environment Information")
    st.write(f"Streamlit version: {st.__version__}")
    
    # Display package status
    st.subheader("Package Status")
    packages = {
        "streamlit": "Installed",
        "pandas": "Installed" if "pandas" not in st.session_state.missing_packages else "Missing",
        "numpy": "Installed" if "numpy" not in st.session_state.missing_packages else "Missing",
        "requests": "Installed" if "requests" not in st.session_state.missing_packages else "Missing", 
        "beautifulsoup4": "Installed" if "bs4" not in st.session_state.missing_packages else "Missing",
        "google-generativeai": "Installed" if "google-generativeai" not in st.session_state.missing_packages else "Missing",
        "scikit-learn": "Installed" if "scikit-learn" not in st.session_state.missing_packages else "Missing",
        "PyPDF2": "Installed" if "PyPDF2" not in st.session_state.missing_packages else "Missing"
    }
    
    for pkg, status in packages.items():
        st.write(f"- **{pkg}**: {status}")
    
    # API Configuration Status
    st.subheader("API Configuration")
    if "GEMINI_API_KEY" in st.secrets:
        st.success("✅ Gemini API key is configured in secrets")
    else:
        st.warning("⚠️ Gemini API key not found in secrets")