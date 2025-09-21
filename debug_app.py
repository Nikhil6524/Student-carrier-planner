import streamlit as st

# Simple debug version to test deployment
st.set_page_config(page_title="Debug Career Agent AI", layout="wide")
st.title("🚀 Career Agent AI: Debug Version")
st.markdown("This is a debug version to troubleshoot Streamlit Cloud deployment.")

# Display environment information
st.subheader("Environment Information")
st.write(f"Streamlit version: {st.__version__}")

# Check if secrets are accessible
st.subheader("API Configuration Check")
try:
    if "GEMINI_API_KEY" in st.secrets:
        st.success("✅ Gemini API key is configured in secrets")
    else:
        st.warning("⚠️ Gemini API key not found in secrets")
except Exception as e:
    st.error(f"Error accessing secrets: {str(e)}")
    
# Basic functionality test
st.subheader("Basic Functionality Test")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! The application is working correctly.")

st.success("Debug app loaded successfully! If you see this message, the basic Streamlit functionality is working.")