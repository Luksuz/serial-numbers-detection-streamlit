import streamlit as st
import os
import tempfile
from CharactersDetector import  CharactersDetector
from RegionDetector import RegionDetector
from PIL import Image


# Set page configuration
st.set_page_config(page_title="Container OCR", layout="wide")

# App Title
st.title("📦 Container Serial Region and Character Detection")
st.write("Upload an image to detect the container's serial region, crop it, and extract characters.")

# Sidebar for settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    # Instantiate models with spinners
    with st.spinner("Initializing Models..."):
        region_model = RegionDetector()
        character_model = CharactersDetector()

    # Button to trigger detection
    if st.button("🔍 Detect Region and Extract Characters"):
        with st.spinner("Detecting Serial Region..."):
            detected_region, cropped_images = region_model.get_serial_region(img)
        
        if detected_region:
            # Display detected serial region
            st.image(detected_region, caption="✅ Detected Serial Region", use_column_width=True)

            # Save the detected region to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                detected_region.save(temp.name)
                temp_path = temp.name

            try:
                # Run character detection on the temporary file
                characters, _ = character_model.sort_and_read_characters(temp_path, conf=conf_threshold, show=True)
                
                # Display detected characters
                st.success(f"✅ Characters Detected: {characters}")
            except Exception as e:
                st.error(f"Error during character detection: {e}")
            finally:
                # Delete the temporary file
                os.remove(temp_path)
        else:
            st.warning("⚠️ No serial region detected.")
