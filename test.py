import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
import RRDBNet_arch as arch

# Set up Streamlit page
st.set_page_config(
    page_title="Image Super-Resolution with ESRGAN and PSNR",
    page_icon="🌟",
    layout="centered",
)

# Ensure directories exist
os.makedirs("LR", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load ESRGAN or PSNR model
@st.cache_resource
def load_model(model_name):
    model_path = f'models/{model_name}.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)
    return model, device

# Image super-resolution function
def super_resolve_image(model, device, input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(output_path, output)

# Streamlit UI
st.title("✨ Image Super-Resolution with ESRGAN and PSNR 🌟")
st.markdown("""
Upload a low-resolution image and upscale it using the ESRGAN or PSNR-oriented model.  
Supports formats like PNG, JPG, and BMP.
""")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    options=["ESRGAN", "PSNR-oriented"],
    index=0,
    help="Select the model you want to use for image super-resolution.",
)
model_map = {
    "ESRGAN": "RRDB_ESRGAN_x4",
    "PSNR-oriented": "RRDB_PSNR_x4"
}

uploaded_file = st.file_uploader("Upload a Low-Resolution Image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # Save uploaded image
    input_path = os.path.join("LR", uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.image(input_path, caption="Uploaded Image", use_container_width=True)
    
    # Load model
    st.write(f"Loading {model_choice} model...")
    try:
        model, device = load_model(model_map[model_choice])
        st.success(f"{model_choice} model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading {model_choice} model: {e}")
        st.stop()
    
    # Perform super-resolution
    if st.button("Upscale Image"):
        output_path = os.path.join("results", f"output_{uploaded_file.name}")
        with st.spinner("Upscaling image... This may take a moment."):
            try:
                super_resolve_image(model, device, input_path, output_path)
                st.success("Upscaling complete!")
                
                # Display and download the output image
                final_image = Image.open(output_path)
                st.image(final_image, caption="Upscaled Image", use_container_width=True)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Upscaled Image",
                        data=file,
                        file_name=f"upscaled_{uploaded_file.name}",
                        mime="image/png",
                    )
            except Exception as e:
                st.error(f"Error during upscaling: {e}")
else:
    st.info("Please upload a low-resolution image to get started.")

# Footer
st.markdown("<br><hr><center>Made with ❤️ using Streamlit and ESRGAN</center><hr>", unsafe_allow_html=True)
