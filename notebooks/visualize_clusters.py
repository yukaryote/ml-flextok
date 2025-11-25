"""
Interactive web visualization for FlexTok first token clusters.

Run with: streamlit run visualize_clusters.py
"""

import streamlit as st
import torch
import sys
import os
import argparse
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="FlexTok First Token Explorer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make everything more compact
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        font-size: 1.3rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stMarkdown p {
        margin-bottom: 0.5rem;
    }
    [data-testid="stImage"] {
        margin-bottom: 0.2rem;
    }
    hr {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# FSQ levels configuration
def parse_args(args):
    parser = argparse.ArgumentParser('Data Diagnostics')
    parser.add_argument('--fsq', default=888555, help='CSV file', required=True)
    parser.add_argument('--output_dir', default="/home/iyu/flextok_first_token_samples/", help='Output directory', required=False)
    return parser.parse_args(args)

args = parse_args(sys.argv[1:])
FSQ_LEVELS = [int(x) for x in str(args.fsq)]
IMG_OUTPUT_DIR = args.output_dir

def get_quant_values(level):
    """Get quantization values for a given FSQ level"""
    return torch.linspace(-1, 1, steps=level).tolist()

# Pre-compute all quantization values
quant_values_per_level = [get_quant_values(level) for level in FSQ_LEVELS]

def get_image_paths(quant_combo):
    """Get paths for 9 sample images for a given quantization combination"""
    quant_str = "_".join([str(float(v)) for v in quant_combo])
    paths = []
    for sample_num in range(1, 10):
        img_path = os.path.join(IMG_OUTPUT_DIR, f"quant_{quant_str}_sample_{sample_num}.png")
        paths.append(img_path)
    return paths

# Title and description
st.title("🎨 FlexTok First Token Explorer")
st.markdown(f"Explore 64,000 possible first tokens. FSQ levels: **{FSQ_LEVELS}**")

# Create sliders in a sidebar
st.sidebar.header("FSQ Controls")

slider_values = []
quant_combo = []

for i, level in enumerate(FSQ_LEVELS):
    quant_vals = quant_values_per_level[i]

    # Create slider
    slider_val = st.sidebar.slider(
        f"Dimension {i}",
        min_value=0,
        max_value=level - 1,
        value=level // 2,  # Start at middle value
        step=1,
        help=f"Range: {quant_vals[0]:.2f} to {quant_vals[-1]:.2f}"
    )

    slider_values.append(slider_val)
    quant_combo.append(quant_vals[slider_val])

    # Display the actual quantization value (more compact)
    st.sidebar.text(f"  → {quant_vals[slider_val]:.2f}")

st.sidebar.markdown("---")
st.sidebar.code(f"[{', '.join([f'{v:.2f}' for v in quant_combo])}]")

# Get image paths
img_paths = get_image_paths(quant_combo)

# Display images in a 3x3 grid (more compact)
cols_per_row = 3
for row in range(3):
    cols = st.columns(cols_per_row, gap="small")
    for col_idx in range(cols_per_row):
        img_idx = row * cols_per_row + col_idx
        img_path = img_paths[img_idx]

        with cols[col_idx]:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            else:
                st.error(f"Not found", icon="⚠️")
