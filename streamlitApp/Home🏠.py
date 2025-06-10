
import streamlit as st
from PIL import Image

# Main Page Configuration
st.set_page_config(
    page_title="AP-Daten Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("AP-Daten Viewer")


    # Main Description
st.markdown(
        """
            This app visualizes samples from the FabriCAD dataset. It displays the plans, geometries, and metadata for each sample.  
            It shows work plan data and the associated 3D models. Additionally, manufacturing information at the feature level can be viewed.

            ### Features of the app:
            - Preview data 🎥: Linked to the directory where preview data is located. Please check if the archive is unpacked correctly.


            The app serves both as a presentation tool to demonstrate results  
            and as a useful tool for debugging and evaluating the generated data.
        """
    )
