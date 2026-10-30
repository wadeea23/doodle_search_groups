import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from streamlit_drawable_canvas import st_canvas
from search_core import preprocess_query, to_model_input, search


# Load model and gallery
@st.cache_resource
def load_model():
    gallery_images = np.load("./index/gallery_images.npy")
    gallery_embeddings = np.load("./index/embeddings.npy")

    backbone = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    backbone.trainable = False

    return gallery_images, gallery_embeddings, backbone


gallery_images, gallery_embeddings, backbone = load_model()


# Website
st.title("DoodleDex Draw 🎨🖼️")

option = st.radio(
    "Choose how to provide your doodle:",
    ["Draw Image", "Upload Image", "Camera Image"],
    horizontal=True
)

image = None

# Option 1: Draw an image
if option == "Draw Image":
    
    canvas_result = st_canvas(
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        drawing_mode="freedraw",
        width=700,
        height=500,
        return_image_data=True,
        key="canvas"
    )
    if canvas_result.image_data is not None:
        canvas_image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        ).convert("RGBA")

        black_background = Image.new(
            "RGBA", canvas_image.size, (0, 0, 0, 255)
        )

        image = Image.alpha_composite(
            black_background, canvas_image
        ).convert("RGB")

        # Avoid searching with an empty drawing
        if np.array(image.convert("L")).max() < 10:
            image = None

# Option 2: Upload an image
elif option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload your doodle",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_file = st.camera_input("Take a picture of your doodle")
            
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")


# Search for similar images
if image is not None:

    st.image(image, caption="Query Image", width=200)

    if st.button("Find Similar Images"):

        # Preprocessing
        query = preprocess_query(image)
        
        if query is None:
            st.warning("Please draw or upload a doodle.")
            st.stop()

        st.image(query, caption="Processed Query (28×28)", width=140)
        
        query = to_model_input(query)

        # Extract features
        query_embedding = backbone.predict(query, verbose=0)

        indices, similarities = search(
            query_embedding,
            gallery_embeddings,
            k=25
        )

        # Display top 25
        st.subheader("Top 25 Similar Images")

        for row in range(5):
            cols = st.columns(5)

            for col in range(5):
                i = row * 5 + col
                idx = indices[0][i]

                with cols[col]:
                    st.image(
                        gallery_images[idx].reshape(28, 28),
                        width=120
                    )
                    st.caption(f"Similarity: {similarities[0][i]:.2f}")