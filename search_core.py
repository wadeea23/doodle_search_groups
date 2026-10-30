import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

def preprocess_query(image, background="white"):

    # Handle transparent images
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, background)
        image = Image.alpha_composite(bg, image)

    # Grayscale
    gray = image.convert("L")

    # White strokes on black background
    if np.array(gray).mean() > 127:
        gray = ImageOps.invert(gray)

    # Thresholding
    gray = np.where(np.array(gray) > 150, 255, 0).astype("uint8")
    gray = Image.fromarray(gray)

    # Check for empty image
    if gray.getbbox() is None:
        return None

    # Crop
    gray = gray.crop(gray.getbbox())

    # Add margin
    margin = max(2, int(max(gray.size) * 0.1))
    gray = ImageOps.expand(gray, border=margin, fill=0)

    # Center, pad and resize
    return ImageOps.pad(gray, (28, 28), color=0)




def to_model_input(images_28):
    
    images = np.array(images_28).reshape(-1, 28, 28, 1)

    images = tf.image.resize(images, (224, 224))
    images = tf.image.grayscale_to_rgb(images)
    images = preprocess_input(images.numpy().astype("float32"))
    
    return images




def embed(model, images_28):
    
    embeddings = []

    for i in range(0, len(images_28), 32):
        images = to_model_input(images_28[i:i+32])
        features = model.predict(images, verbose=0)
        embeddings.append(features)
    
    embeddings = np.concatenate(embeddings)
    embeddings = normalize(embeddings, norm="l2")
    
    return embeddings




def search(query_embedding, gallery_embeddings, k=25):
    
    neighbors = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric="cosine"
    )

    neighbors.fit(gallery_embeddings)
    
    distances, indices = neighbors.kneighbors(query_embedding)
    
    similarities = 1 - distances
    
    return indices, similarities