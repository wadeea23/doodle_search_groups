import os
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from search_core import embed



# Load pretrained MobileNetV2
backbone = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

backbone.trainable = False


# Load gallery images
data = np.load("./data/subset.npz")

gallery_images = data["gallery_images"]

print("Gallery:", gallery_images.shape)


# Extract gallery embeddings
gallery_embeddings = embed(backbone, gallery_images)

print("Embeddings:", gallery_embeddings.shape)


# Save gallery embeddings and images
os.makedirs("./index", exist_ok=True)

np.save("./index/embeddings.npy", gallery_embeddings)
np.save("./index/gallery_images.npy", gallery_images)

print("Index saved successfully!")


