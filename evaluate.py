import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from sklearn.preprocessing import normalize
from search_core import to_model_input, search

# Load saved data
data = np.load("./data/subset.npz")

query_images = data["query_images"]
gallery_images = data["gallery_images"]
gallery_embeddings = np.load("./index/embeddings.npy")



# Load pretrained model
backbone = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

backbone.trainable = False



# Prepare query image
query = query_images[0]
query = to_model_input(query)

# Extract query embedding
query_embedding = backbone.predict(query, verbose=0)
query_embedding = normalize(query_embedding, norm="l2")

print(query_embedding.shape)




# Find top 25 similar images
indices, similarities = search(
    query_embedding,
    gallery_embeddings,
    k=25
)

print(indices.shape)
print(similarities.shape)



# Display query and top 25 similar images
fig, axes = plt.subplots(6, 5, figsize=(12, 12))

axes[0, 0].imshow(query_images[0].reshape(28, 28), cmap="gray")
axes[0, 0].set_title("Query Image")

for i, idx in enumerate(indices[0]):
    ax = axes.flat[i + 1]
    ax.imshow(gallery_images[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Similarity: {similarities[0][i]:.2f}", fontsize=9)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()