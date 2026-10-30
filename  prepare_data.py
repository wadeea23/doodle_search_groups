import os
import numpy as np
import urllib.request
import urllib.parse
import matplotlib.pyplot as plt


# Imports
import os
import numpy as np
import urllib.request
import urllib.parse
import matplotlib.pyplot as plt

# QuickDraw categories
urls = [
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/apple.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/banana.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/strawberry.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/watermelon.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/blackberry.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/blueberry.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/grapes.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/pear.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/pineapple.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/carrot.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/peas.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/broccoli.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/onion.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/potato.npy",
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/peanut.npy"
]

# Download QuickDraw dataset
os.makedirs("./data", exist_ok=True)

for url in urls:
    name = urllib.parse.urlsplit(url).path.split("/")[-1]
    urllib.request.urlretrieve(url, f"./data/{name}")
    print(f"{name} downloaded")





# Sample 500 images from each category
rng = np.random.default_rng(42)
samples = {}

for url in urls:
    name = urllib.parse.urlsplit(url).path.split("/")[-1]

    data = np.load(f"./data/{name}", mmap_mode="r")
    print(name, data.shape)

    idx = np.sort(rng.choice(len(data), size=500, replace=False))
    sample = data[idx]
    print(sample.shape)

    samples[name.replace(".npy", "")] = sample



# Split into Gallery and Query
rng = np.random.default_rng(42)

gallery_images = []
query_images = []

for images in samples.values():
    indices = rng.permutation(len(images))

    gallery_images.extend(images[indices[:450]])
    query_images.extend(images[indices[450:]])

gallery_images = np.array(gallery_images)
query_images = np.array(query_images)

print("Gallery:", gallery_images.shape)
print("Query:", query_images.shape)



# Save dataset
np.savez_compressed(
    "./data/subset.npz",
    gallery_images=gallery_images,
    query_images=query_images
)

data = np.load("./data/subset.npz")

print("Gallery:", data["gallery_images"].shape)
print("Query:", data["query_images"].shape)



# Display 25 gallery images
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(gallery_images[i].reshape(28, 28), cmap="gray")
    ax.axis("off")

plt.show()