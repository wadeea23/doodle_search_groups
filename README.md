# DoodleDex
## Doodle Image Retrieval System
### Project Overview

DoodleDex is an image retrieval system that finds drawings similar to a hand-drawn sketch. It uses deep learning to extract image features and compare drawings to find the closest matches.
The project uses MobileNetV2 for feature extraction and cosine similarity with nearest-neighbor search to retrieve similar images. Unlike image classification, the goal is not to predict the category of a drawing, but to find other drawings that look similar to it.

### App: `https://doodlesearchgroups.streamlit.app/`

### Dataset
The project uses Google's Quick, Draw dataset and includes 15 categories of fruits and vegetables:

- Apple
- Banana
- Strawberry
- Watermelon
- Blackberry
- Blueberry
- Grapes
- Pear
- Pineapple
- Carrot
- Peas
- Broccoli
- Onion
- Potato
- Peanut

The dataset contains 7,500 images, with 500 images selected from each category. From these, 450 images per category are used in the gallery, while the remaining 50 are used for query testing.
This gives us 6,750 gallery images and 750 query images in total.

### Technologies Used
The project was developed using Python and Google Colab. The main libraries and tools include TensorFlow, Keras, MobileNetV2, NumPy, Pandas, Matplotlib, Scikit-learn, and Pillow (PIL).

### How It Works
The process starts by collecting and preparing the doodle images. Each image is cleaned, cropped, centered, and resized before being passed to the feature extraction model.
MobileNetV2 extracts visual features from the images and converts them into numerical representations called embeddings. These embeddings are then compared using cosine distance and nearest-neighbor search to find the most similar drawings.
Finally, the system displays the input sketch alongside the top 25 most similar images from the gallery, including similarity scores.

### Results
DoodleDex retrieves the top 25 visually similar drawings for each query image. The results allow users to compare the original sketch with the retrieved images and see how closely they match.
The system focuses on visual similarity rather than predicting image categories.

### Installation and Setup
The project runs in Google Colab. To get started, open the project notebook, prepare the dataset, and run the cells in order.
The notebook handles image preprocessing, feature extraction, embedding generation, and similarity search. Once these steps are complete, users can test the system with query images and view the retrieval results.

### Future Improvements
There are several ways to improve DoodleDex in the future. These include developing a Streamlit web application, adding more doodle categories, allowing users to draw sketches directly in the application, and improving retrieval speed using FAISS.
Another possible improvement is to experiment with CLIP-based embeddings and explore ways to make the similarity ranking more accurate.

### Authors
Wadeea Almutawaa

Noor Taqi

Fatema Mohammed

Data Science and Machine Learning Project

