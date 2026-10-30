# DoodleDex

## Doodle Image Retrieval System

### Project Overview
DoodleDex is an image retrieval system that finds drawings similar to a hand-drawn sketch. It uses deep learning to extract image features and compare drawings to find the closest matches.
The project uses MobileNetV2 for feature extraction and cosine similarity with nearest-neighbor search to retrieve similar images. Unlike image classification, the goal is not to predict the category of a drawing, but to find other drawings that look visually similar to it.

### Streamlit Application
DoodleDex also includes a Streamlit web application that allows users to interact with the image retrieval system.
Users can upload an image or take a picture directly using their camera. The system processes the input image, searches the gallery, and displays the top 25 most visually similar drawings along with their similarity scores.
**App:** https://doodlesearchgroups.streamlit.app/

### Dataset
The project uses Google's Quick, Draw! dataset and includes 15 categories of fruits and vegetables:

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
The project was developed using Python and Google Colab. The main libraries and tools include:

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- Streamlit

### How It Works

The process starts by collecting and preparing the doodle images. Each image is cleaned, cropped, centered, and resized before being passed to the feature extraction model.
MobileNetV2 extracts visual features from the images and converts them into numerical representations called embeddings. These embeddings are then compared using cosine similarity and nearest-neighbor search to find the most similar drawings.
Users can upload an image or take a picture using the Streamlit application. The system processes the image, compares it with the gallery, and displays the top 25 most similar drawings with their similarity scores.

### Results
DoodleDex retrieves the top 25 visually similar drawings for each query image.
The Streamlit application makes the system easy to use by allowing users to upload or capture their own sketches and view the closest matching drawings.
The system focuses on visual similarity rather than predicting image categories.

### Installation and Setup

The project was developed in Google Colab and deployed as a Streamlit web application.
To explore the project, open the project notebook in Google Colab and run the cells in order. The notebook handles image preprocessing, feature extraction, embedding generation, and similarity search.
Users can also access the deployed Streamlit application directly through the app link and test the image retrieval system.

### Future Improvements
There are several ways to improve DoodleDex in the future:

- Adding more doodle categories.
- Improving retrieval speed using FAISS.
- Experimenting with CLIP-based embeddings.
- Improving the similarity ranking accuracy.
- Enhancing the drawing experience within the application.

### Authors

Wadeea Almutawaa

Noor Taqi

Fatema Mohammed

**Data Science and Machine Learning Project**
