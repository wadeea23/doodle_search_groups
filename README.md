# doodle_search_groups

DoodleDex
Doodle Image Retrieval System

DoodleDex is an image retrieval system designed to find visually similar hand-drawn doodles. It uses deep learning to extract visual features from sketches and retrieve the closest matches from a collection of drawings.

The project uses MobileNetV2 for feature extraction and cosine similarity with nearest-neighbor search to identify visually similar images.

Unlike traditional image classification, DoodleDex focuses on retrieving similar drawings rather than predicting their categories.

Project Overview

The system takes a doodle as input, processes the image, extracts its visual features, and compares them with a gallery of doodles. It then returns the most visually similar results.

The main goal is to make searching for similar hand-drawn sketches simple and efficient.

Dataset

DoodleDex uses Google's Quick, Draw! dataset and includes 15 categories:

Apple
Banana
Strawberry
Watermelon
Blackberry
Blueberry
Grapes
Pear
Pineapple
Carrot
Peas
Broccoli
Onion
Potato
Peanut

A total of 500 images are selected from each category:

450 images for the gallery.
50 images for query testing.

The dataset contains 7,500 doodles in total, divided into 6,750 gallery images and 750 query images.

Technologies Used
Python
TensorFlow / Keras
MobileNetV2
NumPy
Pandas
Matplotlib
Scikit-learn
Pillow
Google Colab
How It Works

The retrieval process consists of the following steps:

Data Collection: Downloading and preparing doodle images from the Quick, Draw! dataset.
Image Preprocessing: Cleaning, cropping, centering, and resizing sketches.
**Feature
