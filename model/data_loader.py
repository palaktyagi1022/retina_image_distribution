# model/data_loader.py
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

# DataLoader class responsible for loading, preprocessing, and augmenting images
class DataLoader:
    def __init__(self, base_image_dir):
        self.base_image_dir = base_image_dir  # Base directory where images are stored
    
    # Load the CSV containing image paths and corresponding DR levels
    def load_labels(self):
        labels_df = pd.read_csv(os.path.join(self.base_image_dir, 'trainLabels.csv'))  # Load CSV
        labels_df['PatientId'] = labels_df['image'].apply(lambda x: x.split('_')[0])  # Extract patient ID from image name
        labels_df['image_path'] = labels_df['image'].apply(lambda x: os.path.join(self.base_image_dir, f'{x}.jpeg'))  # Create full image path
        labels_df['exists'] = labels_df['image_path'].apply(os.path.exists)  # Check if image file exists
        labels_df['eye'] = labels_df['image'].apply(lambda x: 1 if 'left' in x else 0)  # Add left or right eye label
        labels_df['level_cat'] = labels_df['level'].apply(lambda x: tf.keras.utils.to_categorical(x, 5))  # Convert DR level to one-hot encoding
        return labels_df[labels_df['exists']]  # Return rows where images exist
    
    # Augment and preprocess the image: resizing, flipping, and brightness adjustment
    def augment_image(self, image_path, target_size=(512, 512), horizontal_flip=True, random_brightness=True):
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)  # Read and decode the image
        image = tf.image.resize(image, target_size)  # Resize image to target size
        if horizontal_flip:
            image = tf.image.random_flip_left_right(image)  # Randomly flip image horizontally
        if random_brightness:
            image = tf.image.random_brightness(image, max_delta=0.2)  # Randomly adjust brightness
        return preprocess_input(image)  # Preprocess for InceptionV3

    # Generator that loads and augments images in batches for training
    def image_loader(self, df, batch_size=32, augment=True):
        data_size = df.shape[0]  # Get total number of images
        while True:  # Infinite loop to keep yielding data for training
            for start in range(0, data_size, batch_size):
                batch_df = df[start:start + batch_size]  # Select batch of data
                images = [self.augment_image(image_path) for image_path in batch_df['image_path']]  # Augment and load images
                labels = batch_df['level_cat'].tolist()  # Load corresponding labels
                yield np.array(images), np.array(labels)  # Yield images and labels as NumPy arrays
