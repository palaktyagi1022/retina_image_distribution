#Diabetic Retinopathy Detection

This project aims to create an automated analysis system to detect the presence of diabetic retinopathy (DR) in high-resolution retina images. The system assigns a DR score ranging from 0 to 4 based on a provided scale.

Problem Description

Diabetic retinopathy is a condition where high blood sugar levels cause damage to blood vessels in the retina. Early detection through retinal images is crucial for effective treatment. This project utilizes deep learning techniques to classify retina images into five categories based on the severity of diabetic retinopathy:

0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR
Prerequisites

Ensure you have Python and the following libraries installed:

bash
Copy code
pip install tensorflow keras opencv-python scikit-learn pandas numpy matplotlib
Steps to Follow

Data Preparation
Place all retina images in a folder named dataset.
Each image should be named with a pattern like <id>_left.jpeg or <id>_right.jpeg where <id> is the patient identifier.
Image Preprocessing
Resize all images to 224x224 pixels.
Normalize pixel values to range [0, 1].
Apply data augmentation (rotation, flipping, etc.) to improve model generalization.
Model Creation
Build a Convolutional Neural Network (CNN) model using Keras.
Compile the model with the Adam optimizer and categorical cross-entropy loss.
Model Training
Split the dataset into training and validation sets (80-20% split).
Train the model for 25 epochs or until satisfactory accuracy is achieved.
Use data augmentation to improve model performance.
Evaluation and Fine-Tuning
Evaluate the model on the validation set.
Fine-tune hyperparameters, network architecture, and data preprocessing techniques to optimize performance.
Save the Model
Save the trained model to a file (dr_classification_model.h5).
