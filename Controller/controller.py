# controller/controller.py
from model.data_loader import DataLoader
from model.model import RetinopathyModel
from sklearn.model_selection import train_test_split

# RetinopathyController class handles the workflow of loading data, training the model, and saving results
class RetinopathyController:
    def __init__(self, base_image_dir):
        self.data_loader = DataLoader(base_image_dir)  # Initialize DataLoader with the base image directory
        self.model = RetinopathyModel()  # Initialize RetinopathyModel (CNN)
    
    # Prepare the data: load labels, split into training and validation sets
    def prepare_data(self):
        labels_df = self.data_loader.load_labels()  # Load the labels and image paths
        patient_level_df = labels_df[['PatientId', 'level']].drop_duplicates()  # Remove duplicate patients
        
        # Split patient IDs into training and validation sets
        train_ids, valid_ids = train_test_split(
          #optional #loaclly make a database radom id generational 
            patient_level_df['PatientId'], 
            test_size=0.2,  # Use 20% of data for validation
            stratify=patient_level_df['level'],  # Stratify by DR level
            random_state=42  # Set random seed for reproducibility
        )

        train_df = labels_df[labels_df['PatientId'].isin(train_ids)]  # Filter training data
        valid_df = labels_df[labels_df['PatientId'].isin(valid_ids)]  # Filter validation data

        return train_df, valid_df  # Return the training and validation DataFrames

    # Train the CNN model using training and validation data
    def train_model(self, train_df, valid_df):
        train_gen = self.data_loader.image_loader(train_df, batch_size=32)  # Create training data generator
        valid_gen = self.data_loader.image_loader(valid_df, batch_size=32, augment=False)  # Create validation data generator

        steps_per_epoch = len(train_df) // 32  # Calculate steps per epoch for training
        validation_steps = len(valid_df) // 32  # Calculate validation steps

        # Train the model
        #print 
        history = self.model.train(train_gen, valid_gen, steps_per_epoch, validation_steps, epochs=10)
        return history  # Return training history

    # Save the trained model to a file
    def save_model(self, file_path):
        self.model.save(file_path)  # Save the trained model to the specified path
