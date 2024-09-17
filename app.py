# app.py
from controller.controller import RetinopathyController
from view.dashboard import Dashboard

# Main function to run the diabetic retinopathy detection program
def main():
    base_image_dir = '../input/retinopathy-detection'  # Set base directory for images
    controller = RetinopathyController(base_image_dir)  # Initialize the controller

    # Prepare the data
    train_df, valid_df = controller.prepare_data()  # Load and split data into train/validation sets

    # Train the model
    history = controller.train_model(train_df, valid_df)  # Train the model and get training history

    # Display the results
    Dashboard.show_training_results(history)  # Plot the training/validation accuracy

    # Save the trained model
    controller.save_model('../models/retinopathy_model.h5')  # Save the trained model

# Run the program if the script is executed
if __name__ == '__main__':
    main()  # Call the main function
