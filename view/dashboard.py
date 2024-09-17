# view/dashboard.py
import matplotlib.pyplot as plt

# Dashboard class responsible for displaying the training results visually
class Dashboard:
    # Method to plot training and validation accuracy over epochs
    @staticmethod
    def show_training_results(history):
        plt.plot(history.history['accuracy'], label='accuracy')  # Plot training accuracy
        plt.plot(history.history['val_accuracy'], label='val_accuracy')  # Plot validation accuracy
        plt.xlabel('Epoch')  # Label for x-axis
        plt.ylabel('Accuracy')  # Label for y-axis
        plt.legend()  # Add legend to the plot
        plt.show()  # Display the plot
        
        
        
        
        
        
        
