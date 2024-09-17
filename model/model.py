# model/model.py
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# RetinopathyModel class defines the CNN model architecture and its training logic
class RetinopathyModel:
    def __init__(self, input_shape=(512, 512, 3), num_classes=5):
        self.input_shape = input_shape  # Shape of input images (512x512x3)
        self.num_classes = num_classes  # Number of classes for classification (5 DR levels)
        self.model = self.build_model()  # Build the CNN model
    
    # Build the CNN model using InceptionV3 as the base model
    def build_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)  # Load pre-trained InceptionV3 model without top layer
        x = base_model.output  # Get the output of the base model
        x = GlobalAveragePooling2D()(x)  # Apply global average pooling
        x = Dense(256, activation='relu')(x)  # Add a dense layer with 256 units and ReLU activation
        output = Dense(self.num_classes, activation='softmax')(x)  # Add output layer with softmax activation for classification
        model = Model(inputs=base_model.input, outputs=output)  # Create the full model
        
        # Freeze the layers of the base model so they are not trained
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model  # Return the compiled model
    
    # Method to train the model
    def train(self, train_gen, valid_gen, steps_per_epoch, validation_steps, epochs=10):
        history = self.model.fit(
            train_gen,  # Training data generator
            steps_per_epoch=steps_per_epoch,  # Number of steps (batches) per epoch
            validation_data=valid_gen,  # Validation data generator
            validation_steps=validation_steps,  # Number of steps (batches) for validation
            epochs=epochs  # Number of epochs to train
        )
        return history  # Return the training history (loss, accuracy, etc.)

    # Save the trained model to a file
    def save(self, file_path):
        self.model.save(file_path)  # Save the model to the specified file path
