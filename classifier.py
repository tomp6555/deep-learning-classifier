import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# Define image dimensions and batch size
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16 

# Data Generators for training and validation
def create_data_generators(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2 
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training" 
    )

    # We'll use the same generator for now, or create a separate one if a dedicated validation folder exists
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    ) 

    return train_generator, validation_generator

# Build the CNN model
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid") # Binary classification
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Train the model
def train_model(model, train_generator, validation_generator, epochs=10):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator, # Removed validation data for now
        callbacks=[early_stop] # added
    )
    return history

# Main execution for training
if __name__ == "__main__":
    data_directory = "./dataset"
    
    # Create data generators
    train_gen, val_gen = create_data_generators(data_directory)

    # Build and train the model
    model = build_model()
    print("Model Summary:")
    model.summary()

    print("\nTraining Model...")
    train_model(model, train_gen, val_gen, epochs=10)

    # Save the trained model
    model.save("classifier_model.h5")
    print("\nModel saved as classifier_model.h5")

