import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# --- Deep Learning Model Integration --- #
IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_and_preprocess_image_for_dl(image_path):
    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Rescale pixel values
    return img_array

def predict_image_quality_dl(model_path, image_path):
    # Load the trained model
    model = keras.models.load_model(model_path) 

    # Load and preprocess the image
    img_for_prediction = load_and_preprocess_image_for_dl(image_path)

    # Make prediction
    prediction = model.predict(img_for_prediction)[0][0]

    # Interpret prediction
    if prediction > 0.5:
        return "Good", prediction
    else:
        return "Bad", prediction

# --- Main Execution for Deep Learning Based Analysis --- #
if __name__ == '__main__':
    model_file = "classifier_model.h5"
    input_image_path = "/home/dataset/good/pic1.jpg" # Change this to your test image

    if not os.path.exists(model_file):
        print(f"Error: Model file \'{model_file}\' not found. Please train the model first using bead_classifier.py.")
    elif not os.path.exists(input_image_path):
        print(f"Error: Input image \'{input_image_path}\' not found.")
    else:
        # Use the deep learning model for classification
        dl_quality, dl_score = predict_image_quality_dl(model_file, input_image_path)
        print(f"Deep Learning Model Prediction for \'{input_image_path}\' is: {dl_quality} (Confidence: {dl_score:.2f})")

        # Visualize the DL prediction
        img_visual = cv2.imread(input_image_path)
        if img_visual is None:
            raise FileNotFoundError(f"Original image not found at {input_image_path}")

        text = f"DL Prediction: {dl_quality} (Score: {dl_score:.2f})"
        color = (0, 255, 0) if dl_quality == "Good" else (0, 0, 255) # Green for good, Red for bad
        cv2.putText(img_visual, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        output_filename = f"dl_prediction_result_{os.path.basename(input_image_path)}"
        cv2.imwrite(output_filename, img_visual)
        print(f"Deep Learning prediction visualization saved to {output_filename}")
