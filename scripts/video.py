import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import cv2
from sign_detection.sign_detector import SignDetector

def run(args):
    model_dir = f"models/{args.model_name}"
    model_path = os.path.join(model_dir, 'best_model.h5')

    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    source = args.source

    if source == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    print("----------- Starting Video Stream -----------")
    
    while True:
        ret, image = cap.read()
        if not ret:
            print("End of video stream")
            break
        signs = SignDetector().find_signs(image.copy())
        print(f"Found {len(signs)} signs")
        for sign in signs:
            sign_resize = cv2.resize(sign[0], (30, 30), interpolation=cv2.INTER_NEAREST)
            image_array = np.array(sign_resize) / 1.0
            image_array = np.expand_dims(image_array, axis=0)

            list_labels = []
            with open(os.path.join(model_dir, 'labels.txt'), 'r') as f:
                for line in f:
                    list_labels.append(line)

            # Predict the class
            print("Predicting the class of the traffic sign...")
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            print(prediction[0][predicted_class])
            print(f"Predicted Class: {predicted_class}")
            print(f"Predicted Labels: {list_labels[predicted_class]}")

            # Draw the bounding box and label
            cv2.rectangle(image, (sign[1][0], sign[1][1]), (sign[1][0] + sign[1][2], sign[1][1] + sign[1][3]), (0, 255, 0), 2)
            cv2.putText(image, list_labels[predicted_class], (sign[1][0], sign[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    print("----------- Inference Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Inference for Traffic Sign Recognition")
    parser.add_argument('--model_name', type=str, default='vn', help='Directory to save models')
    parser.add_argument('--source', type=str, required=True, help='Path to the video source (0 for webcam or path to video file)')
    args = parser.parse_args()
    run(args)
