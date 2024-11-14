import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import argparse
import random

def augment_image(image):
    # Flip horizontally with 50% probability
    if random.random() > 0.5:
        image = cv2.flip(image, 1) 
    
    # Random rotation between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    # Random zoom by cropping and resizing
    scale = random.uniform(0.9, 1.1)
    h, w = image.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    image = cv2.resize(image, (nw, nh))
    
    # Crop back to original size if scaled up
    if scale > 1.0:
        start_h = (nh - h) // 2
        start_w = (nw - w) // 2
        image = image[start_h:start_h + h, start_w:start_w + w]
    else:
        image = cv2.resize(image, (30, 30))

    return image

def run(args):
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'Train')
    data_info = pd.read_csv(os.path.join(data_dir, 'classes.csv'))
    
    print("----------- Starting Data Preprocessing -----------")
    
    # Load dataset
    data = []
    labels = []

    # Number of classes is determined by the number of columns (excluding filename)
    class_names = data_info.columns[1:]
    num_classes = len(class_names)

    # Load training images
    for idx, row in data_info.iterrows():
        img_file = row['filename']
        label = row[class_names].values.astype(int)  # Multi-hot encoding for classes
        
        img_path = os.path.join(train_dir, img_file)
        img = cv2.imread(img_path, -1)
        
        # Resize image to (30, 30)
        if img is not None:
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
            data.append(img)
            labels.append(label)

            for _ in range(3):  # Change the range to increase the number of augmented samples
                augmented_img = augment_image(img)
                data.append(augmented_img)
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Stratified split for multi-label
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(msss.split(data, labels))
    
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Save preprocessed data
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    print("----------- Data Preprocessing Complete -----------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    args = parser.parse_args()
    run(args)
