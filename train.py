import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# Update these paths to your actual data paths
TRAIN_DIR = '/Users/sanskritiagrawal/Downloads/elderly/archive/train/train'
TEST_DIR = '/Users/sanskritiagrawal/Downloads/elderly/archive/test/test'

def load_dataset(directory):
    """
    Load image paths and labels from directory
    """
    image_paths = []
    labels = []
    
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):  # Make sure it's a directory
            for filename in os.listdir(label_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
                    image_path = os.path.join(label_path, filename)
                    if os.path.isfile(image_path):  # Make sure it's a file
                        image_paths.append(image_path)
                        labels.append(label)
            print(f"Processed {label}")
    
    return image_paths, labels

def extract_features(images):
    """
    Convert images to arrays and normalize
    """
    features = []
    for image_path in images:
        try:
            img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
            img = np.array(img)
            features.append(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            continue
    
    if not features:
        raise ValueError("No valid images were loaded")
    
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

def build_model(input_shape=(48, 48, 1), output_class=7):
    model = Sequential()
    
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_class, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load training data
    print("Loading training data...")
    image_paths, labels = load_dataset(TRAIN_DIR)
    train = pd.DataFrame({'image': image_paths, 'label': labels})
    train = train.sample(frac=1).reset_index(drop=True)  # shuffle
    
    # Load test data
    print("Loading test data...")
    test_paths, test_labels = load_dataset(TEST_DIR)
    test = pd.DataFrame({'image': test_paths, 'label': test_labels})
    
    # Extract features
    print("Extracting features from training data...")
    train_features = extract_features(train['image'])
    print("Extracting features from test data...")
    test_features = extract_features(test['image'])
    
    # Normalize the data
    x_train = train_features/255.0
    x_test = test_features/255.0
    
    # Convert categories to integers
    print("Preparing labels...")
    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)
    
    # Build and train model
    print("Training model...")
    model = build_model()
    history = model.fit(
        x=x_train, 
        y=y_train,
        batch_size=128,
        epochs=50,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Save the model
    model.save('emotion_model.h5')
    print("Model saved as emotion_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Graph')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")

if __name__ == "__main__":
    main()