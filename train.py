# EfficientNet Feature Extractor + RandomForest Classifier (No Epoch Training)

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# CONFIGURATION
data_dir = r"C:/Users/hp340/Downloads/archive/blnw-images-224"  # Folder with subfolders like /nut, /bolt, etc.
IMG_SIZE = (224, 224)

# Load EfficientNetB0 for feature extraction
base_model = EfficientNetB0(include_top=False, pooling='avg', weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features and labels
def extract_features_from_directory(data_dir):
    features = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_index = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, fname)
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                feat = model.predict(img_array, verbose=0)
                features.append(feat.flatten())
                labels.append(class_to_index[class_name])

    return np.array(features), np.array(labels), class_names

print("\n Extracting features (this may take a few minutes)...")
X, y, class_names = extract_features_from_directory(data_dir)
print(" features extracted:", X.shape)
print(" Labels shape:", y.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("\n Model trained with RandomForestClassifier!")

# Evaluate
y_pred = clf.predict(X_test)
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# Save the model
import joblib
joblib.dump((clf, class_names), "rf_parts_classifier.pkl")
print("\n Model saved as 'rf_parts_classifier.pkl'")
