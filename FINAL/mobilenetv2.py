# %%
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split 
import shutil

# %%
# Set random seed for reproducibility
np.random.seed(1337)
tf.random.set_seed(1337)


# %%
# Part 1: Dataset Preparation
def prepare_data(input_dir, output_dir, test_size=0.2):
    # Filter out only the actual class directories (exclude 'train' and 'val')
    classes = [cls for cls in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cls)) and cls not in ['train', 'val']]

    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        train_images, val_images = train_test_split(images, test_size=test_size, random_state=1337)

        train_dir = os.path.join(output_dir, 'train', cls)
        val_dir = os.path.join(output_dir, 'val', cls)

        # Ensure directories exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for img in train_images:
            img_path = os.path.join(cls_path, img)
            try:
                shutil.copy(img_path, os.path.join(train_dir, img))
            except PermissionError:
                print(f"Permission error while copying {img_path}")
                continue

        for img in val_images:
            img_path = os.path.join(cls_path, img)
            try:
                shutil.copy(img_path, os.path.join(val_dir, img))
            except PermissionError:
                print(f"Permission error while copying {img_path}")
                continue



# %%
# Input and output directory
input_dir = r'./dataset_exp'
output_dir = r'./dataset_exp/train_test_split'
prepare_data(input_dir, output_dir)


# %%
# Part 2: Load Pretrained Model and Build the Classifier

# Load MobileNetV2 pretrained on ImageNet
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  

# %%
# Build the classifier
classifier = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  
])

# %%
# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(classifier.summary())


# %%
# Part 3: Data Augmentation and Image Generators

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# %%
# Only rescale for the test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# %%

# Load training and testing data
training_set = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    os.path.join(output_dir, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# %%
# Print class indices
label_map = training_set.class_indices
print(f"Class indices: {label_map}")


# %%
# Visualize augmented data
augmented_images, _ = next(training_set)
plt.figure(figsize=(12, 12))
plt.suptitle("Augmented Images", fontsize=16)
augmentation_titles = [
    "Original", "Horizontal Flip", "Zoom", "Shear", "Brightness Adj.",
    "Rotation", "Width Shift", "Height Shift", "Combination"
]

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[i])
    plt.title(augmentation_titles[i % len(augmentation_titles)])
    plt.axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
# Part 4: Train the Model with Callbacks

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)


# %%
# Train the model
history = classifier.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=1,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size,
    callbacks=[early_stopping, reduce_lr]
)


# %%

# Part 5: Evaluate the Model

# Evaluate on test data
predictions = classifier.predict(test_set, steps=test_set.samples // test_set.batch_size)
y_pred = np.argmax(predictions, axis=1)
y_true = test_set.classes[:len(y_pred)]


# %%
# Calculate metrics
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# %%
# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# %%
# Classification report
report = classification_report(y_true, y_pred, target_names=list(label_map.keys()))
print("Classification Report:")
print(report)

# %%
# Part 6: Convert the Model to TensorFlow Lite (TFLite)

# Convert the model to TFLite
# converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
# tflite_model = converter.convert()


# # %%
# # Save the TFLite model
# tflite_model_path = r'C:\Users\User\Documents\Tenserflow\rice_plant_lacks_nutrients.tflite'
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)

# print(f'TFLite model saved as {tflite_model_path}')



