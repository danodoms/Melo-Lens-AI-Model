import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Set TensorFlow threading
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Set seed
random.seed(42)

# Dataset paths
dataset_path = "dataset/watermelon-disease/Augmented Image/Augmented_Image"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

# Create train/val folders
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Split raw images
classes = [d for d in os.listdir(dataset_path)
           if os.path.isdir(os.path.join(dataset_path, d)) and d not in ["train", "val"]]

for cls in classes:
    class_dir = os.path.join(dataset_path, cls)
    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images, val_images = images[:split_idx], images[split_idx:]

    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(val_path, cls), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_path, cls, img))
    for img in val_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(val_path, cls, img))

    if not os.listdir(class_dir):
        os.rmdir(class_dir)

print("✅ Dataset successfully split.")

# Get number of classes
num_classes = len(classes)

# Data generators
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
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
test_set = test_datagen.flow_from_directory(
    val_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

print("Class labels:", training_set.class_indices)

# Build model using Functional API
input_tensor = Input(shape=(128, 128, 3))
base_model = MobileNetV2(input_tensor=input_tensor, include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train
history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=20,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluation
predictions = model.predict(test_set, steps=test_set.samples // test_set.batch_size)
y_pred = np.argmax(predictions, axis=1)
y_true = test_set.classes[:len(y_pred)]

f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"\n📊 Evaluation Metrics:")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Confusion matrix & classification report
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_true, y_pred, target_names=list(training_set.class_indices.keys()))
print("Classification Report:")
print(report)

# Save Keras model
os.makedirs("model", exist_ok=True)
keras_model_path = "model/melon-disease.keras"
model.save(keras_model_path)
print(f"\n✅ Saved Keras model: {keras_model_path}")

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_path = "model/melon-disease.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Saved TFLite model: {tflite_model_path}")
