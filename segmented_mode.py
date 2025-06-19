"""
Image Classification Application with TensorFlow and PyQt5
-----------------------------------------------------------
A GUI application for training image classification models using Planktoscope images with data augmentation
and handling class imbalance. Supports loading segmented objects, classifying them,
and organizing them into classes for model training.

Current Configuration:
This implementation is optimized for smaller datasets (hundreds of images) with potential
class imbalance. The architecture uses:
- Gentle data augmentation
- Frozen EfficientNet B0 feature extractor
- Small learning rate (0.0001)
- Dropout rate of 0.2
- Class weighting to handle imbalance

Configuration for Larger Datasets:
For datasets with thousands or tens of thousands of images, consider:
- Increasing model capacity (more layers or nodes)
- Unfreezing top layers of the feature extractor for fine-tuning
- Increasing learning rate (0.001)
- Stronger data augmentation
- Batch normalization with larger batch sizes

Helpful Resources:
- TensorFlow Transfer Learning Guide: https://www.tensorflow.org/tutorials/images/transfer_learning
- Keras Data Augmentation: https://www.tensorflow.org/tutorials/images/data_augmentation
- Handling Class Imbalance: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
- EfficientNet Models: https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet

Author: [Adam Larson]
Date: [4.1.2025]
Version: 1.0

History:
[Daniel Elnatan] 4.10.2026 - swapped to using qtpy
"""

import os
import shutil
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from collections import Counter
import tensorflow_hub as hub
from sklearn.utils.class_weight import compute_class_weight

# from PyQt5.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
#     QFileDialog, QGridLayout, QScrollArea, QMessageBox, QSpinBox, QInputDialog, QCheckBox,
#     QApplication, QMainWindow, QProgressBar
# )
# from PyQt5.QtGui import QPixmap, QImage, QColor
# from PyQt5.QtCore import Qt, QThread, pyqtSignal

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGridLayout, QScrollArea, QMessageBox, QSpinBox, QInputDialog, QCheckBox,
    QApplication, QMainWindow, QProgressBar
)
from qtpy.QtGui import QPixmap, QImage, QColor
from qtpy.QtCore import Qt, QThread, Signal # Note: pyqtSignal is standardized to Signal


from keras.utils import get_custom_objects
get_custom_objects()['KerasLayer'] = hub.KerasLayer


class TrainModelThread(QThread):
    """
    A worker thread for training image classification models.
    
    This thread handles dataset preparation, model creation, training with
    class balancing, and model saving to avoid blocking the UI during
    computationally intensive operations for very large datasets.
    
    Signals:
        finished (str): Emitted when training completes with model path or error message
        progress (str): Emitted to provide status updates during training
    """
    finished = Signal(str)
    progress = Signal(str) 

    def __init__(self, data_dir, model_out, img_size=(224, 224), batch_size=32, epochs=15, export_trt=False):
        super().__init__()
        self.data_dir = data_dir
        self.model_out = model_out
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.export_trt = export_trt
        
    def run(self):
        """
        Main execution method that runs when the thread starts.
        
        Performs dataset preparation, model training, evaluation, and saving.
        Emits progress updates and final status when complete.
        """
        try:
            def get_dataset(data_dir):
                """
                Load image paths and labels from a directory structure.
                
                Expects a directory structure where each subdirectory is a class:
                data_dir/
                  ├── class1/
                  │     ├── image1.jpg
                  │     └── image2.jpg
                  └── class2/
                        ├── image3.jpg
                        └── image4.jpg
                
                Args:
                    data_dir (str): Root directory containing class subdirectories
                
                Returns:
                    tuple: (image_paths, labels, class_names)
                
                Raises:
                    ValueError: If directory doesn't exist or no valid images found
                """
                if not os.path.exists(data_dir):
                    raise ValueError(f"The provided data directory does not exist: {data_dir}")
                image_paths = []
                labels = []
                class_names = sorted([d for d in os.listdir(data_dir) 
                                     if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
                if not class_names:
                    raise ValueError("No class subdirectories found in the classification folder.")
                class_indices = {name: i for i, name in enumerate(class_names)}
                for class_name in class_names:
                    folder = os.path.join(data_dir, class_name)
                    for fname in os.listdir(folder):
                        if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                            full_path = os.path.join(folder, fname)
                            if os.path.isfile(full_path):
                                image_paths.append(full_path)
                                labels.append(class_indices[class_name])
                if not image_paths:
                    raise ValueError("No valid image files found in classification folders.")
                return image_paths, np.array(labels), class_names

            def preprocess_image(path):
                img = tf.io.read_file(path)
                img = tf.image.decode_image(img, channels=3)
                img.set_shape([None, None, 3])
                img = tf.image.resize(img, self.img_size)
                img = tf.cast(img, tf.float32) / 255.0
                return img

            def load_dataset(image_paths, labels):
                if len(image_paths) == 0:
                    raise ValueError("No images found in dataset.")
                path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
                image_ds = path_ds.map(lambda x: preprocess_image(x), num_parallel_calls=tf.data.AUTOTUNE)
                label_ds = tf.data.Dataset.from_tensor_slices(labels)
                return tf.data.Dataset.zip((image_ds, label_ds))

            def create_model(num_classes, img_size):
                """
                Create a model optimized for imbalanced datasets with data augmentation.
                
                This model architecture:
                1. Applies gentle data augmentation to prevent overfitting. These values can be changed with large datasets.
                2. Uses EfficientNet B0 as a feature extractor
                3. Adds a classification head with dropout and batch normalization
                4. Compiles with a lower learning rate for stable training. These values can be changed with large datasets. 
                
                Args:
                    num_classes (int): Number of output classes
                    img_size (tuple): Input image dimensions (width, height)
                
                Returns:
                    tf.keras.Model: Compiled model ready for training
                """
                # Define augmentation layers 
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip("horizontal"),
                    # Uses small rotation angle for a smaller (<100 image training set)
                    tf.keras.layers.RandomRotation(0.05),
                    # Conservative zoom for a smaller (<100 image training set)
                    tf.keras.layers.RandomZoom(0.05),
                    # Contrast variation
                    tf.keras.layers.RandomContrast(0.05),
                ])
                
                # Create model inputs
                inputs = tf.keras.Input(shape=img_size + (3,))
                
                # Apply data augmentation only during training
                x = data_augmentation(inputs)
                
                # Load the feature extractor
                model_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
                feature_extractor_layer = hub.KerasLayer(
                    model_url, 
                    input_shape=img_size + (3,),
                    trainable=False  # Keep feature extractor frozen
                )
                
                # Apply feature extraction
                x = feature_extractor_layer(x)
                
                # Classification head
                x = tf.keras.layers.Dropout(0.2)(x)  # Low dropout rate. Can change with large dataset
                x = tf.keras.layers.Dense(256, activation='relu')(x)  # Additional layer
                x = tf.keras.layers.BatchNormalization()(x)  # Adds batch normalization
                x = tf.keras.layers.Dropout(0.2)(x)
                outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
                
                model = tf.keras.Model(inputs, outputs)
                
                # Low learning rate with Adam optimizer, this can be changes with a large dataset
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                
                model.compile(
                    optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                return model

            self.progress.emit("Preparing dataset...")
            image_paths, labels, class_names = get_dataset(self.data_dir)
            
            # Get class distribution statistics
            class_counts = dict(Counter(labels))
            min_samples = min(class_counts.values())
            max_samples = max(class_counts.values())
            imbalance_ratio = max_samples / min_samples
            
            print("Class distribution:", class_counts)
            print(f"Class imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
            self.progress.emit(f"Found {len(image_paths)} images across {len(class_names)} classes")
            self.progress.emit(f"Class distribution: {class_counts}")

            # Compute class weights to handle any imbalance, if one class has significantly more objects this can bias learning
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels),
                y=labels
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            print("Class weights:", class_weight_dict)
            self.progress.emit(f"Using class weights: {class_weight_dict}")
            
            # Data splitting - use stratify to maintain class distribution
            X_train, X_temp, y_train, y_temp = train_test_split(
                image_paths, labels, test_size=0.3, stratify=labels, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
            )
            
            self.progress.emit(f"Training set: {len(X_train)} images")
            self.progress.emit(f"Validation set: {len(X_val)} images")
            self.progress.emit(f"Test set: {len(X_test)} images")

            # Create data loaders
            train_ds = load_dataset(X_train, y_train).shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = load_dataset(X_val, y_val).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = load_dataset(X_test, y_test).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            # Create model 
            self.progress.emit("Creating optimized model architecture...")
            model = create_model(len(class_names), self.img_size)

            print("Training on classes:", class_names)
            self.progress.emit(f"Starting training for {self.epochs} epochs with class weighting...")
            
            # Create callbacks for better training
            callbacks = [
                # Early stopping to prevent overfitting
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                # Reduce learning rate when plateauing
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            # Train with class weights and callbacks
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs,
                verbose=2,
                class_weight=class_weight_dict,
                callbacks=callbacks
            )
            
            self.progress.emit("Evaluating model...")
            test_loss, test_acc = model.evaluate(test_ds)
            print("Test accuracy:", test_acc)
            self.progress.emit(f"Test accuracy: {test_acc:.4f}")

            # If accuracy is very poor, provide a warning
            if test_acc < 0.5:
                self.progress.emit("WARNING: Model accuracy is low. Consider collecting more training data.")
            
            self.progress.emit("Saving model...")
            model.save(self.model_out)

            with open(os.path.splitext(self.model_out)[0] + "_classes.txt", "w") as f:
                f.write("\n".join(class_names))

            if self.export_trt:
                self.progress.emit("Exporting TensorRT model...")
                saved_model_dir = os.path.splitext(self.model_out)[0] + "_savedmodel"
                model.save(saved_model_dir, save_format='tf')
                try:
                    from tensorflow.python.compiler.tensorrt import trt_convert as trt
                    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir)
                    converter.convert()
                    trt_model_path = saved_model_dir + "_trt"
                    converter.save(trt_model_path)
                    self.finished.emit(f"LOAD_MODEL::{trt_model_path}")
                    return
                except Exception as e:
                    self.finished.emit(f"Model saved. TensorRT conversion failed: {str(e)}")
                    return

            self.finished.emit(f"LOAD_MODEL::{self.model_out}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(f"Training failed: {str(e)}")


class PredictThread(QThread):
    prediction_complete = Signal(list)
    progress = Signal(int)  # Progress indicator (0-100)

    def __init__(self, model_path, class_file, image_paths, img_size=(224, 224)):
        super().__init__()
        self.model_path = model_path
        self.class_file = class_file
        self.image_paths = image_paths
        self.img_size = img_size

    def run(self):
        try:
            model = tf.keras.models.load_model(self.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
            with open(self.class_file, 'r') as f:
                class_names = f.read().splitlines()

            def preprocess_image(path):
                img = tf.io.read_file(path)
                img = tf.image.decode_image(img, channels=3)
                img = tf.image.resize(img, self.img_size)
                img = tf.cast(img, tf.float32) / 255.0
                return img

            predictions = []
            total = len(self.image_paths)
            
            # Batch processing 
            batch_size = 8  # Process multiple images at once
            for i in range(0, total, batch_size):
                batch_paths = self.image_paths[i:i+batch_size]
                batch_images = []
                
                # Preprocess batch
                for path in batch_paths:
                    img = preprocess_image(path)
                    batch_images.append(img)
                
                # Make predictions on batch
                if batch_images:
                    batch_tensor = tf.stack(batch_images)
                    batch_preds = model.predict(batch_tensor)
                    
                    # Process batch results
                    for j, path in enumerate(batch_paths):
                        pred = batch_preds[j]
                        pred_class = class_names[np.argmax(pred)]
                        confidence = float(np.max(pred)) * 100
                        predictions.append((path, pred_class, confidence, pred.tolist()))
                
                # Update progress
                self.progress.emit(int(min(100, (i + len(batch_paths)) / total * 100)))

            self.prediction_complete.emit(predictions)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.prediction_complete.emit([f"Prediction failed: {str(e)}"])


class ClassificationLabel(QLabel):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.image_path = path
        self.is_selected = False
        self.prediction = None
        self.confidence = None
        self.setStyleSheet("border: 2px solid transparent;")
        self.setScaledContents(True)
        self.load_thumbnail(path)
        
    def mousePressEvent(self, event):
        self.is_selected = not self.is_selected
        self.update_border()
    
    def update_border(self):
        if self.is_selected:
            self.setStyleSheet("border: 2px solid red;")
        elif self.prediction:
            # Color border based on confidence
            if self.confidence >= 90:
                self.setStyleSheet("border: 2px solid green;")
            elif self.confidence >= 70:
                self.setStyleSheet("border: 2px solid yellow;")
            else:
                self.setStyleSheet("border: 2px solid orange;")
        else:
            self.setStyleSheet("border: 2px solid transparent;")
    
    def set_prediction(self, class_name, confidence):
        self.prediction = class_name
        self.confidence = confidence
        self.update_border()
        self.setToolTip(f"Class: {class_name}\nConfidence: {confidence:.1f}%")
        
        # Show label on the image
        self.class_label.setText(f"{class_name}: {confidence:.1f}%")
        self.class_label.adjustSize()
        self.class_label.show()
        
    def load_thumbnail(self, path):
        img = cv2.imread(path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (120, 120))
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix)
        self.setFixedSize(120, 120)
        
        # Add a placeholder for the class label
        self.class_label = QLabel(self)
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white;")
        self.class_label.setGeometry(0, 100, 120, 20)
        self.class_label.hide()  # Hide initially


class SegmentedModePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.class_dir_label = QLabel("Classification folder: [Not set]")
        layout.addWidget(self.class_dir_label)

        self.btn_set_class_folder = QPushButton("Set Classification Folder")
        self.btn_set_class_folder.clicked.connect(self._on_set_class_folder)
        layout.addWidget(self.btn_set_class_folder)

        self.btn_add_class = QPushButton("Add New Class")
        self.btn_add_class.clicked.connect(self.prompt_new_class)
        layout.addWidget(self.btn_add_class)

        self.btn_load_model = QPushButton("Load Trained Model")
        self.btn_load_model.clicked.connect(self._on_load_model)
        layout.addWidget(self.btn_load_model)

        self.class_buttons_layout = QHBoxLayout()
        layout.addLayout(self.class_buttons_layout)

        self.btn_load_segmented = QPushButton("Load Segmented Objects Folder")
        self.btn_load_segmented.clicked.connect(self.load_segmented_folder)
        layout.addWidget(self.btn_load_segmented)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_widget.setLayout(self.grid_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll)

        self.train_controls_layout = QHBoxLayout()
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(15)
        self.epochs_input.setPrefix("Epochs: ")
        self.train_controls_layout.addWidget(self.epochs_input)

        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 128)
        self.batch_input.setValue(32)
        self.batch_input.setPrefix("Batch: ")
        self.train_controls_layout.addWidget(self.batch_input)

        self.jetson_checkbox = QCheckBox("Export for Jetson (TensorRT)")
        self.train_controls_layout.addWidget(self.jetson_checkbox)
        layout.addLayout(self.train_controls_layout)

        # Classification and training buttons in horizontal layout
        self.action_buttons_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.action_buttons_layout.addWidget(self.train_button)

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_images)
        self.action_buttons_layout.addWidget(self.select_all_btn)
        
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_no_images)
        self.action_buttons_layout.addWidget(self.select_none_btn)

        self.btn_classify = QPushButton("Classify Displayed Images")
        self.btn_classify.clicked.connect(self.classify_current_images)
        self.action_buttons_layout.addWidget(self.btn_classify)
        
        layout.addLayout(self.action_buttons_layout)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        # Add a progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.classification_dir = None
        self.model_path = None
        self.class_file = None
        self.segmented_folder = None
        self.image_paths = []
        self.thumb_labels = []
        self.class_names = []

    def _on_set_class_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Classification Folder")
        if folder:
            self.set_classification_folder(folder)

    def set_classification_folder(self, folder):
        self.classification_dir = folder
        self.class_dir_label.setText(f"Classification folder: {folder}")
        self.model_path = os.path.join(folder, "segmented_model.h5")
        self.class_file = os.path.splitext(self.model_path)[0] + "_classes.txt"

        # Clear existing class buttons
        while self.class_buttons_layout.count():
            child = self.class_buttons_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.class_names = []

        try:
            class_dirs = [d for d in os.listdir(folder)
                          if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')]
            for class_name in class_dirs:
                class_path = os.path.join(folder, class_name)
                self.class_names.append(class_name)
                btn = QPushButton(f"Label as {class_name}")
                btn.clicked.connect(lambda _, name=class_name: self.label_selected(name))
                self.class_buttons_layout.addWidget(btn)
            
            # Report class counts
            if self.class_names:
                class_counts = []
                for class_name in self.class_names:
                    class_path = os.path.join(folder, class_name)
                    img_count = len([f for f in os.listdir(class_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    class_counts.append(f"{class_name}: {img_count}")
                
                self.status_label.setText(f"Classes found: {', '.join(class_counts)}")
            
        except Exception as e:
            print("Error reading classification folder:", e)

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", filter="*.h5")
        if path:
            self.load_model(path)

    def load_model(self, path):
        self.model_path = path
        self.class_file = os.path.splitext(path)[0] + "_classes.txt"
        if not os.path.exists(self.class_file):
            QMessageBox.warning(self, "Missing Class File", "Could not find associated _classes.txt file.")
        else:
            # Read class names from file
            with open(self.class_file, 'r') as f:
                self.class_names = f.read().splitlines()
            QMessageBox.information(self, "Model Loaded", 
                                   f"Loaded model with {len(self.class_names)} classes:\n{', '.join(self.class_names)}")

    def prompt_new_class(self):
        if not self.classification_dir:
            QMessageBox.warning(self, "Set Folder First", "Please set a classification folder first.")
            return
        class_name, ok = QInputDialog.getText(self, "Add New Class", "Enter class name:")
        if ok and class_name:
            if class_name in self.class_names:
                QMessageBox.warning(self, "Duplicate Class", "That class already exists.")
                return
            self.class_names.append(class_name)
            os.makedirs(os.path.join(self.classification_dir, class_name), exist_ok=True)
            btn = QPushButton(f"Label as {class_name}")
            btn.clicked.connect(lambda _, name=class_name: self.label_selected(name))
            self.class_buttons_layout.addWidget(btn)

    def select_all_images(self):
        """Select all displayed images"""
        for label in self.thumb_labels:
            label.is_selected = True
            label.update_border()
    
    def select_no_images(self):
        """Deselect all images"""
        for label in self.thumb_labels:
            label.is_selected = False
            label.update_border()

    def label_selected(self, class_name):
        class_folder = os.path.join(self.classification_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        selected = [l for l in self.thumb_labels if l.is_selected]
        if not selected:
            QMessageBox.information(self, "No Selection", "Please select images to label first.")
            return
            
        for lbl in selected:
            dst = os.path.join(class_folder, os.path.basename(lbl.image_path))
            if os.path.abspath(lbl.image_path) != os.path.abspath(dst):
                shutil.copy2(lbl.image_path, dst)
            # Update visually and reset selection
            lbl.set_prediction(class_name, 100.0)  # 100% confidence for manual labels
            lbl.is_selected = False
            lbl.update_border()
            
        QMessageBox.information(self, "Labeling Complete", f"Labeled {len(selected)} images as {class_name}.")

    def load_segmented_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Segmented Objects")
        if folder:
            self.segmented_folder = folder
            self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.image_paths.sort()
            self.display_thumbnails()
            
            self.status_label.setText(f"Loaded {len(self.image_paths)} images from {folder}")

    def display_thumbnails(self):
        # Clear the grid layout
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        self.thumb_labels = []
        row, col = 0, 0
        for path in self.image_paths:
            lbl = ClassificationLabel(path)
            self.grid_layout.addWidget(lbl, row, col)
            self.thumb_labels.append(lbl)
            col += 1
            if col >= 5:
                col = 0
                row += 1

    def train_model(self):
        if not self.classification_dir:
            QMessageBox.warning(self, "No Folder", "Please set the classification folder first.")
            return
            
        # Check if there are any classes with images
        valid_class_count = 0
        for class_name in self.class_names:
            class_path = os.path.join(self.classification_dir, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    valid_class_count += 1
        


        if valid_class_count < 2:
            QMessageBox.warning(self, "Insufficient Classes", 
                               "You need at least 2 classes with images for training. Please add more labeled images.")
            return
            
        model_path = os.path.join(self.classification_dir, "segmented_model.h5")
        self.status_label.setText("Training in progress...")
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        self.train_thread = TrainModelThread(
            self.classification_dir,
            model_out=model_path,
            batch_size=self.batch_input.value(),
            epochs=self.epochs_input.value(),
            export_trt=self.jetson_checkbox.isChecked()
        )
        self.train_thread.finished.connect(self.show_train_status)
        self.train_thread.progress.connect(self.update_train_progress)
        self.train_thread.start()

    def update_train_progress(self, message):
        self.status_label.setText(message)
        
    def show_train_status(self, message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        if message.startswith("LOAD_MODEL::"):
            model_path = message.split("::")[1]
            self.load_model(model_path)

    def classify_current_images(self):
        """Classify the currently displayed images"""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load segmented images first.")
            return
            
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Missing Model", "Trained model file not found.")
            return
            
        if not self.class_file or not os.path.exists(self.class_file):
            QMessageBox.warning(self, "Missing Class File", "Class names file not found.")
            return

        # Set up progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Classifying images...")
        
        # Start prediction thread
        self.predict_thread = PredictThread(self.model_path, self.class_file, self.image_paths)
        self.predict_thread.prediction_complete.connect(self.show_predictions_on_thumbnails)
        self.predict_thread.progress.connect(self.progress_bar.setValue)
        self.predict_thread.start()

    def show_predictions_on_thumbnails(self, results):
        """Display prediction results directly on the image thumbnails"""
        self.progress_bar.setVisible(False)
        
        if isinstance(results[0], str) and results[0].startswith("Prediction failed"):
            QMessageBox.critical(self, "Prediction Error", results[0])
            self.status_label.setText("")
            return
        
        # Create mapping from file path to prediction result
        result_map = {result[0]: (result[1], result[2]) for result in results}  # path -> (class, confidence)
        
        # Update each thumbnail with its prediction
        for label in self.thumb_labels:
            if label.image_path in result_map:
                class_name, confidence = result_map[label.image_path]
                label.set_prediction(class_name, confidence)
        
        # Show summary message with counts and average confidence by class
        class_counts = {}
        class_confidence = {}
        
        for result in results:
            class_name = result[1]
            confidence = result[2]
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                class_confidence[class_name] = []
            
            class_counts[class_name] += 1
            class_confidence[class_name].append(confidence)
        
        # Create detailed summary
        summary = []
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            avg_conf = sum(class_confidence[class_name]) / len(class_confidence[class_name])
            summary.append(f"{class_name}: {count} images (avg confidence: {avg_conf:.1f}%)")
        
        # Show summary information
        self.status_label.setText(f"Classified {len(results)} images")
        
        QMessageBox.information(self, "Classification Complete", 
                               f"Classified {len(results)} images.\n\nClass distribution:\n" + 
                               "\n".join(summary))


# Example main application code to test the implementation
if __name__ == "__main__":
    import sys
    
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle("Image Classification with Optimized Training")
    main_window.setGeometry(100, 100, 1000, 800)
    
    segmented_panel = SegmentedModePanel()
    main_window.setCentralWidget(segmented_panel)
    main_window.show()
    
    sys.exit(app.exec_())