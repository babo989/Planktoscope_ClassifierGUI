"""
Planktoscope Classifier - Raw Images Mode
-----------------------------------------------------------
This module provides a UI panel for processing full Planktoscope images.
It allows users to:
1. View and browse raw Planktoscope images
2. Segment images to detect objects based on PlanktoScope's segmentation approach
3. Classify detected objects using models trained in the Segmented Objects mode
4. Export segmented objects for further analysis

Author: [Adam Larson]
Date: [4.1.2025]
Version: 1.0

History:
[Daniel Elnatan] 4.10.2026 - swapped to using qtpy
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import tensorflow_hub as hub
from collections import Counter
import json
import shutil
from datetime import datetime
from scipy import ndimage
from skimage import measure, morphology, color

# OLD
# from PyQt5.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
#     QListWidget, QFileDialog, QMessageBox, QProgressBar,
#     QSplitter, QTextEdit, QGroupBox, QCheckBox, QSpinBox,
#     QDoubleSpinBox, QFormLayout, QSlider, QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QTabWidget
# )
# from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
# from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QRect

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QFileDialog, QMessageBox, QProgressBar,
    QSplitter, QTextEdit, QGroupBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QFormLayout, QSlider, QScrollArea, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QTabWidget
)
from qtpy.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from qtpy.QtCore import Qt, QSize, QThread, Signal, QRect


from keras.utils import get_custom_objects
get_custom_objects()['KerasLayer'] = hub.KerasLayer


class SegmentThread(QThread):
    """Worker thread for image segmentation."""
    progress = Signal(int)
    status = Signal(str)
    result = Signal(object, object, list)  # Original image, segmentation mask, object metadata list
    finished_all = Signal(str, int)  # Output folder, object count
    
    def __init__(self, image_paths, output_folder, params=None):
        super().__init__()
        self.image_paths = image_paths
        self.output_folder = output_folder
        self.params = params or {}
        self.stop_requested = False
        
        # Default parameters
        self.min_area = self.params.get('min_area', 100)  # Min pixels for object
        self.pixel_size = self.params.get('pixel_size', 3.45)  # μm per pixel
        self.min_diameter = self.params.get('min_diameter', 20)  # Min diameter in μm
        self.median_recalc_threshold = self.params.get('median_recalc_threshold', 20)
        self.median_window = self.params.get('median_window', 10)
        
        # Calculated values
        self.min_area_pixels = int(np.pi * ((self.min_diameter/2) / self.pixel_size)**2)
        
        # State variables
        self.median_image = None
        self.median_image_index = 0
        self.objects_per_image = []
        self.total_objects = 0
        
    def run(self):
        """Execute the segmentation process."""
        if not self.image_paths:
            self.status.emit("No images to process")
            return
            
        # Prepare output directory
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Calculate initial median image
        self.status.emit("Calculating initial median image...")
        self.calculate_median_image(0)
        
        # Process each image
        total_images = len(self.image_paths)
        for i, image_path in enumerate(self.image_paths):
            if self.stop_requested:
                self.status.emit("Segmentation stopped by user")
                break
                
            self.status.emit(f"Processing image {i+1}/{total_images}: {os.path.basename(image_path)}")
            
            # Load image
            original = cv2.imread(image_path)
            if original is None:
                self.status.emit(f"Error: Could not read image {image_path}")
                continue
                
            # Process image
            try:
                # Apply median correction
                corrected = self.apply_median_correction(original)
                
                # Calculate mask
                mask = self.calculate_mask(corrected)
                
                # Extract objects
                objects_metadata = self.extract_objects(corrected, mask, image_path, i)
                
                # Emit results for the UI to display
                self.result.emit(original, mask, objects_metadata)
                
                # Update progress
                self.progress.emit(int((i + 1) / total_images * 100))
                
                # Check if we need to recalculate median image
                current_object_count = len(objects_metadata)
                self.objects_per_image.append(current_object_count)
                
                if len(self.objects_per_image) > 1:
                    avg_objects = sum(self.objects_per_image) / len(self.objects_per_image)
                    if current_object_count > avg_objects + self.median_recalc_threshold:
                        self.status.emit("Recalculating median image...")
                        next_index = min(i + 1, len(self.image_paths) - 1)
                        self.calculate_median_image(next_index)
            
            except Exception as e:
                self.status.emit(f"Error processing image: {str(e)}")
                import traceback
                traceback.print_exc()
                
        # Emit final completion signal
        self.finished_all.emit(self.output_folder, self.total_objects)
        
    def calculate_median_image(self, start_index):
        """Calculate median image from a sequence of consecutive Planktoscope images."""
        end_index = min(start_index + self.median_window, len(self.image_paths))
        
        # If we're at the end, use previous images
        if end_index - start_index < self.median_window/2:
            start_index = max(0, end_index - self.median_window)
            
        # Use odd number of images as per the PlanktoScope logic
        if (end_index - start_index) % 2 == 0 and end_index > start_index:
            end_index -= 1
            
        if end_index <= start_index:
            self.status.emit("Warning: Not enough images for median calculation")
            return
            
        self.status.emit(f"Calculating median from images {start_index+1}-{end_index}")
        
        # Load all images
        images = []
        for i in range(start_index, end_index):
            img = cv2.imread(self.image_paths[i])
            if img is not None:
                images.append(img)
                
        if not images:
            self.status.emit("Error: Could not load any images for median calculation")
            return
            
        # Stack images and calculate median
        stack = np.stack(images, axis=0)
        self.median_image = np.median(stack, axis=0).astype(np.uint8)
        self.median_image_index = start_index
        
    def apply_median_correction(self, image):
        """Apply median correction to normalize the image."""
        if self.median_image is None:
            return image
            
        # Convert to float to avoid overflow
        image_float = image.astype(np.float32)
        median_float = self.median_image.astype(np.float32)
        
        # Avoid division by zero
        median_float[median_float == 0] = 1.0
        
        # Division and scaling
        corrected = (image_float / median_float) * 127.5  # Scale to middle of 0-255 range
        
        # Clip and convert back to uint8
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
        
    def calculate_mask(self, image):
        """Calculate segmentation mask from corrected image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Triangle threshold for dark objects
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_TRIANGLE)
        
        # Erode with small kernel to remove noise
        erode_kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(thresh, erode_kernel)
        
        # Dilate with larger kernel
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        dilated = cv2.dilate(eroded, dilate_kernel)
        
        # Close to fill holes
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, dilate_kernel)
        
        # Erode back to original size
        final_mask = cv2.erode(closed, dilate_kernel)
        
        return final_mask
        
    def extract_objects(self, image, mask, image_path, image_index):
        """Extract objects from mask and calculate their properties."""
        # Find connected components
        labels = measure.label(mask)
        props = measure.regionprops(labels, intensity_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # Filter by area and extract objects
        objects_metadata = []
        
        for prop in props:
            area = prop.area_filled  # Use filled area (including any holes)
            
            # Skip objects smaller than the minimum area
            if area < self.min_area_pixels:
                continue
                
            # Extract a nice bounding box
            min_row, min_col, max_row, max_col = prop.bbox
            
            # Pad the nice bounding box by 10 pixels
            pad = 10
            min_row = max(0, min_row - pad)
            min_col = max(0, min_col - pad)
            max_row = min(image.shape[0], max_row + pad)
            max_col = min(image.shape[1], max_col + pad)
            
            # Extract and save the object image
            object_img = image[min_row:max_row, min_col:max_col]
            
            # Generate a unique filename
            object_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_obj{prop.label}.png"
            object_path = os.path.join(self.output_folder, object_filename)
            
            # Save object image
            cv2.imwrite(object_path, object_img)
            
            # Calculate object metadata for EcoTaxa compatibility
            metadata = self.calculate_object_metadata(prop, image)
            metadata['image_path'] = object_path
            metadata['source_image'] = image_path
            metadata['object_id'] = f"{image_index}_{prop.label}"
            
            objects_metadata.append(metadata)
            
        # Update total object count
        self.total_objects += len(objects_metadata)
        
        return objects_metadata
        
    def calculate_object_metadata(self, prop, image):
        """Calculate metadata for an object according to PlanktoScope standards."""
        # Basic properties
        metadata = {
            'label': prop.label,
            'area_exc': prop.area,
            'area': prop.area_filled,
            '%area': 1 - (prop.area / prop.area_filled) if prop.area_filled > 0 else 0,
            
            # Equivalent circle
            'equivalent_diameter': prop.equivalent_diameter,
            
            # Equivalent ellipse
            'eccentricity': prop.eccentricity,
            'major': prop.axis_major_length,
            'minor': prop.axis_minor_length,
            'elongation': prop.axis_major_length / prop.axis_minor_length if prop.axis_minor_length > 0 else 0,
            'angle': (prop.orientation * 180 / np.pi) % 180,
            
            # Perimeter
            'perim.': prop.perimeter,
            'perimareaexc': prop.perimeter / prop.area if prop.area > 0 else 0,
            'perimmajor': prop.perimeter / prop.axis_major_length if prop.axis_major_length > 0 else 0,
            'circ.': 4 * np.pi * prop.area_filled / (prop.perimeter * prop.perimeter) if prop.perimeter > 0 else 0,
            'circex': 4 * np.pi * prop.area / (prop.perimeter * prop.perimeter) if prop.perimeter > 0 else 0,
            
            # Bounding box
            'bx': prop.bbox[1],
            'by': prop.bbox[0],
            'width': prop.bbox[3] - prop.bbox[1],
            'height': prop.bbox[2] - prop.bbox[0],
            'bounding_box_area': prop.bbox_area,
            'extent': prop.extent,
            
            # Convex hull
            'convex_area': prop.convex_area,
            'solidity': prop.solidity,
            
            # Centroid
            'x': prop.centroid[1],
            'y': prop.centroid[0],
            'local_centroid_col': prop.local_centroid[1],
            'local_centroid_row': prop.local_centroid[0],
            
            # Topology
            'euler_number': prop.euler_number,
        }
        
        # Calculate HSV statistics (would need to extract region pixels in HSV space)
        # This is not the best approach - for full accuracy, we need to mask the object
        min_row, min_col, max_row, max_col = prop.bbox
        region_img = image[min_row:max_row, min_col:max_col].copy()
        
        # Convert to HSV
        hsv_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        
        # Add HSV statistics
        metadata.update({
            'MeanHue': np.mean(h),
            'StdHue': np.std(h),
            'MeanSaturation': np.mean(s),
            'StdSaturation': np.std(s),
            'MeanValue': np.mean(v),
            'StdValue': np.std(v),
        })
        
        return metadata


class ClassifySegmentedThread(QThread):
    """Worker thread for classifying segmented Planktoscope objects."""
    progress = Signal(int)
    result = Signal(list)  # List of (object_id, class, confidence) tuples
    
    def __init__(self, model, class_names, objects_metadata):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.objects_metadata = objects_metadata
        
    def run(self):
        """Classify all segmented objects."""
        results = []
        total = len(self.objects_metadata)
        
        # Process in batches for efficiency
        batch_size = 8
        for i in range(0, total, batch_size):
            batch_metadata = self.objects_metadata[i:i+batch_size]
            batch_images = []
            
            # Preprocess batch
            for metadata in batch_metadata:
                try:
                    img = cv2.imread(metadata['image_path'])
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error processing {metadata['image_path']}: {e}")
                    continue
            
            if batch_images:
                # Stack images into batch
                batch_tensor = np.stack(batch_images)
                
                # Predict
                batch_preds = self.model.predict(batch_tensor)
                
                # Process predictions
                for j, metadata in enumerate(batch_metadata[:len(batch_images)]):
                    pred = batch_preds[j]
                    class_idx = np.argmax(pred)
                    confidence = float(pred[class_idx]) * 100
                    
                    if class_idx < len(self.class_names):
                        class_name = self.class_names[class_idx]
                    else:
                        class_name = f"Class{class_idx}"
                    
                    results.append((metadata['object_id'], class_name, confidence, metadata))
            
            # Update progress
            self.progress.emit(int(min(100, (i + len(batch_metadata)) / total * 100)))
        
        self.result.emit(results)

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self._zoom = 0
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_pixmap(self, pixmap):
        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self._zoom = 0
        self.resetTransform()

    def wheelEvent(self, event):
        # Define the base scaling factor for one "step" of the wheel.
        zoom_factor_step = 1.25

        # Determine whether we're zooming in or out.
        if event.angleDelta().y() > 0:
            # Zoom in: increase _zoom level if not already too high.
            if self._zoom < 10:  # maximum zoom in level
                self._zoom += 1
                factor = zoom_factor_step
            else:
                factor = 1.0  # do not zoom further
        else:
            # Zoom out: decrease _zoom level if not already too low.
            if self._zoom > -10:  # minimum zoom out level
                self._zoom -= 1
                factor = 1 / zoom_factor_step
            else:
                factor = 1.0  # do not zoom further

        # Apply the scaling.
        self.scale(factor, factor)

class RawModePanel(QWidget):
    """
    Panel for viewing raw Planktoscope images and segmenting them into objects.
    
    Features:
    - Load a folder of raw Planktoscope images
    - View images and their histogram
    - Segment images to detect objects
    - Display segmentation results with object overlays
    - Classify detected objects
    - Export objects for further analysis if you want
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.image_paths = []
        self.current_index = -1
        self.current_image = None
        self.segmented_objects = []
        self.segment_results = {}  # Maps image path to the segmentation results
        
        self.model = None
        self.class_names = []

        self.object_size_slider = None
        self.size_filter_label = None
        
        self.init_ui()
    def clear_layout(self, layout):
        """Clear all widgets from the layout."""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        
        self.btn_load_folder = QPushButton("Load Raw Images")
        self.btn_load_folder.clicked.connect(self.load_image_folder)
        toolbar.addWidget(self.btn_load_folder)
        
        self.btn_segment = QPushButton("Segment Current Image")
        self.btn_segment.clicked.connect(self.segment_current_image)
        toolbar.addWidget(self.btn_segment)
        
        self.btn_segment_all = QPushButton("Segment All Images")
        self.btn_segment_all.clicked.connect(self.segment_all_images)
        toolbar.addWidget(self.btn_segment_all)
        
        self.btn_classify = QPushButton("Classify Segmented Objects")
        self.btn_classify.clicked.connect(self.classify_segmented_objects)
        toolbar.addWidget(self.btn_classify)
        
        self.btn_export = QPushButton("Export Objects")
        self.btn_export.clicked.connect(self.export_segmented_objects)
        toolbar.addWidget(self.btn_export)
        
        main_layout.addLayout(toolbar)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: image list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        left_layout.addWidget(QLabel("Raw Images:"))
        left_layout.addWidget(self.image_list)
        
        # Segmentation parameters
        param_group = QGroupBox("Segmentation Parameters")
        param_layout = QFormLayout(param_group)
        
        self.min_diameter_input = QDoubleSpinBox()
        self.min_diameter_input.setRange(1, 1000)
        self.min_diameter_input.setValue(20)
        self.min_diameter_input.setSuffix(" μm")
        param_layout.addRow("Min Object Diameter:", self.min_diameter_input)
        
        self.pixel_size_input = QDoubleSpinBox()
        self.pixel_size_input.setRange(0.1, 100)
        self.pixel_size_input.setValue(3.45)
        self.pixel_size_input.setSuffix(" μm/px")
        param_layout.addRow("Pixel Size:", self.pixel_size_input)
        
        left_layout.addWidget(param_group)
        # Object Size Filter UI
        size_group = QGroupBox("Min Object Size for Classification")
        size_layout = QVBoxLayout()

        self.size_filter_label = QLabel("Min Object Size: 0 pixels")
        size_layout.addWidget(self.size_filter_label)

        self.object_size_slider = QSlider(Qt.Horizontal)
        self.object_size_slider.setMinimum(0)
        self.object_size_slider.setMaximum(1000)
        self.object_size_slider.setValue(0)
        self.object_size_slider.setTickInterval(50)
        self.object_size_slider.setTickPosition(QSlider.TicksBelow)
        self.object_size_slider.valueChanged.connect(self.update_object_size_filter)
        size_layout.addWidget(self.object_size_slider)

        size_group.setLayout(size_layout)
        left_layout.addWidget(size_group)
        
        # Add visualization options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.show_mask_checkbox = QCheckBox("Show Segmentation Mask")
        self.show_mask_checkbox.setChecked(True)
        self.show_mask_checkbox.stateChanged.connect(self.update_image_display)
        viz_layout.addWidget(self.show_mask_checkbox)
        
        self.show_objects_checkbox = QCheckBox("Show Detected Objects")
        self.show_objects_checkbox.setChecked(True)
        self.show_objects_checkbox.stateChanged.connect(self.update_image_display)
        viz_layout.addWidget(self.show_objects_checkbox)
        
        left_layout.addWidget(viz_group)
        left_layout.addStretch()
        
        content_splitter.addWidget(left_panel)
        
        # Center/right panel: image viewer and results
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        # Image viewer
        self.image_view = ZoomableGraphicsView()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setMinimumSize(600, 400)
        center_layout.addWidget(self.image_view)
        
        # Tabs for results
        results_tabs = QTabWidget()
        
        # Objects tab
        self.objects_tab = QWidget()
        objects_layout = QVBoxLayout(self.objects_tab)
        
        self.objects_text = QTextEdit()
        self.objects_text.setReadOnly(True)
        objects_layout.addWidget(self.objects_text)
        
        results_tabs.addTab(self.objects_tab, "Detected Objects")
        
        # Classification tab
        self.classification_tab = QWidget()
        class_layout = QVBoxLayout(self.classification_tab)
        
        self.classification_text = QTextEdit()
        self.classification_text.setReadOnly(True)
        class_layout.addWidget(self.classification_text)
        
        results_tabs.addTab(self.classification_tab, "Classification Results")
        
        center_layout.addWidget(results_tabs)
        
        content_splitter.addWidget(center_panel)
        
        # Set initial splitter sizes
        content_splitter.setSizes([250, 750])
        
        main_layout.addWidget(content_splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(status_layout)


        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QHBoxLayout()
        self.thumbnail_container.setLayout(self.thumbnail_layout)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        main_layout.addWidget(QLabel("Object Thumbnails:"))
        main_layout.addWidget(self.thumbnail_scroll)    
    
    def load_image_folder(self):
        """Open file dialog to select a folder of raw Planktoscope images."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder of Raw Images")
        if not folder:
            return
            
        # Find image files
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(exts)]
        image_paths.sort()
        
        if not image_paths:
            QMessageBox.warning(self, "No Images", 
                               "No image files found in the selected folder")
            return
            
        # Update UI
        self.image_paths = image_paths
        self.image_list.clear()
        for path in image_paths:
            self.image_list.addItem(os.path.basename(path))
            
        # Select first image
        if self.image_paths:
            self.image_list.setCurrentRow(0)
            
        self.status_label.setText(f"Loaded {len(image_paths)} images from {folder}")
    
    def on_image_selected(self, index):
        """Handle selection of an image in the list."""
        if index < 0 or index >= len(self.image_paths):
            return
            
        self.current_index = index
        path = self.image_paths[index]
        
        # Load image
        self.current_image = cv2.imread(path)
        if self.current_image is None:
            self.image_view.setText(f"Error: Could not read image {os.path.basename(path)}")
            return
            
        # Check if this image has been segmented
        self.update_image_display()
        
        # Show object info if available
        if path in self.segment_results:
            objects = self.segment_results[path]
            self.objects_text.setText(f"Detected {len(objects)} objects in {os.path.basename(path)}\n\n")
            
            # Display the first 10 objects
            for i, obj in enumerate(objects[:10]):
                self.objects_text.append(f"Object {i+1}:\n")
                self.objects_text.append(f"  Size: {obj['area']} pixels\n")
                self.objects_text.append(f"  Position: ({obj['x']:.1f}, {obj['y']:.1f})\n")
                self.objects_text.append(f"  Dimensions: {obj['width']}×{obj['height']} pixels\n")
                self.objects_text.append("\n")
                
            if len(objects) > 10:
                self.objects_text.append(f"... and {len(objects) - 10} more objects.")
        else:
            self.objects_text.setText("No segmentation data for this image.\nUse 'Segment Current Image' to detect objects.")
    def update_object_size_filter(self, value):
        """Update the label and filter objects visually."""
        # Update label
        self.size_filter_label.setText(f"Min Object Size: {value} pixels")
        
        # Immediately update image display to reflect current filter
        self.update_image_display()

    def _handle_classification_result(self, results):
        """Handle classification results from the classification thread."""
        if not results:
            self.status_label.setText("No classification results received")
            return
            
        # Store the total number of classified objects
        total_classified = len(results)
        classified_count_by_class = {}
        
        # Update the classification data in our nicely stored objects
        for obj_id, class_name, confidence, metadata in results:
            # Find this object in both the segmented_objects list and in the results for each image
            # First update in segmented_objects
            for obj in self.segmented_objects:
                if obj['object_id'] == obj_id:
                    obj['classification'] = class_name
                    obj['confidence'] = confidence
                    
                    # Track counts by class
                    if class_name not in classified_count_by_class:
                        classified_count_by_class[class_name] = 0
                    classified_count_by_class[class_name] += 1
                    break
                    
            # Also update in the per-image results
            for path, objects in self.segment_results.items():
                if isinstance(objects, list):  # Skip mask entries which are not lists
                    for obj in objects:
                        if obj.get('object_id') == obj_id:
                            obj['classification'] = class_name
                            obj['confidence'] = confidence
                            break
        
        # Update classification results text view
        self.classification_text.clear()
        self.classification_text.append(f"Classified {total_classified} objects:\n")
        
        # Add summary by class
        for class_name, count in sorted(classified_count_by_class.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_classified) * 100 if total_classified > 0 else 0
            self.classification_text.append(f"{class_name}: {count} objects ({percentage:.1f}%)")
        
        self.classification_text.append("\n\nDetailed results:")
        
        # Add detailed results for first 50 objects
        displayed_results = results[:50] if len(results) > 50 else results
        for obj_id, class_name, confidence, metadata in displayed_results:
            area = metadata.get('area', 'unknown')
            self.classification_text.append(
                f"Object {obj_id}: {class_name} ({confidence:.1f}%) - Area: {area} pixels"
            )
        
        if len(results) > 50:
            self.classification_text.append(f"\n... and {len(results) - 50} more objects.")
        
        # Update status
        self.status_label.setText(f"Classification complete. Classified {total_classified} objects.")
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        # Update the image display to show classifications
        self.update_image_display()
        
        # Update the thumbnails
        self.update_thumbnails()

    def update_image_display(self):
        if self.current_image is None:
            return

        # Create a copy of the current image to draw on
        display_img = self.current_image.copy()
        path = self.image_paths[self.current_index]
        min_size = self.object_size_slider.value() if self.object_size_slider is not None else 0

        # Draw the segmentation mask if requested
        if self.show_mask_checkbox.isChecked() and path in self.segment_results:
            mask_key = path + '_mask'
            if mask_key in self.segment_results:
                mask = self.segment_results[mask_key].copy()
                overlay = np.zeros_like(display_img)
                overlay[:, :, 2] = mask  # red channel for the mask
                alpha = 0.3
                display_img = cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0)

        # Track filtered objects count
        filtered_count = 0
        
        # Draw detected objects if requested
        if self.show_objects_checkbox.isChecked() and path in self.segment_results:
            objects = self.segment_results[path]
            for obj in objects:
                # Apply size filter
                if obj['area'] < min_size:
                    continue
                    
                # Increment filtered count
                filtered_count += 1
                
                x, y = int(obj['bx']), int(obj['by'])
                w, h = int(obj['width']), int(obj['height'])
                
                # Default colors and text for unclassified objects
                box_color = (0, 255, 0)  # green for unclassified
                text_color = (0, 0, 0)   # black text
                
                # For classified objects, use different colors and show class name
                if 'classification' in obj and 'confidence' in obj:
                    # Use orange box for classified objects
                    box_color = (0, 165, 255)  # orange (BGR format)
                    text_color = (255, 255, 255)  # white text
                    
                    # Prepare label with classification info
                    label = f"{obj['classification']} ({obj['confidence']:.1f}%)"
                    
                    # Draw bounding box and classification text
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), box_color, 2)
                    
                    # Add background for text readability
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(
                        display_img, 
                        (x, y - text_size[1] - 5), 
                        (x + text_size[0] + 5, y), 
                        box_color, 
                        -1  # Filled rectangle
                    )
                    
                    # Draw classification text
                    cv2.putText(display_img, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                else:
                    # Just draw bounding box for unclassified objects
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), box_color, 2)
                    
                # Always show object ID and area at the bottom of the box
                id_text = f"ID: {obj.get('object_id', 'N/A')} (Area: {obj['area']:.0f})"
                cv2.putText(display_img, id_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Update the objects text to include filtered count
        if path in self.segment_results:
            objects = self.segment_results[path]
            total_count = len(objects)
            
            # Update the objects text display with both total and filtered counts
            if min_size > 0:
                self.objects_text.setText(f"Detected {filtered_count} objects (of {total_count} total) passing minimum size filter of {min_size} pixels\n\n")
            else:
                self.objects_text.setText(f"Detected {total_count} objects\n\n")
            
            # Display details of the filtered objects (up to 10)
            filtered_objects = [obj for obj in objects if obj['area'] >= min_size]
            for i, obj in enumerate(filtered_objects[:10]):
                self.objects_text.append(f"Object {i+1}:\n")
                self.objects_text.append(f"  Size: {obj['area']} pixels\n")
                self.objects_text.append(f"  Position: ({obj['x']:.1f}, {obj['y']:.1f})\n")
                self.objects_text.append(f"  Dimensions: {obj['width']}×{obj['height']} pixels\n")
                
                # Add classification info if available
                if 'classification' in obj and 'confidence' in obj:
                    self.objects_text.append(f"  Classification: {obj['classification']} ({obj['confidence']:.1f}%)\n")
                    
                self.objects_text.append("\n")
                
            if len(filtered_objects) > 10:
                self.objects_text.append(f"... and {len(filtered_objects) - 10} more objects.")

        # Convert the display image to a QImage and update the QGraphicsView
        h, w, ch = display_img.shape
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        q_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Update the graphics view with the new image
        self.image_view.set_pixmap(pixmap)

    def update_thumbnails(self):
        """Populate the thumbnail area with object images and classification labels."""
        # Clear any existing thumbnails
        self.clear_layout(self.thumbnail_layout)
        
        # Get current size filter
        min_size = self.object_size_slider.value() if self.object_size_slider is not None else 0
        
        # Filter objects by size
        filtered_objects = [obj for obj in self.segmented_objects if obj.get('area', 0) >= min_size]
        
        # Sort objects: classified objects first, then by classification, then by confidence
        def sort_key(obj):
            has_class = 'classification' in obj
            class_name = obj.get('classification', 'ZZZ')  # 'ZZZ' to sort unclassified at the end
            confidence = obj.get('confidence', 0)
            return (not has_class, class_name, -confidence)
        
        filtered_objects.sort(key=sort_key)
        
        # Display a message if no objects match the filter
        if not filtered_objects:
            label = QLabel("No objects match the current size filter")
            label.setAlignment(Qt.AlignCenter)
            self.thumbnail_layout.addWidget(label)
            self.thumbnail_container.setLayout(self.thumbnail_layout)
            return
        
        # Create a thumbnail widget for each filtered object
        for obj in filtered_objects:
            # Create a smaller container for the thumbnail
            container = QWidget()
            container.setFixedSize(120, 120)  # Reduced size to fit more in view
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(2, 2, 2, 2)
            container_layout.setSpacing(1)  # Reduce spacing between elements
            
            # Create the image widget
            image_widget = QLabel()
            image_widget.setAlignment(Qt.AlignCenter)
            pixmap = QPixmap(obj['image_path'])
            if not pixmap.isNull():
                pixmap = pixmap.scaled(100, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_widget.setPixmap(pixmap)
            else:
                image_widget.setText("No Image")
                
            # Simplified metadata - just show the ID
            obj_id = obj.get('object_id', '?')
            info_label = QLabel(f"ID:{obj_id}")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("font-size: 8px; padding: 1px;")
            
            # Add image to container
            container_layout.addWidget(image_widget)
            
            # Add classification if available
            if 'classification' in obj and 'confidence' in obj:
                classification = obj['classification']
                confidence = obj['confidence']
                
                # Create a more compact classification label
                class_label = QLabel(f"{classification} ({confidence:.0f}%)")
                class_label.setAlignment(Qt.AlignCenter)
                class_label.setStyleSheet("""
                    background-color: rgba(0, 165, 255, 0.8); 
                    color: white;
                    font-weight: bold;
                    font-size: 9px;
                    padding: 2px;
                    border-radius: 2px;
                """)
                
                container_layout.addWidget(class_label)
                container_layout.addWidget(info_label)
                
                # Simpler border for classified items
                container.setStyleSheet("border: 2px solid #FFA500; border-radius: 3px;")
            else:
                container_layout.addWidget(info_label)
                container.setStyleSheet("border: 1px solid #CCCCCC; border-radius: 3px;")
            
            # Add the container to the thumbnail layout
            self.thumbnail_layout.addWidget(container)
        
        # Add a stretch at the end
        self.thumbnail_layout.addStretch()
        
        # Refresh the scroll area
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        
        # Make sure the scroll area shows horizontal scrollbar as needed
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)


    def segment_current_image(self):
        """Segment the currently displayed image to detect objects."""
        if self.current_index < 0 or self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load and select an image first")
            return
            
        # Create output folder
        output_folder = self._create_output_folder()
        if not output_folder:
            return
            
        # Get parameters
        params = {
            'min_diameter': self.min_diameter_input.value(),
            'pixel_size': self.pixel_size_input.value(),
        }
        
        # Set up segmentation thread
        current_path = self.image_paths[self.current_index]
        self.segment_thread = SegmentThread([current_path], output_folder, params)
        
        # Connect signals
        self.segment_thread.progress.connect(self.progress_bar.setValue)
        self.segment_thread.status.connect(self.status_label.setText)
        self.segment_thread.result.connect(self._handle_segment_result)
        self.segment_thread.finished_all.connect(self._handle_segment_complete)
        
        # Start processing
        self.progress_bar.setVisible(True)
        self.segment_thread.start()
    
    def segment_all_images(self):
        """Segment all loaded Planktoscope images to detect objects."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first")
            return
            
        # Confirm with user
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText(f"Segment all {len(self.image_paths)} images?")
        msg.setInformativeText("This may take a while depending on the number of images.")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        if msg.exec_() != QMessageBox.Yes:
            return
            
        # Create output folder
        output_folder = self._create_output_folder()
        if not output_folder:
            return
            
        # Get parameters
        params = {
            'min_diameter': self.min_diameter_input.value(),
            'pixel_size': self.pixel_size_input.value(),
        }
        
        # Set up segmentation thread
        self.segment_thread = SegmentThread(self.image_paths, output_folder, params)
        
        # Connect signals
        self.segment_thread.progress.connect(self.progress_bar.setValue)
        self.segment_thread.status.connect(self.status_label.setText)
        self.segment_thread.result.connect(self._handle_segment_result)
        self.segment_thread.finished_all.connect(self._handle_segment_complete)
        
        # Start processing
        self.progress_bar.setVisible(True)
        self.segment_thread.start()
    
    def _create_output_folder(self):
        """Create an output folder for segmented objects."""
        # Use a timestamp to create a unique folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(os.path.dirname(self.image_paths[0]), f"segmented_{timestamp}")
        
        try:
            os.makedirs(output_folder, exist_ok=True)
            return output_folder
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not create output folder: {str(e)}")
            return None
    def classify_segmented_objects(self):
        """Classify only segmented objects that pass the minimum size filter."""
        # Check that there are segmented objects available
        if not self.segmented_objects:
            QMessageBox.warning(self, "No Objects", "Please segment images first.")
            return

        # Check if a model has been loaded, if not, inform the user
        if self.model is None or not self.class_names:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a classification model first.")
            return
            
        # Apply the size filter to get only objects that pass the threshold
        min_size = self.object_size_slider.value() if self.object_size_slider is not None else 0
        filtered_objects = [obj for obj in self.segmented_objects if obj.get('area', 0) >= min_size]
        
        # Check if any objects pass the filter
        if not filtered_objects:
            QMessageBox.warning(
                self, 
                "No Objects Pass Filter", 
                f"No objects meet the minimum size filter of {min_size} pixels.\nReduce the filter or segment larger objects."
            )
            return
            
        # Show confirmation dialog with counts
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText(f"Classify {len(filtered_objects)} objects?")
        msg.setInformativeText(
            f"Only objects with size ≥ {min_size} pixels will be classified.\n"
            f"({len(filtered_objects)} of {len(self.segmented_objects)} total objects pass this filter.)"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        if msg.exec_() != QMessageBox.Yes:
            return

        # Initialize and start a classification thread using the ClassifySegmentedThread
        self.classify_thread = ClassifySegmentedThread(self.model, self.class_names, filtered_objects)
        
        # Connect signals to update the UI
        self.classify_thread.progress.connect(lambda p: self.progress_bar.setValue(p))
        self.classify_thread.result.connect(self._handle_classification_result)
        
        # Show the progress bar
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"Classifying {len(filtered_objects)} objects...")
        
        # Start the classification process
        self.classify_thread.start()

    def _handle_segment_result(self, original_image, mask, objects_metadata):
        """Handle result from segmentation thread for a single image."""
        if not objects_metadata:
            return
            
        # Store the results
        path = objects_metadata[0]['source_image']  # All objects from same source
        self.segment_results[path] = objects_metadata
        
        # Store mask specifically for this image path
        self.segment_results[path + '_mask'] = mask
        
        # If this is the current image, update display
        if path == self.image_paths[self.current_index]:
            self.update_image_display()
            
            # Update objects text
            self.objects_text.setText(f"Detected {len(objects_metadata)} objects\n\n")
            
            # Display first 10 objects
            for i, obj in enumerate(objects_metadata[:10]):
                self.objects_text.append(f"Object {i+1}:\n")
                self.objects_text.append(f"  Size: {obj['area']} pixels\n")
                self.objects_text.append(f"  Position: ({obj['x']:.1f}, {obj['y']:.1f})\n")
                self.objects_text.append(f"  Dimensions: {obj['width']}×{obj['height']} pixels\n")
                self.objects_text.append("\n")
                
            if len(objects_metadata) > 10:
                self.objects_text.append(f"... and {len(objects_metadata) - 10} more objects.")
            
        # Collect all objects for later use
        self.segmented_objects.extend(objects_metadata)
    def _handle_segment_complete(self, output_folder, object_count):
        """Handle completion of the segmentation process."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Reset progress
        self.progress_bar.setValue(0)
        
        # Update status
        self.status_label.setText(f"Segmentation complete. Found {object_count} objects in total.")
        
        # Update thumbnails
        self.update_thumbnails()
        
        # If we're currently viewing an image that was segmented, refresh the view
        if self.current_index >= 0:
            self.update_image_display()
            
        # Show message to user
        QMessageBox.information(
            self, 
            "Segmentation Complete",
            f"Segmentation complete.\nFound {object_count} objects.\nSaved to {output_folder}"
    )
    def export_segmented_objects(self):
        """Export segmented objects with classifications to a CSV file."""
        if not self.segmented_objects:
            QMessageBox.warning(self, "No Objects", "No segmented objects available. Please segment images first.")
            return

        # Ask for export location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Segmented Objects",
            os.path.dirname(self.image_paths[0]),
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            # Prepare data for export
            with open(file_path, 'w') as f:
                # Write header
                header = ["object_id", "source_image", "image_path"]
                
                # Add classification fields if available
                has_classification = any('classification' in obj for obj in self.segmented_objects)
                if has_classification:
                    header.extend(["classification", "confidence"])
                    
                # Add metadata fields
                metadata_fields = [
                    'area', 'area_exc', '%area',
                    'equivalent_diameter', 'eccentricity',
                    'major', 'minor', 'elongation', 'angle',
                    'perim.', 'circ.', 'circex', 
                    'bx', 'by', 'width', 'height', 'extent',
                    'convex_area', 'solidity', 'x', 'y'
                ]
                header.extend(metadata_fields)
                
                f.write(','.join(header) + '\n')
                
                # Write data rows
                for obj in self.segmented_objects:
                    row = [
                        obj['object_id'],
                        os.path.basename(obj['source_image']),
                        os.path.basename(obj['image_path'])
                    ]
                    
                    # Add classification if available
                    if has_classification:
                        row.append(obj.get('classification', ''))
                        row.append(str(obj.get('confidence', '')))
                    
                    # Add metadata
                    for field in metadata_fields:
                        value = obj.get(field, '')
                        row.append(str(value))
                    
                    f.write(','.join(row) + '\n')

            # Create a parallel folder with the object images
            output_folder = os.path.splitext(file_path)[0] + "_images"
            os.makedirs(output_folder, exist_ok=True)

            # Copy object images
            for obj in self.segmented_objects:
                src = obj['image_path']
                dst = os.path.join(output_folder, os.path.basename(src))
                shutil.copy2(src, dst)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(self.segmented_objects)} objects to {file_path}\n"
                f"Images copied to: {output_folder}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting objects: {str(e)}")

    def update_object_size_filter(self, value):
        """Update the label and filter objects visually."""
        # Update label
        self.size_filter_label.setText(f"Min Object Size: {value} pixels")
        
        # Immediately update image display to reflect current filter
        self.update_image_display() 
    def load_model(self, path):
        """Load a trained model from disk."""
        if not path:
            return
            
        try:
            # Load model with TensorFlow Hub support
            self.model = tf.keras.models.load_model(
                path, 
                custom_objects={'KerasLayer': hub.KerasLayer}
            )
            
            # Try to load class names from associated file
            class_file = os.path.splitext(path)[0] + "_classes.txt"
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.class_names = f.read().splitlines()
                self.status_label.setText(
                    f"Model loaded with {len(self.class_names)} classes: {', '.join(self.class_names)}"
                )
            else:
                self.class_names = []
                self.status_label.setText("Model loaded but no class names file found.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))
            self.status_label.setText(f"Error loading model: {str(e)}")
