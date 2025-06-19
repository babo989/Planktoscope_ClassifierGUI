import sys
import tensorflow as tf
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                            QTableWidget, QTableWidgetItem, QFileDialog,
                            QGroupBox, QProgressBar, QTextEdit, QSplitter,
                            QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from pathlib import Path
import json

class ChainClassifierWidget(QWidget):
    """
    A PyQt5 widget for chain classification that integrates with the general planktoscope classifier GUI
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.classifier = None
        self.analyzer = None
        self.current_results = {}
        self.init_ui()

    def select_test_data_directory(self):
        """Select directory for analysis/testing"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Test/Analysis Data Directory"
        )
        if directory:
            self.test_data_path_label.setText(directory)
            self.log(f"Selected test data directory: {directory}")
            
            # Count samples
            count, _ = self.count_samples_and_classes(directory)
            self.log(f"Found {count} images for analysis")   


    def display_image(self, filepath):
        """Display an image in the preview area"""
        try:
            pixmap = QPixmap(filepath)
            if pixmap.isNull():
                self.image_label.setText("Error loading image")
                return
                
            # Scale to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.image_label.setText(f"Error: {str(e)}")

    def display_prediction_info(self, filepath, result, morphology):
        """Display prediction information in the side panel"""
        info_text = f"<h3>Image Analysis</h3>"
        info_text += f"<p><b>File:</b> {Path(filepath).name}</p>"
        info_text += "<hr>"
        
        info_text += f"<h4>Prediction</h4>"
        info_text += f"<p><b>Chain Length:</b> {result['chain_length']} cells</p>"
        info_text += f"<p><b>Confidence:</b> {result['chain_confidence']:.1%}</p>"
        info_text += f"<p><b>Is Chain:</b> {'Yes' if result['is_chain'] else 'No'} "
        info_text += f"({result['chain_probability']:.1%})</p>"
        
        # Show top 3 predictions for a comparison
        info_text += "<p><b>Top Predictions:</b></p><ul>"
        probs = result['length_probabilities']
        sorted_probs = sorted(enumerate(probs, 1), key=lambda x: x[1], reverse=True)[:3]
        for length, prob in sorted_probs:
            info_text += f"<li>{length} cells: {prob:.1%}</li>"
        info_text += "</ul>"
        
        if morphology:
            info_text += "<hr><h4>Morphology</h4>"
            info_text += f"<p><b>Detected Cells:</b> {morphology.get('detected_cells', 'N/A')}</p>"
            info_text += f"<p><b>Chain Length:</b> {morphology.get('skeleton_length', 0):.0f} px</p>"
            info_text += f"<p><b>Area:</b> {morphology.get('area', 0):.0f} px²</p>"
            info_text += f"<p><b>Eccentricity:</b> {morphology.get('eccentricity', 0):.3f}</p>"
        
        self.prediction_info.setHtml(info_text)

    def init_ui(self):
        """Initialize the UI"""
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Chain Classifier")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # Model controls
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNetV2", "EfficientNetB0", "Custom"])
        model_row.addWidget(self.model_combo)
        
        # Load/Save model buttons
        self.load_model_btn = QPushButton("Load Model")
        self.save_model_btn = QPushButton("Save Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.save_model_btn.clicked.connect(self.save_model)
        model_row.addWidget(self.load_model_btn)
        model_row.addWidget(self.save_model_btn)
        
        model_layout.addLayout(model_row)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Training controls
        training_group = QGroupBox("Training Configuration")
        training_layout = QVBoxLayout()
        
        # Data directory selection
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Training Data:"))
        self.data_path_label = QLabel("No directory selected")
        self.select_data_btn = QPushButton("Select Directory")
        self.select_data_btn.clicked.connect(self.select_data_directory)
        data_row.addWidget(self.data_path_label)
        data_row.addWidget(self.select_data_btn)
        training_layout.addLayout(data_row)
        
        # Training parameters
        params_layout = QHBoxLayout()
        
        # Epochs
        params_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        params_layout.addWidget(self.epochs_spin)
        
        # Batch size
        params_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        params_layout.addWidget(self.batch_spin)
        
        # Learning rate
        params_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        params_layout.addWidget(self.lr_spin)
        
        training_layout.addLayout(params_layout)
        
        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        training_layout.addWidget(self.train_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        training_layout.addWidget(self.progress_bar)
        
        training_group.setLayout(training_layout)
        main_layout.addWidget(training_group)
        
        # Analysis controls
        analysis_group = QGroupBox("Chain Analysis")
        analysis_layout = QVBoxLayout()

            # Add test data directory selection
        test_data_row = QHBoxLayout()
        test_data_row.addWidget(QLabel("Test/Analysis Data:"))
        self.test_data_path_label = QLabel("Use training data")
        self.select_test_data_btn = QPushButton("Select Test Directory")
        self.select_test_data_btn.clicked.connect(self.select_test_data_directory)
        test_data_row.addWidget(self.test_data_path_label)
        test_data_row.addWidget(self.select_test_data_btn)
        analysis_layout.addLayout(test_data_row)
        
        # Analysis buttons
        analysis_buttons = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Dataset")
        self.analyze_single_btn = QPushButton("Analyze Single Image")
        self.generate_report_btn = QPushButton("Generate Report")
        
        self.analyze_btn.clicked.connect(self.analyze_dataset)
        self.analyze_single_btn.clicked.connect(self.analyze_single_image)
        self.generate_report_btn.clicked.connect(self.generate_report)
        
        analysis_buttons.addWidget(self.analyze_btn)
        analysis_buttons.addWidget(self.analyze_single_btn)
        analysis_buttons.addWidget(self.generate_report_btn)
        analysis_layout.addLayout(analysis_buttons)
        
        analysis_group.setLayout(analysis_layout)
        main_layout.addWidget(analysis_group)
        
        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        
        # Create main splitter for image preview and the results
        main_splitter = QSplitter(Qt.Vertical)
        
        # Image preview section
        preview_group = QGroupBox("Image Preview")
        preview_layout = QHBoxLayout()
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setMaximumSize(800, 600)
        self.image_label.setScaledContents(False)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setText("No image loaded")
        
        # Prediction info panel
        self.prediction_info = QTextEdit()
        self.prediction_info.setMaximumWidth(300)
        self.prediction_info.setReadOnly(True)
        
        preview_layout.addWidget(self.image_label)
        preview_layout.addWidget(self.prediction_info)
        preview_group.setLayout(preview_layout)
        
        main_splitter.addWidget(preview_group)
        
        # Create horizontal splitter for results table and the log 
        results_splitter = QSplitter(Qt.Horizontal)

        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Image", "True Length", "Predicted Length", "Confidence", "Status"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_splitter.addWidget(self.results_table)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        results_splitter.addWidget(self.log_text)
        
        results_splitter.setSizes([600, 400])
        main_splitter.addWidget(results_splitter)
        
        results_layout.addWidget(main_splitter)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        self.setLayout(main_layout)
    
    def log(self, message):
        """Add message to log display"""
        self.log_text.append(message)
        
    def select_data_directory(self):
        """Select directory containing chain data"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Chain Data Directory"
        )
        if directory:
            self.data_path_label.setText(directory)
            self.log(f"Selected data directory: {directory}")
            
            # Count samples and detect number of classes
            sample_count, max_chain_length = self.count_samples_and_classes(directory)
            self.log(f"Found {sample_count} total images")
            self.log(f"Maximum chain length: {max_chain_length} cells")
            
            # Store max chain length for model building
            self.max_chain_length = max_chain_length
    
    def count_samples_and_classes(self, directory):
        """Count samples in the data directory (handles both training and test formats)"""
        count = 0
        max_chain_length = 0
        data_path = Path(directory)
        
        self.log(f"Looking for data in: {data_path}")
        
        # Check for training-style subdirectories
        has_subdirs = False
        for chain_dir in data_path.iterdir():
            if chain_dir.is_dir() and '_cell' in chain_dir.name.lower():
                has_subdirs = True
                try:
                    chain_length = int(chain_dir.name.split('_')[0])
                    max_chain_length = max(max_chain_length, chain_length)
                    
                    dir_count = 0
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                        files = list(chain_dir.glob(ext))
                        dir_count += len(files)
                    self.log(f"    {chain_dir.name}: {dir_count} images")
                    count += dir_count
                except ValueError:
                    self.log(f"    Skipping {chain_dir.name}: couldn't parse chain length")
        
        # If no subdirectories, count files in the main directory
        if not has_subdirs:
            self.log("  No class subdirectories found - counting all images in directory")
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                files = list(data_path.glob(ext))
                count += len(files)
            self.log(f"  Found {count} images")
        
        return count, max_chain_length
    
    def load_model(self):
        """Load a trained model"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.h5)"
        )
        if filepath:
            try:
                from chain_classifier import ChainClassifier
                self.classifier = ChainClassifier()
                self.classifier.load_model(filepath)
                self.log(f"Model loaded from: {filepath}")
                self.save_model_btn.setEnabled(True)
                QMessageBox.information(self, "Success", "Model loaded successfully!")
            except Exception as e:
                self.log(f"Error loading model: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def save_model(self):
        """Save the trained model"""
        if not self.classifier or not self.classifier.model:
            QMessageBox.warning(self, "Warning", "No model to save!")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "chain_classifier_model.h5", "Model Files (*.h5)"
        )
        if filepath:
            try:
                self.classifier.save_model(filepath.replace('.h5', ''))
                self.log(f"Model saved to: {filepath}")
                QMessageBox.information(self, "Success", "Model saved successfully!")
            except Exception as e:
                self.log(f"Error saving model: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
    
    def train_model(self):
        """Train the chain classifier model"""
        data_dir = self.data_path_label.text()
        if data_dir == "No directory selected":
            QMessageBox.warning(self, "Warning", "Please select a data directory first!")
            return
        
        # Use detected max chain length or default to 9
        num_classes = getattr(self, 'max_chain_length', 9)
        
        # Create training thread
        self.training_thread = TrainingThread(
            data_dir,
            self.epochs_spin.value(),
            self.batch_spin.value(),
            self.lr_spin.value(),
            num_classes
        )
        
        # Connect signals
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.log_message.connect(self.log)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        
        # Disable controls during training
        self.train_btn.setEnabled(False)
        self.select_data_btn.setEnabled(False)
        
        # Start training
        self.training_thread.start()
        self.log("Starting training...")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def training_finished(self, classifier):
        """Handle training completion"""
        self.classifier = classifier
        self.train_btn.setEnabled(True)
        self.select_data_btn.setEnabled(True)
        self.save_model_btn.setEnabled(True)
        self.log("Training completed successfully!")
        QMessageBox.information(self, "Success", "Model training completed!")
    
    def training_error(self, error_message):
        """Handle training error"""
        self.train_btn.setEnabled(True)
        self.select_data_btn.setEnabled(True)
        self.log(f"Training error: {error_message}")
        QMessageBox.critical(self, "Training Error", error_message)
    
    def analyze_dataset(self):
        """Analyze entire dataset - button click handler"""
        # Determine which directory to use
        data_dir = None
        
        # First check if test data directory is selected
        if hasattr(self, 'test_data_path_label'):
            test_dir_text = self.test_data_path_label.text()
            if test_dir_text != "Use training data":
                data_dir = test_dir_text
                self.log(f"Using test directory: {data_dir}")
        
        # If no test directory, use training directory
        if not data_dir:
            data_dir = self.data_path_label.text()
            self.log(f"Using training directory: {data_dir}")
        
        # Check if we have a valid directory
        if not data_dir or data_dir == "No directory selected":
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please select a data directory!\n\n" +
                "Use 'Select Test Directory' in Chain Analysis section\n" +
                "or 'Select Directory' in Training Configuration section."
            )
            return
        
        if not self.classifier:
            QMessageBox.warning(self, "Warning", "Please load or train a model first!")
            return
        
        try:
            from chain_analyzer import ChainAnalyzer
            self.analyzer = ChainAnalyzer(self.classifier)
            
            self.log(f"Starting dataset analysis on: {data_dir}")
            results_df = self.analyzer.analyze_dataset(data_dir)
            
            # Update results table
            self.update_results_table(results_df)
            
            # Calculate and display statistics
            stats = self.analyzer.calculate_statistics()
            self.display_statistics(stats)
            
            self.log("Dataset analysis completed!")
            
        except Exception as e:
            self.log(f"Analysis error: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Analysis Error", str(e))
    
    def analyze_single_image(self):
        """Analyze a single image"""
        if not self.classifier:
            QMessageBox.warning(self, "Warning", "Please load or train a model first!")
            return
            
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        
        if filepath:
            try:
                # Display the image preview
                self.display_image(filepath)
                
                # Make prediction
                result = self.classifier.predict(filepath)
                
                # Analyze morphology
                from chain_analyzer import ChainAnalyzer
                analyzer = ChainAnalyzer()
                morph_features = analyzer.analyze_chain_morphology(filepath)
                
                # Display prediction info in the side panel
                self.display_prediction_info(filepath, result, morph_features)
                
                # Also log to main log
                self.display_single_result(filepath, result, morph_features)
                
            except Exception as e:
                self.log(f"Error analyzing image: {str(e)}")
                QMessageBox.critical(self, "Error", str(e))
        
    def show_image_preview_window(self, filepath, prediction, morphology):
        """Show a popup window with image and results"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Analysis Results - {Path(filepath).name}")
        dialog.setModal(False)  # Non-blocking
        dialog.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Image display
        image_label = QLabel()
        pixmap = QPixmap(filepath)
        scaled_pixmap = pixmap.scaled(700, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)
        
        # Results text
        results_text = QLabel()
        results_text.setText(
            f"Predicted: {prediction['chain_length']} cells "
            f"(Confidence: {prediction['chain_confidence']:.1%})\n"
            f"Morphological cells: {morphology.get('detected_cells', 'N/A') if morphology else 'N/A'}"
        )
        results_text.setAlignment(Qt.AlignCenter)
        results_text.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(results_text)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.show()

    def update_results_table(self, results_df):
        """Update the results table with analysis data"""
        self.results_table.setRowCount(len(results_df))
        
        for idx, row in results_df.iterrows():
            # Image path (shortened style)
            path_item = QTableWidgetItem(Path(row['image_path']).name)
            self.results_table.setItem(idx, 0, path_item)
            
            # For flat directories, true length might be the predicted length
            true_length = row.get('chain_length', 'N/A')
            true_item = QTableWidgetItem(str(true_length))
            self.results_table.setItem(idx, 1, true_item)
            
            # Predicted length and confidence
            if 'predicted_length' in row:
                pred_item = QTableWidgetItem(str(row['predicted_length']))
                conf_item = QTableWidgetItem(f"{row['prediction_confidence']:.2%}")
                
                # Status - for test sets without true labels, just show prediction
                if true_length == 'N/A' or true_length == row['predicted_length']:
                    if true_length == 'N/A':
                        status = "Predicted"
                        status_item = QTableWidgetItem(status)
                        status_item.setForeground(Qt.blue)
                    else:
                        status = "✓ Correct"
                        status_item = QTableWidgetItem(status)
                        status_item.setForeground(Qt.green)
                else:
                    status = "✗ Incorrect"
                    status_item = QTableWidgetItem(status)
                    status_item.setForeground(Qt.red)
                
                self.results_table.setItem(idx, 2, pred_item)
                self.results_table.setItem(idx, 3, conf_item)
                self.results_table.setItem(idx, 4, status_item)
            else:
                # No prediction available
                self.results_table.setItem(idx, 2, QTableWidgetItem("N/A"))
                self.results_table.setItem(idx, 3, QTableWidgetItem("N/A"))
                self.results_table.setItem(idx, 4, QTableWidgetItem("No prediction"))
        
    def display_statistics(self, stats):
        """Display analysis statistics in the log"""
        self.log("\n" + "="*50)
        self.log("ANALYSIS STATISTICS")
        self.log("="*50)
        
        self.log(f"Total images: {stats['total_images']}")
        self.log("\nChain length distribution:")
        for length, count in sorted(stats['chain_length_distribution'].items()):
            self.log(f"  {length} cells: {count} images")
        
        self.log("\nAccuracy by chain length:")
        for length, length_stats in sorted(stats['by_chain_length'].items()):
            if 'accuracy' in length_stats:
                self.log(f"  {length} cells: {length_stats['accuracy']:.2%}")
    
    def display_single_result(self, filepath, prediction, morphology):
        """Display single image analysis results"""
        self.log("\n" + "="*50)
        self.log(f"SINGLE IMAGE ANALYSIS: {Path(filepath).name}")
        self.log("="*50)
        
        self.log(f"Predicted chain length: {prediction['chain_length']} cells")
        self.log(f"Confidence: {prediction['chain_confidence']:.2%}")
        self.log(f"Is chain: {prediction['is_chain']} (prob: {prediction['chain_probability']:.2%})")
        
        if morphology:
            self.log("\nMorphological features:")
            for key, value in morphology.items():
                if isinstance(value, float):
                    self.log(f"  {key}: {value:.3f}")
                else:
                    self.log(f"  {key}: {value}")
    
    def generate_report(self):
        """Generate analysis report"""
        if not self.analyzer or not hasattr(self.analyzer, 'results_df'):
            QMessageBox.warning(self, "Warning", "Please run dataset analysis first!")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "chain_analysis_report.txt", "Text Files (*.txt)"
        )
        
        if filepath:
            try:
                self.analyzer.generate_report(filepath)
                self.log(f"Report saved to: {filepath}")
                
                # Also generate plots
                plot_path = filepath.replace('.txt', '_plots.png')
                self.analyzer.plot_analysis_results(plot_path)
                self.log(f"Plots saved to: {plot_path}")
                
                QMessageBox.information(self, "Success", "Report generated successfully!")
                
            except Exception as e:
                self.log(f"Error generating report: {str(e)}")
                QMessageBox.critical(self, "Error", str(e))


class TrainingThread(QThread):
    """Thread for model training to keep GUI responsive just in case"""
    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(object)  # Emits the trained classifier
    error = pyqtSignal(str)
    
    def __init__(self, data_dir, epochs, batch_size, learning_rate, num_classes=9):
        super().__init__()
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
    def run(self):
        """Run the training process"""
        try:
            from chain_classifier import ChainClassifier
            import tensorflow as tf
            
            # Initialize classifier
            self.log_message.emit("Initializing classifier...")
            classifier = ChainClassifier()
            
            # Build model
            self.log_message.emit("Building model architecture...")
            classifier.build_model(num_classes=self.num_classes)
            
            # Compile model
            self.log_message.emit("Compiling model...")
            classifier.compile_model(learning_rate=self.learning_rate)
            
            # Custom callback to emit progress in case 
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, thread, total_epochs):
                    self.thread = thread
                    self.total_epochs = total_epochs
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress = int((epoch + 1) / self.total_epochs * 100)
                    self.thread.progress.emit(progress)
                    
                    # Log metrics
                    acc = logs.get('chain_length_accuracy', 0)
                    val_acc = logs.get('val_chain_length_accuracy', 0)
                    self.thread.log_message.emit(
                        f"Epoch {epoch+1}/{self.total_epochs} - "
                        f"Acc: {acc:.3f}, Val Acc: {val_acc:.3f}"
                    )
            
            # Create progress callback
            progress_callback = ProgressCallback(self, self.epochs)
            
            # Load data first to check if it's working nice 
            self.log_message.emit("Loading data...")
            images, chain_lengths, is_chains = classifier.load_data(self.data_dir)
            self.log_message.emit(f"Loaded {len(images)} images for training")
            
            # Split data manually since we need to pass the callback
            from sklearn.model_selection import train_test_split
            
            X_train, X_val, y_chain_train, y_chain_val, y_binary_train, y_binary_val = train_test_split(
                images, chain_lengths, is_chains, 
                test_size=0.2, 
                random_state=42,
                stratify=chain_lengths
            )
            
            self.log_message.emit(f"Training samples: {len(X_train)}")
            self.log_message.emit(f"Validation samples: {len(X_val)}")
            
            # Create other callbacks
            callbacks = [
                progress_callback,
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_chain_length_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_chain_classifier.h5',
                    monitor='val_chain_length_accuracy',
                    save_best_only=True
                )
            ]
            
            # Train model
            self.log_message.emit("Starting training...")
            history = classifier.model.fit(
                X_train,
                {'chain_length': y_chain_train, 'is_chain': y_binary_train},
                validation_data=(
                    X_val,
                    {'chain_length': y_chain_val, 'is_chain': y_binary_val}
                ),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            classifier.history = history
            
            # Emit the trained classifier
            self.finished.emit(classifier)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

# Integration function for existing GUI
def add_chain_classifier_tab(main_window):
    """
    Add chain classifier tab to existing GUI
    This function should be called from the main GUI setup
    """
    chain_widget = ChainClassifierWidget(main_window)
    main_window.tab_widget.addTab(chain_widget, "Chain Classifier")
    return chain_widget