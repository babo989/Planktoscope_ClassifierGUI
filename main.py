#!/usr/bin/env python
"""
PLanktoscope Classifier - Main Application
-----------------------------------------------------------
A dual-mode GUI application for training and deploying image classification models
on Planktoscope images. This application provides an interface for both raw Planktoscope
images and segmented objects from the built-in segmentation function, with tools for labeling, training, and classification.

The application uses a PyQt5 tabbed interface with two main panels:
1. Raw Images Panel: For processing and classifying raw Planktoscope images
2. Segmented Objects Panel: For training and classifying individual segmented objects

This is the main entry point of the application that sets up the UI.

Author: [Adam Larson]
Date: [4.1.2025]
Version: 1.0

History:
[Daniel Elnatan] 4.10.2026 - swapped to using qtpy
"""

import sys
import os

# DEPRECATED
# from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QAction, QFileDialog, QMessageBox
from gui_integration import ChainClassifierWidget

from qtpy.QtWidgets import QApplication, QMainWindow, QTabWidget, QAction, QFileDialog, QMessageBox

from raw_mode import RawModePanel
from segmented_mode import SegmentedModePanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planktoscope Classifier - Dual Mode")
        self.setGeometry(100, 100, 1000, 600)

        # Create the central tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Instantiate two mode panels for raw and pre-segmented
        self.raw_mode_panel = RawModePanel(self)
        self.segmented_mode_panel = SegmentedModePanel(self)

        # Add them as tabs
        self.tab_widget.addTab(self.raw_mode_panel, "Raw Images")
        self.tab_widget.addTab(self.segmented_mode_panel, "Segmented Objects")

        self.chain_classifier_widget = ChainClassifierWidget()
        self.tab_widget.addTab(self.chain_classifier_widget, "Chain Classifier")

        # Menu for global actions
        self._create_menus()

    def _create_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        # Set classification folder for Planktoscope images
        set_class_dir_action = QAction("Set Classification Folder", self)
        set_class_dir_action.triggered.connect(self._on_set_class_folder)
        file_menu.addAction(set_class_dir_action)

        # Load model
        load_model_action = QAction("Load Trained Model", self)
        load_model_action.triggered.connect(self._on_load_model)
        file_menu.addAction(load_model_action)

        # Exit
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _on_set_class_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Classification Folder")
        if folder:
            self.segmented_mode_panel.set_classification_folder(folder)

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "H5 or SavedModel (*.h5 *.pb *.savedmodel)")
        if path:
            self.raw_mode_panel.load_model(path)
            self.segmented_mode_panel.load_model(path)
            QMessageBox.information(self, "Model Loaded", f"Model successfully loaded from:\n{path}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
