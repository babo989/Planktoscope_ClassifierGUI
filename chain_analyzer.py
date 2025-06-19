import numpy as np
import pandas as pd
from collections import defaultdict
import cv2
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ChainAnalyzer:
    """
     Analyzer for cell chains that provides:
    - Chain detection and cell counting
    - Basic Statistical analysis of the chain distributions
    - Morphological features of the chains
    - Visualization tools for seeing the nice chains 
    """
    
    def __init__(self, classifier_model=None):
        self.classifier = classifier_model
        self.results = defaultdict(list)
        
    def analyze_chain_morphology(self, image_path):
        """
        Analyze morphological features of a chain including:
        - Chain length (in pixels)
        - Chain curvature
        - Cell spacing
        - Chain orientation
        """
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Binary threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up the binary image
        cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=50)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
        
        # Find the contours
        contours, _ = cv2.findContours(
            cleaned.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get the largest contour (assuming it's the one with the chain)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate features
        features = {}
        
        # Fit ellipse to get orientation
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            features['orientation'] = ellipse[2]  # Angle in degrees
            features['major_axis'] = max(ellipse[1])
            features['minor_axis'] = min(ellipse[1])
            features['eccentricity'] = np.sqrt(1 - (features['minor_axis'] / features['major_axis'])**2)
        
        # Calculate chain length along the skeleton
        skeleton = morphology.skeletonize(cleaned)
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) > 1:
            # Calculate total length along skeleton
            distances = np.sqrt(np.sum(np.diff(skeleton_points, axis=0)**2, axis=1))
            features['skeleton_length'] = np.sum(distances)
        
        # Area and perimeter
        features['area'] = cv2.contourArea(largest_contour)
        features['perimeter'] = cv2.arcLength(largest_contour, True)
        features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter']**2)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        features['aspect_ratio'] = w / h
        features['bbox_area'] = w * h
        features['extent'] = features['area'] / features['bbox_area']
        
        # Detect individual cells within the chain
        distance_transform = cv2.distanceTransform(cleaned.astype(np.uint8), cv2.DIST_L2, 5)
        local_maxima = morphology.h_maxima(distance_transform, h=5)
        markers = ndimage.label(local_maxima)[0]
        features['detected_cells'] = markers.max()
        
        return features
    
    def analyze_dataset(self, data_dir):
        """
        Analyze an entire dataset of chain plankton images
        Supports two modes:
        1. Training-style: organized by chain length subdirectories
        2. Testing-style: flat directory of images (not subdivided)
        """
        data_path = Path(data_dir)
        results = {
            'chain_length': [],
            'image_path': [],
            'features': []
        }
        
        # Check if this is a training-style directory (with the subdirectories)
        has_subdirs = any(d.is_dir() and ('_cell' in d.name.lower() or d.name.isdigit()) for d in data_path.iterdir())
        
        if has_subdirs:
            # Training-style directory structure
            for chain_dir in sorted(data_path.iterdir()):
                if chain_dir.is_dir():
                    # Handle both "1_cell" and "1" naming conventions
                    if '_cell' in chain_dir.name.lower():
                        try:
                            chain_length = int(chain_dir.name.split('_')[0])
                        except ValueError:
                            continue
                    elif chain_dir.name.isdigit():
                        chain_length = int(chain_dir.name)
                    else:
                        continue
                    
                    for img_path in chain_dir.glob('*'):
                        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            # Get morphological features
                            morph_features = self.analyze_chain_morphology(img_path)
                            
                            if morph_features:
                                results['chain_length'].append(chain_length)
                                results['image_path'].append(str(img_path))
                                results['features'].append(morph_features)
                                
                                # If classifier is available, get predictions
                                if self.classifier:
                                    pred = self.classifier.predict(img_path)
                                    morph_features['predicted_length'] = pred['chain_length']
                                    morph_features['prediction_confidence'] = pred['chain_confidence']
        else:
            # Flat directory
            for img_path in data_path.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Get morphological features
                    morph_features = self.analyze_chain_morphology(img_path)
                    
                    if morph_features and self.classifier:
                        # Get prediction
                        pred = self.classifier.predict(img_path)
                        
                        # Use predicted length as the "true" length for the display
                        results['chain_length'].append(pred['chain_length'])
                        results['image_path'].append(str(img_path))
                        results['features'].append(morph_features)
                        
                        morph_features['predicted_length'] = pred['chain_length']
                        morph_features['prediction_confidence'] = pred['chain_confidence']
        
        # Convert to pandas DataFrame for easier analysis
        df_data = []
        for i in range(len(results['chain_length'])):
            row = {
                'chain_length': results['chain_length'][i],
                'image_path': results['image_path'][i]
            }
            row.update(results['features'][i])
            df_data.append(row)
        
        self.results_df = pd.DataFrame(df_data)
        return self.results_df
    
    def calculate_statistics(self):
        """
        Calculate statistics for the analyzed dataset
        """
        if not hasattr(self, 'results_df') or self.results_df is None or self.results_df.empty:
            print("No data to analyze. Run analyze_dataset first.")
            return {
                'total_images': 0,
                'chain_length_distribution': {},
                'by_chain_length': {}
            }
        
        stats = {}
        
        # Overall statistics
        stats['total_images'] = len(self.results_df)
        stats['chain_length_distribution'] = self.results_df['chain_length'].value_counts().to_dict()
        
        # Statistics by chain length
        grouped = self.results_df.groupby('chain_length')
        
        stats['by_chain_length'] = {}
        for chain_length, group in grouped:
            length_stats = {
                'count': len(group),
                'avg_skeleton_length': group['skeleton_length'].mean() if 'skeleton_length' in group else None,
                'avg_area': group['area'].mean(),
                'avg_eccentricity': group['eccentricity'].mean() if 'eccentricity' in group else None,
                'avg_detected_cells': group['detected_cells'].mean() if 'detected_cells' in group else None
            }
            
            # If predictions are available
            if 'predicted_length' in group:
                correct_predictions = (group['predicted_length'] == chain_length).sum()
                length_stats['accuracy'] = correct_predictions / len(group)
                length_stats['avg_confidence'] = group['prediction_confidence'].mean()
            
            stats['by_chain_length'][chain_length] = length_stats
        
        # Correlation analysis
        if 'skeleton_length' in self.results_df and 'detected_cells' in self.results_df:
            stats['length_cell_correlation'] = self.results_df[['skeleton_length', 'detected_cells']].corr().iloc[0, 1]
        
        return stats
    
    def plot_analysis_results(self, save_path=None):
        """
        Create a nice visualization of analysis results for exporting 
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No data to plot. Run analyze_dataset first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Chain length distribution
        ax = axes[0, 0]
        chain_counts = self.results_df['chain_length'].value_counts().sort_index()
        ax.bar(chain_counts.index, chain_counts.values)
        ax.set_xlabel('Chain Length (cells)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Chain Lengths')
        
        # 2. Skeleton length vs chain length
        if 'skeleton_length' in self.results_df:
            ax = axes[0, 1]
            self.results_df.boxplot(column='skeleton_length', by='chain_length', ax=ax)
            ax.set_xlabel('Chain Length (cells)')
            ax.set_ylabel('Skeleton Length (pixels)')
            ax.set_title('Chain Physical Length by Cell Count')
        
        # 3. Eccentricity vs chain length
        if 'eccentricity' in self.results_df:
            ax = axes[0, 2]
            self.results_df.boxplot(column='eccentricity', by='chain_length', ax=ax)
            ax.set_xlabel('Chain Length (cells)')
            ax.set_ylabel('Eccentricity')
            ax.set_title('Chain Eccentricity by Cell Count')
        
        # 4. Area vs chain length
        ax = axes[1, 0]
        self.results_df.boxplot(column='area', by='chain_length', ax=ax)
        ax.set_xlabel('Chain Length (cells)')
        ax.set_ylabel('Area (pixels²)')
        ax.set_title('Chain Area by Cell Count')
        
        # 5. Prediction accuracy by chain length (if available)
        if 'predicted_length' in self.results_df:
            ax = axes[1, 1]
            accuracies = []
            lengths = []
            for length in sorted(self.results_df['chain_length'].unique()):
                subset = self.results_df[self.results_df['chain_length'] == length]
                accuracy = (subset['predicted_length'] == length).mean()
                accuracies.append(accuracy)
                lengths.append(length)
            
            ax.bar(lengths, accuracies)
            ax.set_xlabel('Chain Length (cells)')
            ax.set_ylabel('Classification Accuracy')
            ax.set_title('Model Accuracy by Chain Length')
            ax.set_ylim([0, 1.1])
            
            # Add accuracy values on bars
            for i, (length, acc) in enumerate(zip(lengths, accuracies)):
                ax.text(i, acc + 0.02, f'{acc:.2f}', ha='center')
        
        # 6. Detected cells vs the actual chain length
        if 'detected_cells' in self.results_df:
            ax = axes[1, 2]
            scatter_data = self.results_df.groupby('chain_length')['detected_cells'].apply(list)
            for length, detected_list in scatter_data.items():
                ax.scatter([length] * len(detected_list), detected_list, alpha=0.5, s=30)
            
            # Add a 'perfect' detection line
            max_length = self.results_df['chain_length'].max()
            ax.plot([1, max_length], [1, max_length], 'r--', label='Perfect detection')
            ax.set_xlabel('Actual Chain Length')
            ax.set_ylabel('Detected Cells')
            ax.set_title('Cell Detection Accuracy')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_file='chain_analysis_report.txt'):
        """
        Generate a comprehensive text report of the analysis
        """
        stats = self.calculate_statistics()
        
        with open(output_file, 'w') as f:
            f.write("Chain Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total images analyzed: {stats['total_images']}\n\n")
            
            f.write("Chain Length Distribution:\n")
            for length, count in sorted(stats['chain_length_distribution'].items()):
                f.write(f"  {length} cells: {count} images\n")
            f.write("\n")
            
            f.write("Statistics by Chain Length:\n")
            for length, length_stats in sorted(stats['by_chain_length'].items()):
                f.write(f"\n{length} cells:\n")
                f.write(f"  Count: {length_stats['count']}\n")
                if length_stats['avg_skeleton_length']:
                    f.write(f"  Avg skeleton length: {length_stats['avg_skeleton_length']:.1f} px\n")
                f.write(f"  Avg area: {length_stats['avg_area']:.1f} px²\n")
                if length_stats['avg_eccentricity']:
                    f.write(f"  Avg eccentricity: {length_stats['avg_eccentricity']:.3f}\n")
                if 'accuracy' in length_stats:
                    f.write(f"  Classification accuracy: {length_stats['accuracy']:.2%}\n")
                    f.write(f"  Avg confidence: {length_stats['avg_confidence']:.2%}\n")
            
            if 'length_cell_correlation' in stats:
                f.write(f"\nCorrelation between skeleton length and detected cells: {stats['length_cell_correlation']:.3f}\n")
        
        print(f"Report saved to {output_file}")
    
    def confusion_matrix(self, save_path=None):
        """
        Create confusion matrix for chain length predictions
        """
        if 'predicted_length' not in self.results_df:
            print("No predictions available for confusion matrix")
            return
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Calculate confusion matrix
        cm = confusion_matrix(
            self.results_df['chain_length'], 
            self.results_df['predicted_length']
        )
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Chain Length')
        plt.ylabel('Actual Chain Length')
        plt.title('Chain Length Classification Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ChainAnalyzer()
    
    # Analyze dataset
    # df = analyzer.analyze_dataset('path/to/chain_data')
    
    # Calculate statistics
    # stats = analyzer.calculate_statistics()
    # print(json.dumps(stats, indent=2))
    
    # Generate visualizations
    # analyzer.plot_analysis_results('chain_analysis_plots.png')
    
    # Generate report
    # analyzer.generate_report()
    
    # Show confusion matrix
    # analyzer.confusion_matrix('confusion_matrix.png')