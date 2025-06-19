import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from pathlib import Path
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class ChainClassifier:
    """
    A classifier for detecting and counting cells in Ellie's Chaetoceros chains.
    Uses a pre-trained feature extractor and trains a custom head for:
    1. Chain detection (binary: chain vs single cell)
    2. Chain length classification (1-X cells)
    """
    
    def __init__(self, feature_extractor_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"):
        self.feature_extractor_url = feature_extractor_url
        self.img_size = 224
        self.model = None
        self.chain_encoder = LabelEncoder()
        self.history = None
        
    def build_model(self, num_classes=9):
        """
        Build the model architecture with:
        - Pre-trained feature extractor (frozen)
        - Custom classification head for chain length
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Pre-trained feature extractor
        feature_extractor = hub.KerasLayer(
            self.feature_extractor_url,
            input_shape=(self.img_size, self.img_size, 3),
            trainable=False,  # Freeze the feature extractor
            name='feature_extractor'
        )
        
        # Extract features
        features = feature_extractor(inputs)
        
        # Custom classification head
        x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(features)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layers
        # Chain length classification (1-9 cells)
        chain_length_output = tf.keras.layers.Dense(
            num_classes, 
            activation='softmax', 
            name='chain_length'
        )(x)
        
        # Optional: Binary chain detection (chain vs single)
        is_chain_output = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            name='is_chain'
        )(x)
        
        # Create model with multiple outputs
        self.model = tf.keras.Model(
            inputs=inputs,
            outputs=[chain_length_output, is_chain_output]
        )
        
        return self.model
    

    def load_data(self, data_dir, augment=True, verbose=True):
        """
        Load data from directory structure:
        data_dir/
        ├── 1_cell/
        ├── 2_cell/
        ├── 3_cell/
        ...
        └── N_cell/
        """
        images = []
        chain_lengths = []
        is_chains = []
        debug_info = []
        
        data_path = Path(data_dir)
        
        debug_info.append(f"Loading data from: {data_path}")
        
        # Load images from each subdirectory
        for chain_dir in data_path.iterdir():
            if chain_dir.is_dir() and '_cell' in chain_dir.name.lower():  # Case insensitive
                try:
                    chain_length = int(chain_dir.name.split('_')[0])
                    debug_info.append(f"Processing {chain_dir.name} (length: {chain_length})")
                    
                    image_count = 0
                    # Look for all image files
                    for img_path in chain_dir.glob('*'):
                        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            try:
                                # Load and preprocess image
                                img = Image.open(img_path).convert('RGB')
                                img = img.resize((self.img_size, self.img_size))
                                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                                
                                images.append(img_array)
                                chain_lengths.append(chain_length)
                                is_chains.append(1 if chain_length > 1 else 0)
                                image_count += 1
                            except Exception as e:
                                debug_info.append(f"Error loading {img_path}: {e}")
                    
                    debug_info.append(f"  Loaded {image_count} images from {chain_dir.name}")
                    
                except ValueError as e:
                    debug_info.append(f"Skipping {chain_dir.name}: couldn't parse chain length")
        
        debug_info.append(f"Total images loaded: {len(images)}")
        
        if verbose:
            for msg in debug_info:
                print(msg)
        
        if len(images) == 0:
            # Return debug info with the error
            raise ValueError(f"No images found! Debug info:\n" + "\n".join(debug_info))
        
        # Convert to numpy arrays
        images = np.array(images)
        chain_lengths = np.array(chain_lengths) - 1  # Convert to 0-indexed
        is_chains = np.array(is_chains)
        
        # Data augmentation function
        if augment:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomBrightness(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])
            
            # Apply augmentation to training data
            augmented_images = []
            augmented_lengths = []
            augmented_chains = []
            
            for i in range(len(images)):
                # Original image
                augmented_images.append(images[i])
                augmented_lengths.append(chain_lengths[i])
                augmented_chains.append(is_chains[i])
                
                # Generate augmented versions
                for _ in range(2):  # Create 2 augmented versions per image
                    aug_img = data_augmentation(tf.expand_dims(images[i], 0))
                    augmented_images.append(aug_img[0].numpy())
                    augmented_lengths.append(chain_lengths[i])
                    augmented_chains.append(is_chains[i])
            
            images = np.array(augmented_images)
            chain_lengths = np.array(augmented_lengths)
            is_chains = np.array(augmented_chains)
            

        return images, chain_lengths, is_chains
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate losses and metrics"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'chain_length': 'sparse_categorical_crossentropy',
                'is_chain': 'binary_crossentropy'
            },
            loss_weights={
                'chain_length': 1.0,
                'is_chain': 0.5  # Lower weight for auxiliary task
            },
            metrics={
                'chain_length': ['accuracy'],
                'is_chain': ['accuracy']
            }
        )
    
    def train(self, data_dir, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Load data
        print("Loading data...")
        images, chain_lengths, is_chains = self.load_data(data_dir)
        
        # Split data
        X_train, X_val, y_chain_train, y_chain_val, y_binary_train, y_binary_val = train_test_split(
            images, chain_lengths, is_chains, 
            test_size=validation_split, 
            random_state=42,
            stratify=chain_lengths
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create callbacks
        callbacks = [
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
        self.history = self.model.fit(
            X_train,
            {'chain_length': y_chain_train, 'is_chain': y_binary_train},
            validation_data=(
                X_val,
                {'chain_length': y_chain_val, 'is_chain': y_binary_val}
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Chain length accuracy
        axes[0, 0].plot(self.history.history['chain_length_accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_chain_length_accuracy'], label='Validation')
        axes[0, 0].set_title('Chain Length Classification Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Chain length loss
        axes[0, 1].plot(self.history.history['chain_length_loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_chain_length_loss'], label='Validation')
        axes[0, 1].set_title('Chain Length Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Binary classification accuracy
        axes[1, 0].plot(self.history.history['is_chain_accuracy'], label='Train')
        axes[1, 0].plot(self.history.history['val_is_chain_accuracy'], label='Validation')
        axes[1, 0].set_title('Chain Detection Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        # Total loss
        axes[1, 1].plot(self.history.history['loss'], label='Train')
        axes[1, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[1, 1].set_title('Total Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, image_path):
        """Predict chain length for a single image"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        # Make prediction
        chain_length_pred, is_chain_pred = self.model.predict(img_array)
        
        # Get predicted chain length
        predicted_length = np.argmax(chain_length_pred[0]) + 1
        chain_confidence = np.max(chain_length_pred[0])
        
        # Get chain detection result
        is_chain = is_chain_pred[0][0] > 0.5
        chain_prob = is_chain_pred[0][0]
        
        return {
            'chain_length': predicted_length,
            'chain_confidence': float(chain_confidence),
            'is_chain': bool(is_chain),
            'chain_probability': float(chain_prob),
            'length_probabilities': chain_length_pred[0].tolist()
        }
    
    def predict_batch(self, image_dir, output_file='predictions.json'):
        """Predict chain lengths for all images in a directory"""
        predictions = {}
        image_dir = Path(image_dir)
        
        for img_path in image_dir.glob('*'):  # Adjust extension as needed
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    result = self.predict(img_path)
                    predictions[str(img_path)] = result
                    print(f"Processed: {img_path.name} - Chain length: {result['chain_length']}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Save predictions
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        return predictions
    
    def save_model(self, filepath='chain_classifier_model'):
        """Save the trained model"""
        self.model.save(f'{filepath}.h5')
        print(f"Model saved to {filepath}.h5")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath, custom_objects={'KerasLayer': hub.KerasLayer})
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = ChainClassifier()
    
    # Build model
    model = classifier.build_model(num_classes=9)  # For chains of 1-9 cells
    print(model.summary())
    
    # Compile model
    classifier.compile_model(learning_rate=0.001)
    
    # Train model (adjust path to your data directory)
    # history = classifier.train(
    #     data_dir='path/to/your/chain_data',
    #     epochs=50,
    #     batch_size=32
    # )
    
    # Plot training history
    # classifier.plot_training_history()
    
    # Make predictions
    # result = classifier.predict('path/to/test/image.png')
    # print(f"Predicted chain length: {result['chain_length']} cells")
    # print(f"Confidence: {result['chain_confidence']:.2%}")