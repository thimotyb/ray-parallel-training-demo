"""
Baseline TensorFlow/Keras training script without Ray.
This script trains a CNN model on the CIFAR-10 dataset.
"""

import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse


def create_model():
    """Create a CNN model for CIFAR-10 classification."""
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Second convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model


def load_and_prepare_data():
    """Load and preprocess CIFAR-10 dataset."""
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    return (x_train, y_train), (x_test, y_test)


def train_model(epochs=20, batch_size=128, save_path='baseline_model.h5'):
    """Train the model and return training time and results."""
    print("\n" + "="*60)
    print("BASELINE TRAINING (Without Ray)")
    print("="*60 + "\n")

    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()

    # Create model
    print("\nBuilding model...")
    model = create_model()

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ]

    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Save model
    if save_path:
        print(f"\nSaving model to {save_path}...")
        model.save(save_path)

    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("="*60 + "\n")

    # Save results to JSON
    results = {
        'training_type': 'baseline',
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'epochs_completed': len(history.history['accuracy']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'batch_size': batch_size
    }

    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to baseline_results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline training without Ray')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--save-path', type=str, default='baseline_model.h5',
                       help='Path to save the trained model')

    args = parser.parse_args()

    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
