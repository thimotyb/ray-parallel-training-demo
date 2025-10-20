"""
Baseline TensorFlow/Keras training script without Ray.
This script trains a simple CNN model on the MNIST dataset for fast demonstration.
"""

import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse


def create_model():
    """Create a lightweight CNN model for MNIST classification."""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model


def load_and_prepare_data():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    return (x_train, y_train), (x_test, y_test)


def train_model(epochs=5, batch_size=128, save_path='baseline_model.h5'):
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
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True
        )
    ]

    # Train model
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    print("="*60)

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

    # Get final training metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    epochs_completed = len(history.history['loss'])

    # Save model
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Prepare results
    results = {
        'training_type': 'baseline',
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'epochs_completed': epochs_completed,
        'batch_size': batch_size,
        'final_train_accuracy': float(final_train_acc),
        'final_val_accuracy': float(final_val_acc),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }

    # Save results to JSON
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Training Time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
    print(f"Epochs Completed: {epochs_completed}")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline MNIST training without Ray')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    args = parser.parse_args()

    results = train_model(epochs=args.epochs, batch_size=args.batch_size)
