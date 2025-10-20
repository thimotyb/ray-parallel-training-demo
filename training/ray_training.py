"""
Ray distributed training script using Ray Train with TensorFlow/Keras.
This script demonstrates parallel training across multiple workers.
"""

import time
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.tensorflow import TensorflowTrainer


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
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def train_func_per_worker(config):
    """
    Training function that runs on each Ray worker.
    This function is executed in parallel across all workers.
    """
    epochs = config['epochs']
    batch_size = config['batch_size']

    # Load data on each worker
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()

    # Create TensorFlow MirroredStrategy for multi-worker training
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        # Create and compile model
        model = create_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # Prepare TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Split validation data
    val_size = int(len(x_train) * 0.1)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train[:val_size], y_train[:val_size])
    ).batch(batch_size)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train[val_size:], y_train[val_size:])
    ).batch(batch_size)

    # Custom training loop with Ray Train reporting
    for epoch in range(epochs):
        print(f"Worker {train.get_context().get_world_rank()}: Epoch {epoch + 1}/{epochs}")

        # Training
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        for batch_x, batch_y in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = model.compiled_loss(batch_y, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss += loss.numpy()
            train_acc += tf.keras.metrics.categorical_accuracy(
                batch_y, predictions
            ).numpy().mean()
            num_batches += 1

        train_loss /= num_batches
        train_acc /= num_batches

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = 0

        for batch_x, batch_y in val_dataset:
            predictions = model(batch_x, training=False)
            loss = model.compiled_loss(batch_y, predictions)
            val_loss += loss.numpy()
            val_acc += tf.keras.metrics.categorical_accuracy(
                batch_y, predictions
            ).numpy().mean()
            num_val_batches += 1

        val_loss /= num_val_batches
        val_acc /= num_val_batches

        # Report metrics to Ray Train
        train.report({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc)
        })

    # Final evaluation on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    return {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }


def train_with_ray(num_workers=2, epochs=20, batch_size=128):
    """Train the model using Ray Train with multiple workers."""
    print("\n" + "="*60)
    print(f"RAY DISTRIBUTED TRAINING ({num_workers} Workers)")
    print("="*60 + "\n")

    # Display Ray cluster info
    print("Ray Cluster Information:")
    print(f"  Available nodes: {len(ray.nodes())}")
    print(f"  Available CPUs: {ray.cluster_resources().get('CPU', 0)}")
    print(f"  Available memory: {ray.cluster_resources().get('memory', 0) / (1024**3):.2f} GB")
    print()

    # Configure scaling
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=False,  # Set to True if GPUs are available
        resources_per_worker={"CPU": 1}
    )

    # Configure training run
    run_config = RunConfig(
        name="ray_cifar10_training",
    )

    # Create TensorFlow trainer
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config={
            'epochs': epochs,
            'batch_size': batch_size
        },
        scaling_config=scaling_config,
        run_config=run_config
    )

    # Train the model
    print(f"Starting distributed training with {num_workers} workers...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print()

    start_time = time.time()
    result = trainer.fit()
    training_time = time.time() - start_time

    # Get final metrics
    final_metrics = result.metrics

    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Number of Workers: {num_workers}")
    print(f"Final Training Accuracy: {final_metrics.get('train_accuracy', 'N/A'):.4f}")
    print(f"Final Validation Accuracy: {final_metrics.get('val_accuracy', 'N/A'):.4f}")
    print(f"Test Accuracy: {final_metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"Test Loss: {final_metrics.get('test_loss', 'N/A'):.4f}")
    print("="*60 + "\n")

    # Save results
    results = {
        'training_type': 'ray_distributed',
        'num_workers': num_workers,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'epochs_completed': epochs,
        'final_train_accuracy': float(final_metrics.get('train_accuracy', 0)),
        'final_val_accuracy': float(final_metrics.get('val_accuracy', 0)),
        'test_accuracy': float(final_metrics.get('test_accuracy', 0)),
        'test_loss': float(final_metrics.get('test_loss', 0)),
        'batch_size': batch_size,
        'ray_cluster_cpus': ray.cluster_resources().get('CPU', 0),
        'ray_cluster_nodes': len(ray.nodes())
    }

    with open(f'ray_results_{num_workers}workers.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to ray_results_{num_workers}workers.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ray distributed training')
    parser.add_argument('--ray-address', type=str, default='auto',
                       help='Ray cluster address (use "auto" for local or provide head node address)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of workers for distributed training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')

    args = parser.parse_args()

    # Initialize Ray
    print("Initializing Ray...")
    if args.ray_address == 'auto':
        ray.init()
    else:
        ray.init(address=args.ray_address)

    try:
        train_with_ray(
            num_workers=args.num_workers,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    finally:
        ray.shutdown()
        print("Ray shutdown complete.")
