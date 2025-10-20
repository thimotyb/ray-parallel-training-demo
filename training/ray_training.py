"""
Ray distributed training script using Ray Train with TensorFlow/Keras.
This script demonstrates parallel training across multiple workers using MNIST.
"""

import os
import time
import json
import argparse
import tensorflow as tf

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.tensorflow import TensorflowTrainer


def create_model():
    """Create a lightweight CNN model for MNIST classification."""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model


def load_and_prepare_data():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def train_func_per_worker(config):
    """Training function that runs on each Ray worker."""
    epochs = config['epochs']
    batch_size = config['batch_size']

    # Setup MultiWorkerMirroredStrategy for distributed training
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()

    # Adjust batch size for distributed training
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(global_batch_size)

    # Create model within strategy scope
    with strategy.scope():
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # Train the model
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training loop
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y_batch, predictions)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if step % 50 == 0:
                print(f"  Step {step}, Loss: {loss.numpy():.4f}")

        # Evaluate on a subset
        val_split = int(0.1 * len(x_test))
        val_loss, val_accuracy = model.evaluate(
            x_test[:val_split], y_test[:val_split], verbose=0
        )

        # Report metrics to Ray
        train.report({
            'epoch': epoch + 1,
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy)
        })

    # Final evaluation
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    return {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }


def run_ray_training(num_workers=2, epochs=5, batch_size=128, ray_address='auto'):
    """Run distributed training with Ray."""
    print("\n" + "="*60)
    print("RAY DISTRIBUTED TRAINING")
    print("="*60 + "\n")

    # Initialize Ray
    if ray_address:
        print(f"Connecting to Ray cluster at {ray_address}...")
        ray.init(address=ray_address)
    else:
        print("Starting local Ray cluster...")
        ray.init()

    # Get cluster info
    cluster_resources = ray.cluster_resources()
    print(f"Ray cluster resources: {cluster_resources}")
    print(f"Number of workers: {num_workers}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}\n")

    # Disable default Tune loggers that require tensorboardX (not bundled in Ray image)
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    # Create trainer
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config={
            'epochs': epochs,
            'batch_size': batch_size
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=False
        )
    )

    # Train
    print("Starting distributed training...")
    print("="*60)

    start_time = time.time()
    result = trainer.fit()
    training_time = time.time() - start_time

    # Get results
    metrics = result.metrics
    test_accuracy = metrics.get('test_accuracy', 0.0)
    test_loss = metrics.get('test_loss', 0.0)
    val_accuracy = metrics.get('val_accuracy', 0.0)

    # Prepare results
    results = {
        'training_type': 'ray_distributed',
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'epochs_completed': epochs,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'ray_cluster_cpus': cluster_resources.get('CPU', 0),
        'ray_cluster_nodes': len(ray.nodes()),
        'final_val_accuracy': float(val_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }

    # Save results
    with open('ray_results_2workers.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("RAY TRAINING COMPLETE")
    print("="*60)
    print(f"Training Time: {training_time:.2f}s ({training_time/60:.2f} minutes)")
    print(f"Number of Workers: {num_workers}")
    print(f"Epochs Completed: {epochs}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("="*60 + "\n")

    ray.shutdown()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ray distributed MNIST training')
    parser.add_argument('--ray-address', type=str, default='auto',
                       help='Ray cluster address (default: auto)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of training workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size per worker (default: 128)')
    args = parser.parse_args()

    results = run_ray_training(
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ray_address=args.ray_address
    )
