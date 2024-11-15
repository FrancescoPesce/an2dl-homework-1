import numpy as np
import tensorflow as tf
import re
import os

def average_weights(weight_files, loss_threshold):
    selected_weights = []
    
    # Filter files by loss threshold
    for weight_file in weight_files:
        match = re.match(r'(\d+)-(\d+\.\d+)\.keras', weight_file)
        if match:
            epoch, val_loss = int(match.group(1)), float(match.group(2))
            if val_loss < loss_threshold:
                selected_weights.append(weight_file)
    
    if not selected_weights:
        raise ValueError("No models found with validation loss below the threshold.")
    
    # Load one model to get the architecture
    base_model = tf.keras.models.load_model(selected_weights[0])
    initial_weights = base_model.get_weights()
    averaged_weights = [np.zeros_like(w) for w in initial_weights]

    # Accumulate weights from selected models
    for weight_file in selected_weights:
        model = tf.keras.models.load_model(weight_file)
        weights = model.get_weights()
        for i in range(len(averaged_weights)):
            averaged_weights[i] += weights[i]
    
    # Average the weights
    num_models = len(selected_weights)
    averaged_weights = [w / num_models for w in averaged_weights]

    return base_model, averaged_weights

def create_model_with_averaged_weights(base_model, averaged_weights):
    # Clone the base model to create a new instance with averaged weights
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(averaged_weights)
    return model

def save_model(model, filename="averaged_model.keras"):
    # Save the model to a .keras file
    model.save(filename)
    print(f"Model with averaged weights saved to {filename}")

# Example usage:
# Directory containing your weight files
weight_files = [f for f in os.listdir('.') if f.endswith('.keras')]
loss_threshold = 1.26  # Set your loss threshold

# Get the base model and averaged weights
base_model, averaged_weights = average_weights(weight_files, loss_threshold)

# Create a new model with the averaged weights
model = create_model_with_averaged_weights(base_model, averaged_weights)

# Save the model to a .keras file
save_model(model, "averaged_model.keras")
