import os
from cerebras.cloud.sdk import Cerebras
import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from template import generate_feature_space, generate_prompt
from masking import generate_distractor_features, generate_distractor_prompt, generate_masked_prompt

# Step 1: Update Cerebras Client Initialization to Support Streaming
client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)

def run_inference_streaming(prompt, model_id="llama3.1-8b"):
    """
    Run streaming inference with the Cerebras model.
    
    Args:
    - prompt (str): The input prompt for the model.
    - model_id (str): Model ID to use for inference.
    
    Yields:
    - str: Streaming content from the model.
    """
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_id,
        stream=True,
    )
    response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
        print(content, end="")
    return response

# Step 2: Define a Function for Hierarchical Feature Generation
def generate_hierarchical_features(num_features, depth=2):
    """
    Generate a hierarchical feature space for an object.
    
    Args:
    - num_features (int): Number of top-level features.
    - depth (int): Number of hierarchical levels (sub-features).
    
    Returns:
    - list: A list of (feature_name, feature_value) tuples.
    """
    base_features = ["color", "shape", "material"]
    sub_features = ["brightness", "hue", "texture", "density"]
    
    # Limit the number of base features to the available features
    selected_base_features = random.sample(base_features, num_features)
    hierarchical_features = []
    
    # Create hierarchical features
    for base in selected_base_features:
        # Create sub-features for the base feature
        for _ in range(depth):
            sub_feature_name = f"{base}_{random.choice(sub_features)}"
            sub_feature_value = random.choice(["high", "low", "smooth", "rough", "light", "dark"])
            hierarchical_features.append((sub_feature_name, sub_feature_value))
    
    return hierarchical_features

# Step 3: Update Prompt Generation for Hierarchical Descriptions
def generate_hierarchical_prompt(object_features, label):
    """
    Generate a prompt with hierarchical features.
    
    Args:
    - object_features (list): A list of tuples representing hierarchical feature names and values.
    - label (str): The correct label for the object.
    
    Returns:
    - str: A formatted prompt string.
    """
    feature_description = "; ".join([f"{name}: {value}" for name, value in object_features])
    prompt = f"The object is described as having these characteristics: {feature_description}. What is the associated label? The correct label is: {label}"
    return prompt

# Step 4: Run the Hierarchical Learning Experiment
def run_hierarchical_experiment():
    """
    Run the word-learning experiment in hierarchical vs flat feature spaces.
    """
    conditions = ["flat", "hierarchical"]  # Testing both flat and hierarchical conditions
    num_trials = 10
    num_features = 3
    
    for condition in conditions:
        print(f"Running experiment under {condition} condition:")
        correct_predictions = 0
        
        for trial in range(num_trials):
            # Generate object features based on the condition
            if condition == "flat":
                features = generate_feature_space(num_features)
                prompt = generate_prompt(features, f"label_{trial}")
            else:
                features = generate_hierarchical_features(num_features)
                prompt = generate_hierarchical_prompt(features, f"label_{trial}")
            
            # Run streaming inference
            print(f"\nTrial {trial + 1}/{num_trials}")
            prediction = run_inference_streaming(prompt)
            
            # Evaluate the model's performance
            if f"label_{trial}" in prediction:
                correct_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / num_trials
        print(f"\nAccuracy under {condition} condition: {accuracy * 100:.2f}%\n")

def run_masked_experiment():
    """
    Run the word-learning experiment in hierarchical vs flat feature spaces.
    """
    conditions = ["hierarchical"]  # Testing both flat and hierarchical conditions
    num_trials = 10
    num_features = 3
    
    for condition in conditions:
        print(f"Running experiment under {condition} condition:")
        correct_predictions = 0
        
        for trial in range(num_trials):
            features = generate_hierarchical_features(num_features)
            features = generate_distractor_features(features)
            prompt = generate_masked_prompt(features, f"label_{trial}")
            
            # Run streaming inference
            print(f"\nTrial {trial + 1}/{num_trials}")
            prediction = run_inference_streaming(prompt)
            
            # Evaluate the model's performance
            if f"label_{trial}" in prediction:
                correct_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / num_trials
        print(f"\nAccuracy under {condition} condition: {accuracy * 100:.2f}%\n")

def plot_feature_representations(features, labels, method="pca", title="Feature Representations"):
        """
        Visualize feature representations using PCA or t-SNE.
        
        Args:
        - features (np.array): The high-dimensional feature vectors.
        - labels (list): The corresponding labels for each feature vector.
        - method (str): 'pca' or 'tsne' for dimensionality reduction.
        - title (str): The title of the plot.
        """
        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

        reduced_features = reducer.fit_transform(features)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Label Index")
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()


# Step 5: Execute the Experiment
if __name__ == "__main__":
    # run_hierarchical_experiment()
    run_masked_experiment()