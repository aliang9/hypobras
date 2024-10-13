import os
import cerebras
from cerebras.cloud.sdk import Cerebras

# Step 1: Initialize the Cerebras Inference Client
# Replace "your_api_key" with your actual Cerebras API key.
client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)

# Step 2: Define the prompt construction for the experiment
# This function generates prompts for the LLM based on the number of dimensions (features) of the objects.
def generate_prompt(object_features, label):
    """
    Generate a prompt describing an object with features.
    
    Args:
    - object_features (list): A list of tuples representing feature names and values.
    - label (str): The correct label for the object.
    
    Returns:
    - str: A formatted prompt string.
    """
    feature_description = ", ".join([f"{name}: {value}" for name, value in object_features])
    prompt = f"The object is described as {feature_description}. What is the associated label? The correct label is: {label}"
    return prompt

# Step 3: Define the experimental conditions
# We will test 3D and 5D spaces with random feature-value pairs.
import random

def generate_feature_space(num_features):
    """
    Generate random feature space for an object.
    
    Args:
    - num_features (int): Number of features describing the object.
    
    Returns:
    - list: A list of (feature_name, feature_value) tuples.
    """
    feature_names = ["color", "shape", "size", "material", "function"]
    feature_values = ["red", "blue", "green", "square", "round", "small", "large", "metallic", "plastic", "tool"]
    
    selected_features = random.sample(feature_names, num_features)
    object_features = [(feature, random.choice(feature_values)) for feature in selected_features]
    return object_features

# Step 4: Perform inference using the Cerebras API
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

# Step 5: Run the experiment
def run_experiment():
    """
    Run the word-learning experiment across different dimensionalities.
    """
    dimensions = [3, 5]  # Test with 3D and 5D spaces
    num_trials = 10
    
    for dim in dimensions:
        print(f"Running experiment for {dim}D space:")
        correct_predictions = 0
        
        for trial in range(num_trials):
            # Generate a random object description
            features = generate_feature_space(dim)
            label = f"label_{trial}"  # Use a simple label for demonstration
            
            # Create the prompt and run inference
            prompt = generate_prompt(features, label)
            prediction = run_inference_streaming(prompt)
            
            # Evaluate the model's performance
            if label in prediction:
                correct_predictions += 1
            
            print(f"Trial {trial + 1}/{num_trials}: Predicted '{prediction.strip()}', Actual '{label}'")
        
        # Calculate accuracy
        accuracy = correct_predictions / num_trials
        print(f"Accuracy for {dim}D space: {accuracy * 100:.2f}%\n")

# Step 6: Run the experiment
if __name__ == "__main__":
    run_experiment()