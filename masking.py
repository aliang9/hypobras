import random

def generate_masked_prompt(object_features, label, mask_prob=0.3):
    """
    Generate a prompt with some features masked.
    
    Args:
    - object_features (list): A list of tuples representing feature names and values.
    - label (str): The correct label for the object.
    - mask_prob (float): Probability of masking each feature.
    
    Returns:
    - str: A formatted prompt string with some features masked.
    """
    masked_description = []
    for name, value in object_features:
        if random.random() < mask_prob:
            masked_description.append(f"{name}: [MASKED]")
        else:
            masked_description.append(f"{name}: {value}")
    
    feature_description = "; ".join(masked_description)
    prompt = f"The object is described as {feature_description}. What is the associated label? The correct label is: {label}"
    return prompt

def generate_distractor_features(object_features, num_distractors=2):
    """
    Add distractor features to the object description.
    
    Args:
    - object_features (list): A list of tuples representing the original features.
    - num_distractors (int): Number of distractor features to add.
    
    Returns:
    - list: A list of tuples with both original and distractor features.
    """
    distractor_names = ["noise_level", "random_color", "ambiance"]
    distractor_values = ["high", "low", "purple", "yellow", "warm", "cold"]
    
    distractors = [(random.choice(distractor_names), random.choice(distractor_values)) for _ in range(num_distractors)]
    return object_features + distractors

def generate_distractor_prompt(object_features, label):
    """
    Generate a prompt with distractor features.
    
    Args:
    - object_features (list): A list of tuples representing features with added distractors.
    - label (str): The correct label for the object.
    
    Returns:
    - str: A formatted prompt string with distractors.
    """
    feature_description = "; ".join([f"{name}: {value}" for name, value in object_features])
    prompt = f"The object is described as having these characteristics: {feature_description}. What is the associated label? The correct label is: {label}"
    return prompt