from Embeddings import get_embeddings, mean_pooling

import pickle
import os
import torch
import torch.nn.functional as F

def import_feature_texts(feature_path):
    """
    Import feature texts from two files: 'feature.txt' and 'opposite.txt' in the given path.
    
    Args:
        feature_path (str): Path to the directory containing feature text files.
    
    Returns:
        tuple: (feature_texts, opposite_feature_texts) - Lists of text strings or None if files not found.
    
    Note:
        Expects 'feature.txt' and 'opposite.txt' files in the specified directory.
        Returns None for missing files and prints error messages.
    """

    # Construct file paths
    feature_file_path = os.path.join(feature_path, "feature.txt")
    opposite_file_path = os.path.join(feature_path, "opposite.txt") 
    
    # Read feature texts
    try:
        with open(feature_file_path, 'r') as f:
            feature_texts = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Feature file not found: {feature_file_path}")
        feature_texts = None
    
    # Read opposite feature texts
    try:
        with open(opposite_file_path, 'r') as f:
            opposite_feature_texts = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Opposite file not found: {opposite_file_path}")
        opposite_feature_texts = None
    
    return feature_texts, opposite_feature_texts


def get_mean_layer_embedding(model_name, texts, layer_to_steer, normalize=False):
    """
    Compute the mean embedding of a specific layer for a list of texts.
    
    Args:
        model_name (str): Name of the pre-trained model to use.
        texts (list): List of text strings to embed.
        layer_to_steer (int): Layer index to extract embeddings from.
        normalize (bool, optional): Whether to normalize the mean vector. Defaults to False.
    
    Returns:
        torch.Tensor: Mean embedding vector from the specified layer.
    """
    data = get_embeddings(model_name, texts, normalize=False)
    model_output, encoded_input = data[2], data[4]
    selected_layer = model_output.hidden_states[layer_to_steer]
    pooled = mean_pooling((selected_layer,), encoded_input["attention_mask"])
    mean_vec = torch.mean(pooled, dim=0)
    
    if normalize:
        mean_vec = F.normalize(mean_vec, p=2, dim=0)

    return mean_vec


def get_steering_vector(model_name, sample_texts, layer_to_steer, opposite_texts=None, normalize=False):
    """
    Compute the steering vector for a specific layer based on sample texts.
    
    Args:
        model_name (str): Name of the pre-trained model to use.
        sample_texts (list): List of feature text strings.
        layer_to_steer (int): Layer index to extract embeddings from.
        opposite_texts (list, optional): List of opposite feature texts for contrast. Defaults to None.
        normalize (bool, optional): Whether to normalize the final steering vector. Defaults to False.
    
    Returns:
        torch.Tensor: Steering vector for the specified feature and layer.
    
    Raises:
        ValueError: If sample_texts is None.
    
    Note:
        If opposite_texts provided, steering vector = feature_vector - opposite_vector.
    """
    if sample_texts is None:
        raise ValueError("Feature texts must not be None")
    
    feature_vec = get_mean_layer_embedding(model_name, sample_texts, layer_to_steer, normalize=normalize)
    
    if opposite_texts is not None: # Check if opposite_feature_text is not empty
        opposite_vec = get_mean_layer_embedding(model_name, opposite_texts, layer_to_steer, normalize=normalize)
        feature_vec = feature_vec - opposite_vec

    if normalize:
        feature_vec = F.normalize(feature_vec, p=2, dim=0)

    return feature_vec

def clear_steering_vectors_from_pkl(file_name, feature_name=None):
    """
    Clear steering vectors from a pkl file.
    
    Args:
        file_name (str): Path to the pkl file.
        feature_name (str, optional): If provided, clear only that specific feature. 
                                     If None, clear all features. Defaults to None.
    
    Note:
        Removes specified feature data or clears entire file if no feature specified.
        Handles FileNotFoundError gracefully.
    """
    try:
        with open(file_name, 'rb') as f:
            existing_data = pickle.load(f)
    except FileNotFoundError:
        print(f"File {file_name} not found. Nothing to clear.")
        return
    
    if feature_name:
        if feature_name in existing_data:
            del existing_data[feature_name]
            print(f"Cleared steering vectors for feature '{feature_name}' from {file_name}")
        else:
            print(f"Feature '{feature_name}' not found in {file_name}. Nothing to clear.")
            return
    else:
        # Clear all features
        existing_data.clear()
        print(f"Cleared all steering vectors from {file_name}")
    
    # Save the updated (possibly empty) data back to file
    with open(file_name, 'wb') as f:
        pickle.dump(existing_data, f)

def export_steering_vector_to_pkl(steering_vector, file_path, feature_name, layer_to_steer):
    """
    Export the steering vector to a pkl file with metadata.
    
    Args:
        steering_vector (torch.Tensor): The steering vector to export.
        file_path (str): Path where to save the pkl file.
        feature_name (str): Name of the feature for organization.
        layer_to_steer (int): Layer index for organization.
    
    Note:
        File structure: {feature_name: {layer: vector_tensor, layer2: vector_tensor2, ...}}.
        Preserves existing vectors in the file and adds/updates the specified one.
    """
    # Check if file already exists to preserve existing vectors
    try:
        with open(file_path, 'rb') as f:
            existing_data = pickle.load(f)
    except FileNotFoundError:
        existing_data = {}
    
    # Initialize feature if it doesn't exist
    if feature_name not in existing_data:
        existing_data[feature_name] = {}
    
    # Add or update the steering vector for this layer
    existing_data[feature_name][layer_to_steer] = steering_vector
    
    # Save back to file
    with open(file_path, 'wb') as f:
        pickle.dump(existing_data, f)
    
    print(f"Steering vector for '{feature_name}' (layer {layer_to_steer}) exported to {file_path}")

def import_steering_vector_from_pkl(file_path, feature_name=None, layer_to_steer=None):
    """
    Import steering vector(s) from a pkl file.
    
    Args:
        file_path (str): Path to the pkl file.
        feature_name (str, optional): If provided, return data for that specific feature. Defaults to None.
        layer_to_steer (int, optional): If provided with feature_name, return vector for that specific layer. Defaults to None.
    
    Returns:
        torch.Tensor, dict, or None: 
            - If both feature_name and layer_to_steer provided: returns the steering vector tensor
            - If only feature_name provided: returns dictionary of {layer: vector} for that feature  
            - If neither provided: returns the entire dictionary
            - Returns None if requested data not found
    
    Note:
        Displays available features and layers upon import.
        Handles various error conditions gracefully with informative messages.
    """
    with open(file_path, 'rb') as f:
        steering_data = pickle.load(f)
    
    print(f"Steering vectors imported from {file_path}")
    
    # Display available features and their layers
    available_info = []
    for feat_name, data in steering_data.items():
        if isinstance(data, dict):
            if data.keys() and all(isinstance(k, int) for k in data.keys()):
                # New format: {feature: {layer: vector}}
                layers = sorted(data.keys())
                available_info.append(f"'{feat_name}' (layers: {layers})")
            elif not data.keys():
                # Empty dict
                available_info.append(f"'{feat_name}' (empty - no layers)")
            else:
                available_info.append(f"'{feat_name}' (invalid format - non-integer keys)")
        else:
            available_info.append(f"'{feat_name}' (invalid format - not a dict)")
    
    print(f"Available steering vectors: {', '.join(available_info)}")
    
    if feature_name:
        if feature_name in steering_data:
            data = steering_data[feature_name]
            
            if isinstance(data, dict) and all(isinstance(k, int) for k in data.keys()):
                # New format: {layer: vector}
                if layer_to_steer is not None:
                    if layer_to_steer in data:
                        print(f"Returning steering vector for '{feature_name}' layer {layer_to_steer}")
                        return data[layer_to_steer]
                    else:
                        available_layers = sorted(data.keys())
                        print(f"Warning: Layer {layer_to_steer} not found for '{feature_name}'. Available layers: {available_layers}")
                        print(f"See Notebook 1")
                        print(f"1) Use get_steering_vector(...) to compute the steering vector for this layer")
                        print(f"2) Export it to {file_path} with export_steering_vector_to_pkl(...)\n")
                        raise KeyError(f"Layer {layer_to_steer} not found for feature '{feature_name}'.")
                else:
                    print(f"Returning all layers for '{feature_name}': {sorted(data.keys())}")
                    return data
            else:
                print(f"Error: '{feature_name}' has invalid format. Expected {{layer: vector}} structure.")
                return None
        else:
            print(f"Warning: '{feature_name}' not found in steering vectors")
            return None
    else:
        print("Returning all steering vector data")
        return steering_data


if __name__ == "__main__":
    # Example usage
    # Import feature texts
    feature = "War"
    layer_to_steer = 11


    feature_texts, opposite_feature_texts = import_feature_texts(f"../Features/{feature}")

    # Compute steering vector
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    steering_vector = get_steering_vector(model_name, feature_texts, layer_to_steer=layer_to_steer, opposite_texts=opposite_feature_texts, normalize=True)

    # Export steering vector to pkl
    export_steering_vector_to_pkl(steering_vector, "steering_vector.pkl", feature_name=feature, layer_to_steer=layer_to_steer)

    # Import steering vector from pkl - examples:
    # Get specific vector for specific layer
    #imported_steering_vector = import_steering_vector_from_pkl("steering_vector.pkl", feature_name="War", layer_to_steer=11)

    # Get all layers for a feature (returns {layer: vector} dict)
    #all_love_vectors = import_steering_vector_from_pkl("steering_vector.pkl", feature_name="Love")

    # Get everything (returns full structure)
    #all_data = import_steering_vector_from_pkl("steering_vector.pkl")

    # Clear steering vectors - examples:
    # Clear specific feature
    #clear_steering_vectors_from_pkl("steering_vector.pkl", feature_name="Love")

    # Clear all features (start completely fresh)
    #clear_steering_vectors_from_pkl("steering_vector.pkl")
