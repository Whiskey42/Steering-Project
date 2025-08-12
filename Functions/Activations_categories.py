from Steering_vector import import_steering_vector_from_pkl
from Embeddings import import_embedding_data_from_pkl
from Steering import create_steered_model_output
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_mean_of_categories(embeddings, all_texts_data):
    """
    Calculate the mean embedding vector for each genre category.
    
    Args:
        embeddings (torch.Tensor): Embedding vectors of shape (num_texts, embedding_dim).
        all_texts_data (pd.DataFrame): Text metadata with 'genre', 'title', 'overview' columns.
    
    Returns:
        dict: Genre names mapped to mean embedding vectors (torch.Tensor).
    """
    category_mean = {} # Can be either embeddings or pooled_hidden_states
    for idx, row in enumerate(all_texts_data.itertuples()):
        category = row.genre  # Use the genre column
        if category not in category_mean:
            category_mean[category] = []
        category_mean[category].append(embeddings[idx])
    
    for category in category_mean:
        category_mean[category] = torch.mean(torch.stack(category_mean[category]), dim=0)
    
    return category_mean

def get_category_activations(model_output, encoded_input, all_texts_data, layer_index):
    """
    Extract category activations from the hidden states of a specific layer.
    
    Args:
        model_output: The output from model(**encoded_input, output_hidden_states=True).
        encoded_input (dict): Tokenized input with attention_mask.
        all_texts_data (pd.DataFrame): Text metadata with 'genre', 'title', 'overview' columns.
        layer_index (int): Which layer's hidden state to extract.
    
    Returns:
        dict: Genre names mapped to mean activation vectors (torch.Tensor).
    """
    # Get hidden states from the specified layer
    hidden_states = model_output.hidden_states  # tuple of (layer_num, batch_size, seq_len, hidden_size)
    layer_hidden_states = hidden_states[layer_index]  # (batch_size, seq_len, hidden_size)
    
    # Apply mean pooling to get sentence-level representations
    # This is the same pooling that was used to create the embeddings
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_states.size()).float()
    pooled_hidden_states = torch.sum(layer_hidden_states * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    # Use the existing function to group by category and calculate means
    category_activations = calculate_mean_of_categories(pooled_hidden_states, all_texts_data)

    return category_activations

def plot_category_activations(category_layer_activations, category, info_string="Info"):
    """
    Plot the activations for a specific category in a specific layer.
    
    Args:
        category_layer_activations (dict): Genre names mapped to activation vectors.
        category (str): The category/genre to visualize.
        info_string (str, optional): Title for the plot. Defaults to "Info".
    
    Returns:
        str: "quit" if category not found, None otherwise.
    """
    category_layer_activations
    chosen_category = category  # Change this to the category you want to visualize
    # Ensure the chosen category exists in the activations
    if chosen_category not in category_layer_activations:
        print(f"Category '{chosen_category}' not found in activations.")
        print("Available categories:", list(category_layer_activations.keys()))
        return "quit"
    neurons = list(range(len(category_layer_activations[chosen_category])))
    activations = category_layer_activations[chosen_category].detach().numpy()
    fig = go.Figure(data=go.Bar(x=neurons, y=activations, name=chosen_category, width=0.8))
    fig.update_layout(title=info_string, 
                      xaxis_title='Neuron Index', yaxis_title='Activation Value')
    fig.show()

def find_activation_shift(category_activations, steered_category_activations, category=None, top_n=5):
    """
    Find and print the top N most shifted neurons between original and steered activations.
    
    Args:
        category_activations (dict): Original category activations.
        steered_category_activations (dict): Steered category activations.
        category (str, optional): Specific category to analyze. If None, analyzes all categories.
        top_n (int, optional): Number of top shifted neurons to show. Defaults to 5.
    
    Returns:
        dict: Neuron shifts for each category.
    """
    neuron_shifts = {}
    # If no category is specified, return all categories
    if category is None:
        print(f"\nPrinting top {top_n} most shifted neurons for each category:")
        for cat in category_activations:
            original = category_activations[cat].detach().cpu().numpy()
            steered = steered_category_activations[cat].detach().cpu().numpy()
            difference = steered - original
            neuron_shifts[cat] = difference

            # Find indices of top N neurons with biggest absolute difference
            top_indices = np.argsort(np.abs(difference))[-top_n:][::-1]

            print(f"\nCategory: {cat}")
            print(f"{'Neuron':>6} {'Original':>10} {'Steered':>10} {'Shift':>10}")
            print('-'*40)
            for idx in top_indices:
                print(f"{idx:>6} {original[idx]:>10.4f} {steered[idx]:>10.4f} {difference[idx]:+10.4f}")
        
        return neuron_shifts

    if category in category_activations and category in steered_category_activations:
            print(f"\nPrinting top {top_n} most shifted neurons for category: {category}")
            original = category_activations[category].detach().cpu().numpy()
            steered = steered_category_activations[category].detach().cpu().numpy()
            difference = steered - original
            neuron_shifts[category] = difference

            # Find indices of top N neurons with biggest absolute difference
            top_indices = np.argsort(np.abs(difference))[-top_n:][::-1]

            print(f"\nCategory: {category}")
            print(f"{'Neuron':>6} {'Original':>10} {'Steered':>10} {'Shift':>10}")
            print('-'*40)
            for idx in top_indices:
                print(f"{idx:>6} {original[idx]:>10.4f} {steered[idx]:>10.4f} {difference[idx]:>+10.4f}")

    return neuron_shifts

def create_neuron_shift_heatmap(neuron_shifts, all_texts_data, info_string, top_n=10):
    """
    Create a heatmap showing the top shifted neurons across all categories.
    
    Args:
        neuron_shifts (dict): Neuron shift values for each category.
        all_texts_data (pd.DataFrame): Text metadata for filtering by movie count.
        info_string (str): Title for the heatmap.
        top_n (int, optional): Number of top neurons to include. Defaults to 10.
    
    Note:
        Uses logarithmic scaling for better visualization. Only includes categories with 25+ movies.
    """
    # Create a dictionary to store the top neurons and their shifts for each genre
    genre_data = {}
    
    for genre, shifts in neuron_shifts.items():
        # Filter out genres with less than 25 movies
        genre_count = len(all_texts_data[all_texts_data["genre"] == genre])
        if genre_count < 25:
            print(f"Skipping '{genre}' (only {genre_count} movies, need 25+)")
            continue
            
        # Get top N neurons by absolute shift value
        top_indices = np.argsort(np.abs(shifts))[-top_n:][::-1]
        
        # Store the neuron indices and their shift values
        genre_data[genre] = {}
        for idx in top_indices:
            genre_data[genre][idx] = shifts[idx]
    
    if not genre_data:
        print("No genres have 25+ movies. Skipping heatmap.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(genre_data).fillna(0)
    df.index.name = "Neuron"
    df = df.sort_index()
    
    # Create a transformed version for better color scaling
    # Apply sign-preserving logarithmic transformation
    df_transformed = df.copy()
    
    # Apply transformation: sign(x) * log(1 + |x|) for better visualization
    def log_transform(x):
        return np.sign(x) * np.log1p(np.abs(x))
    
    df_transformed = df_transformed.map(log_transform)
    
    # Plot heatmap with transformed values but original annotations
    plt.figure(figsize=(12, 10))
    
    # Use the transformed data for coloring but original data for annotations
    sns.heatmap(df_transformed, annot=df, cmap="coolwarm", center=0, linewidths=0.5, 
                cbar_kws={'label': 'Log-scaled Neuron Shift'}, fmt='.2f',
                annot_kws={'size': 8})
    
    plt.title(info_string)
    plt.xlabel("Genre")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.show()
    
def plot_change_in_top_shifted_neurons(model, model_output, encoded_input, all_texts_data, layer_to_plot, layer_to_steer, category, steering_coefficient_start, steering_coefficient_end, steering_vector, info_string="Info", number_of_steps=5, number_of_neurons=10):
    """
    Plot how individual neurons change across different steering coefficients for a specific category.
    
    Args:
        model: The transformer model.
        model_output: Original model output with hidden states.
        encoded_input (dict): Tokenized input data.
        all_texts_data (pd.DataFrame): Text metadata with genre information.
        layer_to_plot (int): Layer to extract activations from.
        layer_to_steer (int): Layer to apply steering to.
        category (str): Category/genre to analyze.
        steering_coefficient_start (float): Starting steering coefficient.
        steering_coefficient_end (float): Ending steering coefficient.
        steering_vector (torch.Tensor): Steering vector to apply.
        info_string (str, optional): Plot title. Defaults to "Info".
        number_of_steps (int, optional): Number of steering steps. Defaults to 5.
        number_of_neurons (int, optional): Number of neurons to track. Defaults to 10.
    
    Returns:
        tuple: (neuron_trajectories, steering_coefficients, tracked_indices, category).
    
    Note:
        Tracks the most shifted and most activated neurons within the specified category.
    """
    steering_coefficients = torch.linspace(steering_coefficient_start, steering_coefficient_end, number_of_steps)
    
    # First, identify which neurons to track
    print(f"Analyzing category: {category}")
    print(f"Identifying neurons to track...")
    
    # Get original (unsteered) activations for finding most activated neurons
    original_activations = get_category_activations(model_output, encoded_input, all_texts_data, layer_index=layer_to_plot)

    # Get activations with steering applied to find most shifted neurons
    reference_coeff = steering_coefficient_end  # Use end coefficient as reference for shift calculation
    steered_output = create_steered_model_output(model, encoded_input, layer_to_steer, steering_coefficient_end, steering_vector)
    steered_layer_activations = get_category_activations(steered_output, encoded_input, all_texts_data, layer_index=layer_to_plot)

    # Check if category exists
    if category not in original_activations:
        print(f"Category '{category}' not found in activations.")
        print("Available categories:", list(original_activations.keys()))
        return None
    
    # Get activations for the specific category
    original = original_activations[category].detach().cpu().numpy()
    steered = steered_layer_activations[category].detach().cpu().numpy()
    shift = np.abs(original - steered)
    
    # Find top 5 most shifted neurons within this category (based on reference coefficient)
    top_shifted_indices = np.argsort(shift)[-int(number_of_neurons/2):][::-1]

    # Find top 5 most activated neurons (from ORIGINAL, unsteered activations) within this category
    top_activated_indices = np.argsort(abs(original))[-int(number_of_neurons/2):][::-1]

    print(f"Reference coefficient for shift calculation: {reference_coeff:.3f}")
    print(f"Most activated neurons identified from original (unsteered) activations")
    
    # Combine and remove duplicates while preserving order
    tracked_indices = list(top_shifted_indices) + [idx for idx in top_activated_indices if idx not in top_shifted_indices]
    tracked_indices = tracked_indices[:int(number_of_neurons)]  # Limit to 10 neurons max
    
    print(f"\nTracking {len(tracked_indices)} neurons in category '{category}':")
    for i, neuron_idx in enumerate(tracked_indices):
        shift_val = shift[neuron_idx]
        orig_val = original[neuron_idx]
        neuron_type = "Most Shifted" if neuron_idx in top_shifted_indices else "Most Activated"
        print(f"{i+1}. {neuron_type}: Neuron {neuron_idx} (Shift: {shift_val:.4f}, Original: {orig_val:.4f})")
    
    # Now track these neurons across all steering coefficients
    neuron_trajectories = {}
    
    print(f"\nAnalyzing trajectories across {number_of_steps} steering coefficients from {steering_coefficient_start} to {steering_coefficient_end}")
    
    for i, steering_coefficient in enumerate(steering_coefficients):
        print(f"Processing steering coefficient {i+1}/{number_of_steps}: {steering_coefficient:.3f}")
        
        # Get activations for this steering coefficient
        steered_output = create_steered_model_output(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector)
        steered_layer_activations = get_category_activations(steered_output, encoded_input, all_texts_data, layer_index=layer_to_plot)
        
        # Extract values for our tracked neurons
        for neuron_idx in tracked_indices:
            neuron_key = f"N{neuron_idx}"
            
            if neuron_key not in neuron_trajectories:
                neuron_trajectories[neuron_key] = []
            
            activation_value = steered_layer_activations[category].detach().cpu().numpy()[neuron_idx]
            neuron_trajectories[neuron_key].append(activation_value)
    
    # Create the plot
    fig = go.Figure()
    
    # Sort neurons by their final activation values (highest to lowest)
    final_activations = {}
    for neuron_key, trajectory in neuron_trajectories.items():
        final_activations[neuron_key] = trajectory[-1]  # Last value in trajectory
    
    # Sort by final activation value (descending)
    sorted_neurons = sorted(neuron_trajectories.keys(), 
                          key=lambda x: final_activations[x], 
                          reverse=True)
    
    # Add a line for each tracked neuron in sorted order
    for neuron_key in sorted_neurons:
        trajectory = neuron_trajectories[neuron_key]
        neuron_idx = int(neuron_key[1:])  # Remove 'N' prefix
        
        # Determine if this neuron was in top shifted or top activated
        line_style = dict(width=2, dash='solid')
        if neuron_idx in top_shifted_indices:
            line_style['color'] = 'red'
            neuron_label = f"{neuron_key} (Top Shifted)"
        else:
            line_style['color'] = 'blue'
            neuron_label = f"{neuron_key} (Top Activated)"
        
        fig.add_trace(go.Scatter(
            x=steering_coefficients.numpy(),
            y=trajectory,
            mode='lines+markers',
            name=neuron_label,
            line=line_style,
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=info_string,
        xaxis_title='Steering Coefficient',
        yaxis_title='Neuron Activation Value',
        width=900,
        height=600,
        showlegend=True
    )
    
    fig.show()
    
    return neuron_trajectories, steering_coefficients.numpy(), tracked_indices, category

# Example usage
if __name__ == "__main__":
    # Import the data
    data = import_embedding_data_from_pkl("Test_export_embeddings.pkl", model=True, model_output=True, encoded_input=True, all_texts_data=True)
    model, model_output, encoded_input, all_texts_data = data
    feature = "War"


    layer_to_steer = 10  # Change this to the layer you want to analyze, can not be the last layer. Steering in the next to last layer
    # will affect the activations of neurons in the last layer.
    layer_to_plot = 11  # Choose what layer to plot the category activations from, 

    steering_vector = import_steering_vector_from_pkl("steering_vector.pkl", feature_name=feature, layer_to_steer=layer_to_steer)
    # Get the category activations from the hidden states of a specific layer
    steering_coefficient = 7
    category = "Biography"  # Change this to the category you want to visualize

    print(f"\nSteering | Feature: {feature} | Coefficient: {steering_coefficient} | Layer: {layer_to_steer}\n")


    category_layer_activations = get_category_activations(model_output, encoded_input, all_texts_data, layer_index=layer_to_plot)
    info_string = f'Category Activations for {category} | Count: {len(all_texts_data[all_texts_data["genre"] == category])} | Layer: {layer_to_plot}'
    plot_category_activations(category_layer_activations, category, info_string=info_string)

    steered_model_output = create_steered_model_output(model, encoded_input, layer_to_steer=layer_to_steer, steering_coefficient=steering_coefficient, steering_vector=steering_vector)
    steered_category_layer_activations = get_category_activations(steered_model_output, encoded_input, all_texts_data, layer_index=layer_to_plot)
    info_string = f'Category Activations for {category} | Count: {len(all_texts_data[all_texts_data["genre"] == category])} | Layer: {layer_to_plot} | Steering Coefficient: {steering_coefficient} | Steered layer: {layer_to_steer}'
    plot_category_activations(steered_category_layer_activations, category, info_string=info_string)


    # Find the shift in activations
    neuron_shifts = find_activation_shift(category_layer_activations, steered_category_layer_activations, category=None, top_n=20)

    # Create a heatmap of the top shifted neurons across categories
    info_string = f"Neuron Shift Values Across Genres (25+ Movies) | Layer: {layer_to_plot} | Coefficient: {steering_coefficient} | Steered Layer: {layer_to_steer} | Feature: {feature}"
    create_neuron_shift_heatmap(neuron_shifts, all_texts_data, info_string=info_string, top_n=20)

    info_string = f'Individual Neuron Trajectories | {category} | Layer {layer_to_plot} | Feature: {feature} | Steered Layer: {layer_to_steer}'
    neuron_trajectories, steering_coefficients, tracked_indices, category = plot_change_in_top_shifted_neurons(model, model_output, encoded_input, all_texts_data, layer_to_plot, layer_to_steer=layer_to_steer, 
            category=category, steering_coefficient_start=0.0, steering_coefficient_end=6, steering_vector=steering_vector, info_string=info_string, number_of_steps=10, number_of_neurons=20)