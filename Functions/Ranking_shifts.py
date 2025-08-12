from turtle import st
import torch
import torch.nn.functional as F
from Embeddings import import_embedding_data_from_pkl
from Steering_vector import import_steering_vector_from_pkl
from Steering import get_steered_embeddings_vector, get_steered_embeddings_neuron
import plotly.graph_objects as go
import numpy as np

def title_ranking_difference(original_embedding, steered_embeddings, steering_vector, all_texts_data, type="cosine"):
    """
    Rank and display titles by their distance to the steering vector.
    
    Args:
        original_embedding (torch.Tensor): Original embedding vectors.
        steered_embeddings (torch.Tensor): Steered embedding vectors.
        steering_vector (torch.Tensor): Steering vector for distance calculation.
        all_texts_data (pd.DataFrame): Text metadata with title, genre, and overview columns.
        type (str, optional): Distance metric ('l1', 'l2', 'cosine'). Defaults to "cosine".
    
    Returns:
        tuple: (original_top_10_indices, steered_top_10_indices) - Top 10 closest text indices.
    """
    
    original_differences = []
    steered_differences = []
    if type == "l1":
        for i in range(len(original_embedding)):
            orig_diff = torch.abs(original_embedding[i] - steering_vector).sum().item()
            original_differences.append(orig_diff)
            
            steered_diff = torch.abs(steered_embeddings[i] - steering_vector).sum().item()
            steered_differences.append(steered_diff)

    elif type == "l2":
        for i in range(len(original_embedding)):
            orig_diff = torch.norm(original_embedding[i] - steering_vector).item()
            original_differences.append(orig_diff)
            
            steered_diff = torch.norm(steered_embeddings[i] - steering_vector).item()
            steered_differences.append(steered_diff)
    
    elif type == "cosine":
        for i in range(len(original_embedding)):
            # Subtract from 1 to get similair "orientation" to L1 and L2
            orig_diff = 1 - F.cosine_similarity(original_embedding[i].unsqueeze(0), steering_vector, dim=1).item()
            original_differences.append(orig_diff)
            
            steered_diff = 1 - F.cosine_similarity(steered_embeddings[i].unsqueeze(0), steering_vector, dim=1).item()
            steered_differences.append(steered_diff)

    original_difference_tensor = torch.tensor(original_differences)
    steered_difference_tensor = torch.tensor(steered_differences)

    original_top_10_indices = torch.topk(original_difference_tensor, k=10, largest=False).indices
    steered_top_10_indices = torch.topk(steered_difference_tensor, k=10, largest=False).indices

    print(f"\nOriginal Ranking: Top 10 texts with closest {type} distance to steering vector:\n")
    print(f"{'Text':<5} {'Title':<40} {'Genre':<15} {f'{type.capitalize()}':>10}")
    print('-'*75)
    for idx in original_top_10_indices:
        i = idx.item()
        title = all_texts_data['title'][i][:37] + '...' if len(all_texts_data['title'][i]) > 40 else all_texts_data['title'][i]
        genre = all_texts_data['genre'][i]
        diff_val = original_difference_tensor[i].item()
        print(f"{i+1:<5} {title:<40} {genre:<15} {diff_val:>10.4f}")

    print("\nTop result")
    print("Title: " + all_texts_data['title'][original_top_10_indices[0].item()])
    print("Description: " + all_texts_data['overview'][original_top_10_indices[0].item()])

    print(f"\nSteered Ranking: Top 10 texts with closest {type} distance to steering vector:\n")
    print(f"{'Text':<5} {'Title':<40} {'Genre':<15} {f'{type.capitalize()}':>10}")
    print('-'*75)
    for idx in steered_top_10_indices:
        i = idx.item()
        title = all_texts_data['title'][i][:37] + '...' if len(all_texts_data['title'][i]) > 40 else all_texts_data['title'][i]
        genre = all_texts_data['genre'][i]
        diff_val = steered_difference_tensor[i].item()
        print(f"{i+1:<5} {title:<40} {genre:<15} {diff_val:>10.4f}")
    
    print("\nTop result")
    print("Title: " + all_texts_data['title'][steered_top_10_indices[0].item()])
    print("Description: " + all_texts_data['overview'][steered_top_10_indices[0].item()])


    return original_top_10_indices, steered_top_10_indices

def all_differences(original_embedding, steered_embedding, texts=None, Print=True):
    """
    Compute and print the differences between original and steered embeddings.
    
    Args:
        original_embedding (torch.Tensor): Original embedding vectors.
        steered_embedding (torch.Tensor): Steered embedding vectors.
        texts (pd.DataFrame, optional): Text metadata for display. Defaults to None.
        Print (bool, optional): Whether to print results. Defaults to True.
    
    Returns:
        tuple: (mean_shift_l1, mean_shift_l2, mean_cosine, per_sentence_diff_l1, per_sentence_diff_l2, per_sentence_diff_cosine).
    """
    if original_embedding.dim() == 1:
        original_embedding = original_embedding.unsqueeze(0)
        steered_embedding = steered_embedding.unsqueeze(0)

    # Compute L1 norm per sentence (Manhattan distance)
    per_sentence_diff_l1 = torch.abs(steered_embedding - original_embedding).sum(dim=1)
    mean_shift_l1 = per_sentence_diff_l1.mean().item()

    # Compute L2 norm per sentence (Euclidean distance)
    per_sentence_diff_l2 = torch.norm(steered_embedding - original_embedding, dim=1)
    mean_shift_l2 = per_sentence_diff_l2.mean().item()

    # Cosine similarity per sentence
    per_sentence_diff_cosine = F.cosine_similarity(steered_embedding, original_embedding, dim=1)
    mean_cosine = per_sentence_diff_cosine.mean().item()

    if Print:
        print(f"Mean L1 shift (Manhattan): {mean_shift_l1:.4f}")
        print(f"Mean L2 shift (Euclidian): {mean_shift_l2:.4f}")
        print(f"Mean cosine similarity: {mean_cosine:.4f}")

        if texts is not None:
            print("\nL1: Manhattan distance")
            most_shifted_idx_l1 = torch.argmax(per_sentence_diff_l1).item() #Keep in mind: this is the most shifted based on L1
            print(f"Most shifted (L1) text index: Text nr. {most_shifted_idx_l1+1}") #index + 1
            print(f"Title: {texts['title'][most_shifted_idx_l1]}")
            print(f"MOST: L1 shift (Manhattan): {per_sentence_diff_l1[most_shifted_idx_l1].item():.4f}")
            print(f"L2 shift (Euclidian): {per_sentence_diff_l2[most_shifted_idx_l1].item():.4f}")
            print(f"Cosine similarity: {per_sentence_diff_cosine[most_shifted_idx_l1].item():.4f}")
            print("\nL2: Euclidean distance")
            most_shifted_idx_l2 = torch.argmax(per_sentence_diff_l2).item() #Keep in mind: this is the most shifted based on L2
            print(f"Most shifted (L2) text index: Text nr. {most_shifted_idx_l2+1}") #index + 1
            print(f"Title: {texts['title'][most_shifted_idx_l2]}")
            print(f"L1 shift (Manhattan): {per_sentence_diff_l1[most_shifted_idx_l2].item():.4f}")
            print(f"MOST: L2 shift (Euclidian): {per_sentence_diff_l2[most_shifted_idx_l2].item():.4f}")
            print(f"Cosine similarity: {per_sentence_diff_cosine[most_shifted_idx_l2].item():.4f}")


    return mean_shift_l1, mean_shift_l2, mean_cosine, per_sentence_diff_l1, per_sentence_diff_l2, per_sentence_diff_cosine


def calculate_specific_differences(original_embeddings, steered_embeddings, steering_vector, type):
    """
    Calculate specific differences between original and steered embeddings based on type (l1, l2, cosine).
    
    Args:
        original_embeddings (torch.Tensor): Original embedding vectors.
        steered_embeddings (torch.Tensor): Steered embedding vectors.
        steering_vector (torch.Tensor): Steering vector for distance calculation.
        type (str): Distance metric ('l1', 'l2', 'cosine').
    
    Returns:
        tuple: (original_differences, steered_differences) - Lists of distance values.
    
    Raises:
        ValueError: If invalid type is specified.
    """
    original_differences = []
    steered_differences = []
    for i in range(len(original_embeddings)):
        if type == "l1":
            orig_diff = torch.abs(original_embeddings[i] - steering_vector).sum().item() #skal man ha .item her 
            diff = torch.abs(steered_embeddings[i] - steering_vector).sum().item()
        elif type == "l2":
            orig_diff = torch.norm(original_embeddings[i] - steering_vector).item()
            diff = torch.norm(steered_embeddings[i] - steering_vector).item()
        elif type == "cosine":
            # Again we flip the cosine to match the orientation of L1 and L2
            orig_diff = 1 - F.cosine_similarity(original_embeddings[i].unsqueeze(0), steering_vector, dim=1).item()
            diff = 1 - F.cosine_similarity(steered_embeddings[i].unsqueeze(0), steering_vector, dim=1).item()
        else:
            raise ValueError("Invalid type specified. Use 'l1', 'l2', or 'cosine'.")
        original_differences.append(orig_diff) # eller her (diff.item())
        steered_differences.append(diff)
    return original_differences, steered_differences



def rank_categories_by_difference(all_texts_data, original_differences, steered_differences, type, info_string, print_results=True, 
                                  remove_low_count=True, plot_graph=True):
    """
    Rank categories (genres) by the mean difference in embeddings.
    
    Args:
        all_texts_data (pd.DataFrame): Text metadata with 'genre' column.
        original_differences (list): Original distance values.
        steered_differences (list): Steered distance values.
        type (str): Distance metric type for labeling.
        info_string (str): Information string for plot title.
        print_results (bool, optional): Whether to print ranking results. Defaults to True.
        remove_low_count (bool, optional): Whether to filter genres with <25 items. Defaults to True.
        plot_graph (bool, optional): Whether to create visualization. Defaults to True.
    
    Returns:
        tuple: Varies based on whether original_differences is provided.
               If provided: (filtered_genres, original_filtered_genres, genre_differences, original_genre_differences, genre_means)
               If None: (filtered_genres, genre_differences, genre_means)
    """

    genre_differences = {}
    for i, difference in enumerate(steered_differences):
        genre = all_texts_data['genre'][i]
        if genre not in genre_differences:
            genre_differences[genre] = []
        genre_differences[genre].append(difference)
    
        # Calculate mean difference for each genre
    genre_means = {}
    for genre, sims in genre_differences.items():
        genre_means[genre] = sum(sims) / len(sims)
    
    # Sort genres by mean difference
    sorted_genres = sorted(genre_means.items(), key=lambda x: x[1], reverse=True)
    
    if original_differences is not None:
 
        # If original differences are provided, organize them by genre too
        original_genre_differences = {}
        for i, difference in enumerate(original_differences):
            genre = all_texts_data['genre'][i]
            if genre not in original_genre_differences:
                original_genre_differences[genre] = []
            original_genre_differences[genre].append(difference)
        

        original_genre_means = {}
        for genre, sims in original_genre_differences.items():
            original_genre_means[genre] = sum(sims) / len(sims)
        
        original_sorted_genres = sorted(original_genre_means.items(), key=lambda x: x[1], reverse=True)

    if remove_low_count:
        filtered_genres = [(genre, mean_sim) for genre, mean_sim in sorted_genres
                           if len(genre_differences[genre]) >= 25]
        if original_differences is not None:
            # Filter original genres similarly
            original_filtered_genres = [(genre, mean_sim) for genre, mean_sim in original_sorted_genres
                                        if len(original_genre_differences[genre]) >= 25]
    else:
        filtered_genres = sorted_genres
        if original_differences is not None:
            original_filtered_genres = original_sorted_genres


    if print_results:

        # Also print the original genre means if provided
        if original_differences is not None:
            print(f"\nOriginal Category Ranking by {type} difference:") # Endre på teksten her
            print("=" * 50)
            print(f"{'Genre':<20} {'Mean diff':<15} {'Count':<10}")
            print("-" * 50)
            for genre, mean_sim in original_filtered_genres:
                count = len(original_genre_differences[genre])
                print(f"{genre:<20} {mean_sim:<15.4f} {count:<10}")

        print(f"\nSteered Category Ranking by {type} difference:") # Endre på teksten her
        print("=" * 50)
        print(f"{'Genre':<20} {'Mean diff':<15} {'Count':<10}")
        print("-" * 50)
        for genre, mean_sim in filtered_genres:
            count = len(genre_differences[genre])
            print(f"{genre:<20} {mean_sim:<15.4f} {count:<10}")



    if plot_graph:
        plot_bar_category_ranking(filtered_genres, original_genre_differences, genre_differences, type, info_string, remove_low_count=remove_low_count)

    if original_differences is not None:
        # If original differences were provided, return both filtered lists
        return filtered_genres, original_filtered_genres, genre_differences, original_genre_differences, genre_means
    else:
        # If no original differences, just return the filtered genres and their differences
        return filtered_genres, genre_differences, genre_means

def export_ranked_categories_to_txt(filtered_genres, original_filtered_genres, 
                                    genre_differences, original_genre_differences,type, 
                                    file_name = "ranked_categories.txt", info_string = "Experiment info"):
    """
    Export the ranked categories to a text file.
    
    Args:
        filtered_genres (list): Ranked steered genre data.
        original_filtered_genres (list): Ranked original genre data.
        genre_differences (dict): Steered genre difference data.
        original_genre_differences (dict): Original genre difference data.
        type (str): Distance metric type for filename.
        file_name (str, optional): Base filename. Defaults to "ranked_categories.txt".
        info_string (str, optional): Experiment information. Defaults to "Experiment info".
    """
    with open(f"{file_name}_{type}.txt", 'a') as f:
        if original_filtered_genres is not None:
            f.write("\nOriginal Category Ranking:\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Genre':<20} {'Mean diff':<15} {'Count':<10}\n")
            f.write("-" * 50 + "\n")
            for genre, mean_sim in original_filtered_genres:
                count = len(original_genre_differences[genre])
                f.write(f"{genre:<20} {mean_sim:<15.4f} {count:<10}\n")

        f.write(f"\n{info_string}\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Genre':<20} {'Mean diff':<15} {'Count':<10}\n")
        f.write("-" * 50 + "\n")
        for genre, mean_sim in filtered_genres:
            count = len(genre_differences[genre])
            f.write(f"{genre:<20} {mean_sim:<15.4f} {count:<10}\n")



def plot_bar_category_ranking(sorted_genres, original_genre_differences, genre_differences, type, info_string, remove_low_count=True):
    """
    Plot the category ranking by mean difference.
    
    Args:
        sorted_genres (list): Sorted genre data by mean difference.
        original_genre_differences (dict): Original genre difference data.
        genre_differences (dict): Steered genre difference data.
        type (str): Distance metric type for labeling.
        info_string (str): Information string for plot title.
        remove_low_count (bool, optional): Whether to filter genres with <25 items. Defaults to True.
    
    Returns:
        plotly.graph_objects.Figure or None: Figure object if successful, None if no valid data.
    """

    # Flip the cosine differences back for plotting if type is cosine
    if type == "cosine":
        # Apply 1 - x to each value in the lists within the dictionaries
        for genre in original_genre_differences:
            original_genre_differences[genre] = [1 - x for x in original_genre_differences[genre]]
        for genre in genre_differences:
            genre_differences[genre] = [1 - x for x in genre_differences[genre]]
    
    if remove_low_count:
        filtered_genres = [(genre, mean_sim) for genre, mean_sim in sorted_genres 
                           if len(genre_differences[genre]) >= 25]
    else:
        filtered_genres = sorted_genres
    
    if not filtered_genres:
        print("No genres have 25 or more movies. Skipping plot.")
        return None
    
    genres = [genre for genre, _ in filtered_genres]
    mean_sims = [mean_sim for _, mean_sim in filtered_genres]
    counts = [len(genre_differences[genre]) for genre, _ in filtered_genres]
    
    hover_text_steered = [f"Genre: {genre}<br>Steered Mean Difference: {mean_sim:.4f}<br>Count: {count}" 
                          for genre, mean_sim, count in zip(genres, mean_sims, counts)]
    
    original_mean_sims = []
    hover_text_original = []
    
    for genre, _ in filtered_genres:
        if genre in original_genre_differences:
            original_mean = sum(original_genre_differences[genre]) / len(original_genre_differences[genre])
            original_mean_sims.append(original_mean)
            hover_text_original.append(f"Genre: {genre}<br>Original Mean Difference: {original_mean:.4f}<br>Count: {len(original_genre_differences[genre])}")
        else:
            original_mean_sims.append(0)
            hover_text_original.append(f"Genre: {genre}<br>Original Mean Difference: N/A<br>Count: 0")
    
    fig = go.Figure()
    
    for i, genre in enumerate(genres):
        steered_val = mean_sims[i]
        original_val = original_mean_sims[i]
        diff = steered_val - original_val

        # Decide draw order based on which bar should be in front
        if diff < 0:
            # Steered is smaller -> draw steered second (on top)
            fig.add_trace(go.Bar(
                x=[genre],
                y=[original_val],
                marker_color='grey',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[hover_text_original[i]],
                name='Original',
                opacity=1,
                showlegend=(i == 0)
            ))
            fig.add_trace(go.Bar(
                x=[genre],
                y=[steered_val],
                marker_color='blue',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[hover_text_steered[i]],
                name='Steered',
                opacity=1,
                showlegend=(i == 0)
            ))
        else:
            # Original is smaller or equal -> draw original second (on top)
            fig.add_trace(go.Bar(
                x=[genre],
                y=[steered_val],
                marker_color='blue',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[hover_text_steered[i]],
                name='Steered',
                opacity=1,
                showlegend=(i == 0)
            ))
            fig.add_trace(go.Bar(
                x=[genre],
                y=[original_val],
                marker_color='grey',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[hover_text_original[i]],
                name='Original',
                opacity=1,
                showlegend=(i == 0)
            ))

    fig.update_layout(
        title=f'Category Ranking by Mean Difference (25+ Movies) | {info_string}' if remove_low_count else f'Category Ranking by Mean Difference (All Genres) | {info_string}',
        xaxis_title='Genre',
        yaxis_title=f'Absolute Mean Difference ({type})',
        xaxis_tickangle=-45,
        width=900,
        height=500,
        showlegend=True,
        barmode='overlay'
    )
    
    fig.show()
    return fig


def get_most_shifted_categories(all_texts_data, original_differences, steered_differences, remove_low_count=True, top_n=10, print_results=True):
    """
    Print the most shifted categories based on the differences in embeddings.
    
    Args:
        all_texts_data (pd.DataFrame): Text metadata with 'genre' column.
        original_differences (list): Original distance values.
        steered_differences (list): Steered distance values.
        remove_low_count (bool, optional): Whether to filter genres with <25 items. Defaults to True.
        top_n (int, optional): Number of top shifted categories to return. Defaults to 10.
        print_results (bool, optional): Whether to print results. Defaults to True.
    
    Returns:
        list or None: Top shifted categories data, or None if no valid data.
    """
    steered_genre_differences = {}
    original_genre_differences = {}
    
    # Organize cosine similarities by genre
    for i, (steered_sim, original_sim) in enumerate(zip(steered_differences, original_differences)):
        genre = all_texts_data['genre'][i]
        
        if genre not in steered_genre_differences:
            steered_genre_differences[genre] = []
            original_genre_differences[genre] = []
        
        steered_genre_differences[genre].append(steered_sim)
        original_genre_differences[genre].append(original_sim)
    
    # Calculate mean cosine similarities and shifts for each genre
    genre_shifts = {}
    for genre in steered_genre_differences.keys():
        steered_mean = sum(steered_genre_differences[genre]) / len(steered_genre_differences[genre])
        original_mean = sum(original_genre_differences[genre]) / len(original_genre_differences[genre])
        shift = steered_mean - original_mean
        count = len(steered_genre_differences[genre])
        
        genre_shifts[genre] = {
            'shift': shift,
            'steered_mean': steered_mean,
            'original_mean': original_mean,
            'count': count
        }
    
    # Filter by count if requested
    if remove_low_count:
        genre_shifts = {k: v for k, v in genre_shifts.items() if v['count'] >= 25}
    
    if not genre_shifts:
        print("No genres have 25 or more movies. Skipping analysis.")
        return None
    
    # Sort by absolute shift (largest changes first)
    sorted_shifts = sorted(genre_shifts.items(), key=lambda x: abs(x[1]['shift']), reverse=True)
    
    # Take top N most shifted
    top_shifts = sorted_shifts[:top_n]
    
    # Print summary
    if print_results:
        filter_text = " (25+ Movies)" if remove_low_count else " (All Genres)"
        print(f"\nTop {top_n} Most Shifted Categories{filter_text}:")
        print("=" * 70)
        print(f"{'Genre':<20} {'Shift':<10} {'Original':<10} {'Steered':<10} {'Count':<8}")
        print("-" * 70)
        for genre, data in top_shifts:
            print(f"{genre:<20} {data['shift']:<10.4f} {data['original_mean']:<10.4f} {data['steered_mean']:<10.4f} {data['count']:<8}")
    
    return top_shifts


def get_top_shift_for_steering_coefficient(model, encoded_input, all_texts_data, original_embeddings, layer_to_steer, steering_coefficient, steering_vector, type):
    """
    Get the shift data for all categories at a specific steering coefficient.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        all_texts_data (pd.DataFrame): Text metadata with 'genre' column.
        original_embeddings (torch.Tensor): Original embedding vectors.
        layer_to_steer (int): Layer index to apply steering to.
        steering_coefficient (float): Steering strength.
        steering_vector (torch.Tensor): Steering vector to apply.
        type (str): Distance metric type.
    
    Returns:
        dict: Dictionary mapping genre names to shift values.
    """
    steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, normalize=True)
    original_differences, steered_differences = calculate_specific_differences(original_embeddings, steered_embeddings, steering_vector, type)
    top_shifts = get_most_shifted_categories(all_texts_data, original_differences, steered_differences, remove_low_count=True, top_n=10, print_results=False)
    
    # Convert to dictionary for easier access
    shift_dict = {}
    for genre, data in top_shifts:
        shift_dict[genre] = data['shift']
    
    return shift_dict
    


def plot_change_in_top_shift(model, encoded_input, all_texts_data, original_embeddings, layer_to_steer, steering_vector, type, feature, steering_coefficient_start, steering_coefficient_end, number_of_steps=5):
    """
    Plot how the top shifted categories change across different steering coefficients.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        all_texts_data (pd.DataFrame): Text metadata with 'genre' column.
        original_embeddings (torch.Tensor): Original embedding vectors.
        layer_to_steer (int): Layer index to apply steering to.
        steering_vector (torch.Tensor): Steering vector to apply.
        type (str): Distance metric type.
        feature (str): Feature name for plot title.
        steering_coefficient_start (float): Starting steering coefficient.
        steering_coefficient_end (float): Ending steering coefficient.
        number_of_steps (int, optional): Number of coefficient steps. Defaults to 5.
    
    Returns:
        tuple: (category_shifts, steering_coefficients) - Shift data and coefficient values.
    """
    steering_coefficients = torch.linspace(steering_coefficient_start, steering_coefficient_end, number_of_steps)
    
    # Dictionary to store shifts for each category across all steering coefficients
    category_shifts = {}
    
    print(f"Analyzing shifts across {number_of_steps} steering coefficients from {steering_coefficient_start} to {steering_coefficient_end}")
    
    for i, steering_coefficient in enumerate(steering_coefficients):
        print(f"\nProcessing steering coefficient {i+1}/{number_of_steps}: {steering_coefficient:.3f}")
        
        # Get shifts for this steering coefficient
        shift_dict = get_top_shift_for_steering_coefficient(model, encoded_input, all_texts_data, original_embeddings, layer_to_steer, steering_coefficient, steering_vector, type)
        
        # Store results for each category
        for category, shift in shift_dict.items():
            if category not in category_shifts:
                category_shifts[category] = []
            category_shifts[category].append(shift)
    
    # Create the plot
    fig = go.Figure()
    
    # Sort categories by their final shift value (highest shift first)
    categories_with_final_shifts = [(cat, shifts[-1]) for cat, shifts in category_shifts.items()]
    categories_with_final_shifts.sort(key=lambda x: x[1], reverse=True)
    
    # Add a line for each category in sorted order
    for category, _ in categories_with_final_shifts:
        shifts = category_shifts[category]
        fig.add_trace(go.Scatter(
            x=steering_coefficients.numpy(),
            y=shifts,
            mode='lines+markers',
            name=category,
            line=dict(width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=f'Change in Genre Shift vs Steering Coefficient | Layer: {layer_to_steer} | Feature: {feature} | Diff: {type}',
        xaxis_title='Steering Coefficient',
        yaxis_title=f'Shift in Mean {type} Distance',
        width=900,
        height=600,
        showlegend=True
    )
    
    fig.show()
    
    return category_shifts, steering_coefficients.numpy()



def plot_rank_plot(category_shifts, steering_coefficients, info_string="Experiment Info"):
    """
    Plot rank changes of categories over steering coefficients.
    
    Args:
        category_shifts (dict): Dictionary mapping categories to their shift values across coefficients.
        steering_coefficients (numpy.ndarray): Array of steering coefficient values.
        info_string (str, optional): Information string for plot title. Defaults to "Experiment Info".
    """
    categories = list(category_shifts.keys())
    num_steps = len(steering_coefficients)
    
    # Build a rank matrix: rows = categories, cols = coefficients
    ranks_over_coeffs = []
    for step in range(num_steps):
        # Get the shift values at this coefficient for all categories
        shifts_at_step = [category_shifts[cat][step] for cat in categories]
        # Get sorted indices (highest shift gets rank 1)
        sorted_idx = np.argsort(shifts_at_step)
        # Map indices to ranks
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, len(categories)+1)
        ranks_over_coeffs.append(ranks)
    
    # Transpose to get list of ranks for each category
    ranks_over_coeffs = np.array(ranks_over_coeffs).T  # shape: (num_categories, num_steps)
    
    # Get final ranking (at last steering coefficient) to sort legend
    final_ranks = ranks_over_coeffs[:, -1]  # Last column contains final ranks
    # Sort categories by their final rank (rank 1 first)
    sorted_category_indices = np.argsort(final_ranks)
    
    # Plot
    fig = go.Figure()
    for idx in sorted_category_indices:
        cat = categories[idx]
        fig.add_trace(go.Scatter(
            x=steering_coefficients,
            y=ranks_over_coeffs[idx],
            mode='lines+markers',
            name=cat,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f'Rank Plot: Category rank vs Steering Coefficient | {info_string}',
        xaxis_title='Steering Coefficient',
        yaxis_title='Rank (1=highest shift)',
        yaxis_autorange='reversed',  # so rank 1 is at the top
        width=900,
        height=600,
        showlegend=True
    )
    fig.show()


def experimentation_function(all_texts_data, type, layer_to_steer, steering_coefficient, steering_vector, print_results=False, 
                             remove_low_count=True, plot_graph=False):
    """
    Experimental function to test steering effects across all neurons in a layer.
    
    Args:
        all_texts_data (pd.DataFrame): Text metadata with 'genre' column.
        type (str): Distance metric type.
        layer_to_steer (int): Layer index to apply steering to.
        steering_coefficient (float): Steering strength.
        steering_vector (torch.Tensor): Steering vector to compare against.
        print_results (bool, optional): Whether to print detailed results. Defaults to False.
        remove_low_count (bool, optional): Whether to filter genres with <25 items. Defaults to True.
        plot_graph (bool, optional): Whether to create visualizations. Defaults to False.
    
    Note:
        Requires 'model', 'encoded_input', 'original_embeddings', and 'feature' variables in global scope.
        Exports results to a text file for each neuron tested.
    """

    print("\nStarting experimentation function...\n")
    # Run through for first neuron
    node_to_steer = 0
    print(f"\nExperimenting with steering node {node_to_steer} in layer {layer_to_steer} with coefficient {steering_coefficient}...")
            
    # Get steered embeddings for the specific neuron
    steered_embeddings = get_steered_embeddings_neuron(model, encoded_input, layer_to_steer=layer_to_steer, 
                                                node_to_steer=node_to_steer, steering_coefficient=steering_coefficient, normalize=True, verbose=False)
    original_differences, steered_differences = calculate_specific_differences(original_embeddings, steered_embeddings, steering_vector=steering_vector, type=type)

    info_string = f'Steered layer: {layer_to_steer} | Coeff: {steering_coefficient}'

    filtered_genres, original_filtered_genres, genre_differences, original_genre_differences, genre_means = rank_categories_by_difference(
        all_texts_data, original_differences, steered_differences, info_string, type=type,
        print_results=print_results, remove_low_count=remove_low_count, plot_graph=plot_graph
    )

    export_ranked_categories_to_txt(filtered_genres, original_filtered_genres, genre_differences, original_genre_differences, type=type,
                                    file_name=f"ranked_categories_layer_{layer_to_steer}_coeff_{steering_coefficient}_feature_{feature}" ,
                                    info_string=f"Experimenting in layer {layer_to_steer} with coefficient {steering_coefficient}\nSteering node: {node_to_steer}")

    original_filtered_genres = None
    original_genre_differences = None

    for node_to_steer in range(1, len(original_embeddings[0])):
        print(f"Experimenting with steering node {node_to_steer} in layer {layer_to_steer} with coefficient {steering_coefficient}...\nPercentage done: {round((node_to_steer / (len(original_embeddings[0]) - 1)) * 100, 2)}%", end="\r")
        # Get steered embeddings for the specific neuron
        steered_embeddings = get_steered_embeddings_neuron(model, encoded_input, layer_to_steer=layer_to_steer, 
                                                   node_to_steer=node_to_steer, steering_coefficient=steering_coefficient, normalize=True, verbose=False)
        original_differences, steered_differences = calculate_specific_differences(original_embeddings, steered_embeddings, steering_vector=steering_vector, type=type)

        original_differences = None

        info_string = f'Steered layer: {layer_to_steer} | Coeff: {steering_coefficient}'

        filtered_genres, genre_differences, genre_means = rank_categories_by_difference(
            all_texts_data, original_differences, steered_differences, info_string, type=type,
            print_results=print_results, remove_low_count=remove_low_count, plot_graph=plot_graph
        )
        
        export_ranked_categories_to_txt(filtered_genres, original_filtered_genres, genre_differences, original_genre_differences, type,
                                        file_name=f"ranked_categories_layer_{layer_to_steer}_coeff_{steering_coefficient}_feature_{feature}", 
                                        info_string=f"Steering node: {node_to_steer}")

    print("\n Exported all results to "f"ranked_categories_layer_{layer_to_steer}_coeff_{steering_coefficient}_feature_{feature}.txt")
    


# Example usage
if __name__ == "__main__":
    # Import data
    data = import_embedding_data_from_pkl('Test_export_embeddings.pkl', model=True, embeddings=True, encoded_input=True, all_texts_data=True)
    model, original_embeddings, encoded_input, all_texts_data = data

    feature = "War"
    type = "cosine" # l1, l2 or cosine
    layer_to_steer = 11  # Change this to the layer you want to analyze
    steering_coefficient = 1  # Adjust steering coefficient as needed

    steering_vector = import_steering_vector_from_pkl('steering_vector.pkl', layer_to_steer=layer_to_steer, feature_name=feature)

    # Steer using a vector
    steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer=layer_to_steer, steering_coefficient=steering_coefficient, 
                                               steering_vector=steering_vector, normalize=True)

    # Steer on specific neuron
    #steered_embeddings = get_steered_embeddings_neuron(model, encoded_input, layer_to_steer=layer_to_steer, 
    #                                                   node_to_steer=92, steering_coefficient=steering_coefficient, normalize=True)

    print(f"\nSteering | Feature: {feature} | Coefficient: {steering_coefficient} | Layer: {layer_to_steer}\n")

    # Calculate cosine similarities and rankings
    print("Calculating differences between original and steered embeddings...")
    original_differences, steered_differences = calculate_specific_differences(original_embeddings, steered_embeddings, steering_vector, type=type)
    print("[Done]")

    # Print top 10 titles based on differences
    print("\nRanking titles by differences...")
    original_top_10_indices, steered_top_10_indices = title_ranking_difference(original_embeddings, steered_embeddings, 
                                                                               steering_vector, all_texts_data, type=type)

    # Bar plot of category rankings
    print("\nRanking categories by mean difference...")
    info_string = f'Steered layer: {layer_to_steer} | Coeff: {steering_coefficient}'
    filtered_genres, original_filtered_genres, genre_differences, original_genre_differences, genre_means = rank_categories_by_difference(all_texts_data, 
                                                                original_differences, steered_differences, type, info_string, print_results=True, 
                                                            remove_low_count=True, plot_graph=True)

    # Print most shifted categories
    print("\nPrinting most shifted categories...\n")
    get_most_shifted_categories(all_texts_data, original_differences, steered_differences, remove_low_count=True, top_n=10, print_results=True)


    #experimentation_function(all_texts_data, type=type, layer_to_steer=2, steering_coefficient=10, steering_vector=steering_vector)


    # Plot change in top shifts across steering coefficients
    print("\n\nPlotting change in top shifts across steering coefficients...\n")
    category_shifts, steering_coefficients = plot_change_in_top_shift(model, encoded_input, all_texts_data, original_embeddings, layer_to_steer, steering_vector, type, 
                                                                    feature, steering_coefficient_start=0.0, steering_coefficient_end=15, number_of_steps=20)

    # Plot rank changes of categories over steering coefficients
    print("\nPlotting rank changes of categories over steering coefficients...\n")
    info_string = f'Layer: {layer_to_steer} | Feature: {feature} | Diff: {type}'
    plot_rank_plot(category_shifts, steering_coefficients, info_string)