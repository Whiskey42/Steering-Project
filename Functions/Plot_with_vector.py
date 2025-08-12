from Ranking_shifts import calculate_specific_differences
from Steering import get_steered_embeddings_vector
from Steering_vector import get_steering_vector, import_feature_texts
from Embeddings import import_embedding_data_from_pkl

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch


def create_hover_data(all_texts_data, number_of_texts):
    """
    Create hover data for visualizations with title and genre information.
    
    Args:
        all_texts_data (pd.DataFrame): Text metadata with 'title' and 'genre' columns.
        number_of_texts (int): Number of texts to create hover data for.
    
    Returns:
        tuple: (hover_data_original, hover_data_steered) - Lists of formatted hover strings.
    """
    indices = all_texts_data.index[:number_of_texts]
    return [
        f"Original | {all_texts_data.loc[i, 'title']} | {all_texts_data.loc[i, 'genre']}"
        for i in indices
    ], [
        f"Steered | {all_texts_data.loc[i, 'title']} | {all_texts_data.loc[i, 'genre']}"
        for i in indices
    ]


def plot_distance_projection(model, encoded_input, original_embeddings, all_texts_data, layer_to_steer, steering_coefficient, steering_vector, 
                             feature, text_range=None, type="l2", normalize=True, print_differences=True, print_average=True, Show=True, Write=False, return_fig=False):
    """
    Plot the distance projection of original and steered embeddings for a given feature.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        original_embeddings (torch.Tensor): Original embedding vectors.
        all_texts_data (pd.DataFrame): Text metadata with title and genre information.
        layer_to_steer (int): Layer index to apply steering to.
        steering_coefficient (float): Strength of the steering intervention.
        steering_vector (torch.Tensor): Steering vector to apply.
        feature (str): Feature name for plot title and labeling.
        text_range (tuple, optional): Range of texts to analyze. Defaults to None.
        type (str, optional): Distance metric ('l2', 'l1', 'cosine'). Defaults to "l2".
        normalize (bool, optional): Whether to normalize embeddings. Defaults to True.
        print_differences (bool, optional): Whether to print most/least changed items. Defaults to True.
        print_average (bool, optional): Whether to print average distances. Defaults to True.
        Show (bool, optional): Whether to display the plot. Defaults to True.
        Write (bool, optional): Whether to save plot as HTML file. Defaults to False.
        return_fig (bool, optional): Whether to return the figure object. Defaults to False.
    
    Returns:
        plotly.graph_objects.Figure or None: Figure object if return_fig=True, otherwise None.
    """
    # Apply slicing if requested
    if text_range is not None:
        s = slice(*text_range)
        encoded_input = {key: val[s] for key, val in encoded_input.items()}
        all_texts_data = all_texts_data.iloc[s]
        original_embeddings = original_embeddings[s]

    steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, normalize=normalize, verbose=False)

    original_differences, steered_differences = calculate_specific_differences(original_embeddings, steered_embeddings, steering_vector, type)

    # Prepare data
    hover_data_original, hover_data_steered = create_hover_data(all_texts_data, len(original_differences))
    genres = all_texts_data['genre'].tolist()[:len(original_differences)]
    if text_range is not None:
        y_positions = list(range(text_range[0], text_range[1]))
    else:
        y_positions = list(range(len(original_differences)))

    df_original = pd.DataFrame({
        'Distance': original_differences,
        'Y': y_positions,
        'Cluster': genres,
        'Title': hover_data_original,
        'Type': 'Original'
    })

    df_steered = pd.DataFrame({
        'Distance': steered_differences,
        'Y': y_positions,
        'Cluster': genres,
        'Title': hover_data_steered,
        'Type': 'Steered'})

    df = pd.concat([df_original, df_steered], ignore_index=True)

    # Plot points using px.scatter
    fig = px.scatter(
        df,
        x='Distance',
        y='Y',
        color='Cluster',
        symbol='Type',
        hover_data=['Title', 'Cluster'],
        labels={'Distance': f'Distance to {feature}', 'Y': 'Movie Index'})

    # Set transparency and show only steered in legend
    for trace in fig.data:
        if 'Original' in trace.name:
            trace.opacity = 0.3
            trace.showlegend = False
        else:
            trace.opacity = 1.0
            trace.showlegend = True

    # Add dotted lines
    for i in range(len(original_differences)):
        fig.add_trace(go.Scatter(
            x=[original_differences[i], steered_differences[i]],
            y=[y_positions[i], y_positions[i]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='none'))

    # Update layout
    fig.update_layout(
        title=f'Distance Projection to {feature} ({type.upper()} Distance) | Layer: {layer_to_steer} | Coeff: {steering_coefficient}',
        width=900,
        height=700,
        xaxis_title=f'Distance to {feature} using {type.upper()}',
        yaxis=dict(title='Movie Index', showticklabels=True, range=[min(y_positions) - 5, max(y_positions) + 5]))

    if Show:
        fig.show()
    if Write:
        fig.write_html(f'Distance_Projection_{type}.html')
    if return_fig:
        return fig
    if print_differences:
        print_most_and_least_changed(original_differences, steered_differences, all_texts_data, global_indices=all_texts_data.index.tolist())
    if print_average:
        steered_avg = torch.tensor(steered_differences).mean().item()
        print(f"Average Steered Distance to {feature}: {steered_avg:.4f}")
        if type == "cosine":
            print(f"Average Steered 1- Cosine Similarity to {feature}: {1 - steered_avg:.4f}")



def print_most_and_least_changed(original_distances, steered_distances, texts=None, top_k=3, global_indices=None):
    """
    Print the most and least changed distances after steering.
    
    Args:
        original_distances (list): Original distance values.
        steered_distances (list): Steered distance values.
        texts (pd.DataFrame, optional): Text metadata for display. Defaults to None.
        top_k (int, optional): Number of top items to display. Defaults to 3.
        global_indices (list, optional): Global indices for the texts. Defaults to None.
    """
    original = torch.tensor(original_distances)
    steered = torch.tensor(steered_distances)
    delta = torch.abs(steered - original)

    most_idx = torch.topk(delta, top_k).indices.tolist()
    least_idx = torch.topk(-delta, top_k).indices.tolist()

    if global_indices is None:
        global_indices = list(range(len(original)))

    print("\n\nMOST CHANGED (based on difference)")
    print("-" * 50)
    for i in most_idx:
        global_idx = global_indices[i]
        title = texts.loc[global_idx, 'title']
        genre = texts.loc[global_idx, 'genre']
        print(f"Index: {global_idx} | Title: {title} | Genre: {genre}")
        print(f"Original distance: {original[i].item():.4f}")
        print(f"Steered distance:  {steered[i].item():.4f}")
        print(f"CHANGE:            {delta[i].item():.4f}\n")

    print("LEAST CHANGED (based on distance difference)")
    print("-" * 50)
    for i in least_idx:
        global_idx = global_indices[i]
        title = texts.loc[global_idx, 'title']
        genre = texts.loc[global_idx, 'genre']
        print(f"Index: {global_idx} | Title: {title} | Genre: {genre}")
        print(f"Original distance: {original[i].item():.4f}")
        print(f"Steered distance:  {steered[i].item():.4f}")
        print(f"CHANGE:            {delta[i].item():.4f}\n")




def plot_2D_distance_projection(model, encoded_input, original_embeddings, all_texts_data, steering_vector, comparison_vector, layer_to_steer, steering_coefficient,
                                             steering_feature, comparison_feature, text_range=None, type="l2", normalize=True, print_differences=True, Show=True, Write=False, return_fig=False):
    """
    Plot a 2D projection of original vs steered embeddings based on distances to two semantic vectors.
    
    Args:
        model: The transformer model.
        encoded_input (dict): Tokenized input data.
        original_embeddings (torch.Tensor): Original embedding vectors.
        all_texts_data (pd.DataFrame): Text metadata with title and genre information.
        steering_vector (torch.Tensor): Primary steering vector to apply.
        comparison_vector (torch.Tensor): Secondary vector for comparison dimension.
        layer_to_steer (int): Layer index to apply steering to.
        steering_coefficient (float): Strength of the steering intervention.
        steering_feature (str): Name of the steering feature for labeling.
        comparison_feature (str): Name of the comparison feature for labeling.
        text_range (tuple, optional): Range of texts to analyze. Defaults to None.
        type (str, optional): Distance metric ('l2', 'l1', 'cosine'). Defaults to "l2".
        normalize (bool, optional): Whether to normalize embeddings. Defaults to True.
        print_differences (bool, optional): Whether to print most/least changed items. Defaults to True.
        Show (bool, optional): Whether to display the plot. Defaults to True.
        Write (bool, optional): Whether to save plot as HTML file. Defaults to False.
        return_fig (bool, optional): Whether to return the figure object. Defaults to False.
    
    Returns:
        plotly.graph_objects.Figure or None: Figure object if return_fig=True, otherwise None.
    """
    # Slice if needed
    if text_range is not None:
        s = slice(*text_range)
        encoded_input = {key: val[s] for key, val in encoded_input.items()}
        original_embeddings = original_embeddings[s]
        all_texts_data = all_texts_data.iloc[s]
    
    global_indices = all_texts_data.index.tolist()

    # Compute steered embeddings
    steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer, steering_coefficient, steering_vector, normalize=normalize,verbose=False)

    # Calculate distances
    orig_x, steer_x = calculate_specific_differences(original_embeddings, steered_embeddings, steering_vector, type)
    orig_y, steer_y = calculate_specific_differences(original_embeddings, steered_embeddings, comparison_vector, type)

    # Hover and labels
    hover_data_original, hover_data_steered = create_hover_data(all_texts_data, len(orig_x))
    genres = all_texts_data['genre'].tolist()[:len(orig_x)]

    df_original = pd.DataFrame({
        'X': orig_x,
        'Y': orig_y,
        'Cluster': genres,
        'Title': hover_data_original,
        'Type': 'Original'})

    df_steered = pd.DataFrame({
        'X': steer_x,
        'Y': steer_y,
        'Cluster': genres,
        'Title': hover_data_steered,
        'Type': 'Steered'})

    df = pd.concat([df_original, df_steered], ignore_index=True)

    # Plot
    fig = px.scatter(
        df,
        x='X',
        y='Y',
        color='Cluster',
        symbol='Type',
        hover_data=['Title', 'Cluster'],
        labels={
            'X': f'Distance to "{steering_feature}"',
            'Y': f'Distance to "{comparison_feature}"'})

    # Add dotted lines between original and steered
    for trace in fig.data:
        if 'Original' in trace.name:
            trace.opacity = 0.3
            trace.showlegend = False
        else:
            trace.opacity = 1.0
            trace.showlegend = True

    for i in range(len(orig_x)):
        fig.add_trace(go.Scatter(
            x=[orig_x[i], steer_x[i]],
            y=[orig_y[i], steer_y[i]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='none'))

    fig.update_layout(
        title=f'2D Projection - Steering: {steering_feature} | Comparison: {comparison_feature} | Distance: {type.upper()} | Layer: {layer_to_steer} | Coeff: {steering_coefficient}',
        width=900,
        height=700,
        xaxis_title=f'Distance to "{steering_feature}" ({type.upper()})',
        yaxis_title=f'Distance to "{comparison_feature}" ({type.upper()})')

    if Show:
        fig.show()
    if Write:
        fig.write_html(f'Semantic_2D_Projection_{steering_feature}_{comparison_feature}_{type}.html')
    if return_fig:
        return fig
    if print_differences:
        print_most_and_least_changed_2D(orig_x, orig_y, steer_x, steer_y, all_texts_data, steering_feature, comparison_feature, global_indices, top_k=3)




def print_most_and_least_changed_2D(orig_x, orig_y, steer_x, steer_y, texts, steering_feature, comparison_feature, global_indices, top_k=3):
    """
    Print the most and least changed movies in 2D distance space after steering.
    
    Args:
        orig_x (list): Original distances to steering feature.
        orig_y (list): Original distances to comparison feature.
        steer_x (list): Steered distances to steering feature.
        steer_y (list): Steered distances to comparison feature.
        texts (pd.DataFrame): Text metadata for display.
        steering_feature (str): Name of the steering feature.
        comparison_feature (str): Name of the comparison feature.
        global_indices (list): Global indices for the texts.
        top_k (int, optional): Number of top items to display. Defaults to 3.
    """
    orig_x = torch.tensor(orig_x)
    orig_y = torch.tensor(orig_y)
    steer_x = torch.tensor(steer_x)
    steer_y = torch.tensor(steer_y)

    delta = torch.sqrt((steer_x - orig_x)**2 + (steer_y - orig_y)**2)

    most_idx = torch.topk(delta, top_k).indices.tolist()
    least_idx = torch.topk(-delta, top_k).indices.tolist()

    print("\nMOST CHANGED in 2D distance space")
    print("-" * 50)
    for i in most_idx:
        global_idx = global_indices[i]
        title = texts.loc[global_idx, 'title'] if texts is not None else f"Movie {global_idx}"
        print(f"Index: {global_idx} | Title: {title}")
        print(f"Original ({steering_feature}, {comparison_feature}): ({orig_x[i].item():.4f}, {orig_y[i].item():.4f})")
        print(f"Steered  ({steering_feature}, {comparison_feature}): ({steer_x[i].item():.4f}, {steer_y[i].item():.4f})")
        print(f"Change: {delta[i].item():.4f}\n")

    print("\nLEAST CHANGED in 2D distance space")
    print("-" * 50)
    for i in least_idx:
        global_idx = global_indices[i]
        title = texts.loc[global_idx, 'title'] if texts is not None else f"Movie {global_idx}"
        print(f"Index: {global_idx} | Title: {title}")
        print(f"Original ({steering_feature}, {comparison_feature}): ({orig_x[i].item():.4f}, {orig_y[i].item():.4f})")
        print(f"Steered  ({steering_feature}, {comparison_feature}): ({steer_x[i].item():.4f}, {steer_y[i].item():.4f})")
        print(f"Change: {delta[i].item():.4f}\n")



def print_movie_info(index, data):
    """
    Print movie title and overview for a given index from the dataset.
    
    Args:
        index (int): Index of the movie in the dataset.
        data (pd.DataFrame): Dataset containing 'title' and 'overview' columns.
    """
    try:
        title = data.loc[index, "title"]
        overview = data.loc[index, "overview"]
        print(f"Index: {index}\nTitle: {title}\nOverview: {overview}")
    except KeyError:
        print(f"Index {index} not found in the dataset.")



#Example usage:
if __name__ == "__main__":
    model_name = "sentence-transformers/all-MiniLM-L12-v2"

    data = import_embedding_data_from_pkl('Test_export_embeddings.pkl', model=True, embeddings=True, encoded_input=True, all_texts_data=True)
    model, original_embeddings, encoded_input, all_texts_data = data

    # 1D projecttion to "War"
    layer_to_steer = 11
    steering_coefficient = 1
    feature = "War"
    normalize = True

    feature_texts, opposite_feature_texts = import_feature_texts(f"../Features/{feature}")
    steering_vector = get_steering_vector(model_name, feature_texts, layer_to_steer, opposite_texts=opposite_feature_texts, normalize=True)

    plot_distance_projection(model, encoded_input, original_embeddings, all_texts_data, layer_to_steer, steering_coefficient, 
                                steering_vector, feature, text_range=(900,1000), type="l2", normalize=normalize, print_differences=True)

    print_movie_info(996, all_texts_data)


    # 2D projection to "War" vs "Love"
    layer_to_steer = 11
    steering_coefficient = 1
    normalize = True

    steering_feature = "War"
    comparison_feature = "Love"

    steering_feature_texts, steering_opposite_feature_texts = import_feature_texts(f"../Features/{feature}")
    steering_vector = get_steering_vector(model_name, feature_texts, layer_to_steer, opposite_texts=opposite_feature_texts, normalize=True)

    comparison_feature_texts, comparison_opposite_feature_texts = import_feature_texts(f"../Features/{comparison_feature}")
    comparison_vector = get_steering_vector(model_name, comparison_feature_texts, layer_to_steer, opposite_texts=comparison_opposite_feature_texts, normalize=True)

    plot_2D_distance_projection(model, encoded_input, original_embeddings, all_texts_data, steering_vector, comparison_vector, layer_to_steer, 
                                steering_coefficient, steering_feature, comparison_feature, text_range=(900,1000), type="cosine", normalize=normalize, print_differences=True) #Legg til print_differences 
                            

