from Embeddings import import_embedding_data_from_pkl, mean_pooling
from Steering_vector import import_steering_vector_from_pkl
from Steering import get_steered_embeddings_vector
from Activations_categories import calculate_mean_of_categories

from sklearn.cluster import KMeans
import torch
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_hover_data(all_texts_data, number_of_texts):
    """
    Create hover data for PCA visualizations with title and genre information.
    
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

def do_kmeans(embeddings, n_clusters):
    """
    Perform K-means clustering on embeddings.
    
    Args:
        embeddings (numpy.ndarray): Embedding vectors to cluster.
        n_clusters (int): Number of clusters to create.
    
    Returns:
        numpy.ndarray: Cluster labels for each embedding.
    """
    # Cluster with KMeans
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    print(f"Did kmeans clustering with {n_clusters} clusters")
    return labels

def plot_pca_fixed_kmeans(original_embeddings, steered_embeddings, all_texts_data, info_string,
                           text_range=None, projected=True, n_clusters=5, lines=True, Write=False):
    """
    Plot PCA visualization of original vs steered embeddings with K-means clustering.
    
    Args:
        original_embeddings (torch.Tensor): Original embedding vectors.
        steered_embeddings (torch.Tensor): Steered embedding vectors.
        all_texts_data (pd.DataFrame): Text metadata with title and genre information.
        info_string (str): Information string for plot title and filename.
        text_range (tuple, optional): Range of texts to analyze. Defaults to None.
        projected (bool, optional): If True, fit PCA only on original embeddings. Defaults to True.
        n_clusters (int, optional): Number of K-means clusters. Defaults to 5.
        lines (bool, optional): Whether to draw lines between original and steered points. Defaults to True.
        Write (bool, optional): Whether to save plot as HTML file. Defaults to False.
    
    Note:
        If projected=True, fits PCA only on original embeddings and projects steered embeddings.
        If projected=False, fits PCA on combined embeddings.
    """
    # Apply slicing if requested
    if text_range is not None:
        s = slice(*text_range)
        all_texts_data = all_texts_data.iloc[s]
        original_embeddings = original_embeddings[s]
        steered_embeddings = steered_embeddings[s]

    hover_data_original, hover_data_steered = create_hover_data(all_texts_data, len(original_embeddings))

    # Prepare data
    original_np = original_embeddings.cpu().numpy()
    steered_np = steered_embeddings.cpu().numpy()
    
    # Fit K-Means on original embeddings
    original_labels = do_kmeans(original_np, n_clusters)

    if projected:
        # Fit PCA only on original and transform both sets
        pca = PCA(n_components=2)
        pca_original = pca.fit_transform(original_np)
        pca_steered = pca.transform(steered_np)
        title_prefix = "PCA Projection:"
    else:
        # PCA for all points combined
        combined_embeddings = torch.cat([original_embeddings, steered_embeddings], dim=0).cpu().numpy()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(combined_embeddings)
        
        # Split transformed points
        pca_original = X_pca[:len(original_embeddings)]
        pca_steered = X_pca[len(original_embeddings):]
        title_prefix = "PCA:"

    # Create DataFrame
    df_original = pd.DataFrame(pca_original, columns=['PCA 1', 'PCA 2'])
    df_original['Cluster'] = original_labels.astype(str)
    df_original['Title'] = hover_data_original
    df_original['Type'] = 'Original'

    df_steered = pd.DataFrame(pca_steered, columns=['PCA 1', 'PCA 2'])
    df_steered['Cluster'] = original_labels.astype(str)
    df_steered['Title'] = hover_data_steered
    df_steered['Type'] = 'Steered'

    df = pd.concat([df_original, df_steered], ignore_index=True)

    fig = px.scatter(
        df,
        x='PCA 1',
        y='PCA 2',
        color='Cluster',
        symbol='Type',
        hover_name='Title',
        hover_data={'PCA 1': ':.3f', 'PCA 2': ':.3f', 'Cluster': True, 'Type': True},
        color_discrete_sequence=px.colors.qualitative.Set1)


    # Lower opacity for original points
    for trace in fig.data:
        if 'Original' in trace.name:
            trace.opacity = 0.3  # Transparent for original
        else:
            trace.opacity = 1.0  # Solid for steered

    if lines:
        # Add lines between original and steered points
        for i in range(len(df_original)):
            fig.add_trace(go.Scatter(
                x=[df_original.iloc[i]['PCA 1'], df_steered.iloc[i]['PCA 1']],
                y=[df_original.iloc[i]['PCA 2'], df_steered.iloc[i]['PCA 2']],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='none'))
    
    if Write:
        fig.write_html(f"{title_prefix} {info_string}.html")

    fig.update_layout(title=f'{title_prefix} Original vs. Steered Embeddings {info_string}', width=900, height=700)
    fig.show()



def plot_pca_labeled_projection(original_embeddings, steered_embeddings, all_texts_data, info_string, steering_vector=None, text_range=None, lines=True, Write=False):
    """
    Plot PCA visualization with genre-based labels and optional steering vector projection.
    
    Args:
        original_embeddings (torch.Tensor): Original embedding vectors.
        steered_embeddings (torch.Tensor): Steered embedding vectors.
        all_texts_data (pd.DataFrame): Text metadata with 'genre' column for labeling.
        info_string (str): Information string for plot title and filename.
        steering_vector (torch.Tensor, optional): Steering vector to project and display. Defaults to None.
        text_range (tuple, optional): Range of texts to analyze. Defaults to None.
        lines (bool, optional): Whether to draw lines between original and steered points. Defaults to True.
        Write (bool, optional): Whether to save plot as HTML file. Defaults to False.
    
    Note:
        Fits PCA only on original embeddings and projects steered embeddings for consistency.
        Uses genre information for color-coding points.
    """

    # Apply slicing if requested
    if text_range is not None:
        s = slice(*text_range)
        all_texts_data = all_texts_data.iloc[s]
        original_embeddings = original_embeddings[s]
        steered_embeddings = steered_embeddings[s]

    hover_data_original, hover_data_steered = create_hover_data(all_texts_data, len(original_embeddings))

    # Prepare PCA projection from original embeddings
    original_np = original_embeddings.cpu().numpy()
    steered_np = steered_embeddings.cpu().numpy()

    # Get the labels from the genre 
    original_labels = all_texts_data['genre'].tolist()[:len(original_np)]

    # Fit PCA only on original and transform both sets
    pca = PCA(n_components=2)
    pca_original = pca.fit_transform(original_np)
    pca_steered = pca.transform(steered_np)
    if steering_vector is not None:
        pca_mean = pca.transform(steering_vector.unsqueeze(0).cpu().numpy())

    # Create DataFrames
    df_original = pd.DataFrame(pca_original, columns=['PCA 1', 'PCA 2'])
    df_original['Cluster'] = [str(label) for label in original_labels]
    df_original['Title'] = hover_data_original
    df_original['Type'] = 'Original'
    df_original['show_legend'] = False

    df_steered = pd.DataFrame(pca_steered, columns=['PCA 1', 'PCA 2'])
    df_steered['Cluster'] = [str(label) for label in original_labels]  # same clusters
    df_steered['Title'] = hover_data_steered
    df_steered['Type'] = 'Steered'
    df_steered['show_legend'] = True

    df = pd.concat([df_original, df_steered], ignore_index=True)

    # Create scatter plot
    fig = px.scatter(df, 
                     x='PCA 1', 
                     y='PCA 2', 
                     color='Cluster', 
                     symbol='Type', 
                     hover_name='Title',
                     hover_data={'PCA 1': ':.3f', 'PCA 2': ':.3f', 'Cluster': True, 'Type': True},
                     color_discrete_sequence=px.colors.qualitative.Set1)

    if steering_vector is not None:
        fig.add_trace(go.Scatter(
        x=[pca_mean[0, 0]],
        y=[pca_mean[0, 1]],
        mode='markers+text',
        marker=dict(size=12, color='black', symbol='x'),
        text=["Mean point"],
        textposition="top center",
        name='Steering Vector'
        ))
    
    # Transparency
    for trace in fig.data:
        if 'Original' in trace.name:
            trace.showlegend = False
            trace.opacity = 0.3
        else:
            trace.opacity = 1.0

    # Optionally draw dotted lines between original and steered
    if lines:
        for i in range(len(df_original)):
            fig.add_trace(go.Scatter(
                x=[df_original.iloc[i]['PCA 1'], df_steered.iloc[i]['PCA 1']],
                y=[df_original.iloc[i]['PCA 2'], df_steered.iloc[i]['PCA 2']],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='none',
            ))

    fig.update_layout(
        title=f'PCA Projection: Original vs. Steered Embeddings {info_string}',
        width=900,
        height=700
    )

    if Write:
        fig.write_html(f"PCA Projection: {info_string}.html")

    fig.show()


if __name__ == "__main__":
    # Example usage
    # Import data
    data = import_embedding_data_from_pkl('Test_export_embeddings.pkl', model=True, embeddings=True, encoded_input=True, all_texts_data=True)
    model, original_embeddings, encoded_input, all_texts_data = data

    feature = "Love"
    layer_to_steer = 11
    steering_coefficient = 0.5

    steering_vector = import_steering_vector_from_pkl('steering_vector.pkl', feature_name=feature, layer_to_steer=layer_to_steer)
    steered_embeddings = get_steered_embeddings_vector(model, encoded_input, layer_to_steer=layer_to_steer, steering_coefficient=steering_coefficient, steering_vector=steering_vector, normalize=True)

    # Use movie titles for hover data instead of token IDs
    hover_titles = all_texts_data['title'].tolist()

    info_string = f"| Layer: {layer_to_steer} | Feature: {feature} | Steering: {steering_coefficient}" # Vector steering

    # Plot PCA with fixed KMeans clustering
    print("Plotting PCA with fixed KMeans clustering...")
    plot_pca_fixed_kmeans(original_embeddings, steered_embeddings, all_texts_data, info_string,
                           text_range=None, projected=True, n_clusters=5, lines=True, Write=False)
                        
    
    # Plot PCA with labeled projection
    print("Plotting PCA with labeled projection...")
    plot_pca_labeled_projection(original_embeddings, steered_embeddings, all_texts_data, info_string, 
                                steering_vector=None, text_range=None, lines=True, Write=False)
