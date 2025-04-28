# Standard library imports
import os
import numpy as np
import random
import pandas as pd

# Third-party imports
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from sklearn.metrics import ndcg_score

# Define constants for relevance labels
EXACT = "Exact"
SUBSTITUTE = "Substitute"
COMPLEMENT = "Complement"
IRRELEVANT = "Irrelevant"

# Define a dictionary that maps ESCI labels to numerical relevance scores.
LABEL_TO_SCORE = {
    EXACT: 3,  # Highest relevance
    SUBSTITUTE: 2,
    COMPLEMENT: 1,
    IRRELEVANT: 0  # Lowest relevance
    }

def create_ndcg_df(dataset, model):
    """
    Compute NDCG@10 scores for a dataset of queries and product titles.

    This function calculates the Normalized Discounted Cumulative Gain (NDCG) at rank 10
    for each query in the dataset. It uses a pre-trained SentenceTransformer model to
    compute embeddings for queries and product titles, calculates cosine similarity
    scores, and evaluates the ranking quality using NDCG.

    Args:
        dataset (pd.DataFrame): A Pandas DataFrame containing the following columns:
            - 'query': The query text.
            - 'product_title': A list of product titles associated with the query.
            - 'esci_label': A list of relevance labels (e.g., 'exact', 'substitute').
        model (SentenceTransformer): A pre-trained SentenceTransformer model used to
            encode queries and product titles into embeddings.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'query': The query text.
            - 'ndcg_score': The NDCG@10 score for the query.

    """
    # Initialize a list to store NDCG scores and queries
    ndcg_scores = []
    queries = []

    # Iterate through each row in dateset
    for _, row in dataset.iterrows():
        # Get the query and product titles
        query = row['query']
        product_titles = row['product_title']
        
        # Compute embeddings for the query and product titles
        query_embedding = model.encode(query, convert_to_tensor=True)
        title_embeddings = model.encode(product_titles, convert_to_tensor=True)
        
        # Calculate cosine similarity as predicted relevance scores
        cos_sim = util.pytorch_cos_sim(query_embedding, title_embeddings).cpu().numpy().flatten()
        
        # Get true relevance scores
        true_relevance = [LABEL_TO_SCORE[label] for label in row['esci_label']]

        # Compute NDCG@10 for the current query
        ndcg = ndcg_score([true_relevance], [cos_sim], k=10)
        ndcg_scores.append(ndcg)
        queries.append(query)

    # Create a DataFrame with queries and their NDCG scores and Sort by NDCG scores in ascending order
    ndcg_df = pd.DataFrame({
        'query': queries,
        'ndcg_score': ndcg_scores
        }).reset_index(drop=True)
    
    return ndcg_df

def main():
    # Load ESCI dataset
    esci_dataset = load_dataset("tasksource/esci")

    # Define columns to be used
    cols = ['query', 'product_title', 'product_description', 'esci_label']

    # Select relevant columns from the test dataset
    test_dataset = esci_dataset['test'].select_columns(cols)

    # Since the dataset is large, I will sample 1000 unique queries for evaluation.
    # Another reason for sampling is that I do not have sufficient GPU resources to process the entire dataset.
    unique_queries = list(set(test_dataset['query']))
    random.seed(42)
    random_queries = random.sample(unique_queries, 1000)

    # Filter the test dataset to include only the sampled queries
    filtered_test = test_dataset.filter(lambda row: row['query'] in random_queries)

    # Convert filtered_test to a Pandas DataFrame
    filtered_test_df = pd.DataFrame(filtered_test)

    # Group by query and aggregate product_title and esci_label into lists
    query_df = filtered_test_df.groupby('query').agg({
        'product_title': list,
        'esci_label': list
        }).reset_index()

    # Keep rows where esci_label has no NaN values and length > 10
    filtered_query_df = query_df[query_df['esci_label'].apply(lambda x: pd.notna(x).all() and len(x) > 10)]

    # Load the model
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # Compute NDCG scores
    ndcg_df = create_ndcg_df(filtered_query_df, model)

    # Merge the NDCG DataFrame with the filtered query_df DataFrame
    query_ndcg_df = pd.merge(ndcg_df, filtered_query_df, on='query')
    query_ndcg_df.sort_values(by='ndcg_score', ascending=True).reset_index(drop=True)

    # Ensure the output folder exists
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder, "query_ndcg_df.parquet")
    # Export the NDCG DataFrame to a parquet file
    query_ndcg_df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    main()