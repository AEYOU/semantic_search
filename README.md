# semantic_search


This project implements a semantic search pipeline for evaluating query relevance using the ESCI dataset. It leverages the `SentenceTransformer` model for embedding queries and product titles and computes NDCG (Normalized Discounted Cumulative Gain) scores to assess ranking quality.

## Project Structure

- **`script.py`**: The main script for processing the ESCI dataset, computing NDCG scores, and saving the results to a Parquet file.
- **`analysis.ipynb`**: A Jupyter Notebook for analyzing the computed NDCG scores and discussing results.

## Install the required dependencies:
`pip install -r requirements.txt`

## Usage
1. Run the script:
`python script.py`<br>
The output will be saved in the `output` directory as `query_ndcg_df.parquet`.

2. Run the notebook:
`analysis.ipynb`