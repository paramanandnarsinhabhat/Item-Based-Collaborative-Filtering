# Item-Based Collaborative Filtering

This project implements an item-based collaborative filtering recommender system using Python. It utilizes user-item interactions to predict user preferences based on item similarity.

## Requirements

The project requires the following packages:

- pandas
- numpy
- scikit-learn
- scikit-surprise

You can install these packages using the provided `requirements.txt` file with the following command:

```
pip install -r requirements.txt
```

## Usage

The recommender system is implemented in a Python script which includes the following steps:

1. Import necessary libraries.
2. Read user ratings and movie information datasets.
3. Merge movie information into the ratings dataframe.
4. Combine movie ID and movie title and store it in a new column.
5. Retain only the essential columns in the ratings dataframe.
6. Split data into training and test sets and define the evaluation metric (RMSE).
7. Implement a simple baseline using the average of all ratings.
8. Perform item-based collaborative filtering using:
   - Simple item mean
   - Similarity weighted mean (using cosine similarity)
9. Conduct a grid search to optimize the neighborhood size and similarity measure for the KNNWithMeans algorithm.

## Data

The data consists of the following:

- User ratings
- Movie information

The data should be prepared as described in the script's comments to ensure compatibility with the system.

## Evaluation

The system's performance is evaluated using the root mean squared error (RMSE) metric.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


