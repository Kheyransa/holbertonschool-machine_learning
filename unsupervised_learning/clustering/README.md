# Clustering

This project introduces **clustering**, an unsupervised learning technique used to group similar data points together.

## What is Clustering?

Clustering is used when we have data without predefined labels and want to discover natural groups within the dataset.

For example, given a set of points, a clustering algorithm can identify groups such as:

```text
Dataset
   ↓
Clustering
   ↓
Cluster 1   Cluster 2   Cluster 3
```

Points within the same cluster should be more similar to each other than to points in other clusters.

## K-means Clustering

This project focuses mainly on **K-means**, a popular clustering algorithm.

The basic process is:

1. Initialize `k` cluster centroids.
2. Assign each data point to its closest centroid.
3. Recalculate the centroids based on the assigned points.
4. Repeat until the centroids stabilize or the maximum number of iterations is reached.

### Initialization

The initial centroids are randomly generated within the minimum and maximum values of each feature.

```python
centroids = initialize(X, k)
```

The result has shape:

```text
(k, d)
```

where `k` is the number of clusters and `d` is the number of dimensions.

## Project Structure

```text
clustering/
├── 0-initialize.py
├── 1-kmeans.py
├── 2-bimodal.py
├── 3-optimum.py
├── ...
└── README.md
```

## Key Idea

K-means tries to minimize the distance between data points and the centroids of their assigned clusters.

```text
Data
 ↓
Initialize centroids
 ↓
Assign points
 ↓
Update centroids
 ↓
Repeat
 ↓
Final clusters
```

Clustering is useful for tasks such as **customer segmentation, image grouping, anomaly detection, and exploratory data analysis**.
