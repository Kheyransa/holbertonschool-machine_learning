# Dimensionality Reduction

This project introduces **dimensionality reduction** using **Principal Component Analysis (PCA)**.

Dimensionality reduction is an important technique in machine learning because real-world datasets can contain hundreds or thousands of features. Working with too many dimensions can increase computation time, make visualization difficult, and sometimes introduce redundant information.

The goal of PCA is to represent the data using fewer dimensions while preserving as much of the important information as possible.

---

## What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms the original features into a new set of variables called **principal components**.

Each principal component represents a direction in the data.

The first component captures the largest possible amount of variance, the second captures the next largest amount, and so on.

For example, if we have:

```text
X → (2500, 784)
```

we can reduce it to:

```text
T → (2500, 50)
```

This means that each of the 2500 data points is now represented using only 50 dimensions instead of 784.

---

## Why Reduce Dimensions?

Dimensionality reduction can be useful because it can:

* Reduce computational cost
* Reduce memory usage
* Remove redundant information
* Make datasets easier to visualize
* Help simplify machine learning models
* Reduce the effect of highly correlated features

For example, suppose a dataset contains 100 features, but many of them contain almost the same information. PCA can find a smaller number of directions that represent most of the variation in the original dataset.

---

# PCA and Variance

PCA is based on **variance**.

Variance tells us how much the data changes along a particular direction.

PCA searches for directions where the data has the largest variance.

The principal components are ordered from the most important to the least important:

```text
PC1 → largest variance
PC2 → second largest variance
PC3 → third largest variance
...
```

Therefore, if we keep only the first few components, we can reduce the dimensionality while retaining most of the information.

---

# PCA Using SVD

PCA can be implemented using **Singular Value Decomposition (SVD)**.

For a matrix `X`:

[
X = U\Sigma V^T
]

where:

* `U` contains the left singular vectors
* `Σ` contains the singular values
* `Vᵀ` contains the right singular vectors

The rows of `Vᵀ` correspond to the principal directions.

The singular values determine how much variance is associated with each component. The variance explained by a component is proportional to:

[
s_i^2
]

where (s_i) is the corresponding singular value.

---

# PCA Transformation

After selecting the desired principal components, we create a transformation matrix `W`.

The original dataset can then be transformed using:

[
T = XW
]

where:

* `X` = original dataset
* `W` = selected principal component directions
* `T` = transformed, lower-dimensional dataset

For example:

```text
X = (2500, 784)
W = (784, 50)

T = XW

T = (2500, 50)
```

The number of rows stays the same because we still have the same data points. Only the number of dimensions changes.

---

# Tasks

## 0. PCA

File:

```text
0-pca.py
```

Function:

```python
def pca(X, var=0.95):
```

This version of PCA determines **how many principal components should be kept** based on the desired fraction of preserved variance.

For example:

```python
W = pca(X, 0.95)
```

means:

> Keep enough principal components to preserve at least 95% of the original variance.

The function returns the transformation matrix:

```text
W → (d, nd)
```

where `nd` is automatically determined based on `var`.

### Process

1. Perform SVD on `X`.
2. Obtain the singular values.
3. Calculate the variance explained by each component.
4. Calculate cumulative explained variance.
5. Find the minimum number of components needed to reach the requested variance.
6. Return the corresponding principal directions.

---

## 1. PCA v2

File:

```text
1-pca.py
```

Function:

```python
def pca(X, ndim):
```

This version is slightly different.

Instead of specifying how much variance to preserve, we directly specify the desired number of dimensions.

For example:

```python
T = pca(X, 50)
```

means:

> Reduce the original dataset to exactly 50 dimensions.

If:

```text
X.shape = (2500, 784)
```

then:

```text
T.shape = (2500, 50)
```

### Process

1. Perform SVD on `X`.
2. Select the first `ndim` principal components.
3. Construct the transformation matrix.
4. Project the original data onto the selected components.

---

# Difference Between Task 0 and Task 1

The main difference is **how we choose the number of components**.

### Task 0

We specify the amount of variance:

```python
W = pca(X, 0.95)
```

The algorithm decides:

```text
How many components do I need to preserve 95%?
```

### Task 1

We specify the number of dimensions:

```python
T = pca(X, 50)
```

We are saying:

```text
I want exactly 50 components.
```

So:

```text
Task 0 → variance → determine dimensions

Task 1 → dimensions → directly transform data
```

---

# Example

The provided dataset contains:

```text
2500 samples
784 features
```

We can reduce it using:

```python
T = pca(X, 50)
```

The resulting matrix has:

```text
T.shape
```

```text
(2500, 50)
```

Instead of processing 784 features for every data point, we now work with only 50 principal components.

---

# Files

```text
dimensionality_reduction/
│
├── 0-pca.py
├── 0-main.py
│
├── 1-pca.py
├── 1-main.py
│
└── README.md
```

---

# Running the Tests

Run the provided main programs:

```bash
./0-main.py
```

and:

```bash
./1-main.py
```

For task 0, the returned matrix `W` can be used to transform the data:

```python
T = np.matmul(X, W)
```

For task 1, the function directly returns the transformed data:

```python
T = pca(X, 50)
```

---

# Key Takeaways

* **PCA** is used to reduce the number of dimensions in a dataset.
* PCA finds directions that capture the greatest amount of variance.
* These directions are called **principal components**.
* SVD can be used to calculate the principal components.
* Larger singular values correspond to components containing more variance.
* The transformed data is obtained using:

[
T = XW
]

* Task 0 chooses the number of components based on a required variance.
* Task 1 uses a specified number of components.

### In short:

```text
High-dimensional data
        ↓
       PCA
        ↓
Find important directions
        ↓
Keep the most important components
        ↓
Lower-dimensional data
```

PCA allows us to simplify a dataset while attempting to preserve its most important structure.
