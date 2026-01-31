# Fashion-MNIST: Naive Bayes vs Random Forest

This project applies two classical machine learning algorithms to the Fashion-MNIST dataset:
Gaussian Naive Bayes and Random Forest, and compares their performance.

## Dataset

- Fashion-MNIST from `tensorflow.keras.datasets`
- 60,000 train images and 10,000 test images
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Methods

1. **Gaussian Naive Bayes**
   - Input features: 784 pixel values scaled to [0, 1]
   - Model: `GaussianNB(var_smoothing=1e-2)`

2. **Random Forest**
   - Input features: raw 784-dimensional pixel vectors
   - Model: `RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)`

## Results

- GaussianNB test accuracy: **0.6715**
- RandomForest test accuracy: **0.8778**

The Random Forest improves accuracy by roughly 20 percentage points over Gaussian Naive Bayes on the same dataset.

## What the notebook shows

- Loading and reshaping Fashion-MNIST
- Training and evaluating GaussianNB and RandomForest
- Classification report for Random Forest
- Confusion matrices for both models
- Visualization of some misclassified examples

## How to run

1. Open the notebook directly in Google Colab (link at the top of the `.ipynb` file), or clone the repo and run locally.
2. Install dependencies:
   - `numpy`
   - `matplotlib`
   - `scikit-learn`
   - `tensorflow` (for loading Fashion-MNIST)
3. Run all cells in order.
