#  Water Pump Project
# Sabrina Abdukadirova
# CMPSC 463


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Step 0: Load Dataset
# -----------------------
df = pd.read_csv("rul_hrs.csv").iloc[:10000]  # First 10k rows
print("Dataset loaded. Shape:", df.shape)


# Step 1: Transform RUL into 4 categories
# -----------------------
Q10 = df['rul'].quantile(0.10)
Q50 = df['rul'].quantile(0.50)
Q90 = df['rul'].quantile(0.90)


def get_rul_category(rul):
    if rul < Q10:
        return "Extremely Low RUL"
    elif rul < Q50:
        return "Moderately Low RUL"
    elif rul < Q90:
        return "Moderately High RUL"
    else:
        return "Extremely High RUL"


df['rul_category'] = df['rul'].apply(get_rul_category)
print("RUL categories added.")



# Step 2: Divide-and-Conquer Segmentation
# -----------------------
def segment_signal(signal, threshold=0.1):
    """Recursive segmentation based on variance."""
    if len(signal) <= 1:
        return [(0, len(signal) - 1)]
    var = np.var(signal)
    if var > threshold:
        mid = len(signal) // 2
        left_segments = segment_signal(signal[:mid], threshold)
        right_segments = segment_signal(signal[mid:], threshold)
        right_segments = [(start + mid, end + mid) for start, end in right_segments]
        return left_segments + right_segments
    else:
        return [(0, len(signal) - 1)]



# Automatically detect all sensor columns
# -----------------------
sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
print(f"Using {len(sensor_cols)} sensors: {sensor_cols}")

# Initialize segmentation scores
segmentation_scores = {}

# Plot first 10 sensors for readability
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    signal = df[col].values
    segments = segment_signal(signal, threshold=0.1)
    segmentation_scores[col] = len(segments)

    # Plot only first 10 sensors
    if i < 10:
        ax = axes[i]
        ax.plot(signal, label=col, color='blue')
        for start, end in segments:
            ax.axvline(x=end, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"{col} ({len(segments)} segments)")
        ax.set_ylabel("Value")
        ax.set_xticks([])

plt.tight_layout()
plt.show()

print("Segmentation complexity scores (number of segments per sensor):")
for k, v in segmentation_scores.items():
    print(f"{k}: {v}")


# Step 3: Simple Clustering (from scratch)
# -----------------------
def simple_kmeans(X, k=4, max_iters=50):
    # Randomly initialize centroids
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        # Assign each point to nearest centroid
        distances = np.sqrt(((X[:, None] - centroids[None, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        # Update centroids
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = X[labels == i].mean(axis=0)
    return labels


# Use all available sensors for clustering
X = df[sensor_cols].values
clusters = simple_kmeans(X, k=4)
df['cluster'] = clusters

# Count majority RUL category per cluster
print("\nCluster → Majority RUL category mapping:")
for c in range(4):
    cluster_ruls = df[df['cluster'] == c]['rul_category']
    if len(cluster_ruls) > 0:
        majority = cluster_ruls.value_counts().idxmax()
        print(f"Cluster {c} majority RUL category: {majority}")
    else:
        print(f"Cluster {c} is empty")



# Step 4: Maximum Subarray (Kadane)
# -----------------------
def kadane(arr):
    max_sum = current_sum = 0
    start = end = s = 0
    for i in range(len(arr)):
        current_sum += arr[i]
        if current_sum < 0:
            current_sum = 0
            s = i + 1
        elif current_sum > max_sum:
            max_sum = current_sum
            start = s
            end = i
    return max_sum, start, end


print("\nKadane max-deviation intervals and early indicators:")
early_indicators = []

for col in sensor_cols:
    signal = df[col].values
    # Preprocess: absolute first-difference minus mean
    diff_signal = np.abs(np.diff(signal)) - np.mean(np.abs(np.diff(signal)))
    max_sum, start_idx, end_idx = kadane(diff_signal)

    interval_rul = df['rul_category'].iloc[start_idx:end_idx + 1].value_counts()
    print(f"{col}: interval {start_idx}-{end_idx}, sum={max_sum}")
    print("RUL categories in interval:\n", interval_rul, "\n")

    # Identify potential early indicators of low RUL
    total_interval = end_idx - start_idx + 1
    if "Extremely Low RUL" in interval_rul and interval_rul["Extremely Low RUL"] > total_interval / 2:
        early_indicators.append(col)

print("Sensors that may serve as early indicators of low RUL (based on max deviation interval):")
print(early_indicators)