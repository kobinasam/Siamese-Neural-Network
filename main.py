import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense
from tensorflow import keras


from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

# Prepare the labeled dataset
data = pd.read_csv('./data/train.csv')
#data = [[-8.618643,41.141412],[-8.618499,41.141376],[-8.620326,41.14251],[-8.622153,41.143815],[-8.623953,41.144373],[-8.62668,41.144778],[-8.627373,41.144697],[-8.630226,41.14521],[-8.632746,41.14692],[-8.631738,41.148225],[-8.629938,41.150385],[-8.62911,41.151213],[-8.629128,41.15124],[-8.628786,41.152203],[-8.628687,41.152374],[-8.628759,41.152518],[-8.630838,41.15268],[-8.632323,41.153022],[-8.631144,41.154489],[-8.630829,41.154507],[-8.630829,41.154516],[-8.630829,41.154498],[-8.630838,41.154489]]
X_train = np.array(data.POLYLINE)
# Generating paris for similar and dissimilar
# Cluster-based Labeling and we need to create 2 Clusters
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(X_train)

# Generate pairs and assign labels based on cluster membership
X_pairs = []
y_labels = []
for i in range(num_clusters):
    cluster_indices = np.where(cluster_labels == i)[0]
    num_cluster_samples = len(cluster_indices)

    # Generate pairs within the cluster similar pairs
    for j in range(num_cluster_samples - 1):
        for k in range(j + 1, num_cluster_samples):
            traj_idx1 = cluster_indices[j]
            traj_idx2 = cluster_indices[k]
            X_pairs.append((X_train[traj_idx1], X_train[traj_idx2]))
            y_labels.append(1)

    # Generate pairs between different clusters dissimilar pairs
    for j in range(num_cluster_samples):
        other_cluster_indices = np.where(cluster_labels != i)[0]
        random_cluster_idx = np.random.choice(other_cluster_indices)
        traj_idx1 = cluster_indices[j]
        traj_idx2 = random_cluster_idx
        X_pairs.append((X_train[traj_idx1], X_train[traj_idx2]))
        y_labels.append(0)

y_train = y_labels

# 4. Create the Siamese network model
def create_siamese_model(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    shared_network = tf.keras.Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu')
    ])

    output_a = shared_network(input_a)
    output_b = shared_network(input_b)

    # Define a distance metric layer to compute the similarity
    distance = tf.keras.layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1, keepdims=True))
    similarity = distance([output_a, output_b])

    model = Model(inputs=[input_a, input_b], outputs=similarity)
    return model

# Create the Siamese network model
input_shape = X_train.shape[1:]  # Update with the appropriate shape of trajectory features
siamese_model = create_siamese_model(input_shape)
print(siamese_model)

# 5. Train the model
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
siamese_model.fit([X_pairs[:, 0], X_pairs[:, 1]], y_labels, epochs=10, batch_size=32)

# 6. Use the trained model to predict similarity
def predict_similarity(model, trajectory_a, trajectory_b):
    similarity_score = model.predict([[trajectory_a], [trajectory_b]])
    return similarity_score[0][0]

# Here we pass the test data for both trajectories
trajectory1 = []
trajectory2 = []
similarity_score = predict_similarity(siamese_model, trajectory1, trajectory2)
print(f"Similarity Score: {similarity_score}")
