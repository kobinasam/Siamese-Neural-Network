# THIS IS IMPORTANT TO ACHIEVE REPRODUCIBILITY WITH TENSORFLOW. MUST HAPPEN BEFORE TENSORFLOW IMPORT
import os

# SHOULD HAVE THIS ENVIRONMENT VARIABLE SET BEFORE PYTHON EVEN BEGINS EXECUTION
os.environ['PYTHONHASHSEED']=str(1)

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

# Prepare the labeled dataset
data = pd.read_csv('./data/testdata.csv')
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

# Create the Siamese network model
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


input_shape = X_train.shape[1:]
siamese_model = create_siamese_model(input_shape)
print(siamese_model)

# Train the model
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
siamese_model.fit([X_pairs[:, 0], X_pairs[:, 1]], y_labels, epochs=10, batch_size=32)


# Use the trained model to predict similarity
def predict_similarity(model, trajectory_a, trajectory_b):
    similarity_score = model.predict([[trajectory_a], [trajectory_b]])
    return similarity_score[0][0]


# Here we pass the test data for both trajectories
trajectory1 = []
trajectory2 = []
similarity_score = predict_similarity(siamese_model, trajectory1, trajectory2)
print(f"Similarity Score: {similarity_score}")
