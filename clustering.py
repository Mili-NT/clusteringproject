import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from fcmeans import FCM
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz

# TODO: DETERMINE APPROPRIATE NUMBER OF CLUSTERS INSTEAD OF HARDCODING

def boxplot_associate(df, indexes, graph_info_array):
    """
    :param df: The pandas dataframe containing the data to graph
    :param indexes: a tuple containing the x and y indexes to associate
    :param graph_info_array: an array containing graph information -> [xlabel, ylabel, title]

    Example:
        To create a boxplot associating gender and spending score:
            boxplot_associate(df, (1, 4), ["Gender", "Spending Score", "Gender-Spending Score Association])
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df.iloc[:, indexes[0]], y=df.iloc[:, indexes[1]], data=df)
    plt.xlabel(graph_info_array[0])
    plt.ylabel(graph_info_array[1])
    plt.title(graph_info_array[2])
    plt.grid(True)
    plt.show()

def import_data():
    df = pd.read_csv('Mall_Customers.csv')
    # One Hot Encoding
    df = pd.get_dummies(df, columns=["Genre"])
    # MinMax Scale 0-1
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]] = MinMaxScaler().fit_transform(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
    return df

def explore_data(df):
    # Information about the dataset including the mean values and standard deviations for each column (check here for normalization info)
    print(f"{df.describe()}\n")
    # 200 entries, 200 non-null counts for each column means there are no missing values that need to be accounted for
    print(f"{df.info()}\n")
    print(f"Missing values:\n{df.isnull().sum()}\n")

def visual_eda(df):
    # Value distribution (histogram)
    df.hist(bins=30, figsize=(15, 10))
    plt.show()
    sns.displot(df.iloc[:, 3], kde=True)
    sns.displot(df.iloc[:, 2], kde=True)
    sns.displot(df.iloc[:, 4], kde=True)
    # Seaborn scatterplots
    sns.pairplot(df)
    plt.show()
    # Gender-Spending Score Association (Boxplot)
    boxplot_associate(df, (1,4), ["Gender", "Spending Score", "Gender-Spending Score Association"])
    # Gender-Annual Income Association (Boxplot)
    boxplot_associate(df, (1,3), ["Gender", "Annual Income", "Gender-Annual Income Association"])

# Placeholder Function for determining correct cluster count
def nclusters_by_elbow_method(selected_features):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=50, random_state=0)
        kmeans.fit(selected_features)
        wcss.append(kmeans.inertia_)
    print(wcss)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    # TODO: FINISH IMPLEMENTING ELBOW METHOD NCLUSTER SELECTION
    # Currently, this returns the previous hardcoded ncluster value until we implement selecting from the inertia scores
    # by calculating the gradient with diff
    return 5

# Function to apply KMeans and calculate silhouette score
def kmeans_silhouette(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    return score


def k_cluster(selected_features, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=50, random_state=0)
    clusters = kmeans.fit_predict(selected_features)
    clusters_n = len(np.unique(clusters))
    return clusters, clusters_n

#Applying fuzzy-k
def fuzzy_cmeans(selected_features, n_clusters):
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(selected_features.values)
    clusters = fcm.predict(selected_features.values)
    clusters_n = len(np.unique(clusters))
    return clusters, clusters_n

# Function to apply Fuzzy C-Means and calculate silhouette score
def fuzzy_cmeans_silhouette(selected_features, n_clusters):
    # Fuzzy C-Means algorithm using skfuzzy
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(selected_features.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    labels = np.argmax(u, axis=0)  # Assign each point to the cluster with the highest membership
    score = silhouette_score(selected_features, labels)
    return score

#Applying DBSCAN Optimizing
def dbscan_opt(selected_features, minPts, k=10 ):
    """
    Plots the k-distance graph to help estimate the optimal eps for DBSCAN.
    Parameters:
    - data: The scaled dataset for clustering
    - minPts: The number of nearest neighbors to consider (default = 4)
    """

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(selected_features)
    distances, _ = neighbors_fit.kneighbors(selected_features)
    k_distances = np.sort(distances[:, k - 1], axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.xlabel('Data Points (sorted by distance)')
    plt.ylabel(f'{minPts}-th Nearest Neighbor Distance')
    plt.title('k-distance Graph for DBSCAN')
    plt.grid(True)
    plt.show()

    # Choosing an epsilon based on the elbow point from the graph (adjustable)
    optimal_eps = np.percentile(k_distances, 90)

    # Step 2: Optimize min_samples using silhouette score for the optimized eps
    best_score = -1
    best_min_samples = None
    best_labels = None

    for min_samples in minPts:
        dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
        labels = dbscan.fit_predict(selected_features)

        # Only calculate silhouette score if more than 1 cluster is found
        if len(set(labels)) > 1:
            score = silhouette_score(selected_features, labels)
            if score > best_score:
                best_score = score
                best_min_samples = min_samples
                best_labels = labels

    return best_labels, optimal_eps, best_min_samples, best_score

    #Applying DBSCAN
def dbscan(features, minPts, ep):
    dbscan_model = DBSCAN(eps=ep, min_samples=minPts)
    final_labels = dbscan_model.fit_predict(features)
    count_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)

    return final_labels, count_clusters

# Function to apply GMM and calculate silhouette score
def gmm_silhouette(X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(X)
    score = silhouette_score(X, labels)
    return score

#Finiding Optimial n components for Gaussian Matrix Mixture
def find_best_n_components(selected_features):
    bic = []
    aic = []
    n_components_range = range(1, 11)  # Try cluster counts from 1 to 10
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(selected_features)
        bic.append(gmm.bic(selected_features))
        aic.append(gmm.aic(selected_features))

    # Plot BIC and AIC
    plt.plot(n_components_range, bic, label='BIC')
    plt.plot(n_components_range, aic, label='AIC')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC/AIC')
    plt.legend()
    plt.show()

    # Return the optimal number of components
    optimal_n_components = n_components_range[bic.index(min(bic))]
    return optimal_n_components


def apply_gmm(X_scaled, n_components=3):
    # Apply GMM with a specified number of components (clusters)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    clusters = gmm.fit_predict(X_scaled)

    return clusters


def calculate_silhouette_scores(X, max_clusters=10):
    kmeans_scores = []
    gmm_scores = []
    fuzzy_scores = []

    for n in range(2, max_clusters + 1):
        kmeans_scores.append(kmeans_silhouette(X, n))
        gmm_scores.append(gmm_silhouette(X, n))
        fuzzy_scores.append(fuzzy_cmeans_silhouette(X, n))

    return kmeans_scores, gmm_scores, fuzzy_scores

def visualize_clusters(df, feature_x, feature_y, cluster_label, count_cluster, sil_score):
    plt.scatter(df[feature_x], df[feature_y], c=df[cluster_label], cmap='viridis', s=50)
    plt.title(cluster_label)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.text(0.95, 0.95, f'Clusters: {count_cluster}, \n Silhouette_Score: {sil_score}', fontsize=12, ha='center', va='center',
             transform=plt.gca().transAxes)
    plt.show()


# Function to plot silhouette score comparisons
def plot_silhouette_comparison(kmeans_scores, gmm_scores, fuzzy_scores, best_scores):
    clusters_range = range(2, len(kmeans_scores) + 2)

    plt.figure(figsize=(10, 6))
    plt.plot(clusters_range, kmeans_scores, marker='o', label='KMeans')
    plt.plot(clusters_range, gmm_scores, marker='x', label='GMM')
    plt.plot(clusters_range, fuzzy_scores, marker='s', label='Fuzzy C-Means')
    plt.plot(clusters_range, fuzzy_scores, marker='d', label='DBSCAN')
    plt.title('Silhouette Score Comparison for Clustering Methods')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.show()


def main():
    df = import_data()
    explore_data(df)
    #visual_eda(df)

    selected_features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    nclusters = nclusters_by_elbow_method(selected_features)
    df['Kmeans_Cluster'], count_clusters_kmeans = k_cluster(selected_features, n_clusters=nclusters)

    df['FCM_Cluster'], count_clusters_FCM = fuzzy_cmeans(selected_features, n_clusters=nclusters)

    minPts = range(3, 15)
    labels, optimal_eps, best_min_samples, best_score = dbscan_opt(selected_features, minPts)
    df['DBSCAN_Cluster'], count_clusters_dbscan = dbscan(selected_features, best_min_samples, optimal_eps)

    ncomponents = find_best_n_components(selected_features)
    df['GMM_Cluster'] = apply_gmm(selected_features, n_components=ncomponents)

    kmeans_scores, gmm_scores, fuzzy_scores = calculate_silhouette_scores(selected_features)

    visualize_clusters(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'Kmeans_Cluster', count_clusters_kmeans, kmeans_scores)
    visualize_clusters(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'FCM_Cluster', count_clusters_FCM, fuzzy_scores)
    visualize_clusters(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'DBSCAN_Cluster', count_clusters_dbscan,best_score)
    visualize_clusters(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'GMM_Cluster', ncomponents, gmm_scores)

    plot_silhouette_comparison(kmeans_scores, gmm_scores, fuzzy_scores, best_score)

if __name__ == '__main__':
    main()
