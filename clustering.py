import pandas as pd
import seaborn as sns
from fcmeans import FCM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def import_data():
    """
    Imports data from Mall_Customers.csv and performs encoding and scaling.

    :return: df -> scaled and encoded dataframe
    """
    df = pd.read_csv('Mall_Customers.csv')
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["Genre"])
    # MinMax Scale 0-1
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]] = MinMaxScaler().fit_transform(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
    return df

def explore_data(df):
    """
    Performs basic data description and checks for missing values

    :param df: pandas dataframe containing mall customer data
    """
    # Information about the dataset including the mean values and standard deviations for each column (check here for normalization info)
    print(f"{df.describe()}\n")
    # 200 entries, 200 non-null counts for each column means there are no missing values that need to be accounted for
    print(f"{df.info()}\n")
    print(f"Missing values:\n{df.isnull().sum()}\n")

def visual_eda(df):
    """
    Performs visual EDA and generates graphs

    :param df: pandas dataframe containing mall customer data
    """
    # Value distribution (histogram)
    df.hist(bins=30, figsize=(15, 10))
    plt.show()
    # Seaborn scatterplots
    sns.pairplot(df)
    plt.show()
    explore_data(df)
    # Annual Income-Spending Score Association (jointplot)
    sns.jointplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, kind='kde', fill=True, cmap='coolwarm')
    plt.show()

def get_optimal_nclusters_elbow(selected_features):
        """
        :param selected_features: dataframe containing the selected features to cluster
        :return: integer value representing the elbow point
        """
        wcss = []
        for i in range(1, 12):
            kmeans = KMeans(n_clusters=i, init='k-means++', n_init=50)
            kmeans.fit(selected_features)
            wcss.append(kmeans.inertia_)
        # From visual analysis we can tell the elbow point is 5, as that is where the inertia starts decreasing linearly
        # Get rate of change for values in WCSS list:
        wcss_derivative = np.diff(wcss)
        # Get rate of change for the wcss_derivative value. This lets us find the elbow point.
        wcss_derivative_2 = np.diff(wcss_derivative)
        # Get minimum value of the second derivative to find the optimal elbow point (offset by 1)
        elbow_point = np.argmin(wcss_derivative_2) + 1
        # Plot to confirm
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 12), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.xticks(range(1, 12))
        plt.show()
        return elbow_point


def get_optimal_nclusters_silhouette(selected_features):
    """
    :param selected_features: dataframe containing the selected features to cluster
    :return: an integer representing the silhouette score taken from the scores list
    """
    scores = []
    for i in range(2, 12):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=50)
        labels = kmeans.fit_predict(selected_features)
        score = silhouette_score(selected_features, labels)
        scores.append(score)
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 12), scores)
    plt.title('Silhouette Scores by Ncluster')
    plt.xlabel('Ncluster')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, 12))
    plt.grid()
    plt.show()
    return range(2, 12)[np.argmax(scores)]

def k_cluster(selected_features, n_clusters):
    """
    Uses k-means to label clusters

    :param selected_features: dataframe containing the features to cluster
    :param n_clusters: optimal number of clusters determined by elbow and silhouette methods
    :return: cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(selected_features)
    return clusters

#Applying fuzzy-k
def fuzzy_cmeans(selected_features, n_clusters):
    """
    Uses fuzzy c-means to label clusters

    :param selected_features: dataframe containing the features to cluster
    :param n_clusters: optimal number of clusters determined by elbow and silhouette methods
    :return: cluster labels
    """
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(selected_features.values)
    clusters = fcm.predict(selected_features.values)
    return clusters

#Applying DBSCAN Optimizing

def dbscan_opt(selected_features, min_pts, k=10):
    """
    Plots the k-distance graph to help estimate the optimal eps value for DBSCAN.

    :param selected_features: The dataset containing the selected features for clustering
    :param min_pts: minimum number of points required to form a cluster
    :param k: nearest neighbor value
    :return: best_labels, best_eps, best_min_samples, best_score
    """
    # Compute k-nearest neighbors distances
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(selected_features)
    distances, _ = neighbors_fit.kneighbors(selected_features)
    k_distances = np.sort(distances[:, k - 1], axis=0)

    # Plot the k-distance graph
    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.xlabel('Data Points (sorted by distance)')
    plt.ylabel(f'{min_pts}-th Nearest Neighbor Distance')
    plt.title('k-distance Graph for DBSCAN')
    plt.grid(True)
    plt.show()


    # Try different percentiles for `eps`
    best_eps = None
    best_score = -1
    best_min_samples = None
    best_labels = None

    for percentile in [55, 60, 80, 85, 90]:
        eps = np.percentile(k_distances, percentile)
        print(f"Testing with eps = {eps} (percentile: {percentile})")


        # Optimize `min_samples` using silhouette score for each `eps`
        for min_samples in min_pts:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(selected_features)

            # Only calculate silhouette score if more than 1 cluster is found
            if len(set(labels)) > 1:
                score = silhouette_score(selected_features, labels)
                if score > best_score:
                    best_score = score
                    best_min_samples = min_samples
                    best_eps = eps
                    best_labels = labels

    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best silhouette score: {best_score}")
    return best_labels, best_eps, best_min_samples, best_score

#Applying DBSCAN
def dbscan_clustering(selected_features, min_pts, eps):
    """
    Runs the DBSCAN algorithm to cluster the features according to the optimal number of clusters.

    :param selected_features: the dataframe containing the selected features for clustering.
    :param min_pts: the optimized minimum points to form a cluster
    :param eps: the optimized epsilon value
    :return: dbscan cluster labels
    """
    print('EPS:', {eps})
    dbscan_model = DBSCAN(eps=eps, min_samples=min_pts)
    final_labels = dbscan_model.fit_predict(selected_features)
    return final_labels

def visualize_clusters(df, feature_x, feature_y, cluster_label):
    plt.scatter(df[feature_x], df[feature_y], c=df[cluster_label], cmap='viridis', s=50)
    plt.title(cluster_label)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()


def main():
    df = import_data()
    explore_data(df)
    visual_eda(df)

    selected_features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    nclusters_silhouette = get_optimal_nclusters_silhouette(selected_features)
    nclusters_elbow = get_optimal_nclusters_elbow(selected_features)
    print(f"nclusters according to silhouette method: {nclusters_silhouette}\nnclusters according to elbow method: {nclusters_elbow}")
    # Because the two agree we can conclusively say that the optimal ncluster value is likely to be 5
    kmeans_nclusters = nclusters_elbow
    # K-Means
    df['Kmeans_Cluster'] = k_cluster(selected_features, n_clusters=kmeans_nclusters)
    visualize_clusters(df,'Annual Income (k$)','Spending Score (1-100)', 'Kmeans_Cluster')
    # Fuzzy C-Means
    df['FCM_Cluster'] = fuzzy_cmeans(selected_features, n_clusters=kmeans_nclusters)
    visualize_clusters(df, 'Annual Income (k$)' ,'Spending Score (1-100)','FCM_Cluster' )
    # DBSCAN
    min_pts = range(3, 15)
    labels, optimal_eps, best_min_samples, best_score = dbscan_opt(selected_features, min_pts)
    df['DBSCAN_Cluster'] = dbscan_clustering(selected_features, best_min_samples, optimal_eps)
    visualize_clusters(df, 'Annual Income (k$)',  'Spending Score (1-100)','DBSCAN_Cluster')

if __name__ == '__main__':
    main()
