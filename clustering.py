import pandas as pd
import seaborn as sns
from fcmeans import FCM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


# TODO: Determine ncluster via some automated method (ncluster=5 is optimal, proved via testing)
# TODO: Improve DBSCAN accuracy

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
    # One-Hot Encoding
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
    sns.displot(df.iloc[:, 4], kde=True)
    # Seaborn scatterplots
    sns.pairplot(df)
    plt.show()
    # Annual Income-Spending Score Association (Boxplot)
    boxplot_associate(df, (3,4), ["Annual Income", "Spending Score", "Annual Income-Spending Score Association"])


def determine_elbow_point(wcss):
    # Calculate the first derivative
    first_derivative = np.diff(wcss)

    # Calculate the second derivative
    second_derivative = np.diff(first_derivative)

    # Find the index of the maximum value in the second derivative
    elbow_index = np.argmin(second_derivative) + 1  # +1 to account for the shift due to np.diff

    return elbow_index
def get_optimal_nclusters(selected_features):
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=50, random_state=0)
            kmeans.fit(selected_features)
            wcss.append(kmeans.inertia_)
        print(wcss)
        # From visual analysis we can tell the elbow point is 5, as that is where the inertia starts decreasing linearly
        # Get rate of change for values in WCSS list:
        wcss_derivative = np.diff(wcss)
        # Get rate of change for the wcss_derivative value, if positive then the rate of change is decreasing.
        # If negative the RoC is still decreasing, but at a slower rate. This lets us find the elbow point
        wcss_derivative_2 = np.diff(wcss_derivative)
        # Get minimum value of the second derivative to find the optimal elbow point
        elbow_point = np.argmin(wcss_derivative_2) + 1
        print("Optimal number of clusters (elbow point):", elbow_point)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.xticks(range(1, 11))
        plt.show()

def k_cluster(selected_features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(selected_features)
    return clusters

#Applying fuzzy-k
def fuzzy_cmeans(selected_features, n_clusters):
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(selected_features.values)
    clusters = fcm.predict(selected_features.values)
    return clusters

#Applying DBSCAN Optimizing
def dbscan_opt(selected_features, min_pts, k=10):
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
    plt.ylabel(f'{min_pts}-th Nearest Neighbor Distance')
    plt.title('k-distance Graph for DBSCAN')
    plt.grid(True)
    plt.show()

    # Choosing an epsilon based on the elbow point from the graph (adjustable)
    optimal_eps = np.percentile(k_distances, 90)

    # Step 2: Optimize min_samples using silhouette score for the optimized eps
    best_score = -1
    best_min_samples = None
    best_labels = None

    for min_samples in min_pts:
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
def dbscan_clustering(features, min_pts, ep):
    dbscan_model = DBSCAN(eps=ep, min_samples=min_pts)
    final_labels = dbscan_model.fit_predict(features)
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
    print(get_optimal_nclusters(selected_features))
    exit()
    kmeans_nclusters = 5

    df['Kmeans_Cluster'] = k_cluster(selected_features, n_clusters=kmeans_nclusters)
    visualize_clusters(df,'Annual Income (k$)','Spending Score (1-100)', 'Kmeans_Cluster')

    df['FCM_Cluster'] = fuzzy_cmeans(selected_features, n_clusters=kmeans_nclusters)
    visualize_clusters(df, 'Annual Income (k$)' ,'Spending Score (1-100)','FCM_Cluster' )

    min_pts = range(3, 15)
    labels, optimal_eps, best_min_samples, best_score = dbscan_opt(selected_features, min_pts)
    df['DBSCAN_Cluster'] = dbscan_clustering(selected_features, best_min_samples, optimal_eps)
    visualize_clusters(df, 'Annual Income (k$)',  'Spending Score (1-100)','DBSCAN_Cluster')

if __name__ == '__main__':
    main()
