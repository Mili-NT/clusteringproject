import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from fcmeans import FCM
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

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
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
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
    #visual_eda(df)

    selected_features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    nclusters = nclusters_by_elbow_method(selected_features)

    df['Kmeans_Cluster'] = k_cluster(selected_features, n_clusters=nclusters)
    visualize_clusters(df,'Annual Income (k$)','Spending Score (1-100)', 'Kmeans_Cluster')

    df['FCM_Cluster'] = fuzzy_cmeans(selected_features, n_clusters=nclusters)
    visualize_clusters(df, 'Annual Income (k$)' ,'Spending Score (1-100)','FCM_Cluster' )

    minPts = range(3, 15)
    labels, optimal_eps, best_min_samples, best_score = dbscan_opt(selected_features, minPts)
    df['DBSCAN_Cluster'] = dbscan(selected_features, best_min_samples, optimal_eps)
    visualize_clusters(df, 'Annual Income (k$)',  'Spending Score (1-100)','DBSCAN_Cluster')




if __name__ == '__main__':
    main()
