import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



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

#applying k-means
def k_cluster(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X)
    # Finding the optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
        wcss.append(kmeans.inertia_)

    return clusters

#Applying fuzzy-k
def fuzzy_cmeans(X, n_clusters=5):
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(X)
    clusters = fcm.predict(X)
    return clusters

def visualize_clusters(df, feature_x, feature_y, cluster_label):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature_x], df[feature_y], c=df[cluster_label], cmap='viridis', s=50)
    plt.title('K-Means Clustering with Gender')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()


#

def main():
    df = import_data()
    explore_data(df)
    visual_eda(df)
    df['KMeans_Cluster'] = k_cluster(X_scaled, n_clusters=5)
    visualize_clusters(df, 'Annul Income (k$)', 'Spending Score (1-100', 'k-Means Cluster')
    df['FCM_Cluster'] = fuzzy_cmeans(X)


if __name__ == '__main__':
    main()
