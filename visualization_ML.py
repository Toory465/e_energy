import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import argparse


def generate_artificially_clusters(sample_size, n_clusters, file_path='./data.npy', low_bound=5.0, high_bound=15.0):
    """
    This function is responsible to generate random samples around four randomly generated datapoints.
    In this method, we create four samples. Then, we populate randomly generated samples from a normal distribution
    with standard deviation of 0.3. The results will be saved at the given path as a npy file.
    :param sample_size: number of samples that should be generated for eacch cluster.
    :param n_clusters: number of clusters.
    :param file_path: is the path to save the generated data as .npy file
    :param low_bound: is the x-axis lower bound for sample generation.
    :param high_bound: is the x-axis upper bound for sample generation.
    """
    # we create four random data points as the center of each cluster.
    data_pints_x_axis = np.random.uniform(low=low_bound, high=high_bound, size=(n_clusters,))
    data_pints_y_axis = np.random.uniform(low=low_bound, high=high_bound, size=(n_clusters,))
    data_points = np.array((data_pints_x_axis, data_pints_y_axis)).T.reshape(-1, 2)

    population = None
    # generate random samples in the vicinity of each cluster.
    for i, point in enumerate(data_points):
        x = np.random.normal(loc=point[0], size=sample_size, scale=0.3)
        y = np.random.normal(loc=point[1], size=sample_size, scale=0.3)
        stack_points = np.stack((x, y), axis=1)
        if population is None:
            population = stack_points
        else:
            population = np.concatenate((population, stack_points))
    # save the populated data points in a numpy data file at the given path.
    np.save(file_path, population)


def plot_data_points(file_path):
    """
    This function is responsible for ploting the generated data that is already saved in the given path.
    :param file_path: path to the numpy data file with .npy format.
    :return: it shows the plot of data points.
    """
    data = np.load(file_path)
    plt.figure(figsize=(9, 6))
    plt.title('Population', fontsize=20)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    pyplot.scatter(data[:, 0], data[:, 1])
    pyplot.show()


def visualize_k_distance(file_path):
    """
    This function is used for hyperparameter tuning for the epsilon parameter that is used in DBScan algorithm.
    This function is responsible to illustrate data points sorted by K-distance. It uses the Nearest Neighbors model
    to find the pair-wise distance for the given data points. Then we can plot the data points sorted by their k-distance.
    Then, we can consider the k-distance that gives the maximum curvature of the plot as a epsilon value
    for the training of DBScan algorithm.
    :param file_path: path to the numpy data file with .npy format.
    :return: the illustration of data points sorted by K-distance.
    """
    data = np.load(file_path)
    nn_model = NearestNeighbors(n_neighbors=2)
    nn_out = nn_model.fit(data)
    distances, indices = nn_out.kneighbors(data)
    # plotting the Data points sorted by K-distance
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.figure(figsize=(9, 6))
    plt.plot(distances)
    plt.title('Data points sorted by K-distance', fontsize=20)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Epsilon', fontsize=14)
    plt.show()


def dbscan_clustering(file_path, epsilon, min_samples=3):
    """
    This function is responsible to fit data into a DBScan model to cluster our datapoints.
    As a result, it plot the clustered data point visually seperated in different colors.
    :param file_path: path to the numpy data file with .npy format.
    :param epsilon: the radius of circle that is used for the grouping datapoints in the DBScan algorithm.
    :param min_samples: the minimum number of samples that can be grouped in the same cluster and lies in the circle
    with the radius of epsilon.
    :return: the illustration f the clustering algorithm outcome as a python plot.
    """

    data = np.load(file_path)
    # Model training.
    dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan_model.fit_predict(data)
    # clusters visualization.
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis")
    plt.title('DBscan algorithm result', fontsize=20)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    pyplot.show()

def main():
    """
    The main method to run the solution for the question 3 (visualization and Machine learning par
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_data', dest='generate_data', action='store_true',
                        help='It should be set if you want to generate new data points and save as a numpy data file.')
    parser.add_argument('--plot_samples', dest='plot_samples', action='store_true',
                        help='It should be set if you want to plot the data that is stored in the file.')
    parser.add_argument('--ht_tuning', dest='ht_tuning', action='store_true',
                        help='It should be set if you want to see the data points sorted by k-distance plot '
                             'in order to find the best hyper parammeters for epsilon.')
    parser.add_argument('--plot_clusters', dest='plot_clusters', action='store_true',
                        help='It should be set if you want to see the plot of the clustered data points.')
    parser.add_argument('--sample_size', type=int, default=30,
                        help='number of random samples that should be generated for each cluster.')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--n_clusters', type=int, default=4, help='The number of cluster that is used'
                                                                  ' to generate random datapoints.')
    parser.add_argument('--file_path', default='./data.npy')
    parser.add_argument('--dbscan_min_sample', type=int, default=3, help='hyperparameter for the DBScan model')
    parser.add_argument('--dbscan_epsilon', type=float, default=2.5, help='the epsilon hyperparameter ths is requiered '
                                                                          'for the DBScan model')

    config = parser.parse_args()

    # For reproducibility, we should set the seed.
    np.random.seed(config.seed)

    if config.generate_data:
        generate_artificially_clusters(sample_size=config.sample_size, n_clusters=config.n_clusters)

    if config.plot_samples:
        plot_data_points(file_path=config.file_path)

    if config.ht_tuning:
        visualize_k_distance(file_path=config.file_path)

    if config.plot_clusters:
        dbscan_clustering(file_path=config.file_path, min_samples=config.dbscan_min_sample,
                          epsilon=config.dbscan_epsilon)


if __name__ == '__main__':
    main()

