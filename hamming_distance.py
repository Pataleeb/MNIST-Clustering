import numpy as np
from collections import Counter
import scipy.io
from sklearn.cluster import KMeans


class MNISTClusteringHamming:
    def __init__(self, n_clusters=10, data_file='mnist_10digits.mat'):
        self.n_clusters = n_clusters
        self.data_file = data_file
        self.data = None
        self.labels = None
        self.predicted_labels = None


    def load_data(self):
        mat = scipy.io.loadmat(self.data_file)
        self.data = mat['xtrain'] / 255.0
        self.labels = mat['ytrain'].flatten()

    def binary_data(self):
        self.data=np.where(self.data>128,1,0)

        def hamming_distance(self, X, centroids):

            distances = np.array([
                np.sum(X != centroid, axis=1) for centroid in centroids
            ]).T
            return distances

        def majority_vote_centroid(self, cluster_points):

            return np.round(np.mean(cluster_points, axis=0)).astype(int)

        def perform_clustering(self, max_iter=10):

            indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
            centroids = self.data[indices]

            for _ in range(max_iter):
                distances = self.hamming_distance(self.data, centroids)
                self.predicted_labels = np.argmin(distances, axis=1)


                new_centroids = []
                for cluster in range(self.n_clusters):
                    cluster_points = self.data[self.predicted_labels == cluster]
                    if len(cluster_points) == 0:
                        new_centroids.append(centroids[cluster])  # No update if empty
                    else:
                        new_centroids.append(self.majority_vote_centroid(cluster_points))
                centroids = np.array(new_centroids)

            print(f"KMeans clustering with Hamming distance and {self.n_clusters} clusters.")

        def purity_scores(self):
            clusters = set(self.predicted_labels)
            cluster_purity = {}
            correct_num = 0

            for cluster in clusters:
                cluster_indices = np.where(self.predicted_labels == cluster)[0]
                true_labels = self.labels[cluster_indices]
                majority_label, count = Counter(true_labels).most_common(1)[0]

                purity = count / len(cluster_indices)
                cluster_purity[cluster] = purity
                correct_num += count

                print(f"Cluster {cluster}: Majority Label = {majority_label}, Purity = {purity:.4f}")

            overall_purity = correct_num / len(self.labels)
            print(f"\nOverall Purity Score: {overall_purity:.4f}")

            return {
                'cluster_purity': cluster_purity,
                'overall_purity': overall_purity
            }

        def run(self):
            self.load_data()
            self.binarize_data()
            self.perform_clustering()
            results = self.purity_scores()
            return results

    if __name__ == "__main__":
        clustering_hamming = MNISTClusteringHamming(n_clusters=10, data_file='mnist_10digits.mat')
        results = clustering_hamming.run()

        print("\nFinal Cluster-wise Purity Scores:")
        for cluster, purity in results['cluster_purity'].items():
            print(f"Cluster {cluster}: Purity = {purity:.4f}")

        print(f"\nFinal Overall Purity Score: {results['overall_purity']:.4f}")

