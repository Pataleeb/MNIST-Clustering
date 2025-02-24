import numpy as np
from collections import Counter
import scipy.io
from sklearn.cluster import KMeans

class MNISTClustering:
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



    def perform_clustering(self):
        kmeans = KMeans(n_clusters=self.n_clusters, algorithm='lloyd')
        self.predicted_labels = kmeans.fit_predict(self.data)
        print(f"KMeans clustering with {self.n_clusters} clusters.")


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
        self.perform_clustering()
        results = self.purity_scores()
        return results


if __name__ == "__main__":
    clustering = MNISTClustering(n_clusters=10, data_file='mnist_10digits.mat')
    results = clustering.run()


    print("\nFinal Cluster-wise Purity Scores:")
    for cluster, purity in results['cluster_purity'].items():
        print(f"Cluster {cluster}: Purity = {purity:.4f}")

    print(f"\nFinal Overall Purity Score: {results['overall_purity']:.4f}")




