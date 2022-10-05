# Import required packages
import numpy as np


class Searcher:
    def __init__(self, index):
        # Store index
        self.index = index

    def search(self, queryFeatures):
        # Initialize results
        results = {}

        # loop over the index
        for (k, features) in self.index.items():
            # Compute chi squared distance
            d = self.chi2_distance(features, queryFeatures)

            # Store distances
            results[k] = d

        # Sort Results
        results = sorted([(v, k) for (k, v) in results.items()])

        # Return the results
        return results

    def chi2_distance(self, histA, histB, eps=1e-10):
        # Compute the chi squred distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        
        # Retrun the chi squared distance
        return d
