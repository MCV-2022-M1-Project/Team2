# Import required packages
import numpy as np
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--matcher", type=str, default="BruteForce",
                help="Feature matcher to use. Options ['BruteForce', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']")
args = vars(ap.parse_args())


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
        # Compute the chi squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

        # Return the chi squared distance
        return d

    def euclidean_distance(self, histA, histB):
        # Compute the euclidean distance
        d = np.sqrt(np.sum([((a - b) ** 2) for (a, b) in zip(histA, histB)]))

        # Return the euclidean distance
        return d

    def l1_distance(self, histA, histB):
        # Compute the l1 distance
        d = np.sum([np.abs(a - b) for (a, b) in zip(histA, histB)])

        # Return the l1 distance
        return d

    def histogram_intersection(self, histA, histB):
        # Compute histogram intersection
        d = np.sum([np.minimum(a, b) for (a, b) in zip(histA, histB)])

        # Return the metric
        return d

    def hellinger_kernel(self, histA, histB):
        # Compute the hellinger kernel
        d = np.sum([np.sqrt(a * b) for (a, b) in zip(histA, histB)])

        # Return the metric
        return d


class SearchFeatures:
    def __init__(self, index):
        # Store index
        self.index = index

    def search(self, queryDescriptors):
        results = {}
        matcher = DescriptorMatcher_create(args["matcher"])

        # loop over the index
        for (k, features) in self.index.items():
            rawMatches = matcher.knnMatch(queryDescriptors, features, 2)
            matches = []

            if rawMatches is not None:
                # loop over the raw matches
                for m in rawMatches:
                    # ensure the distance passes David Lowe's ratio test
                    if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                        matches.append((m[0].trainIdx, m[0].queryIdx))

            # Append it to results
            results[k] = len(matches)
            # print("For{}, number of matches{}".format(k, results[k]))

        # Sort Results
        results = sorted([(v, k) for (k, v) in results.items()], reverse=True)

        # Return the results
        return results
