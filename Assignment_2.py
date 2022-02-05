# K-means and K-medians algorithms implementation for Assignment 2 of COMP527
# This .py file contains all relvant methods required for clustering and
# can be ran independently using the if __name__ == '__main__' method at the
# the bottom.

# The Classes include:
#   - KMeans        -- use K-means clustering to best fit centroids to a dataset
#   - KMedians      -- inherits from KMeans, altering only the _distance and _average methods
#   - GetData       -- import data from a file
#   - Preprocessing -- concatenate the data from selected files and apply l2 normalisation (param)
#   - Questions     -- provide outputs for the questions of Assignment 2

# Rob Lamprell
# Last Modified: 2021/05/05    


# Packages
import numpy                as np
import matplotlib.pyplot    as plt




# use K-means clustering to best fit centroids to a dataset
class KMeans():
    def __init__(self, k=5, epochs=20):

        # the number of clusters(k) and the number of iterations(epochs)
        self.k              = k
        self.epochs         = epochs

        # setup contains for the clusters, centroids and labels
        self.clusters       = []
        self.centroids      = []
        self.cluster_labels = []

        # number of rows and columns in the dataset
        self.n_rows         = 0
        self.n_features     = 0


    # find centroids to fit the data
    #   -- X is the array of our features 
    def fit(self, X):

        self.X = X
        self.n_rows, self.n_features = X.shape
        
        # initialise randomly selected representatives
        self.__initialiseRandomCentroids()

        # run the clustering algorithm
        # -- iterate and change the clustering positions until they
        #    have converged, or the number of epochs has been reached
        for _ in range(self.epochs):

            # assign each data item/row to their nearest centroid
            self.clusters = self.__assign()

            # store the positions from the previous round
            temp = self.centroids

            # calc the positions for this round
            self.centroids = self.__getNewCents()
            
            # check get the difference between old and new positions
            distances = []
            for i in range(self.k):

                thisDistance = self._distance(temp[i], self.centroids[i])
                distances.append(thisDistance)

            # end the algorithm if the centroids/clusters assignments have not moved
            if sum(distances)==0:
                break

        # return the clusters
        # -- these contain the indexes of each row/item
        return self.clusters    


    # randomly choose centroids
    # -- for KMeans we could choose completely random points but, choosing
    #    from the dataset means we can also reuse this for KMedians
    def __initialiseRandomCentroids(self):
        
        # choose random start position
        random_centroids    = np.random.choice(self.n_rows, self.k, replace=False)

        # ensure the centroids container is empty and append the random choices
        self.centroids      = []
        for i in random_centroids:
            self.centroids.append(self.X[i])


    # assign each row/item to the nearest centroid
    # -- this is the formation of the clusters
    def __assign(self):

        # create a blank array to store our cluster assignments
        new_clusters = []
        for i in range(self.k):
            new_clusters.append([])

        # for each item/row in the dataset:
        # -- get the centroid it is closest to
        # -- assign it to that centroid along with its index in the dataset
        for i, data in enumerate(self.X):
           
            centroid = self.__getClosestCent(data)#, centroids)
            new_clusters[centroid].append(i)

        # return the newly formed clusters
        return new_clusters


    # get the closest centroid for this item/row in the datastet
    def __getClosestCent(self, row):
        
        # create a container to store all the distances
        distances = []

        # for each of the centroids
        # -- record the distance (KMeans - squared euclidean, KMedians - manhattan)
        #    and append to the distances array
        for i in range(len(self.centroids)):

            thisDistance = self._distance(row, self.centroids[i])
            distances.append(thisDistance)

        # get the centroid with the smallest of these distances and return its index
        closest_centroid = np.argmin(distances)

        return closest_centroid


    # get the centroid of a cluster (k)
    # -- using the _average method (mean or median)
    def __getNewCents(self):

        # create a k*features container
        centroids = np.zeros((self.k, self.n_features))

        # for each of the clusters
        # -- calculate the centroid position based on the _average method
        # -- append these values to the centroids container
        for i, cluster in enumerate(self.clusters):

            centroid        = self._average(X, cluster)
            centroids[i]    = centroid

        return centroids


    # squared euclidean distance (as in the lecture notes)
    def _distance(self, X, Y):
        return np.sum((X - Y)**2)


    # mean calculation for the position of centroids
    def _average(self, X, cluster):
        return np.mean(self.X[cluster], axis=0)


    # evaulate the clustering classification exstrinicly
    def bCubedMetrics(self, true_labels):
        
        # containers for all metrics
        precisions  = []
        recalls     = []
        f_scores    = []

        # get the unique labels withn the true label set
        unique_labels   = np.unique(true_labels)

        # get a list of of true labels in all clusters -- needed for recall
        allClusters     = self.__createTrueLabelClusters_All(true_labels)

        # iterate through each cluster to
        # -- assign the
        for c in range(len(self.clusters)):

            # convert this cluster and the true label arrays to numpy
            this_cluster    = np.array(self.clusters[c], dtype=np.int32)
            npTrue_labels   = np.array(true_labels,      dtype=np.int32)

            # this is a container for the true labels contained within this cluster
            # create an array which contrains the full list of true labels (rather than indexes)
            clustered_true_labels = []
            for i in range(len(this_cluster)):
                
                clustered_true_labels.append(npTrue_labels[this_cluster[i]])

            # apply the precision, recall and f_score formulas to each index
            # and add them to their respective array
            for i in range(len(clustered_true_labels)):
                
                # what is the current true label
                thisValue           = clustered_true_labels[i]

                # how many in this cluster match the true label
                # -- (1) in this cluster
                # -- (2) in all clusters
                thisCluster_Matches = np.count_nonzero(clustered_true_labels==thisValue)
                allCluster_Matches  = np.count_nonzero(allClusters==thisValue)

                # append each container with the related formula
                # -- precision = No. of items in C(x) with A(x) / No. of items in C(x)
                precision = thisCluster_Matches / len(clustered_true_labels)
                precisions.append(precision)

                # -- recall    = No. of items in C(x) with A(x) / No. of items with A(x)
                recall = thisCluster_Matches / allCluster_Matches
                recalls.append(recall)

                # -- f_score   = 2*Recall(x) * precision(x) / Recall(x) + precision(x)
                f_score = (2*recall*precision) / (recall + precision)
                f_scores.append(f_score)

        # calculate the averages for each set
        avg_precision   = np.average(precisions)
        avg_recalls     = np.average(recalls)
        avg_f_scores    = np.average(f_scores)

        return avg_precision, avg_recalls, avg_f_scores


    # get all the true labels across all the clusters 
    def __createTrueLabelClusters_All(self, true_labels):
    
        # get the unique labels in the data
        unique_labels   = np.unique(true_labels)
        
        # this is a container for the true labels contained within all clusters
        clustered_true_labels = []
        for c in range(len(self.clusters)):
            
            # for this cluster create an np array and true label array
            this_cluster    = np.array(self.clusters[c], dtype=np.int32)
            npTrue_labels   = np.array(true_labels,      dtype=np.int32)

            # for each index append the true label        
            for i in range(len(this_cluster)):
                
                clustered_true_labels.append(npTrue_labels[this_cluster[i]])

        return clustered_true_labels




# K-Medians clustering -- inherits most of its class from KMeans
# -- alters distance to manhanttan (instead of squared euclidean)
# -- alters averaging method to median (rather than mean).
#    This will force the alogorithm to align the centroids with
#    points from the dataset.
class KMedians(KMeans):

    # manhattan distance
    def _distance(self, X, Y):
        return np.sum(abs(X - Y))

    # numpy median
    def _average(self, X, cluster):
        return np.median(self.X[cluster], axis=0)




# Get the training or testing data from the files
# -- This is a modified version of what I submitted for Assignment 1 of COMP527.
class GetData():    
    # -- Note: this assumes the class is in the first index of every row
    def __init__(self, fileName, featureCount=300):

        self.featureCount   = featureCount
        self.X, self.y      = self.__load(fileName)


    # load all data from the files
    def __load(self, fileName):

        # assumed data format:
        # -- [0]        = class name
        # -- [1, len+1] = features
        # (+1 to ensure the final value is captured)
        X = np.loadtxt(fileName, delimiter=" ", usecols=np.arange(1, self.featureCount+1))
        y = np.loadtxt(fileName, delimiter=" ", usecols=[0], dtype='str')

        return X, y


    # return features and labels from the file
    def load(self):
        return self.X, self.y




# Preprocessing of the clustering data.  This can:
#   - concatenate all the input file(s) data and return them in the form X, y, cat.
#     Where X is the feature score, y is the label name and cat is the class (int, 0-3)
#   - apply l2normalisation to the feature set (X)
class Preprocessing:
    def __init__(self, fileNames, l2Norm=False):
        
        # files to load data from
        # -- assumes they all have the same formatting
        self.fileNames  = fileNames

        # containers for data ouputs
        # - features
        # - labels (string, example: "apple")
        # - classification (int, default: [0,3])
        self.X          = []
        self.y          = []
        self.cats       = []
        
        # import and concatenate all the data
        self.prepareData()

        # apply l2 normalisation to the feature-set if requested
        if l2Norm:
            self.__l2Normalisation()


    # combine all the imported data files
    # -- return all four as an array (animals, fruits, veggies and countries)
    def prepareData(self):

        self.X       = np.empty((0, 300), float)
        self.y       = np.empty(0, str)
        self.cats    = np.empty(0, int)

        for i in range(len(self.fileNames)):

            data    = GetData(self.fileNames[i])
            xi, yi  = data.load()
            cati    = np.full(len(yi), i)

            self.X       = np.concatenate((self.X,      xi))
            self.y       = np.concatenate((self.y,      yi))
            self.cats    = np.concatenate((self.cats,   cati))


    # function to preprocess the data using l2 normalisation
    def __l2Normalisation(self):

        # nomarlise the data (l2) with respect to each row (by object)
        # -- get the magnitude of the vector, v = sqrt(x[0]**2 + x[1]**2 + ... + x[n]**2)
        # -- then divide each feature by v
        feature_norms   = np.linalg.norm(self.X, ord=2, axis=1)
        data_normalised = self.X/feature_norms[:, np.newaxis] 

        # replace our feature set with our normalised feature set
        self.X = np.copy(data_normalised)
        
        # check our l2 normalisation was correctly implemented
        self.__checkL2_Magnitudes()
    

    # check the magnitudes post l2 normalisation
    # -- this is just precautionary code
    # -- the magnitude of all vectors should be 1
    def __checkL2_Magnitudes(self):
        
        # for every object
        for j in range(len(self.X)):

            # create a container to store the squared values
            squares = []
            for i in range(len(self.X[j])):
                squares.append(self.X[j][i]**2)

            # calculate the magnitude of the vector
            vector_magnitude = (np.sum(squares))**(1/2)

            # check the values in this vector add up to 1.0 (1dp to account for rounding errors on the floats)
            if np.round(vector_magnitude, 1) != 1.0:
                raise Exception("Error - Post L2 normalisation the vectors magnitudes do not all sum to 1.")


    # return the data
    def load(self):
        return self.X, self.y, self.cats




# Answer the questions from Assignment 2 of COMP527
class Questions:

    # template function for running questions from the assignment
    def run(self, X, y, clusterAlgorithm=KMeans, question="", lower_b=1, upper_b=10):

        # inform the user what class we're using
        print("==============================================")
        print(question)
        print("----------------------------------------------")
        print("Running for...", clusterAlgorithm)

        # containers for all the metrics to be added to out graphs
        precisions  = []
        recalls     = []
        f_scores    = []
        clusts      = []

        # iterate through each cluster value selected
        for i in range(lower_b, upper_b):

            # record the number of clusters (k) used in each 
            # -- append these to an array so they can be graphed
            clusters = i
            clusts.append(clusters)

            # Run the data for either KMeans or KMedians (KMeans by default)
            k = clusterAlgorithm(k=clusters, epochs=100)
            k.fit(X)

            # obtain all the B-CUBED metrics
            # -- append each cluster's (k's) results to an array to be graphed
            prec, rec, fs = k.bCubedMetrics(y)
            precisions.append(prec)
            recalls.append(rec)
            f_scores.append(fs)

        # plot and display all the metrics
        self.__plot_bcubed(clusts, precisions, recalls, f_scores)
        print("----------------------------------------------")
        print("clusters",   clusts)
        print("precisions", np.round(precisions, 2))
        print("recalls",    np.round(recalls, 2))
        print("f_scores",   np.round(f_scores, 2))
        print("==============================================")
        print()
        print()


    # plotter function for B-CUBED metrics
    def __plot_bcubed(self, k, prec, rec, fs):
        
        # create x and axis
        x = k
        xi = list(range(len(x)))
        plt.ylim(0.0, 1.1)

        # plot the B-Cubed metrics of precision, recall and f_score
        plt.plot(xi, prec,  marker='o', linestyle='--', color='r', label='Preicison') 
        plt.plot(xi, rec,   marker='o', linestyle='--', color='g', label='Recall') 
        plt.plot(xi, fs,    marker='o', linestyle='--', color='b', label='F_Score') 
        plt.ylabel('Performance')
        plt.xlabel('Cluster Count (k)') 
        plt.xticks(xi, x)
        plt.title('B-CUBED Metrics')
        plt.legend() 
        plt.show()




# Run from here to provide solutions for Assignment_2
if __name__ == '__main__':

    # numpy seed to ensure consistency across testing
    np.random.seed(54)


    # import the data from provided four files
    # -- then concatenate them into a single array
    files           = ["veggies", "countries", "fruits", "animals"]
    data            = Preprocessing(files)
    X, y, cats      = data.load()


    # create an l2 normalised form of the features - y and cats are unaffected by the processing
    data_norm       = Preprocessing(files, l2Norm=True)
    X_norm, y, cats = data_norm.load()

 
    # provide graphs for each of the four questions (q3, q4, q5, q6) from the assignment
    # -- KMeans & KMedians for both raw & normalised feature data
    Questions().run(X,      cats, KMeans,   "Question 3 -- B-CUBED for standard K-Means")
    Questions().run(X_norm, cats, KMeans,   "Question 4 -- B-CUBED for normalised K-Means")
    Questions().run(X,      cats, KMedians, "Question 5 -- B-CUBED for standard K-Medians")
    Questions().run(X_norm, cats, KMedians, "Question 6 -- B-CUBED for normalised K-Medians")