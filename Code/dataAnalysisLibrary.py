import base64
import csv
import io
import math
import os
import pickle
import random
import sys
from io import BytesIO

import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib import gridspec

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import scipy
from scipy import signal
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.interpolate import (
    CubicSpline,
    PchipInterpolator,
    Akima1DInterpolator,
    interp1d
)
from scipy.io import loadmat
from scipy.optimize import Bounds, direct
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


####################################
#                                  #
#               Hey!               #
#       If you find any bugs       #
#    or errors in documentation    #
#       Please let me know!        #
#     Ill come in and fix them     #
#              -Andy               #
#                                  #
####################################


################################################ Quality of Life/Miscellanious #############################################################################################################################################
def sanityCheck():
    print("you are sane")


def clearDirectory(direc='./Images for simulation'):
    """
    Description
    -----------
    Deletes the contents of a target directory.

    Parameters
    ----------
    direc : string
       string containing the path to your target directory. Default is the essential 'Images for Simulation' directory since thats what we use the most, but you can put in whatever you want.

    Returns
    -------
    Nothing. But! This function does delete the contents of another folder.
    """
    dir = direc
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def collapseFrames(cellMaskData):
    """
    Description
    -----------
    Given a 3D array, collapseFrames() will collapse all frames of the volume into one plane. Like projecting all the drawing on a stack of paper on to one single sheet.
    Translated into Python from Franklins MATLAB code.

    Parameters
    ----------
    cellMaskData : 3D array-like
       Data set that youre working with (most likely call mask data). cellMaskData[i] MUST be a square matrix, otherwise this function wont work.

    Returns
    -------
    projectedFrame : 2D array-like
        Square matrix that has all the cellMaskData frames projected on to one
    """    
    overlap = np.sum(cellMaskData,0)>1 # Find which indices are greater than 1 (That is, indices with overlap)
    cellMaskData = np.multiply(cellMaskData,~overlap) # Multiply our projected frame with the negated overlap index. This puts all overlap indices at 0 
    indexedROIprojection = np.zeros((len(cellMaskData[0]), len(cellMaskData[0][0]),1)) # Initialize the new volume that will carry the index for that indices ROI 
    for i in range(len(cellMaskData)):
        indexedROIprojection = indexedROIprojection+cellMaskData[:][:][i]*i
    
    return(indexedROIprojection)

def sortMatrixByThreshold(data, threshold):
    """
    Description
    -----------
    sortMatrixByThreshold() sorts the rows of a 0-1 normalized matrix based off which row hits a certain threshold first reading left to right

    Parameters
    ----------
    data : 2D array-like
       Data set
    threshold : float
        Numerical value for the sorting threshold. Should be between 0-1
    
    Returns
    -------
    sorted_matrix : 2D array-like (same shape as data array)
        Sorted data array
    """
    matrix_thresh = data > threshold
    exceeds = []
    
    # Rank all of the rows in the data matrix based on what hits the threshold first
    for i in range(len(data)):
        all_exceeds = [idx for idx, val in enumerate(matrix_thresh[i]) if val != 0]
        exceeds.append(all_exceeds[0])
        
    # Sort the rows based on the earliest exceed index
    sorted_indices = np.argsort(exceeds)
    
    # Return the sorted matrix
    sorted_matrix = data[sorted_indices]
    
    return([sorted_matrix, sorted_indices])

def iscellTrim(data, iscellData):
    """
    Description
    -----------
    iscellTrim() cuts the original data set (data) according to what ROIs are deemed cells by iscellData

    Parameters
    ----------
    data : 2D array-like
       Data set
    iscellData : 2D array-like, size (len(data),2)
        The unique iscell array for the given data set
    d
    Returns
    -------
    iscell_data : 2D array-like
        Trimmed data set
    iscellKey : dictionary
        Maps the raw data trace index to its trimmed index
    reverseIscellKey : dictionary
        Maps the trimmed trace index to its raw index
    """
    #Create the key for mapping where the old traces went after trimming. Works both ways, so we can go from raw -> trim and trim -> raw. Two dictionaries
    iscellKey = {}
    reverseIscellKey = {}
    newTraceNumber = 0
    
    for i in range(len(iscellData)):
        if iscellData[i][0] == 1:
            iscellKey[i] = newTraceNumber
            reverseIscellKey[newTraceNumber] = i
            newTraceNumber += 1
    
    # Thanks to Franklin for this one, using list comprehension and converting it into a numpy array we're able to perform this in a single line with some magic
    iscell_data = np.array([data[i] for i in range(len(iscellData)) if iscellData[i][0] == 1])
    return([iscell_data, iscellKey, reverseIscellKey])

# Problem with this function: it doesnt account for duplication. It returns pairs on both sides of the diagonal, not just one like we want it to
def matrixThreshold(dataset, threshold=0.2):
    """
    Description
    -----------
    matrixThreshold() returns coordinate pairs for indices in a matrix beneath a certain threshold.
    Namely this was made to find which traces are below a certain cosine similarity but has been generalized for more than that

    Parameters
    ----------
    dataset : 2D array-like
       datset or matrix in question
    threshold : float
        cutoff threshold, will return everything lower than this

    Returns
    -------
    arr : (,2) array-like
        list of coordinates where each entry is an entry beneath the threshold
    """
    [x,y]=np.where(dataset < threshold)
    
    # Combine X and Y coordinates
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    duplicates = np.argwhere(coordinates[:, 0] == coordinates[:, 1]).flatten()
    
    # Remove duplicates
    arr = np.delete(coordinates, duplicates, axis=0)
    return(arr)

def savePlotyPlotsToPDF(plot_list, pdf_name='combined_plots.pdf', dpi=300):
    """
    Description
    -----------
    savePlotsToPDF() will save a list of Plotly plots as a single PDF using a lot of os magic.

    Parameters
    ----------
    plot_list : python list
       List of Plotly plots. Needs to have atleast one plot inside. All objects in this list must be plotly plots.

    Returns
    -------
    Nothing. But! It saves all the individual plots as a PDF in the "PDFs" folder,
    """
    temp_dir = './temp_images'
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    
    for i, plot in enumerate(plot_list):
        image_path = os.path.join(temp_dir, f'plot_{i}.png')
        print(f"Saving image to {image_path}")  # Print path for debugging
        plot.write_image(image_path, scale=dpi/96, engine="kaleido")
        image_paths.append(image_path)
    
    images = [Image.open(image_path) for image_path in image_paths]
    
    print(f"Saving PDF to {pdf_name}")  # Print path for debugging
    images[0].save(pdf_name, save_all=True, append_images=images[1:], resolution=dpi, quality=95)
    
    for image_path in image_paths:
        os.remove(image_path)
    os.rmdir(temp_dir)

def interpolateData(data, newLength):
    # Create a linearly spaced array x_old representing the original indices of the input array
    xOld = np.linspace(0, 1, len(data))
    
    # Create a linearly spaced array x_new representing the new indices for the resampled array
    xNew = np.linspace(0, 1, newLength)
    
    # Create an interpolation function f based on the original array and indices
    f = interp1d(xOld, data, kind='linear')
    
    # Use the interpolation function to calculate the values of the resampled array at the new indices
    return f(xNew)

############################################# This marks the start of the Normalization sutff #######################################################################################################################

#Self explanatory
def normalize(a):
    """
    Description
    -----------
    Normalize each vector in a 2D array relative to each component. That is, each vector is normalized to itself rather than a global maximum for the matrix

    Parameters
    ----------
    a : 2D array-like
       Data set that youre working with.

    Returns
    -------
    NormalizedDataset : 2D array-like
        Normalized data set
    """
    normalizedDataset = np.zeros((len(a), len(a[0])))
    for i in range(len(a)):
        normalizedDataset[i]=a[i]/np.linalg.norm(a[i])
    
    return(normalizedDataset)

def normalizeDeltaF(trace):
    """
    Description
    -----------
    normalizedArray() calculates deltaD/F0 of a given trace

    Parameters
    ----------
    trace : array-like, shape (1,)
       Single ROI trace array.

    Returns
    -------
    NormalizedArray : array-like, shape (1,) (same shape as trace parameter)
        Normalized trace array
    """
    f0 = np.percentile(trace,5) # Set F0 to the fifth percentile (As per franklins instruction)
    normalizedArray = np.zeros(len(trace)) # Initialize the normalized array
    for i in range(len(trace)):
        normalizedArray[i] = (trace[i]-f0)/f0 # Calculating deltaF/F0 at each index on the array
    
    return(normalizedArray)

def normalizeDatasetDeltaF(dataSet):
    """
    Description
    -----------
    normalizeDataset() performs flourescenceNormalization() on each ROI in a given data set.

    Parameters
    ----------
    data : 2D array-like
       ROI Data set. Each row is a single ROI trace.

    Returns
    -------
    normalizedDataSet : 2D array-like (same shape as dataSet parameter)
        Normalized trace array
    """
    normalizedDataSet = np.zeros((len(dataSet),len(dataSet[0])))
    for i in range(len(dataSet)):
        normalizedDataSet[i] = normalizeDeltaF(dataSet[i])
        
    return(normalizedDataSet)

def normalizePercentile(dataSet):
    """
    Description
    -----------
    Normalize a dataset (collection of traces) but dividing each trace by the 95th percentile of that specific trace

    Fun fact, when creating this function I saw the absolute weirdest bug. No matter what I did I could NOT load this function into my code no matter what I did. Eventually I just deleted it and replaced it with my github backup and now everything works fine! Litereally never seen anything like it

    Parameters
    ----------
    dataSet : 2D array-like
       ROI Data set. Each row is a single ROI trace.

    Returns
    -------
    normalizedDataSet : 2D array-like (same shape as dataSet parameter)
        Normalized trace array
    """
    normalizedDataSet = np.zeros((len(dataSet),len(dataSet[0])))
    for i in range(len(dataSet)):
        normalizedDataSet[i] = dataSet[i]/np.percentile(dataSet[i],95)
    return(normalizedDataSet)

def normalizeMax(dataSet):
    """
    Description
    -----------
    Normalize a dataset (collection of traces) to the max of the individual trace
    
    Parameters
    ----------
    dataSet : 2D array-like
       ROI Data set. Each row is a single ROI trace.

    Returns
    -------
    normalizedDataSet : 2D array-like (same shape as dataSet parameter)
        Normalized trace array
    """
    normalizedDataSet = np.zeros((len(dataSet),len(dataSet[0])))
    for i in range(len(dataSet)):
        normalizedDataSet[i] = dataSet[i]/np.max(dataSet[i])
    return(normalizedDataSet)

################################################ This marks the start of our correlation stuff ######################################################################################################################################################

def cosineSimilarity(a,b):
    """
    Description
    -----------
    Computer the cosine similarity between two vectors. This documentation is like 10 times the length of just the function but here we are.

    Parameters
    ----------
    a, b : array-like, shape (1,)
       The two vectors that you want to find the cosine similarity between.
       a and b must be the same shape.

    Returns
    -------
    Cosine similairty : float
    """
    return(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def cosDistMatrix(dataSet):
    """
    Description:
    ------------ 
    costDistMatrix() is a function where given a data set this function will return the cosine similarity distance matrix of said data set.

    Parameters
    ----------
    dataSet : 2D array-like
       Data set array.

    Returns
    -------
    distMatrix : 2D array-like
        The cosine similarity matrix. Each entry of distMatris[i,j] is the cosine distance between data[i] and data[j]. 
    """
    distMatrix = np.empty([len(dataSet),len(dataSet)])
    for i in range(len(dataSet)):
        for j in range(len(dataSet)):
            #1-cosSimilarity to make it a "distance", since angles arent really a distance (Unless you get philosophical, then youre on your own)
            distMatrix[i,j]=1-cosineSimilarity(dataSet[i],dataSet[j])
    
    return(distMatrix)

def multiexperimentCosDistMatrix(dataList, cutoff=10):
    """
    Description
    -----------
    multiexperimentCosDistMatrix() Creates a cosine distance matrix between multiple experiments of the same framerate

    Parameters
    ----------
    dataList : pythonList
       A python list of each experiments data set (a python list of 2D numpy arrays)
    cutoff : int
        Optional. cutoff threshold for removing the first few elements due to edge effect from denoising. Ususally determined by eye. Default value is 10 for no good reason

    Returns
    -------
    arr : (,2) array-like
        list of coordinates where each entry is an entry beneath the threshold
    """
    # Find the shortest trace and trim everything down to match that size
    shortestTraceLength = float('inf') 
    for dataSet in dataList:
        if len(dataSet[0]) < shortestTraceLength:
            shortestTraceLength = len(dataSet[0])

    trimmedTraces = [np.array([trace[:shortestTraceLength] for trace in dataSet]) for dataSet in dataList]

    # Trim off the first few data points so that we can get rid of the denoising edge effects
    noEdgeEffectTraces = [np.array([trace[cutoff:] for trace in dataSet]) for dataSet in trimmedTraces]

    # Concatenate all the data into one big mama
    concatenatedData = np.vstack(noEdgeEffectTraces)
    
    return(cosDistMatrix(concatenatedData))

def magnitudeSimilarity(a, b):
    """
    Description
    -----------
    Computer the difference in magnitude between two vectors

    Parameters
    ----------
    a, b : array-like, shape (1,)
       The two vectors that you want to find the difference in magnitude between.
       a and b must be the same shape.

    Returns
    -------
    Difference in Magnitude : float
    """
    return(np.abs(np.linalg.norm(a)-np.linalg.norm(b)))

def magnDiffMatrix(dataSet):
    """
    Description:
    ------------ 
    magnitudeSimilarityMatrix() is a function where given a data set this function will return the magnitude similarity distance matrix of said data set.

    Parameters
    ----------
    data : 2D array-like
       Data set array.

    Returns
    -------
    distMatrix : 2D array-like
        The cosine similarity matrix. Each entry of distMatrix[i,j] is the difference in magnitude between data[i] and data[j]. 
    """
    distMatrix = np.empty([len(dataSet),len(dataSet)])
    for i in range(len(dataSet)):
        for j in range(len(dataSet)):
            #1-cosSimilarity to make it a "distance", since angles arent really a distance (Unless you get philosophical, then youre on your own)
            distMatrix[i,j]=magnitudeSimilarity(dataSet[i], dataSet[j])
    
    return(distMatrix)

def spearmanMatrix(dataSet):
    """
    Description:
    ------------ 
    spearmanMatrix() is a function where given a data set this function will return the Spearman Correlation Coefficient matrix of said data set.

    Parameters
    ----------
    data : 2D array-like
       Data set array.

    Returns
    -------
    distMatrix : 2D array-like
        The Spearman Correlation Coefficient matrix. Each entry of distMatris[i,j] is the Spearman Correlation Coefficient between data[i] and data[j]. 
    """
    distMatrix = np.empty([len(dataSet),len(dataSet)])
    for i in range(len(dataSet)):
        for j in range(len(dataSet)):
            distMatrix[i,j]=scipy.stats.spearmanr(dataSet[i],dataSet[j])[0]

    return(distMatrix)

# Maybe shouldnt use this since the ROIs might night be linear
def pearsonMatrix(dataSet):
    """
    Description:
    ------------ 
    spearmanMatrix() is a function where given a data set this function will return the Spearman Correlation Coefficient matrix of said data set.

    Parameters
    ----------
    data : 2D array-like
       Data set array.

    Returns
    -------
    distMatrix : 2D array-like
        The Spearman Correlation Coefficient matrix. Each entry of distMatris[i,j] is the Spearman Correlation Coefficient between data[i] and data[j]. 
    """
    distMatrix = np.empty([len(dataSet),len(dataSet)])
    for i in range(len(dataSet)):
        for j in range(len(dataSet)):
            distMatrix[i,j]=scipy.stats.pearsonr(dataSet[i],dataSet[j])[0]

    return(distMatrix)

def maxDifference(a,b):
    """
    Description
    -----------
    maxDifference() computes the absolute magnitude of difference between two vectors. Useful for trying to find out how much they deviate between eachother

    Parameters
    ----------
    dataset : 2D array-like
       ROI Data set. Each row is a single ROI trace.

    Returns
    -------
    normalizedDataSet : 2D array-like (same shape as dataSet parameter)
        Normalized trace array
    """
    return(np.max(np.absolute(np.subtract(a,b))))

def differenceMatrix(dataSet):
    """
    Description
    -----------
    differenceMatrix() computes the max difference matrix, where each entry is maximum of the element-wise difference vector between vector i and vector j.

    Parameters
    ----------
    dataset : 2D array-like
       ROI Data set. Each row is a single ROI trace.

    Returns
    -------
    normalizedDataSet : 2D array-like (same shape as dataSet parameter)
        Normalized trace array
    """
    differenceDataset = np.zeros((len(dataSet),len(dataSet)))
    for i in range(len(dataSet)):
        for j in range(len(dataSet)):
            if i != j:
                differenceDataset[i][j] = maxDifference(dataSet[i], dataSet[j])
            else:
                differenceDataset[i][j]=7
        
    return(differenceDataset)

def correlationHistogramPlot(dataSet, numBins = 30, title="Correlation Histogram", show=False):
    """
    Description
    -----------
    histogramCorrelationPlot() makes a multiplot of the histogram of the given dataSet and the cosine correlation matrix of that dataSet

    Parameters
    ----------
    dataSet : 2D array-like
       datset in question

    Returns
    -------
    Nothing! But it makes a multiplot of the histogram of the given dataSet and the cosine correlation matrix of that dataSet
    """
    similarities = np.triu(dataSet,1).flatten()
    nonzeroCosines =similarities[np.nonzero(similarities)]
    
    # Create subplots with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot histogram of correlations
    ax1.hist(nonzeroCosines, bins=numBins, color='blue', alpha=0.5)
    ax1.set_title('Histogram')
    
    # Plot the cosineMatrix
    image_plot = ax2.imshow(dataSet, cmap='viridis', interpolation='nearest')
    ax2.set_title('Cosine Distance Matrix')
    
    # Add color bar for the imshow
    cbar = fig.colorbar(image_plot, ax=ax2)
    cbar.set_label('Cosine Similarity')
    
    fig.suptitle(title)
    if show == True:
        plt.show()

# Noteiced problem: The highest correlations eems to be pairs correlated with themselves, want to get rid of this
# Noticed problem: Noticing some of the low correlation pairs have weird spikes in them. Normalize through zscore and use a sort of zscore threshold to get rid of pairs like this
def representativePairs(dataSet, numBins=30, printIndices = False):
    """
    Description
    -----------
    representativePairs() makes three subplots stacked on top of eachother showing random representative pairs from different bins in the histogram plot of cosine matrices. This should be used in conjunction with (and ideally directly under) the use of the correlationHistogramPlot() function
    

    Parameters
    ----------
    dataSet : 2D array-like
       datset in question
    numBines : int
        Number of bins in the histogram, should be the same as in correlationHistogramPlot() for most accurate data capture
    printIndices : Boolean
        If true, the function will print what values the bins represent along with their random representative pair. This is more of a convenient exploratory tool.

    Returns
    -------
    random_indices : List of 3-tuples
        -First element in each tuple is value that the bin represents
        -Second element in each tuple is the index of one trace in the pair
        -Third element is the other trace index
    """
    # Create a sample 2D numpy array
    data = dataSet
    
    # Flatten the array
    flattened_array = data.flatten()
    
    # Compute the histogram
    hist, bin_edges = np.histogram(flattened_array, bins=numBins)  # Adjust the number of bins as needed
    
    # Digitize the flattened array to get the bin indices
    bin_indices = np.digitize(flattened_array, bin_edges, right=True)
    
    # Create a dictionary to store the indices for each bin
    indices_per_bin = {i: [] for i in range(1, len(bin_edges))}
    
    # Iterate over the bin indices and store the original indices
    for idx, bin_idx in np.ndenumerate(bin_indices):
        # Make sure bin_idx is within the valid range of bin keys
        if bin_idx in indices_per_bin:
            original_index = np.unravel_index(idx[0], data.shape)
            indices_per_bin[bin_idx].append(original_index)
    
    # List to store one random pair of indices from each bin
    random_indices = []
    
    # Select one random pair of indices from each bin and save it in the list
    for bin_idx, indices in indices_per_bin.items():
        if indices:  # Check if the list of indices is not empty
            random_index = random.choice(indices)
            random_indices.append((round((bin_edges[bin_idx-1]+bin_edges[bin_idx])/2,2), random_index[0], random_index[1]))
    
    # Print the list of random indices
    if printIndices == True:
        print("Random indices from each bin in order:", random_indices)
    
    # Create a figure with 3 subplots stacked vertically
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    # Plot data on the first subplot
    axs[0].plot(data[random_indices[0][1]])
    axs[0].plot(data[random_indices[0][2]])
    axs[0].set_title("cosineSimilarity="+str(random_indices[0][0]))
    
    # Plot data on the second subplot, which comes from the bin with the most elements in it
    axs[1].plot(data[random_indices[max(indices_per_bin, key=lambda k: len(indices_per_bin[k]))][1]])
    axs[1].plot(data[random_indices[max(indices_per_bin, key=lambda k: len(indices_per_bin[k]))][2]])
    axs[1].set_title("cosineSimilarity="+str(random_indices[10][0]))
    
    # Plot data on the third subplot
    axs[2].plot(data[random_indices[-1][1]])
    axs[2].plot(data[random_indices[-1][2]])
    axs[2].set_title("cosineSimilarity="+str(random_indices[-1][0]))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    plt.show()

    return(random_indices)

def interactiveCorrelationMatrix(dataSet):
    """
    Description
    -----------
    interactiveCorrelationMatrix() makes an interactive plot of the cosine matrix provided in the input dataSet. This is useful for zooming in on a specific section of the matrix to find the indices of highly correlated ROIs

    Parameters
    ----------
    dataSet : 2D array-like
       datset in question (Should be the cosine matrix, or any other form of correlation matrix)

    Returns
    -------
    Nothing! But it makes an interactive plot of the cosine matrix provided in the input dataSet. This is useful for zooming in on a specific section of the matrix to find the indices of highly correlated ROIs
    """
    fig = go.Figure(data=go.Heatmap(z=dataSet))
    
    # Update layout
    fig.update_layout(
        title='Heatmap',
        xaxis=dict(title='X Axis', scaleanchor='y', scaleratio=1),
        yaxis=dict(title='Y Axis', scaleanchor='x', scaleratio=1)
    )
    
    # Show the plot
    fig.show()

############################################# This marks the start of the Clustering sutff #######################################################################################################################

def maxClusterSize(cutClusterPoints):
    """
    Description:
    ------------ 
    maxClusterSize() is a function where given the cut cluster points after cutting the dendrogram, get the max cluster size so that you can make the index matrix

    Parameters
    ----------
    cutClusterPoiunts : 2D array-like
       Clusters of points cut under a certain threshold value; obtained from linkage() from scipy.cluster.hierarchy

    Returns
    -------
    maxValue : int
        Size of the largest cluster
    """
    flattenedClusterPoints = cutClusterPoints.flatten()
    unique, counts = np.unique(flattenedClusterPoints, return_counts=True)
    occuranceDict = dict(zip(unique, counts))
    maxValue = max(occuranceDict.values())
    return(maxValue)

def fillIndexMatrix(cutClusterPoints):
    """
    Description:
    ------------ 
    fillIndexMatrix(): Will take the cut cluster points array and returns a(number  of clusters)x(length of longest cluster) matrix where each column holds in the ROI index of each cluster, then padded with 0's
    Read numbers from left to right per column until you hit a zero, then these numbers are the ROI indices for that cluster
        ^Maybe fill with NaNs instead of 0s to avoid bug?

    Parameters
    ----------
    cutClusterPoiunts : 2D array-like
       Clusters of points cut under a certain threshold value; obtained from linkage() from scipy.cluster.hierarchy

    Returns
    -------
    indexMatrix : 2D array-like
        Matrix containing index points for heirarchical clustering
    """
    indexMatrix = np.zeros(((int(np.amax(cutClusterPoints))+1),maxClusterSize(cutClusterPoints)))
    
    for i in range(int(np.max(cutClusterPoints)+1)):
        clusterIndex = np.where(cutClusterPoints == [i])[0] #Take the [0] element because this has the indices of which points are in the cluster. The other elements contain extraneous data we dont care about.
        clusterIndex = [int(index) for index in clusterIndex]
        clusterLength = len(clusterIndex)
        maxClust = maxClusterSize(cutClusterPoints)
        paddedClust = np.pad(clusterIndex, (0,(maxClust-clusterLength)), 'constant', constant_values=(0))
        indexMatrix[i,:]=paddedClust
        
    return(indexMatrix)

def heirarchicalClustering(data,cutValue):
    """
    Description:
    ------------ 
    heirarchicalClustering(): Function that takes the data thats going to get clusters, the height at which you want to cut the tree and the specific axon you want to look at and will return an array with the ROI indices
    use like [clusters, linkage] = heirarchicalClustering(rawData,cutValue)

    Parameters
    ----------
    data : 2D array-like
       Data set array

    Returns
    -------
    [filledMatrix, linkage_data]
        FilledMatrix : 2D array-like
            Matrix where each row represents the HC information for that indices ROI
        linkage_data : object
            Feed into dendrogram() function from the HC scipy module to produce the classic HC correlation tree
    """
    cosMatrix = cosDistMatrix(data)
    linkage_data = linkage(cosMatrix)
    cutClusterPoints = cut_tree(linkage_data, height=cutValue)
    filledMatrix = fillIndexMatrix(cutClusterPoints)
    return([filledMatrix, linkage_data])

def variableClustering(data, stepSize = 0.05, minCut = 0.3, maxCut = 2.4):
    """
    Description:
    ------------ 
    variableClustering() performs the heirarchicalClustering() function multiple times at varying cut levels and returns a list where each entry is the returned [filledMatrix, linkage_data] list from heirarchicalClustering()

    Parameters
    ----------
    data : array-like, shape (1,)
        Data set you want to perform hc on.
    stepSize : float
        The step size between cut values on the dendrogram. Default value is tailored to the pre-stimuls phase of the Ground Truth data set.
    minCut : float
        Minimum cut value. Usually anything below this value is just clusters of size 1. Default value is tailored to the pre-stimuls phase of the Ground Truth data set.
    maxCut : float
        Maximum cut value. Usually at this value there is just one cluster the size of the data set. Default value is tailored to the pre-stimuls phase of the Ground Truth data set.

    Returns
    -------
    maxValue : int
        Size of the largest cluster
    """
    hcList = [] # Initialize a blank list for each execution of heirarchicalClustering(). Want a list because we will have elements of variable size between loop iterations.
    i=minCut # Initialize starting value

    # Loop over variable cut values and store them all in hcList to be returned at the end
    while i <= maxCut:
        [clustMat,dend] = heirarchicalClustering(data, i)
        hcList.append([clustMat,dend])
        i+=stepSize
    return(hcList)

############################################# This marks the start of the PCA sutff #######################################################################################################################

def pcaAnalysis2D(data, pdfSave = False, pdfName = 'default', colorscale = 'viridis', pointSize = 1):
    """
    Description
    -----------
    pcaAnalysis2D() performs PCA on a 2D data set (Like 'F.npy' files from suite2p) and returns the first three principal components in array form.
    It also makes a 1x3 subplot of all the different combinations of the first 3 principal components together (three combinations total).

    Parameters
    ----------
    data : 2D array
        Data set array, should be deltaF normalized before being put into this function
    pdfSave : Boolean
        If true then this one plot will be saved to PDF in the working directory
    pdfName : String
        Only used if pdfSave is 'True'. This is the name of the pdf if saved. Do NOT add '.pdf' to this tring
    colorscale : string
        String dictating the colorscale/map you want for the plot. Default is 'viridis' as to avoid cyclic nature (wanted to show beginning and end times at the time of implementation)
    pointSize : float
        Full disclosure, I dont know if its actually a float. But this is the point size of the PCA plot.

    Returns
    -------
    [pca.components_, pca.explained_variance_ratio_, fig] : Python List. Size is [(len(data), 3) numpy array, (1, 3) numpy array]
        pca.components_ : numpy array shape (3,)
            First three principal components in numpy array form
        pca.explained_variance_ratio_ : numpy array
            Corresponding eigenvalue ratios for the first three principal components (Remember, this can be interpreted as the weight of importance for each principal component, sorted descendingly)
        fig : Plotly Figure Object
            PCA plot object
    """
    # Initialize the PCA object for the first three components
    pca = PCA(n_components=3)

    # Standardize and fit
    standardData = StandardScaler().fit_transform(data)
    pcaData = pca.fit_transform(standardData)
    transPCAData = np.transpose(pcaData)  # Transpose them for the sake of plotting (Theres a lot of transpose flip-flopping throughout this, im so sorry)

    # Define objects for pca plot
    x = transPCAData[0]
    y = transPCAData[1]
    z = transPCAData[2]

    # Create color range for plotting
    colors = np.zeros(len(transPCAData[0]))
    for i in range(len(transPCAData[0])):
        colors[i] = i

    # Create the figure
    fig = make_subplots(
        rows = 1,
        cols = 3,
        subplot_titles=(
            'PC1 vs PC2',
            'PC1 vs PC3',
            'PC2 vs PC3'
        )
    )

    # Add first scatter plot
    fig.add_trace(
        go.Scatter(
            x = x,
            y = y,
            mode = 'markers',
            name = 'PC1 vs PC2',
            marker = dict(
                size = pointSize,
                color = colors,
                colorscale = colorscale,
                opacity = 0.8
            ),
            showlegend = False
        ),
        row = 1,
        col = 1
    )

    # Add second scatter plot
    fig.add_trace(
        go.Scatter(
            x = x,
            y = z,
            mode = 'markers',
            marker = dict(
                size = pointSize,
                color = colors,
                colorscale = colorscale,
                opacity = 0.8
            ),
            showlegend = False
            
        ),
        row = 1,
        col = 2
    )

    # Add third scatter plot
    fig.add_trace(
        go.Scatter(
            x = y,
            y = z,
            mode = 'markers',
            marker = dict(
                size = pointSize,
                color = colors,
                colorscale = colorscale,
                colorbar = dict(
                    title = 'Length of Movie',
                    titleside ='right',
                    tickvals = [],
                    ticktext = [],
                    ticks = "",
                    len = 1
                ),
                opacity = 0.8
            ),
            showlegend = False
        ),
        row = 1,
        col = 3
    )


    # Update layout
    fig.update_layout(
        title_text='PCA Analysis 2D',
        height=400,
        width=900,
        annotations=[
            dict(
                x=0.98,
                y=-0.25,
                xref='paper',
                yref='paper',
                text='Beginning of<br>Experiment',
                showarrow=False,
                xanchor='left',
                yanchor='bottom'
            ),
            dict(
                x=0.99,
                y=1.25,
                xref='paper',
                yref='paper',
                text='End of<br>Experiment',
                showarrow=False,
                xanchor='left',
                yanchor='top'
            )
        ]
    )

    # (1,1) x axis
    fig.update_xaxes(
        title_text = "PC1",
        row = 1,
        col = 1,
        title_standoff = 1
    )

    # (1,1) y axis
    fig.update_yaxes(
        title_text = "PC2",
        row = 1,
        col = 1,
        title_standoff = 0
    )

    # (1,2) x axis
    fig.update_xaxes(
        title_text = "PC1",
        row = 1,
        col = 2,
        title_standoff = 1
    )

    # (1,2) y axis
    fig.update_yaxes(
        title_text = "PC3",
        row = 1,
        col = 2,
        title_standoff = 0
    )

    # (1,3) x axis
    fig.update_xaxes(
        title_text = "PC2",
        row = 1,
        col = 3,
        title_standoff = 1
    )

    # (1,3) y axis
    fig.update_yaxes(
        title_text = "PC3",
        row = 1,
        col = 3,
        title_standoff = 0
    )
    
    # Show plot
    fig.show()

    if pdfSave == True:
        working_directory = os.getcwd()
        pdf_path = os.path.join(working_directory, pdfName+'.pdf')
        fig.write_image(pdf_path)
    
    return([pca.components_, pca.explained_variance_ratio_, fig])

def pcaAnalysis3D(data, experimentMetadata, pdfSave = False, pdfName = 'default', colorscale = 'viridis', pointSize = 1):
    """
    Description
    -----------
    pcaAnalysis3D() performs PCA on a 2D data set (Like 'F.npy' files from suite2p) and returns the first three principal components in array form.
    It also makes an interactive 3D Scatter plot of the first three principal components plotted against themselves

    Parameters
    ----------
    data : 2D array
        Data set array, should be deltaF normalized before being put into this function
    experimentMetadata : String
        Metadata of the experiment, like date, mouseID, age, sex, etc.
    pdfSave : Boolean
        If true then this one plot will be saved to PDF in the working directory
    pdfName : String
        Only used if pdfSave is 'True'. This is the name of the pdf if saved. Do NOT add '.pdf' to this tring
    colorscale : string
        String dictating the colorscale/map you want for the plot. Default is 'viridis' as to avoid cyclic nature (wanted to show beginning and end times at the time of implementation)
    pointSize : float
        Full disclosure, I dont know if its actually a float. But this is the point size of the PCA plot.

    Returns
    -------
    [pca.components_, pca.explained_variance_ratio_, fig] : Python List. Size [(len(data), 3) numpy array, (1, 3) numpy array, 1]
        pca.components_ : numpy array shape (3,)
            First three principal components in numpy array form
        pca.explained_variance_ratio_ : numpy array
            Corresponding eigenvalue ratios for the first three principal components (Remember, this can be interpreted as the weight of importance for each principal component, sorted descendingly)
        fig : Plotly Figure Object
            PCA plot object
    """
    # Initialize the PCA object for the first three components
    pca = PCA(n_components=3)

    # Standardize and fit
    standardData = StandardScaler().fit_transform(data)
    pcaData = pca.fit_transform(standardData)
    transPCAData = np.transpose(pcaData)  # Transpose them for the sake of plotting (Theres a lot of transpose flip-flopping throughout this, im so sorry)

    # Define objects for pca plot
    x = transPCAData[0]
    y = transPCAData[1]
    z = transPCAData[2]

    # Create color range for plotting
    colors = np.zeros(len(transPCAData[0]))
    for i in range(len(transPCAData[0])):
        colors[i] = i

    # Create the 3D scatter plot (FIRST (TOP) PLOT OF THE SUBPLOTS)
    scatter = go.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode = 'markers',
        marker = dict(
            size = pointSize,
            color = colors,  # Assign the colors here
            colorscale = colorscale,  # Choose a colorscale
            opacity = 0.8,
            colorbar = dict(
                title = 'Length of Movie',
                titleside = 'right',
                tickvals = [colors.min(), colors.max()],
                ticktext = ['', ''],
                ticks = ""
            )
        )
    )

    fig = go.Figure(data = [scatter])

    # This first update_layout is what we are using to set the overall size of the image. We are also plotting the first plot (scatter plot) in this step
    fig.update_layout(
        width=800,
        height=800,
        title=experimentMetadata,
        scene=dict(
            xaxis_title='PC 1',
            yaxis_title='PC 2',
            zaxis_title='PC 3',
            camera=dict(
                eye=dict(
                    x=1.75,
                    y=1.75,
                    z=1.75
                )
            )
        ),
        # Using annotations here so that we can put the "beginning of experiment" stuff above and below the color bar. Better readability
        annotations=[
            dict(
                x=0.99,
                y=-0.05,
                xref='paper',
                yref='paper',
                text='Beginning of<br>Experiment',
                showarrow=False,
                xanchor='left',
                yanchor='bottom'
            ),
            dict(
                x=0.99,
                y=1.05,
                xref='paper',
                yref='paper',
                text='End of<br>Experiment',
                showarrow=False,
                xanchor='left',
                yanchor='top'
            )
        ]
    )

    
    # Show plot
    fig.show()

    if pdfSave == True:
        working_directory = os.getcwd()
        pdf_path = os.path.join(working_directory, pdfName+'.pdf')
        # print(f'Saving PDF to: {pdf_path}')
        fig.write_image(pdf_path)
    
    return([pca.components_, pca.explained_variance_ratio_, fig])

def pcPlotWithTimeStepImage(principalComponents, imagePath='./Images/time step.png', pdfSave=False, pdfName='default', colorscale='viridis'):
    """
    Description
    -----------
    pcPlotWithTimeStepImage() creates a 2x1 figure where the top plot is the staircase of light steps and their respective strengths and the bottom plot is the heatmap of the first 3 principal components.
    Meant to be used directly in conjunction with pcaAnalysis2D() or pcaAnalysis3D()

    Parameters
    ----------
    principalComponents : numpy array shape (3,)
        principal components obtained by either thepcaAnalysis2D()[0] or pcaAnalysis3D()[0] objects.
    imagePath : String
        Path to the light-step image to be imported.
    pdfSave : Boolean
        If true then this one plot will be saved to PDF in the working directory
    pdfName : String
        Only used if pdfSave is 'True'. This is the name of the pdf if saved. Do NOT add '.pdf' to this tring
    colorscale : string
        String dictating the colorscale/map you want for the plot. Default is 'viridis' as to avoid cyclic nature (wanted to show beginning and end times at the time of implementation)

    Returns
    -------
    fig : Plotly figure object
        Object of the figure in question
    """
    try:
        # Create subplots with two subplots: one for image, and one for the heatmap
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{'type': 'image'}], [{'type': 'heatmap'}]],  # Manually set the plot type for each subplot
            subplot_titles=("Time Steps", "Heatmap of First Three Principal Components"),
            vertical_spacing=0.1  # Adjust vertical spacing if needed
        )

        # Add the image
        img = Image.open(imagePath)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        image_trace = go.Image(
            source=f'data:image/png;base64,{img_str}',
            dy=2.5,  # Adjust width if needed (dx in the x direction and dy in the y direction)
            dx=1.3
        )
        fig.add_trace(image_trace, row=1, col=1)

        # Setting the heatmap data as the first 3 principal components obtained by the .components_ attribute
        heatmap_data1 = principalComponents[0][np.newaxis]
        heatmap_data2 = principalComponents[1][np.newaxis]
        heatmap_data3 = principalComponents[2][np.newaxis]
        heatmap_data = np.vstack((heatmap_data1, heatmap_data2, heatmap_data3))

        # Create the heatmap (SECOND (MIDDLE) PLOT OF THE SUBPLOTS)
        heatmap_trace = go.Heatmap(
            z=heatmap_data,
            colorscale=colorscale,
            colorbar=dict(
                title='PC Value',
                orientation='v',
                tickvals=[np.min(heatmap_data), np.max(heatmap_data)],
                ticktext=[str(np.min(heatmap_data)), str(np.max(heatmap_data))],
                len=0.5,  # Adjust length of colorbar
                y=0.25
            )
        )

        # Add the first heatmap
        fig.add_trace(heatmap_trace, row=2, col=1)

        ##################################################################### Manual updating of each subplot ######################################################################################################

        # This first update_layout is what we are using to set the overall size of the image. We are also plotting the first plot (scatter plot) in this step
        fig.update_layout(
            width=800,
            height=600,
            title="",
            scene=dict(
                xaxis_title='PC 1',
                yaxis_title='PC 2',
                zaxis_title='PC 3'
            )
        )

        # (1,1) x axis
        fig.update_xaxes(
            title_text="Light time steps",
            tickvals=[0, 68, 140, 210, 280, 351, 423, 495, 565],  # Manually set tick values
            ticktext=['0', '1', '2', '3', '4', '5', '6', '7', '8'],
            tickangle=0,
            row=1,
            col=1
        )

        # (1,1) y axis
        fig.update_yaxes(
            title_text="Irradiance",
            tickvals=[0, 180],
            ticktext=['Max', '0'],
            tickangle=0,
            row=1,
            col=1
        )

        # (2,1) x axis
        fig.update_xaxes(
            title_text="ROI indices",
            row=2,
            col=1
        )

        # (2,1) y axis
        fig.update_yaxes(
            title_text="Principal Components",
            tickvals=[0, 1, 2],
            ticktext=['PC1', 'PC2', 'PC3'],
            tickangle=0,
            row=2,
            col=1
        )
        
        # Show plot
        fig.show()

        if pdfSave:
            working_directory = os.getcwd()
            pdf_path = os.path.join(working_directory, pdfName + '.pdf')
            print(f'Saving PDF to: {pdf_path}')  # Debugging statement
            fig.write_image(pdf_path)

        return(fig)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return(None)
    
def roiHeatmap(data, colorscale = 'viridis'):
    """
    Description
    -----------
    roiHeatmap() is relatively simple and just produces the heatmap for all the ROIs in our data set.

    Parameters
    ----------
    data : numpy array
        2D data set. Should be configured as such where the rows are individual ROIs and the columns are frames.
    colorscale : string
        String dictating the colorscale/map you want for the plot. Default is 'viridis' as to avoid cyclic nature (wanted to show beginning and end times at the time of implementation)

    Returns
    -------
    fig : Plotly figure object
        Object of the figure in question
    """
    fig = go.Figure(data=go.Heatmap(
        z = data,
        colorscale=colorscale,
        colorbar = dict(
            title = 'Relative Brightness',
            titleside ='right',
            tickvals = [],
            ticktext = [],
            ticks = "",
            len = 1
                )
        )
    )

    fig.update_layout(
        width = 800,
        height = 1000,
        title = "ROI Heatmap"
    )

    # (2,1) x axis
    fig.update_xaxes(
        title_text = "Frames",
    )

    # (2,1) y axis
    fig.update_yaxes(
        title_text = "ROIs",
    )
    
    fig.show()

    return(fig)

def pcaFigureGenerator(data, experimentMetadata, pdfName):
    """
    Description
    -----------
    pcaFigureGenerator() generates a PDF of all the pca plots related to a specific data set.

    Parameters
    ----------
    data : numpy array
        2D data set in question
    experimentMetadata : String
        Metadata of the experiment, like date, mouseID, age, sex, etc.
    pdfName : string
        Custom name of the pdf. Do NOT include ".pdf" in this part of the name.

    Returns
    -------
    eigenvalueArray : array-like, shape (1,)
        Same name but this is now a normalized eigenvalue array so we can really see "percent contribution" Of a specific eigvector
    """
    pcaPlot3D = pcaAnalysis3D(data.T, experimentMetadata = experimentMetadata, pointSize = 2)
    pcaPlot2D = pcaAnalysis2D(data.T, pointSize = 2)
    imagePlot = pcPlotWithTimeStepImage(pcaPlot3D[0])
    heatmap = roiHeatmap(data)
    plotList = [pcaPlot3D[2], pcaPlot2D[2], imagePlot, heatmap]
    savePlotyPlotsToPDF(plotList, pdf_name="./PDFs/"+str(pdfName)+".pdf")

def pcaEigvalRatio(eigenvalueArray):
    """
    Description
    -----------
    pcaEigvalRatio() takes in the raw eigenvalue vector and spits out the ratio of eigenvalue to total, so eigenvalue/sum of eigenvalues

    Parameters
    ----------
    eigenvalueArray : array-like, shape (1,)
       eigenvalue array, preferablle gotten from pcaEigvals()

    Returns
    -------
    eigenvalueArray : array-like, shape (1,)
        Same name but this is now a normalized eigenvalue array so we can really see "percent contribution" Of a specific eigvector
    """
    tot = np.sum(eigenvalueArray)
    for i in range(len(eigenvalueArray)):
        eigenvalueArray[i]=eigenvalueArray[i]/tot
        
    return(eigenvalueArray)

################################################################## Start of Denoising Stuff ########################################################################################

# Proposed Change: Instead of cleaning the trace using these indices, collect these indices for all ROIs and then have some method of "voting" on which indices are kept. Something like a "global averaging" on what frequency indices should be dropped.
# Maybe start by looking at how much agreement there is between ROIs. Have a matrix where rows are ROIs and columns are PSD indices and we are basically tracking which indices this procedure is saying to keep for each ROI
def fourierDenoising(signal, threshold):
    """
    Description
    -----------
    fourierDenoising() takes in a signal array and a thehold value you want to amplitude cut at and returns the denoised signal array

    Parameters
    ----------
    signal : array-like, shape (1,)
       Signal array
    threshold : float
        Treshold that you want to cut your PSD spikes at
    Returns
    -------
    filteredSignal : array-like, shape (1,)
        Filtered signal array
    """
    sig = signal #Get signal
    dt = 3 #Sample rate, needs to be taken from metadata. Maybe a way to automate this straight from the excel sheet?
    n = len(sig) #Length of sample
    fhat = np.fft.fft(sig, n) #compute the fft
    psd = fhat*np.conj(fhat)/n #Compute the power spectrum density (?) (This is what makes everything positive so that we can do the amplitude cutting)
    freq = (1/(dt*n))*np.arange(n) #Create an array for the frequencies (?)
    L = np.arange(1,np.floor(n/2), dtype='int') #Length of the fft we want to look at (Because its symetric about the origin we only are gonna look at the first half)

    indices = psd > threshold #Create an array with the same length as psd where each index is either a 1 or a 0 based off of the inrequality requirement (This is checking to see if the array values are above our threshold value)
    psdclean = psd*indices #Multiply our idices array by psd, this will multiply all the indices that dont match our threshold with a '0', thus annihilating them
    fhat = indices*fhat #Clear these same indices from the original fourier transform (Since each index in the fhat array is a fourier coefficient, we basically just set those to 0)
    filteredSignal = np.fft.ifft(fhat) #Invert
    return(filteredSignal)

def fourierDenoiseDataset(dataSet, threshold):
    denoisedData = np.zeros((len(dataSet),len(dataSet[0])))
    for i in range(len(dataSet)):
        denoisedData[i]=fourierDenoising(dataSet[i], threshold)

    return(denoisedData)


# fourierHighpass() does essentially the same thing as fourierDenoising(), infact the code is nearly identical, but the inequality sign is flipped in the definition of the 'indices' variable
# DISCLAIMER: THIS IS NOT A TRUE HIGH PASS FILTER. This is a 'rough approximation' of a high pass filter that just so happens to work for us.
# Instead of checking to see what frequencies have an amplitude ABOVE a crtain threshold, this shows the frequencies BELOW a certain threshold.
# Since the PSD for most of the data we work with is heavily favored for the lower frequencies, this is kind of a trivial "poor mans" method of high-pass filtering. Its not exactly the same, and I intend to make a better algorithm, but Im using this for now
def fourierHighpass(signal, threshold, sampleRate = 3):
    """
    Description
    -----------
    fourierHighpass() does essentially the same thing as fourierDenoising(), infact the code is nearly identical, but the inequality sign is flipped in the definition of the 'indices' variable
    DISCLAIMER: THIS IS NOT A TRUE HIGH PASS FILTER. This is a 'rough approximation' of a high pass filter that just so happens to work for us.
    Instead of checking to see what frequencies have an amplitude ABOVE a crtain threshold, this shows the frequencies BELOW a certain threshold.
    Since the PSD for most of the data we work with is heavily favored for the lower frequencies, this is kind of a trivial "poor mans" method of high-pass filtering. Its not exactly the same, and I intend to make a better algorithm, but Im using this for now

    Parameters
    ----------
    signal : array-like, shape (1,)
       Signal array
    threshold : float
        Treshold that you want to cut your PSD spikes at
    sampleRate : float
        Sample rate of the data of the signal. A priori knowledge.
        Default sample rate is 3
    Returns
    -------
    filteredSignal : array-like, shape (1,)
        Same name as filteredSignal, but this returns the 'high-pass' signal
    """
    sig = signal #Get signal
    dt = sampleRate #Sample rate, needs to be taken from metadata. Maybe a way to automate this straight from the excel sheet?
    n = len(sig) #Length of sample
    fhat = np.fft.fft(sig, n) #compute the fft
    psd = fhat*np.conj(fhat)/n #Compute the power spectrum density (?) (This is what makes everything positive so that we can do the amplitude cutting)
    freq = (1/(dt*n))*np.arange(n) #Create an array for the frequencies (?)
    L = np.arange(1,np.floor(n/2), dtype='int') #Length of the fft we want to look at (Because its symetric about the origin we only are gonna look at the first half)

    indices = psd < threshold #Create an array with the same length as psd where each index is either a 1 or a 0 based off of the inrequality requirement (This is checking to see if the array values are below our threshold value)
    psdclean = psd*indices #Multiply our idices array by psd, this will multiply all the indices that dont match our threshold with a '0', thus annihilating them
    fhat = indices*fhat #Clear these same indices from the original fourier transform (Since each index in the fhat array is a fourier coefficient, we basically just set those to 0)
    filteredSignal = np.fft.ifft(fhat) #Invert
    return(filteredSignal)

def highpassFilterButter(data, order, criticalFreq, frameRate):
    """
    Description
    -----------
    highpassFilterButter() performs a Butterworth highpass filter on a desired signal.

    Parameters
    ----------
    data : array-like, shape (1,)
       Signal array
    order : int
        Order of the Butterworth signal. Higher order leads to "crisper" results.
    criticalFreq : float
        Cutoff for the critical freuency. Beacuse of some weird signal processing stuff, it must be in the interval [0,criticalFreq/2]
    frameRate : float
        Framerate of the experiment.
    
    Returns
    -------
    filteredSignal : array-like, shape (1,)
        Filtered signal array
    """
    butterworthStuff = signal.butter(order,criticalFreq, btype='high', output='sos', fs=frameRate)
    filteredSignal = signal.sosfilt(butterworthStuff,data)
    return(filteredSignal)

def lowpassFilterButter(data, order, criticalFreq, frameRate):
    """
    Description
    -----------
    lowpassFilterButter() performs a lowpass Butterworth filter on a desired signal.

    Parameters
    ----------
    data : array-like, shape (1,)
       Signal array
    order : int
        Order of the Butterworth signal. Higher order leads to "crisper" results.
    criticalFreq : float
        Cutoff for the critical freuency. Beacuse of some weird signal processing stuff, it must be in the interval [0,criticalFreq/2]
    frameRate : float
        Framerate of the experiment.
    
    Returns
    -------
    filteredSignal : array-like, shape (1,)
        Filtered signal array
    """

    # Check for NaNs and infinite values in the data
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Input data contains NaNs or infinite values.")
    
    # Ensure criticalFreq is within the valid range
    if not (0 < criticalFreq < frameRate / 2):
        raise ValueError("criticalFreq must be in the interval (0, frameRate/2).")
    
    butterworthStuff = signal.butter(order,criticalFreq, btype='low', output='sos', fs=frameRate) # Create the buttwrworth filter (matrices and scalars)
    filteredSignal = signal.sosfiltfilt(butterworthStuff,data) # Actually perform the filtering on your signal. Use 'sosfiltfilt()' to reduce artifacts in the signal (Like those massive spike we've been getting)
    return(filteredSignal)

def cleanDataLow(data, fr, cf, st=0.8):
    """
    Description
    -----------
    cleanData() cleans and organizes a data set first by implementing a butterworth filter then normalizing each trace by their 95th percentile and finally sorting the data.

    Parameters
    ----------
    data : 2D array-like
       Data Set thats going to be cleaned.
    fr : float
        frameRate of the data set, collected from the metadata
    cf : float
        Cutoff for the critical freuency. Beacuse of some weird signal processing stuff, it must be in the interval [0,criticalFreq/2]
    st : float in the interval (0,1)
        Sorting threshold. This is the value that needs to be hit so that it can sort correctly
    
    Returns
    -------
    sortedData : 2D array-like, same shape as data
        Cleaned and Sorted Data
    """
    filteredData = lowpassFilterButter(data, 10, frameRate=fr, criticalFreq=cf)
    normalizedData = normalizePercentile(filteredData)
    sortedData = sortMatrixByThreshold(normalizedData, st)
    return(sortedData)

def cleanDataHigh(data, fr, cf, st=0.8):
    """
    Description
    -----------
    cleanData() cleans and organizes a data set first by implementing a butterworth filter then normalizing each trace by their 95th percentile and finally sorting the data.

    Parameters
    ----------
    data : 2D array-like
       Data Set thats going to be cleaned.
    fr : float
        frameRate of the data set, collected from the metadata
    cf : float
        Cutoff for the critical freuency. Beacuse of some weird signal processing stuff, it must be in the interval [0,criticalFreq/2]
    st : float in the interval (0,1)
        Sorting threshold. This is the value that needs to be hit so that it can sort correctly
    
    Returns
    -------
    sortedData : 2D array-like, same shape as data
        Cleaned and Sorted Data
    """
    filteredData = highpassFilterButter(data, 10, frameRate=fr, criticalFreq=cf)
    normalizedData = normalizePercentile(filteredData)
    sortedData = sortMatrixByThreshold(normalizedData, st)
    return(sortedData)