import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

############################################# This marks the start of the Heirarchical Clustering sutff #######################################################################################################################

# a and b have to be vectors of the same length
def cosineSimilarity(a,b):
    return(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

# costDistMatrix(): Given the data set this function will return the cosine similarity distance matrix
def cosDistMatrix(data):
    distMatrix = np.empty([len(data),len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            #1-cosSimilarity to make it a "distance", since angles arent really a distance (Unless you get philosophical, then youre on your own)
            distMatrix[i,j]=1-cosineSimilarity(data[i],data[j])
    
    return(distMatrix)

# maxClusterSize(): Given the cut cluster points after cutting the dendrogram, get the max cluster size so that you can make the index matrix
def maxClusterSize(cutClusterPoints):
    flattenedClusterPoints = cutClusterPoints.flatten()
    unique, counts = np.unique(flattenedClusterPoints, return_counts=True)
    occuranceDict = dict(zip(unique, counts))
    maxValue = max(occuranceDict.values())
    return(maxValue)

# fillIndexMatrix(): Will take the cut cluster points array and returns a (number of clusters)x(length of longest cluster) matrix where each column holds in the ROI index of each cluster, then padded with 0's
# Read numbers from left to right per column until you hit a zero, then these numbers are the ROI indices for that cluster
# Maybe fill with NaNs instead of 0s to avoid bug?
def fillIndexMatrix(cutClusterPoints):
    indexMatrix = np.zeros(((np.max(cutClusterPoints)+1),maxClusterSize(cutClusterPoints)))
    
    for i in range(np.max(cutClusterPoints)+1):
        clusterIndex = np.where(cutClusterPoints == [i])[0]
        clusterLength = len(clusterIndex)
        maxClust = maxClusterSize(cutClusterPoints)
        paddedClust = np.pad(clusterIndex, (0,(maxClust-clusterLength)), 'constant', constant_values=(0))
        indexMatrix[i,:]=paddedClust
        
    return(indexMatrix)

# assignROItoAxon(): Function that takes the data thats going to get clusters, the height at which you want to cut the tree and the 
# specific axon you want to look at and will return an array with the ROI indices
# use like [clusters, linkage] = assignROItoAxon(rawData,cutValue)
def assignROItoAxon(data,cutValue):
    cosMatrix = cosDistMatrix(data)
    linkage_data = linkage(cosMatrix)
    cutClusterPoints = cut_tree(linkage_data, height=cutValue)
    filledMatrix = fillIndexMatrix(cutClusterPoints)
    return([filledMatrix, linkage_data])

############################################# This marks the start of the PCA sutff #######################################################################################################################

# pcaAnalysis(): Takes in the ROI data set and returns a pca graph with the first three principal components
def pcaAnalysis(data):

    #Initialize the pca object for the first three components
    pca = PCA(n_components = 3)

    #Standardize and fit
    standardData = StandardScaler().fit_transform(data)
    pcaData = pca.fit_transform(standardData)
    transPCAData = np.transpose(pcaData)

    #Define objects for
    x = transPCAData[0]
    y = transPCAData[1]
    z = transPCAData[2]

    #create a default list filled with black 1702 times, then go in with transformed ground truth ROIs and flip the appropriate indices to red or somethin
    colors = ['black']*1702
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size = 3, color=colors, opacity=0.8))])

    # Update layout
    fig.update_layout(scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
                      title='3D Scatter Plot',
                      autosize=False,
                      width=800, height=800)

    # Show the plot
    fig.show()

# pcaAnalysis(): Takes in the ROI data set and returns a pca graph with the first three principal components and a color scale. Default colorscale is hsv.
# This is good for PCA on the transposed data matrix which takes ROIs as the variables and gives frames as the points on the graph. Shows the wrap-around nature of the staircase experiments
def pcaAnalysisColored(data, clrscl='hsv', cbar=False):

    #Initialize the pca object for the first three components
    pca = PCA(n_components = 3)

    #Standardize and fit
    standardData = StandardScaler().fit_transform(data)
    pcaData = pca.fit_transform(standardData)
    transPCAData = np.transpose(pcaData)

    #Define objects for
    x = transPCAData[0]
    y = transPCAData[1]
    z = transPCAData[2]
    colors = np.zeros(len(transPCAData[0]))
    for i in range(len(transPCAData[0])):
        colors[i] = i

    # Create a 3D scatter plot
    fig = go.Figure()
    if cbar == True:
        scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=3,
                color=colors,  # Assign the colors here
                colorscale=clrscl,  # Choose a colorscale
                opacity=0.8,
                colorbar=dict(title='Colormap') #Create colorbar
            )
        )
    else:
        scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=3,
                color=colors,  # Assign the colors here
                colorscale=clrscl,  # Choose a colorscale
                opacity=0.8,
            )
        )

    fig.add_trace(scatter)

    # Set layout
    fig.update_layout(scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'))

    # Show the plot
    fig.show()


# pcaNormalizedAnalysis(): Same thing as pcaAnalysis() except every tace is normalized to itself before being ran through PCA
def pcaNormalizedAnalysis(data):
    pca = PCA(n_components = 3)
    #Standardize
    normalData = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        normalizedTrace = (1/data[i].max())*data[i]
        normalData[i] = normalizedTrace
        
    standardData = StandardScaler().fit_transform(normalData)
    pcaData = pca.fit_transform(standardData)
    transPCAData = np.transpose(pcaData)

    x = transPCAData[0]
    y = transPCAData[1]
    z = transPCAData[2]
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size = 3, opacity=0.8))])

    # Update layout
    fig.update_layout(scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
                      title='3D Scatter Plot',
                      autosize=False,
                      width=800, height=800)

    # Show the plot
    fig.show()

# pcaNormalizedAnalysis: Same thing as pcaAnalysisColored() except every tace is normalized to itself before being ran through PCA. 
# Default colorscale is hsv.
# This is good for PCA on the transposed data matrix which takes ROIs as the variables and gives frames as the points on the graph. Shows the wrap-around nature of the staircase experiments
def pcaNormalizedAnalysisColored (data, clrscl = 'hsv'):
    pca = PCA(n_components = 3)
    #Standardize
    normalData = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        normalizedTrace = (1/data[i].max())*data[i]
        normalData[i] = normalizedTrace
        
    standardData = StandardScaler().fit_transform(normalData)
    pcaData = pca.fit_transform(standardData)
    transPCAData = np.transpose(pcaData)

    x = transPCAData[0]
    y = transPCAData[1]
    z = transPCAData[2]

    #create a default list filled with black 1702 times (length of trace), then go in with transformed ground truth ROIs and flip the appropriate indices to red or somethin
    colors = np.zeros(len(transPCAData[0]))
    for i in range(len(transPCAData[0])):
        colors[i] = i
    #Going to normalize each data point
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size = 3, color=colors, colorscale=clrscl, opacity=0.8, colorbar=dict(title='Colormap')))])

    # Update layout
    fig.update_layout(scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
                      title='3D Scatter Plot',
                      autosize=False,
                      width=800, height=800)

    # Show the plot
    fig.show()

# pcaAnalysis(): Takes in the ROI data set and returns a pca graph with the first three principal components
def pcaAnalysisVariableComponents(data, component1, component2, component3):

    #Initialize the pca object for the first three components
    pca = PCA(n_components = 3)

    #Standardize and fit
    standardData = StandardScaler().fit_transform(data)
    pcaData = pca.fit_transform(standardData)
    transPCAData = np.transpose(pcaData)

    #Define objects for
    x = transPCAData[component1]
    y = transPCAData[component2]
    z = transPCAData[component3]

    #create a default list filled with black 1702 times, then go in with transformed ground truth ROIs and flip the appropriate indices to red or somethin
    colors = ['black']*1702
    axon2idx = [1686, 1660, 1638, 1630, 1613, 1593, 1553, 1536]
    for j in axon2idx:
        colors[j] = 'blue'

    axon1idx = [1691, 1622, 1612, 1588, 1558, 1538]
    for i in axon1idx:
        colors[i] = 'red'

    axon3idx = [1338, 1347, 1367, 1394, 1405, 1417, 1423, 1429, 1435, 1441, 1450]
    for k in axon3idx:
        colors[k] = 'green'

    axon4idx = [1488, 1458, 1436, 1426, 1416]
    for l in axon4idx:
        colors[l] = 'cyan'    
    #Right under here do the transformation
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size = 3, color=colors, opacity=0.8))])

    # Update layout
    fig.update_layout(scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
                      title='3D Scatter Plot',
                      autosize=False,
                      width=800, height=800)

    # Show the plot
    fig.show()

# pcaEigvals(): takes in the 2D data array and returns the sorted descending eigenvalues and eigenvectors of the covarience for the matrix used in PCA
def pcaEigvals(data):
    centered_matrix = data - data.mean(axis=1)[:, np.newaxis]
    cov = np.dot(centered_matrix, centered_matrix.T)
    eigenValues, eigenVectors = np.linalg.eig(cov)

    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    eigenValues = eigenValues[::-1]
    eigenVectors = eigenVectors[::-1]
    return(eigenValues, eigenVectors)

# pcaEigvalRatio() takes in the raw eigenvalue vector and spits out the ratio of eigenvalue to total, so eigenvalue/sum of eigenvalues
def pcaEigvalRatio(eigenvalueArray):
    tot = np.sum(eigenvalueArray)
    for i in range(len(eigenvalueArray)):
        eigenvalueArray[i]=eigenvalueArray[i]/tot
        
    return(eigenvalueArray)

################################################################## Start of Fourier Stuff ########################################################################################

# fourierDenoising() takes in a signal array and a thehold value you want to amplitude cut at and returns the denoised signal array
def fourierDenoising(signal, threshold):
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