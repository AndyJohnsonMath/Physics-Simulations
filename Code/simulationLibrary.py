import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from tqdm import tqdm
import matplotlib as mpl
import sys
from PyPDF2 import PdfMerger
import re

################################################ Quality of Life/Miscellanious #############################################################################################################################################

def testFunction():
    print("Yup, im here!")

def clearDirectory(direc='../Images for simulation'):
    """
    Description
    -----------
    Deletes the contents of a target directory.

    Parameters
    ----------
    direc : string
       string containing the path to your target directory. Default is the essential 'Images for Simulation' library since thats what we use the most, but you can put in whatever you want.

    Returns
    -------
    Nothing. But! This function does delete the contents of another folder.
    """
    dir = direc
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def combinePDFs(directory = r'../Images for simulation', filename="combined.pdf"):
    """
    Description
    -----------
    Combines all of the pdf's in a target directory into a single pdf ordered by number.

    Parameters
    ----------
    directory : string
        The directory containing the pdf's to be merged. Default argument is a file that comes along with the github.
    filename : string
        Name of the combined pdf file.

    Returns
    -------
    Nothing. But! This function does produce the combined pdf inside of the target directory.
    """
    # Create a PdfMerger object
    merger = PdfMerger()
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    # Sort files by the numerical value in their names
    def extract_number(filename):
        match = re.search(r'\d+', filename)  # Find the first number in the filename
        return int(match.group()) if match else 0  # Return 0 if no number is found
    
    pdf_files.sort(key=lambda x: extract_number(x))
    
    # Merge each PDF file
    for pdf in pdf_files:
        pdf_path = os.path.join(directory, pdf)
        print(f"Adding {pdf}...")
        merger.append(pdf_path)
    
    # Write the combined PDF to a file
    output_path = os.path.join(directory, filename)
    merger.write(output_path)
    merger.close()
    
    print(f"All PDFs combined into {output_path}")

################################################ Numerical Methods #########################################################################################################

def eulersMethod(function, stepSize, initialPair, intervalLength):
    """
    Description
    -----------
    eulersMethod(): Generalized Function that takes in the stepsize, initial conditions and said function from previous cell and returns
    a 2d array where the first entry is the array of x-values and the second entry is the array of approximated y-values.

    Parameters
    ----------
    function : function (?, not sure if this is a valid data type, but this parameter is a function)
        Numerical array for the function on the right side of the first order ODE set up for this method (Look it up, itll make more sense).
    stepSize : float
        Parameter defining the step size of the algorithm.
    initialPair : array-like, shape (1,2)
        Starting point for the iterative scheme.
    intervalLength : float
        Length of the interval from the initialPair[] array to the right. 

    Returns
    -------
    solution : array-like, shape (2,len(function))
        Approximated solution array found using Euler's method. Each element is a coordinate pair, thus the 2D array.
    """
    y0=initialPair[1]
    x0=initialPair[0]
    deltat=stepSize

    lasty = y0
    lastx = x0

    solutionx = np.array([x0])
    solutiony = np.array([y0])
    iterations = math.floor(intervalLength/deltat)
    for i in range(iterations):
        nexty = lasty+(deltat*(function(lastx,lasty)))
        nextx = lastx+deltat
        solutionx = np.append(solutionx, [nextx],axis=0)
        solutiony = np.append(solutiony, [nexty],axis=0)
        
        lasty = nexty
        lastx += deltat
        
    solution = np.array([solutionx, solutiony])
    return(solution)


def rungeKutta(function, stepSize, initialPair, intervalLength):
    """
    Description
    -----------
    rungeKutta(): Function set up exactly the same as eulersMethod(), just runs the Runge-Kutta algorithm instead

    Parameters
    ----------
    function : function (?, not sure if this is a valid data type, but this parameter is a function)
        string containing the path to your target directory. Default is the essential 'Images for Simulation' library since thats what we use the most, but you can put in whatever you want.
    stepSize : float
        Parameter defining the step size of the algorithm.
    initialPair : array-like, shape (1,2)
        Starting point for the iterative scheme.
    intervalLength : float
        Length of the interval from the initialPair[] array to the right. 

    Returns
    -------
    solution : array-like, shape (2,len(function))
        Approximated solution array found using the Runge-Kutta method. Each element is a coordinate pair, thus the 2D array.
    """
    y0=initialPair[1]
    x0=initialPair[0]
    
    lasty = y0
    lastx = x0
    
    solutionx = np.array([x0])
    solutiony = np.array([y0])
    iterations = math.floor(intervalLength/stepSize)
    for i in range(iterations):
        k1 = function(lastx,lasty)
        k2 = function(lastx+(stepSize/2), lasty+(stepSize*(k1/2)))
        k3 = function(lastx+(stepSize/2), lasty+(stepSize*(k2/2)))
        k4 = function(lastx+stepSize, lasty+(stepSize*k3))
        
        nexty = lasty+((stepSize/6)*(k1+2*k2+2*k3+k4))
        nextx = lastx+stepSize
        solutionx = np.append(solutionx, [nextx],axis=0)
        solutiony = np.append(solutiony, [nexty],axis=0)
        
        lasty = nexty
        lastx += stepSize
        
    solution = np.array([solutionx, solutiony])
    return(solution)


def rkf45(function, initialStepSize, initialPair, intervalLength, minStepSize, maxStepSize, TOL):
    """
    Description
    -----------
    rkf45(): This function runs the Runge-Kutta-Fehlberg fourth-order-fifth-order scheme.
    You might notice this function takes significantly more function arguments than the previous function, that is just a result of this function being more intense in general.
    Takes in a function f(x,y), initial step size you want to work with (if the step size is too big this algorithm will correct it), initialPair which is a numpy array with the initial values for the IVP.
    minStepSize and maxStepSize are the bounds for the variable step size and TOL is the user-defined tolerance measuring the maximum amount of error allowed in solving. In testing, TOL=5*10^-7.

    Parameters
    ----------
    function : function (?, not sure if this is a valid data type, but this parameter is a function)
       string containing the path to your target directory. Default is the essential 'Images for Simulation' library since thats what we use the most, but you can put in whatever you want.
    initialStepSize : float
        Parameter defining the initial step size of the algorithm.
    initialPair : array-like, shape (1,2)
        Starting point for the iterative scheme.
    intervalLength : float
        Length of the interval from the initialPair[] array to the right.
    minStepSize : float
        Minimum stepsize the algorithm will use if needed.
    maxStepSize : float
        Maximum stepsize the algorithm will use if needed.
    TOL : float
        Tolerance measuring the maximum amount of error allowed in solving.

    Returns
    -------
    solution : array-like, shape (2,len(function))
        Approximated solution array found using the Runge-Kutta method. Each element is a coordinate pair, thus the 2D array
    """
    #Classic initialization, since this method has a variable step-size it makes more sense to iterate over the length of the interval rather than the number of steps
    totalLengthComputed = 0
    stepSize = initialStepSize
    
    y0=initialPair[1]
    x0=initialPair[0]
    
    lasty = y0
    lastx = x0
    
    solutionx = np.array([x0])
    solutiony = np.array([y0])
    
    while totalLengthComputed <= intervalLength:        
        # Compute coefficients
        k1 = stepSize*function(lastx, lasty)
        k2 = stepSize*function(lastx+(1/4)*stepSize, lasty+(1/4)*k1)
        k3 = stepSize*function(lastx+(3/8)*stepSize, lasty+(3/32)*k1+(9/32)*k2)
        k4 = stepSize*function(lastx+(12/13)*stepSize, lasty+(1932/2197)*k1-(7200/2197)*k2+(7296/2197)*k3)
        k5 = stepSize*function(lastx+stepSize, lasty+(439/216)*k1-8*k2+(3680/513)*k3-(845/4104)*k4)
        k6 = stepSize*function(lastx+(1/2)*stepSize, lasty-(8/27)*k1+2*k2-(3544/2565)*k3+(1859/4104)*k4-(11/40)*k5)
        
        # Calculate actual values
        nexty = lasty+(25/216)*k1+(1408/2565)*k3+(2197/4101)*k4-(1/5)*k5
        # Dont believe the line under this is necessary, but im gonna keep it around just in case
        # predictedy = lasty+(16/135)*k1+(6656/12825)*k3+(28561/56430)*k4-(9/50)*k5+(2/55)*k6
        truncatedError = ((1/360)*k1-(128/4275)*k3-(2197/75240)*k4+(1/50)*k5+(2/55)*k6)/stepSize
        q = (TOL/(2*np.absolute(truncatedError)))**(1/4)
        
        # This checks if the local error is larger than the tolerance set by the user. If it is, it calculates the new step size using the step size adjustment value, q
        # After new stepSize is defined, jump ahead to the next iteration without calculating anything else.
        # Once the stepSize is good enough then the rest of the calculations will be ran
        if truncatedError > TOL:
            stepSize = stepSize*q
            continue
        
        # Run stepSize checks
        stepSize = stepSize*q
        if stepSize >= maxStepSize:
            stepSize = maxStepSize
        if stepSize <= minStepSize:
            stepSize = minStepSize
        else:
            stepSize = stepSize
            
        nextx = lastx+stepSize
        
        solutionx = np.append(solutionx, [nextx],axis=0)
        solutiony = np.append(solutiony, [nexty],axis=0)
        
        lasty = nexty
        lastx += stepSize
        totalLengthComputed = lastx
        
    solution = np.array([solutionx, solutiony])
    return(solution)

###################################################### gravity stuff #######################################################################################################################################################################


# PointMassBody() is the class for point masses
# position, velocity and acceleration should all be 1x2 arrays
class PointMassBody:
    """
    Description
    -----------
    Class defining gravitational point masses.

    Parameters
    ----------
    mass : float
        Mass of object in kg.
    position: array-like, shape (1,2)
        2D velocity array with x and y components.
    acceleration: array-like, shape (1,2)
        2D acceleration array with x and y components.
    """
    def __init__(self,mass,position,velocity,acceleration):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

def randParticle(massMean=1*10**12, massStdDev=100000000000, posMean = 0, posStdDev = 15, velMean = 0, velStdDev = 0.2, type='gravity'):
    """
    Description
    -----------
    randParticleGravity() just generates a random particle with Gaussian-ly distributed mass, position and velocity.

    Parameters
    ----------
    massMean : float
        mean of the randomly decided mass. Default argument arbitrarily chosen tailored for gravity simulations.
    massStdDev : float
        standard deviation mass of the randomly decided mass. Default argument arbitrarily chosen tailored for gravity simulations.
    posMean : float
        mean of the randomly decided position. Default argument arbitrarily chosen tailored for gravity simulations.
    posStdDev : float
        standard deviation mass of the randomly decided position. Default argument arbitrarily chosen tailored for gravity simulations.
    velMean : float
        mean of the randomly decided velocity. Default argument arbitrarily chosen tailored for gravity simulations.
    velStdDev : float
        standard deviation mass of the randomly decided velocity. Default argument arbitrarily chosen tailored for gravity simulations.
    tpye : string
       Parameter that determines the kind of particle generated.
       Current support for:
           -'gravity': Creates an object of the PointMassBody() class.
           -default : Creates an object of the Particle() class. 


    Returns
    -------
    particle : PointMassBody() object, dtype='object'
        Returns a PointMassBody() object with randomly decided mass, position and velocity.
    """
    #Declare randomly assigned mass
    mass = np.random.normal(loc=massMean,scale=massStdDev)
    
    #Declare randomly assigned positions and make sure it doesnt go out of bounds
    posx = np.random.normal(loc=posMean, scale=posStdDev)
    posy = np.random.normal(loc=posMean, scale=posStdDev)
    if posx < -10 or posx > 10:
        posx = 80
    if posy >= 80:
        posy = 80
    position = np.array([posx,posy])
    
    #Declare randomly assigned positions and make sure it doesnt go out of bounds
    velx = np.random.normal(loc=velMean, scale=velStdDev)
    vely = np.random.normal(loc=velMean, scale=velStdDev)
    velocity = np.array([velx,vely])

    #Generate and return the particle
    if type == 'gravity':
        particle = PointMassBody(mass,position,velocity,np.array([0,0]))
    else:
        particle = Particle(mass,position,velocity,np.array([0,0]))
    return(particle)

# generateParticles() takes in an integer argument and generates an array of random particles generated from randParticleGravity()
def generateParticles(num):
    """
    Description
    -----------
    generateParticles() takes in an integer argument and generates an array of random particles generated from randParticleGravity().

    Parameters
    ----------
    num : int
       Parameter defining the number of particles generated.

    Returns
    -------
    pointMassArray : array-like, shape (1,), dtype='object'
        An array of PointMassBody() objects.
    """
    pointMassArray=np.zeros(num,dtype='object')
    for i in range(len(pointMassArray)):
        pointMassArray[i]=randParticle(type='gravity')
    return(pointMassArray)

def calculateGravity(obj1,obj2):
    """
    Description
    -----------
    calculateGravity() calculates the gravitational force vector between two objects from the PointMassBody() class.
    Takes in two objects, returns a 1x2 force vector.

    Parameters
    ----------
    obj1, obj2 : PointMassBody() objects
       The two gravitational bodies that we want to calculate the force between.

    Returns
    -------
    F12 : array-like, shape(1,2)
       Returns a 2D force vector with x and y components.
    """
    G = 6.674*pow(10,-11)
    F12Hat = (obj2.position-obj1.position)/np.linalg.norm(obj2.position-obj1.position)
    F12 = ((G*obj1.mass*obj2.mass)/pow(np.linalg.norm(obj2.position-obj1.position),2))*F12Hat
    return(F12)

def update(objects,dt=1/30):
    """
    Description
    -----------
    update() takes in a list of PointMassBody() objects, along with a time interval, and updates the gravitational system by one step. Does not return anything.
    Big problem with this function is that its brute force and is the least computationally efficient way for doing this, but hey its a start.

    Parameters
    ----------
    objects : array-like, shape (1,)
       Numpy array of PointMassBody() class objects.
    dt : float
       Parameters defining the update step size.

    Returns
    -------
    Nothing. But! This function does update the member variables of the given array of objects. This is useful for running multiple updates. Specifically useful in the next function, gravitySimulation().
    """
    # Initialize a force matrix, populate it with the forces of the corresponding index. So, at poition (i,j) is Fij
    forceMatrix=np.zeros((len(objects),len(objects),2))
    for j in range(len(objects)):
        for i in range(len(objects)):
            if i == j:
                forceMatrix[i][j]=0 #Force between an object and itself is zero, so Fij = 0 iff i == j
            else:
                forceMatrix[i][j]=calculateGravity(objects[i],objects[j])
    
    # Initialize these arrays
    totForceArray = np.zeros((len(objects),2))
    acceleration = np.zeros((len(objects),2))
    # Iterate over every object
    for i in range(len(objects)):
        # Calculate total forces so that we can then find net acceleration for a single object
        totForceArray[i]=forceMatrix[i].sum(axis=0)
        acceleration[i]=(1/objects[i].mass)*totForceArray[i]
        
        #Actually update the object information
        objects[i].acceleration = acceleration[i]
        objects[i].velocity = objects[i].velocity+(objects[i].acceleration*dt)
        objects[i].position = objects[i].position+(objects[i].velocity*dt)

def gravitySimulation(kind='random', numParticles = 10, numFrames=500, bod=np.array([0],dtype='object'), clean=True):
    """
    Description
    -----------
    Runs an entire gravity simulation by generating frames and saving them to the essential 'Images for Simulation' folder. 

    Parameters
    ----------
    kind : string
       Parameter that determines the kind of gravity simultion.
       Current support for:
           -'random': A random selection of point masses selected Gaussianly. Default Value
           -'polygonal': symmetrically distributed identical point masses. Must be hardcoded in
           -'custom': Input a custom 'bodies' object array under the 'bod' argument.
    numParticles : int
        Number of particles in the simulation. Default is 10 point mass bodies.
    bod : Array-like, shape (1,)
        Numpy array of data type 'object' that has PointMassBody objects as each entry. Default is np.array([0],dtype='object') just as a dummy object meant to be written over.
    numframes : int
       Length of simulation. Default is 500.
    clean : Boolean True or False
        True plots the bodies with no gridlines or axes, false plots the bodies with gridlines and axes.

    Returns
    -------
    Nothing. But! This function does fill your 'Images for Simulation' folder that is essential for this function to work.
    Its these images that you run the ffmpeg command on
    """
    #The path below should be the path that YOU are saving every frame to. I didnt want to provide my personal one, so unfortunately this is the one thing you will have to do yourself
    clearDirectory()
    
    # This block checks the kind of simulation you want to do
    # Ultimately these conditional statements just determine what the 'bodies' variable is set to. Then 'bodies' gets passed to the update function
    if kind == 'custom':
        bodies = bod
    elif kind=='polygonal':
        bodies = np.zeros(numParticles,dtype='object')
        for i in range(numParticles):
            bodies[i]=PointMassBody(1*10**12.5,np.array([0,0]),np.array([0,0]),np.array([0,0]))
            bodies[i].position=np.array([6*np.cos((2*np.pi/numParticles)*i),6*np.sin((2*np.pi/numParticles)*i)])
            bodies[i].velocity=np.array([6*(-1)*np.sin((2*np.pi/numParticles)*i),6*np.cos((2*np.pi/numParticles)*i)])
    else:
        bodies = generateParticles(numParticles)

    # Calculate the force between each every body and every other body
    #Start the main loop
    for i in range(numFrames):
        figure, axes = plt.subplots()
        update(bodies)

        for j in range(len(bodies)):
            axes.scatter(bodies[j].position[0], bodies[j].position[1], s=1)

        if clean == True:
            plt.grid(None)
            plt.axis('off')
        else:
            pass
        axes.set_aspect('auto')
        plt.xlim(-200,1500)
        plt.ylim(-700,700)

        # figure.savefig('./Images for simulation/graph'+str(i)+'.png', bbox_inches='tight', pad_inches=0 ,dpi=300)
        figure.savefig('./Images for simulation/graph'+str(i)+'.png', bbox_inches='tight', pad_inches=0)
        plt.close('all')


###################################################### Attractor Stuff ################################################################################################################################################

# Not my own code, got it from DrM at https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html     
def lorenz(xyz, s=10, r=28, b=2.667):
    """
    Description
    -----------
    Computes the next step in the Lorenz attractor system.

    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns 
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def thomas(xyz, b=0.208186):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    b : float
       Parameters defining the Thomas' attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Thomas' attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = np.sin(y)-(b*x)
    y_dot = np.sin(z)-(b*y)
    z_dot = np.sin(x)-(b*z)
    return np.array([x_dot, y_dot, z_dot])

def rossler(xyz, a=0.2, b=0.2, c=5.7):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    a, b, c : float
       Parameters defining the Rossler attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Rossler attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = -y-z
    y_dot = x+(a*y)
    z_dot = b+(z*(x-c))
    return np.array([x_dot, y_dot, z_dot])

def lorenzAttractorImage(length, s=10, r=28, b=2.667, save=False, clean=False, initPos=np.array([0,1,1.05])):
    """
    Parameters
    ----------
    length : int
       Number of iterations (Segments) you want to have the path length be
    s,r,b : float
       Parameters defining the Lorenz attractor
    save : boolean
        If true, image will be saved to 'Imeages for simulation' directory
    clean : boolean
        If true then all other graph elements will be removed except for what is just being plotted, makes it very clean
    initPos : arra-like, shape (1,3)
        Position vector describing the initial position of the system. 

    Returns
    -------
    Nothing! But this will produce an single image of the attractor, mainly used for seeing if the attractor can be reproduced at all.
    """
    dt = 0.01
    num_steps = length

    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = initPos  # Set initial values
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i],s,r,b) * dt

    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.text2D(0.05, 0.95, r"$\sigma$="+str("{:.2f}".format(s)), transform=ax.transAxes)
    ax.text2D(0.05, 1.0, r"$\rho$="+str("{:.2f}".format(r)), transform=ax.transAxes)
    ax.text2D(0.05, 1.05, r"$\beta$="+str("{:.2f}".format(b)), transform=ax.transAxes)
    ax.text2D(0.75, 0.95, r"$x$="+str("{:.3f}".format(xyzs[0][0])), transform=ax.transAxes)
    ax.text2D(0.75, 1.0, r"$y$="+str("{:.2f}".format(xyzs[0][1])), transform=ax.transAxes)
    ax.text2D(0.75, 1.05, r"$z$="+str("{:.2f}".format(xyzs[0][2])), transform=ax.transAxes)
    plt.xlim((-25,25))
    plt.ylim((-30,35))
    if clean == True:
        ax.grid(None)
        ax.axis('off')
    else:
        ax.set_title("Lorenz Attractor")
        pass
    
    if save == True:
        plt.savefig('../Images for simulation/graph'+str(j)+'.png', dpi=300)
        plt.close('all')
    else:
        plt.show()

def thomasAttractorImage(length, b=0.208186, save=False, clean=False, initPos=np.array([0,0,0])):
    """
    Parameters
    ----------
    length : int
       Number of iterations (Segments) you want to have the path length be
    b : float
       Parameters defining the Lorenz attractor
    save : boolean
        If true, image will be saved to 'Imeages for simulation' directory
    clean : boolean
        If true then all other graph elements will be removed except for what is just being plotted, makes it very clean
    initPos : arra-like, shape (1,3)
        Position vector describing the initial position of the system. 

    Returns
    -------
    Nothing! But this will produce an single image of the attractor, mainly used for seeing if the attractor can be reproduced at all.
    """
    dt = 0.01
    num_steps = length

    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = initPos  # Set initial values
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + thomas(xyzs[i],b) * dt

    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    # ax.text2D(0.05, 0.95, r"$\sigma$="+str("{:.2f}".format(s)), transform=ax.transAxes)
    # ax.text2D(0.05, 1.0, r"$\rho$="+str("{:.2f}".format(r)), transform=ax.transAxes)
    # ax.text2D(0.05, 1.05, r"$\beta$="+str("{:.2f}".format(b)), transform=ax.transAxes)
    # ax.text2D(0.75, 0.95, r"$x$="+str("{:.3f}".format(xyzs[0][0])), transform=ax.transAxes)
    # ax.text2D(0.75, 1.0, r"$y$="+str("{:.2f}".format(xyzs[0][1])), transform=ax.transAxes)
    # ax.text2D(0.75, 1.05, r"$z$="+str("{:.2f}".format(xyzs[0][2])), transform=ax.transAxes)
    plt.xlim((-5,5))
    plt.ylim((-5,5))
    # plt.zlim((-5,5))
    if clean == True:
        ax.grid(None)
        ax.axis('off')
    else:
        ax.set_title("Thomas Attractor")
        pass
    
    if save == True:
        plt.savefig('../Images for simulation/graph.png', dpi=300)
        plt.close('all')
    else:
        plt.show()

def rosslerAttractorImage(length, a=0.2, b=0.2, c=5.7, save=False, clean=False, initPos=np.array([0,0,0])):
    """
    Parameters
    ----------
    length : int
       Number of iterations (Frames) you want the animation to be
    a,b,c : float
       Parameters defining the Rossler attractor
    save : boolean
        If true, image will be saved to 'Imeages for simulation' directory
    clean : boolean
        If true then all other graph elements will be removed except for what is just being plotted, makes it very clean
    initPos : arra-like, shape (1,3)
        Position vector describing the initial position of the system. 

    Returns
    -------
    Nothing! But this will produce an single image of the attractor, mainly used for seeing if the attractor can be reproduced at all.
    """
    dt = 0.01
    num_steps = length

    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = initPos  # Set initial values
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + rossler(xyzs[i],b) * dt

    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    # ax.text2D(0.05, 0.95, r"$\sigma$="+str("{:.2f}".format(s)), transform=ax.transAxes)
    plt.xlim((-12,12))
    plt.ylim((-12,12))
    # plt.zlim((-5,5))
    if clean == True:
        ax.grid(None)
        ax.axis('off')
    else:
        ax.set_title("Thomas Attractor")
        pass
    
    if save == True:
        plt.savefig('../Images for simulation/graph.png', dpi=300)
        plt.close('all')
    else:
        plt.show()

# lorenzAttractorTrace() takes in the amount of frames you want to video to be along with the system paramters s, r and b. Default value is just a known value that gives a known result to use as sanity checks
# Will save images to target directory where you will then have to run ffmpeg through the command line to use. Ffmpeg comand is given in the next line
# ffmpeg -start_number 0 -framerate 60 -i graph%01d.png video.webm

def lorenzAttractorTrace(frames, s=10, r=28, b=2.667, clean=False, rotation=False,initPos=np.array([0,1,1.05])):
    """
    Parameters
    ----------
    frames : int
       Number of iterations (Frames) you want the animation to be
    s,r,b : float
       Parameters defining the Lorenz attractor
    clean : boolean
        If true then all other graph elements will be removed except for what is just being plotted, makes it very clean
    rotation : boolean
        If true, the produced animation will rotate about the origin. Good for artistic reasons
    initPos : arra-like, shape (1,3)
        Position vector describing the initial position of the system. 

    Returns
    -------
    Nothing! But this will produce all the frames for an animation. Use the ffmpeg command on 'Images for simulation" to finish the animation production
    """
    #Empty the target directory
    clearDirectory()

    #Calculate the array of points according to the lorenz system
    #Do this outside the main loop so that we only calculate it once rather than a bazillion times and annihilate memory
    dt = 0.01
    numSteps = frames

    xyzs = np.empty((numSteps+1, 3))  # Need one more for the initial values
    xyzs[0] = initPos  # Set initial values
    for i in range(numSteps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i],s,r,b) * dt

    # Checking if the attractor is clean or not to determine what the first frame should look like     
    if clean == True:
        #plot the first frame outside of the main loop, same idea as initial conditions just with a frame
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(*xyzs[0].T, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.grid(None)
        ax.axis('off')
        plt.savefig('../Images for simulation/graph.png'+str(0)+'.png')
        plt.close('all')
    else:
        #plot the first frame outside of the main loop, same idea as initial conditions just with a frame
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(*xyzs[0].T, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        plt.savefig('../Images for simulation/graph.png'+str(0)+'.png')
        plt.close('all')
    
    #Non-rotation video
    if rotation == False:
        #Initialize frame to 1 so that our indexing for xyzs in the main loop prints from 0-frame. If frame was 0 then we would be plotting xysz from xyzs[0] ot xyzs[0] which we cant do. We need atleast xyzs[0] to xyzs[1]
        frame = 1
        while frame < numSteps:
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(*xyzs[:frame].T, lw=0.5) #Recall this [:frame] notion means we plot the array from xyzs[0] to xyzs[frame]
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            plt.xlim((-25,25))
            plt.ylim((-30,35))
            ax.set_zlim(0,60)
            if clean == True:
                ax.grid(None)
                ax.axis('off')
            else:
                ax.set_title("Lorenz Attractor")
                pass
            plt.savefig('../../Physics-Simulations/Images for simulation/graph.png'+str(frame)+'.png', dpi=300) # dpi argument increases resolution
            plt.close('all')
            frame = frame + 1
    # To rotate video, add in the ax.view_init() function which takes in spherical coordinate
    else:
        #Initialize frame to 1 so that our indexing for xyzs in the main loop prints from 0-frame. If frame was 0 then we would be plotting xysz from xyzs[0] ot xyzs[0] which we cant do. We need atleast xyzs[0] to xyzs[1]
        frame = 1
        angle = 0
        while frame < numSteps:
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(*xyzs[:frame].T, lw=0.5) #Recall this [:frame] notion means we plot the array from xyzs[0] to xyzs[frame]
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            plt.xlim((-25,25))
            plt.ylim((-30,35))
            ax.set_zlim(0,60)
            ax.view_init(30,angle)
            if clean == True:
                ax.grid(None)
                ax.axis('off')
            else:
                ax.set_title("Lorenz Attractor")
                pass
            plt.savefig('../..//Physics-Simulations/Images for simulation/graph.png'+str(frame)+'.png', dpi=300) # dpi argument increases resolution
            plt.close('all')
            frame = frame + 1
            angle = angle + 1

def rosslerAttractorTrace(frames, a=0.2, b=0.2, c=10, clean=False, rotation=False,initPos=np.array([0,0,0])):
    """
    Parameters
    ----------
    frames : int
       Number of iterations (Frames) you want the animation to be
    a,b,c : float
       Parameters defining the Rossler attractor
    clean : boolean
        If true then all other graph elements will be removed except for what is just being plotted, makes it very clean
    rotation : boolean
        If true, the produced animation will rotate about the origin. Good for artistic reasons
    initPos : arra-like, shape (1,3)
        Position vector describing the initial position of the system. 

    Returns
    -------
    Nothing! But this will produce all the frames for an animation. Use the ffmpeg command on 'Images for simulation" to finish the animation production
    """
    #Empty the target directory
    clearDirectory()

    #Enable latex
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

    #Calculate the array of points according to the lorenz system
    #Do this outside the main loop so that we only calculate it once rather than a bazillion times and annihilate memory
    dt = 0.01
    numSteps = 25*frames

    xyzs = np.empty((numSteps+1, 3))  # Need one more for the initial values
    xyzs[0] = initPos  # Set initial values
    for i in range(numSteps):
        xyzs[i + 1] = xyzs[i] + rossler(xyzs[i]) * dt

    # Checking if the attractor is clean or not to determine what the first frame should look like     
    if clean == True:
        #plot the first frame outside of the main loop, same idea as initial conditions just with a frame
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(*xyzs[0].T, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.grid(None)
        ax.axis('off')
        plt.savefig('../Images for simulation/graph'+str(0)+'.png')
        plt.close('all')
    else:
        #plot the first frame outside of the main loop, same idea as initial conditions just with a frame
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(*xyzs[0].T, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Rossler Attractor")
        plt.savefig('../Images for simulation/graph'+str(0)+'.png')
        plt.close('all')
    
    #Non-rotation video
    if rotation == False:
        #Initialize frame to 1 so that our indexing for xyzs in the main loop prints from 0-frame. If frame was 0 then we would be plotting xysz from xyzs[0] ot xyzs[0] which we cant do. We need atleast xyzs[0] to xyzs[1]
        frame = 1
        while frame < numSteps/25:
            ax = plt.figure(figsize=(8,6)).add_subplot(projection='3d')
            ax.plot(*xyzs[:(frame*25)].T, lw=0.5) #Recall this [:frame] notion means we plot the array from xyzs[0] to xyzs[frame]
            ax.text2D(0.3,1, "Rossler Attractor", transform=ax.transAxes)
            ax.text2D(0,0.85, r"$\frac{d}{dx} \begin{bmatrix} x \\ y \\ z \end{bmatrix}=\begin{bmatrix} 0 & -1 & -1 \\  1 & \alpha & 0 \\ z & 0 & -\gamma \end{bmatrix}\begin{bmatrix} x \\ y \\z \end{bmatrix}+\begin{bmatrix}0 \\ 0 \\ \beta \end{bmatrix}$", transform=ax.transAxes)
            ax.text2D(-0.2, 0.65, "Paramaters:", transform=ax.transAxes)
            ax.text2D(-0.2, 0.55, r"$a$="+str("{:.2f}".format(a)), transform=ax.transAxes)
            ax.text2D(-0.2, 0.45, r"$b$="+str("{:.2f}".format(b)), transform=ax.transAxes)
            ax.text2D(-0.2, 0.35, r"$c$="+str("{:.2f}".format(c)), transform=ax.transAxes)
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            plt.xlim((-12,12))
            plt.ylim((-12,12))
            ax.set_zlim(0,50)
            if clean == True:
                ax.grid(None)
                ax.axis('off')
            else:
                ax.set_title("Rossler Attractor")
                pass
            plt.savefig('../Images for simulation/graph'+str(frame)+'.png', dpi=300) # dpi argument increases resolution
            plt.close('all')
            frame = frame + 1
    #Rotation video, add in the ax.view_init() function which takes in spherical coordinate
    else:
        #Initialize frame to 1 so that our indexing for xyzs in the main loop prints from 0-frame. If frame was 0 then we would be plotting xysz from xyzs[0] ot xyzs[0] which we cant do. We need atleast xyzs[0] to xyzs[1]
        frame = 1
        angle = 0
        while frame < numSteps/25:
            ax = plt.figure(figsize=(8,6)).add_subplot(projection='3d')
            ax.plot(*xyzs[:(frame*25)].T, lw=0.5) #Recall this [:frame] notion means we plot the array from xyzs[0] to xyzs[frame]
            ax.text2D(0.3,1, "Rossler Attractor", transform=ax.transAxes)
            ax.text2D(0,0.85, r"$\frac{d}{dx} \begin{bmatrix} x \\ y \\ z \end{bmatrix}=\begin{bmatrix} 0 & -1 & -1 \\  1 & \alpha & 0 \\ z & 0 & -\gamma \end{bmatrix}\begin{bmatrix} x \\ y \\z \end{bmatrix}+\begin{bmatrix}0 \\ 0 \\ \beta \end{bmatrix}$", transform=ax.transAxes)
            ax.text2D(-0.2, 0.65, "Paramaters:", transform=ax.transAxes)
            ax.text2D(-0.2, 0.55, r"$a$="+str("{:.2f}".format(a)), transform=ax.transAxes)
            ax.text2D(-0.2, 0.45, r"$b$="+str("{:.2f}".format(b)), transform=ax.transAxes)
            ax.text2D(-0.2, 0.35, r"$c$="+str("{:.2f}".format(c)), transform=ax.transAxes)
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            plt.xlim((-12,12))
            plt.ylim((-12,12))
            ax.set_zlim(0,50)
            ax.view_init(30,angle)
            if clean == True:
                ax.grid(None)
                ax.axis('off')
            else:
                ax.set_title("Rossler Attractor")
                pass
            plt.savefig('../Images for simulation/graph'+str(frame)+'.png', dpi=300) # dpi argument increases resolution
            plt.close('all')
            frame = frame + 1
            angle = angle + 1

######################################### This marks the start of the collision detection stuff #########################################################################################################################################################################################################

class Particle:
    """
    Description
    -----------
    A class defining a particle to be used in collision detection.

    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns 
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    def __init__(self, mass, position, velocity, acceleration, box):
        
        self.mass = mass
        self.radius = math.sqrt(self.mass/(math.pi*1.5))
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.KE = (1/2)*self.mass*np.dot(self.velocity,self.velocity)

    # Use properties for the particle boundaries
    @property
    def left(self):
        return self.position[0] - self.radius
    
    @property
    def right(self):
        return self.position[0] + self.radius
    
    @property
    def top(self):
        return self.position[1] + self.radius
    
    @property
    def bottom(self):
        return self.position[1] - self.radius
    
    #Method that detects if there is a colision between the particle and the box
    def handleBoxCollision(self, box):
        #cor is the coefficient of restitution
        cor = 0.9
        if self.left <= box.left or self.right >= box.right:
            self.velocity[0]=-self.velocity[0]
        
        if self.bottom <= box.bottom or self.top >= box.top:
            self.velocity[1]=-self.velocity[1]
    
        #These statements make sure that the ball is put back into the box, rather than letting it clip though
        if self.top >= box.top:
            self.position[1] = box.top - self.radius
        
        if self.bottom <= box.bottom:
            self.position[1] = box.bottom + self.radius
        
        if self.left <= box.left:
            self.position[0] = box.left + self.radius
        
        if self.right >= box.right:
            self.position[0] = box.right - self.radius

############################################################### Collision Detection stuff ######################################################################################################################################################
        
#just defining the box tha5t the simulation takes place in
class Box:
    def __init__(self):
        self.left = -10
        self.right = 10
        self.bottom = -10
        self.top = 10

def handleParticleCollision(pa,pb):
    if math.dist(pa.position,pb.position) <= (pa.radius+pb.radius):
        P1 = pa.position
        P2 = pb.position
        v1 = pa.velocity
        v2 = pb.velocity
        m1 = pa.mass
        m2 = pb.mass

        r1=pa.radius
        r2=pb.radius

        a = np.linalg.norm(v1-v2)**2
        b = 2*np.inner(v1-v2,P1-P2)
        c= np.linalg.norm(P1-P2)**2-(r1+r2)**2

        # calculating  the discriminant
        dis = (b**2) - (4 * a*c)
        
        # The first result is when they make contact
        ans1 = (-b-math.sqrt(dis))/(2 * a)
        
        temp3 = pa.position+pa.velocity*ans1
        temp4 = pb.position+pb.velocity*ans1 
        
        temp1 = v1-((2*m2)/(m1+m2))*(np.inner(v1-v2,P1-P2)/np.linalg.norm(P1-P2)**2)*(P1-P2)
        temp2 = v2-((2*m1)/(m1+m2))*(np.inner(v2-v1,P2-P1)/np.linalg.norm(P2-P1)**2)*(P2-P1)
        pa.velocity = temp1
        pb.velocity = temp2
        pa.position = temp3
        pb.position = temp4
        
# #Since this is a discreet colission simulation this function is for updating the state of the simulation
# def updateCollisionBruteDorce(dt):
#     p1.velocity = p1.velocity+(p1.acceleration*dt)
#     p1.position = p1.position+(p1.velocity*dt)
#     p1.KE = (1/2)*p1.mass*np.dot(p1.velocity,p1.velocity)
    
#     p2.velocity = p2.velocity+(p2.acceleration*dt)
#     p2.position = p2.position+(p2.velocity*dt)
#     p2.KE = (1/2)*p2.mass*np.dot(p2.velocity,p2.velocity)
    
#     p1.handleBoxCollision(box)
#     p2.handleBoxCollision(box)
    
#     handleParticleCollision(p1,p2)

######################################################################## Start of wave equation stuff #################################################################################################

def waveSimulation(initialFunc, c=1, length=1, totalTime=5, Nx=200, Nt=1000):
    """
    Description
    -----------
    A function for numerically solving the 1D wave equation and plotting each frame of its simulation.

    Parameters
    ----------
    initialFunc : Function
        Function defining the first initial condition.
    c : Float
       Wave eq parameter.
    length : Float
        Length of the spatial dimension
    totalTime : Float
        Total time of the simulation
    Nx : int
        Total number of discrete points in the spatial dimension
    Nt : int
        Total number of discrete points in the temporal dimention
    
    Returns 
    -------
    Nothing. But! Creates all the frames of the simulation in the titular "Images for Simulation" file to then be used with ffmpeg (ffmpeg -start_number 0 -framerate 60 -i graph%01d.png video.webm)
    """
    # Lets create a meshed function
    length = length
    c=c
    totalTime = totalTime
    Nx = Nx # x axis
    Nt = Nt # t axis
    deltax = length/(Nx-1) # THIS IS THE IMPORTANT LINE OR ELSE IT BLOWS UP!!!!!!!!!!!!!!!!!!!! I HAVE NO CLUE WHY THAT -1 NEEDS TO BE THERE BUT IT DOES OR EVERYTHING GOES TO SHIT
    deltat = totalTime/Nt
    C=c*(deltat/deltax)
    if C >=1 :
        print("C is greater than 1. C="+str(C))
        sys.exit(1)
    
    x=np.linspace(0,length,Nx)
    t=np.linspace(0,totalTime, Nt)
    
    u=np.zeros((Nt, Nx))
    u[0]=initialFunc(x)
    u[1]=u[0]
    for n in tqdm(range(1,Nt-1)):
        for i in range(1,Nx-1): # All interior points
            u[n+1,i]=(2 * (1 - C**2) * u[n, i] - u[n - 1, i] + C**2 * (u[n, i + 1] + u[n, i - 1]))

    clearDirectory()
    for i in tqdm(range(Nt)):
        plt.plot(u[i])
        plt.xlim(0,Nx)
        plt.ylim(-5,5)
        plt.savefig('../..//Physics-Simulations/Images for simulation/graph'+str(i)+'.png') # dpi argument increases resolution
        plt.close('all')

def dampedWaveSimulation(initialFunc, c=1, gamma=1, length=1, totalTime=5, Nx=200, Nt=1000):
    """
    Description
    -----------
    A function for numerically solving the 1D wave equation and plotting each frame of its simulation.

    Parameters
    ----------
    initialFunc : Function
        Function defining the first initial condition.
    c : Float
       Wave eq parameter.
    gamma : Float
        Dampening Parameter
    length : Float
        Length of the spatial dimension
    totalTime : Float
        Total time of the simulation
    Nx : int
        Total number of discrete points in the spatial dimension
    Nt : int
        Total number of discrete points in the temporal dimention
    
    Returns 
    -------
    Nothing. But! Creates all the frames of the simulation in the titular "Images for Simulation" file to then be used with ffmpeg (ffmpeg -start_number 0 -framerate 60 -i graph%01d.png video.webm)
    """
    # Lets create a meshed function
    length = length
    c=c
    gamma = gamma
    totalTime = totalTime
    Nx = Nx # x axis
    Nt = Nt # t axis
    deltax = length/(Nx-1) # THIS IS THE IMPORTANT LINE OR ELSE IT BLOWS UP!!!!!!!!!!!!!!!!!!!! I HAVE NO CLUE WHY THAT -1 NEEDS TO BE THERE BUT IT DOES OR EVERYTHING GOES TO SHIT
    deltat = totalTime/Nt
    C=c*(deltat/deltax)
    if C >=1 :
        print("C is greater than 1. C="+str(C))
        sys.exit(1)
    
    x=np.linspace(0,length,Nx)
    t=np.linspace(0,totalTime, Nt)
    
    u_initial = np.exp(-((x - 0.1)**2) / (2 * sigma**2))
    v_initial = (A * v * (x - x_0) / sigma**2) * np.exp(-((x - x_0)**2) / (2 * sigma**2))
    
    u=np.zeros((Nt, Nx))
    u[0]=initialFunc(x)
    u[1]=u[0]
    for n in tqdm(range(1,Nt-1)):
        for i in range(1,Nx-1): # All interior points
            u[n+1,i]=(1/(1+gamma*deltat))*((2*u[n, i]*(1-gamma*deltat))-(u[n-1,i]*(1-gamma*deltat))+(C**2)*(u[n,i+1]-2*u[n,i]+u[n,i-1]))
            
    sim.clearDirectory()
    for i in tqdm(range(Nt)):
        plt.plot(u[i])
        plt.xlim(0,Nx)
        plt.ylim(-5,5)
        plt.savefig('../..//Physics-Simulations/Images for simulation/graph'+str(i)+'.png') # dpi argument increases resolution
        plt.close('all')