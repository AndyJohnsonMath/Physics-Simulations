import numpy as np
import matplotlib.pyplot as plt
import math
import os

################################################ Numerical Methods #########################################################################################################

# eulersMethod(): Generalized Function that takes in the stepsize, initial conditions and said function from previous cell and returns
# a 2d array where the first entry is the array of x-values and the second entry is the array of approximated y-values

def eulersMethod(function, stepSize, initialPair, intervalLength):
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

# rungeKutta(): Function set up exactly the same as eulersMethod(), just runs the Runge-Kutta algorithm instead
def rungeKutta(function, stepSize, initialPair, intervalLength):
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

# rkf45(): This function runs the Runge-Kutta-Fehlberg fourth-order-fifth-order scheme
# You might notice this function takes significantly more function arguments than the previous function, that is just a result of this function being more intense in general
# Takes in a function f(x,y), initial step size you want to work with (if the step size is too big this algorithm will correct it), initialPair which is a numpy array with the initial values for the IVP
# minStepSize and maxStepSize are the bounds for the variable step size and TOL is the user-defined tolerance measuring the maximum amount of error allowed in solving. In testing, TOL=5*10^-7
def rkf45(function, initialStepSize, initialPair, intervalLength, minStepSize, maxStepSize, TOL):
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

###################################################### gravity stuff ########################################################################################################


# PointMassBody() is the class for point masses
class PointMassBody:
    def __init__(self,mass,position,velocity,acceleration):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration


###################################################### Lorenz Attractor Stuff ########################################################################################################
                
def lorenz(xyz, s=10, r=28, b=2.667):
    """
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

# lorenzAttractorImage() produces an image of a Lorenz Attractor System after 'length' amount of iterations. Also takes system parameters s, r, and b. Default value is just a known value that gives a known result to use as sanity checks
def lorenzAttractorImage(length, s=10, r=28, b=2.667):
    dt = 0.01
    num_steps = length

    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = (0., 1., 1.05)  # Set initial values
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
    ax.set_title("Lorenz Attractor")

    plt.show()

# lorenzAttractorTrace() takes in the amount of frames you want to video to be along with the system paramters s, r and b. Default value is just a known value that gives a known result to use as sanity checks
# Will save images to target directory where you will then have to run ffmpeg through the command line to use. Ffmpeg comand is given in the next line
# ffmpeg -start_number 0 -framerate 60 -i graph%01d.png video.webm

def lorenzAttractorTrace(frames, s=10, r=28, b=2.667):
    #Empty the target directory
    dir = './Images for simulation'

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    #Calculate the array of points according to the lorenz system
    #Do this outside the main loop so that we only calculate it once rather than a bazillion times and annihilate memory
    dt = 0.01
    numSteps = frames

    xyzs = np.empty((numSteps+1, 3))  # Need one more for the initial values
    xyzs[0] = (0., 1., 1.05)  # Set initial values
    for i in range(numSteps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i],s,r,b) * dt

    #plot the first frame outside of the main loop, same idea as initial conditions just with a frame
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*xyzs[0].T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.savefig('./Images for simulation/graph'+str(0)+'.png')
    plt.close('all')

    #Initialize frame to 1 so that our indexing for xyzs in the main loop prints from 0-frame. If frame was 0 then we would be plotting xysz from xyzs[0] ot xyzs[0] which we cant do. We need atleast xyzs[0] to xyzs[1]
    frame = 1
    while frame < numSteps:
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(*xyzs[:frame].T, lw=0.5) #Recall this [:frame] notion means we plot the array from xyzs[0] to xyzs[frame]
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            ax.set_title("Lorenz Attractor")

            plt.savefig('./Images for simulation/graph'+str(frame)+'.png')
            plt.close('all')

            frame = frame + 1