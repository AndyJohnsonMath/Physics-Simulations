# We are running a memory profile on our physics simulation software.
# This file allows you to perform a memory analysis on a bit of code

# Runs as follows:
# 1) Open terminal
# 2) Go to working directory
# 3) first run: python3 -m mprof run memoryExperiment.py
# 4) Wait for 3) to finish
# 5) run: python3 -m mprof plot

# On my laptop I have to run: python -m mprof run memoryExperiment.py
# On my desktop I have to run: python3 -m mprof run memoryExperiment.py
# No clue why i need the 3 in the desktop but not on the laptop, but thats what i need to do

# Final result is a plot of the memory consumption (In MiB) over time taken at 0.1s intervals

# Necessary libraries, just a hodgepodge of ones ive used
import simulationLibrary as sim
import time
import numpy as np

# definition body of memFunc() is the function you actually want to run
def memFunc():
    i = 0
    while i <= 70:
        sim.gravitySimulation(numParticles=(i+48), numFrames=300)
        i += 5

# Need this __name__ conditional to run the memFunc() function
if __name__ == '__main__':
    memFunc()

# def memFunc():
#     timeArray = np.array([])
#     i = 0
#     while i <= 300:
#     # for i in range(10000000):
#         start = time.time()
#         sim.gravitySimulation(numParticles=(i+2), numFrames=10)
#         end=time.time()
#         tot = end-start
#         timeArray = np.append(timeArray,tot)
#         i += 10

#     while i >= 0:
#         start = time.time()
#         sim.gravitySimulation(numParticles=(i+2), numFrames=10)
#         end=time.time()
#         tot = end-start
#         timeArray = np.append(timeArray,tot)
#         i -= 10


