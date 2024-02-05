# About Physics-Simulations

Welcome to my repository for physics simulations! I call this repository "Physics Simulations", but in reality its just any sort of scientific computing that I'm currently interested in. This repository is filled with a variety of things mainly consisting of Jupyter Notebooks and custom made libraries. In this README you will find a brief description of this repositories contents, required dependencies, and general tips and tricks.

# Contents
- **Basic Collision Detection [Notebook].ipynb** : Jupyter Notebook detailing the construction of a basic collision detection engine.
- **Lorenz Attractor [Notebook].ipynb** : Jupyter Notebook exploring the details of the the Lorenz Attractor system, mainly creating animations of the Attractor.
- **N-Body Problem [Notebook].ipynb** : Jupyter Notbook detailing how to implement the gravity simulation engine.
- **memoryExperiment.py** : Python file designed to be used in memory profiling via the terminal
- **simulationLibrary.py** : Python file housing the entire physics engine that all of the jupyter notebooks are dependent on. This contains all generalized functions developed in the notebooks. It is split into parts depending on specific field of physics or numerical analysis that those specific functions match with.
- **Collision Detection [Notebook].ipynb** : Jupyter notebook documenting our exploration into collision detection simulations and our attempt at creating a homebrew one.

# Required python packages

I installed these all on my local machine using ```pip install```. Run these following lines in your terminal to make sure you have the required dependencies
- ```pip install numpy```
- ```pip install matplotlib```
- ```pip install os```
- ```pip install tqdm```
- ```pip install ffmpeg```

# General procedure for running on your own machine
- The main idea with these notebooks is to produce frames for an animation. These frames are saved to a folder called "Images for Simulation". **You need to create this file on your own device.** Once you have it made you will need to replace all the references to its path in the code with your own unique path on your machine. This is a current problem I am addressing and trying to fix so that this process can become more streamlined.

- The "high level" procedure goes as follows: $\text{Simulation runs in notebook} \rightarrow \text{saves frames to "Images for Simulation"} \rightarrow \text{Navigate to "Images for Simulation"} \rightarrow \text{Run ffpmeg commands}$


