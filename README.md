# Penrose Diagram Plotter

A Python project for generating and plotting Penrose-Carter diagrams for the Schwarzschild spacetime.

A Penrose diagram is a tool used in general relativity to visualise the causal structure of a stationary black hole on a 2D map. It transforms the coordinates so that the infinities of the black hole spacetime are mapped to a finite distance. This allows us to clearly see the causal relationships between different regions, such as the event horizon, the singularity, and the different "universes" or infinities.

This script performs this transformation by first moving from Schwarzschild-like (R,T) coordinates to Kruskal-Szekeres coordinates and then applying a final conformal transformation to create the compact Penrose diagram.

# Includes
plots.py: which contains all the functions for converting time and radial coordinates into Penrose-Carter coordinates and plotting on a Penrose diagram template.
penrose.ipynb: which demonstrates the usage of plots.py on Scwarzschild, isotropic and trumpet coordinates.

# Requirements
numpy, matplotlib, scipy, sympy 
