from .hyperbolic_barnes_hut import uniform_grid

# This file contains Python definitions for the C functions related to the Uniform Grid

def divide_points_over_grid(points, n):
    return uniform_grid.py_divide_points_over_grid(points, n)

def distance_py(u0, u1, v0, v1):
    return uniform_grid.distance(u0, u1, v0, v1)

def py_poincare_to_euclidean(x, y):
    return uniform_grid.py_poincare_to_euclidean(x, y)

def py_euclidean_to_poincare(x, y):
    return uniform_grid.py_euclidean_to_poincare(x, y)