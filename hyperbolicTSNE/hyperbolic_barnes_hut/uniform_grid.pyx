# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, log, acosh, cosh, cos, sin, M_PI, atan2, tanh, atanh, isnan, fabs, fmin, fmax

np.import_array()

cdef double EPSILON = 0.0
cdef double BOUNDARY = 1 - EPSILON

cdef double clamp(double n, double lower, double upper) nogil:
    cdef double t = lower if n < lower else n
    return upper if t > upper else t

def py_divide_points_over_grid(points, n):
    cdef int num_points = points.shape[0]
    cdef int grid_size = n * n
    cdef np.ndarray[np.int32_t, ndim=1] result_indices = np.empty(num_points, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] result_starts_counts = np.empty((grid_size * 2), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] max_distances = np.empty(grid_size, dtype=np.float64)
    c_divide_points_over_grid(points, n, result_indices, result_starts_counts, max_distances)
    return result_indices, result_starts_counts, max_distances

cpdef double distance(double u0, double u1, double v0, double v1):
    if fabs(u0 - v0) <= EPSILON and fabs(u1 - v1) <= EPSILON:
        return 0.

    cdef:
        double uv2 = ((u0 - v0) * (u0 - v0)) + ((u1 - v1) * (u1 - v1))
        double u_sq = clamp(u0 * u0 + u1 * u1, 0, BOUNDARY)
        double v_sq = clamp(v0 * v0 + v1 * v1, 0, BOUNDARY)
        double alpha = 1. - u_sq
        double beta = 1. - v_sq
        double result = acosh( 1. + 2. * uv2 / ( alpha * beta ) )

    return result

cdef int valid_point(double x, double y):
    if x*x+y*y < 1.0:
        return 1

    return 0

cdef double max_distance_in_grid_square(int grid_x, int grid_y, double grid_width, double px, double py):
    cdef bl_x =     grid_x * grid_width - 1.0
    cdef bl_y =     grid_y * grid_width - 1.0
    cdef tr_x = (grid_x+1) * grid_width - 1.0
    cdef tr_y = (grid_y+1) * grid_width - 1.0

    cdef double max_dist = 0.0
    cdef double dist = 0.0

    # Checking bottom left distance
    if (valid_point(bl_x, bl_y) == 1):
        dist = distance(px, py, bl_x, bl_y)
        if (dist > max_dist):
            max_dist = dist
        
    # Checking top right distance
    if (valid_point(tr_x, tr_y) == 1):
        dist = distance(px, py, tr_x, tr_y)
        if (dist > max_dist):
            max_dist = dist
        
    # Checking top left distance
    if (valid_point(bl_x, tr_y) == 1):
        dist = distance(px, py, bl_x, tr_y)
        if (dist > max_dist):
            max_dist = dist
        
    # Checking bottom right distance
    if (valid_point(tr_x, bl_y) == 1):
        dist = distance(px, py, tr_x, bl_y)
        if (dist > max_dist):
            max_dist = dist

    return max_dist

cdef void c_divide_points_over_grid(double[:,:] points, int n,
        np.ndarray[np.int32_t, ndim=1] result_indices,
        np.ndarray[np.int32_t, ndim=1] result_starts_counts,
        np.ndarray[np.float64_t, ndim=1] max_distances):
    cdef int num_points = points.shape[0]
    cdef int grid_size = n * n
    cdef double grid_width = 2.0 / n
    cdef int[:, :] grid_indices = np.empty((n, n), dtype=np.int32)
    cdef int i, j, k, index
    cdef double x, y
    
    # Steps of this algorithm:
    # 1. decide for each point which grid square index it belongs to (saved into indices)
    # 2. find for each grid square how many members it has (grid_counts)
    # 3. using the member count to find the start index int he final array for each grid square
    # 4. using these values as the start values, fill the final result indices array

    # Initialise the max distances at 0
    for i in range(grid_size):
        max_distances[i] = distance(-0.9999, 0.0, 0.0, 0.0)

    # ======= STEP 1 =======
    
    cdef int* indices = <int*>malloc(sizeof(int) * num_points) # = [0, grid_size-1]
    cdef double dist = 0.0

    for p in range(num_points):
        x = points[p, 0] # valid
        y = points[p, 1] # valid

        i = int((x + 1.0) / grid_width)
        j = int((y + 1.0) / grid_width)

        index = i * n + j

        if (index < 0):
            index = 0
        
        if (index >= grid_size):
            index = grid_size-1
        

        indices[p] = index # valid

        dist = max_distance_in_grid_square(i, j, grid_width, x, y)
        #dist = distance(0.0, 0.0, x, y)
        #if (dist > max_distances[index]):
            #max_distances[index] = dist


    # ======= STEP 2 =======

    cdef int* grid_counts = <int*>malloc(sizeof(int) * grid_size)

    # Initialise the counts at zero
    for i in range(grid_size):
        grid_counts[i] = 0

    # Counting the members
    for p in range(num_points):
        index = indices[p]

        grid_counts[index] += 1


    # ======= STEP 3 =======

    cdef int* grid_start_indices = <int*>malloc(sizeof(int) * grid_size)
    cdef int current_start_index = 0

    for i in range(grid_size):
        # Saving the index and stride data per grid cell
        result_starts_counts[i*2+0] = current_start_index
        result_starts_counts[i*2+1] = grid_counts[i]

        grid_start_indices[i] = current_start_index
        current_start_index += grid_counts[i]


    # ======= STEP 4 =======

    for p in range(num_points):
        index = indices[p] # valid

        # Moving the current pointer over
        result_index = grid_start_indices[index]

        result_indices[result_index] = p
        grid_start_indices[index] += 1

    
    # Free memory for temporary arrays
    free(indices)
    free(grid_counts)
    free(grid_start_indices)



