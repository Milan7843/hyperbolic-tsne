# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

np.import_array()

def py_divide_points_over_grid(points, n):
    cdef int num_points = points.shape[0]
    cdef int grid_size = n * n
    cdef np.ndarray[np.int32_t, ndim=1] result_indices = np.empty(num_points, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] result_starts_counts = np.empty((grid_size * 2), dtype=np.int32)
    c_divide_points_over_grid(points, n, result_indices, result_starts_counts)
    return result_indices, result_starts_counts

cdef void c_divide_points_over_grid(double[:,:] points, int n, np.ndarray[np.int32_t, ndim=1] result_indices, np.ndarray[np.int32_t, ndim=1] result_starts_counts):
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

    # ======= STEP 1 =======
    
    cdef int* indices = <int*>malloc(sizeof(int) * num_points) # = [0, grid_size-1]
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



