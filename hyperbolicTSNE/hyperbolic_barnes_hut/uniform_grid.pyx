# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

cimport numpy as np
import numpy as np
import time
include "grid_gen_gpu.py"
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, log, acosh, cosh, cos, sin, M_PI, atan2, tanh, atanh, isnan, fabs, fmin, fmax

np.import_array()

cdef double EPSILON = 0.0
cdef double BOUNDARY = 1 - EPSILON
cdef double MACHINE_EPSILON = np.finfo(np.double).eps

def get_current_time():
    return time.time() * 1000.0

cdef double clamp(double n, double lower, double upper) nogil:
    cdef double t = lower if n < lower else n
    return upper if t > upper else t

def py_divide_points_over_grid(points, n):
    cdef int num_points = points.shape[0]
    cdef int grid_size = n * n
    cdef np.ndarray[np.int32_t, ndim=1] result_indices = np.empty(num_points, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] new_points = np.empty(points.shape, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] grid_square_indices_per_point = np.empty(num_points, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] result_starts_counts = np.empty((grid_size * 2), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] max_distances = np.empty(grid_size, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] square_positions = np.zeros((grid_size * 2), dtype=np.float64)
    cdef double x_min, width, y_min, height
    c_divide_points_over_grid(points, new_points, n, grid_square_indices_per_point, result_indices, result_starts_counts, max_distances, square_positions, &x_min, &width, &y_min, &height)
    return result_starts_counts, square_positions, x_min, width, y_min, height

def py_poincare_to_euclidean(x, y):
    cdef double ex, ey
    poincare_to_euclidian(x, y, &ex, &ey)
    return ex, ey

def py_euclidean_to_poincare(x, y):
    cdef double px, py
    euclidean_to_poincare(x, y, &px, &py)
    return px, py

cdef void poincare_to_klein(double px, double py, double* kx, double* ky):
    cdef double denominator = 1.0 + px*px + py*py
    kx[0] = 2.0 * px / denominator
    ky[0] = 2.0 * py / denominator
    return

cdef void klein_to_poincare(double kx, double ky, double* px, double* py):
    cdef double denominator = 1.0 + np.sqrt(1.0 - kx*kx - ky*ky)
    px[0] = kx / denominator
    py[0] = ky / denominator
    return

cdef double gamma(double v0, double v1):
    cdef double norm_v_sq = v0 * v0 + v1 * v1
    return 1.0 / np.sqrt(1.0 - norm_v_sq)

cdef void poincare_to_euclidian(double x, double y, double* ox, double* oy):
    if (x == 0.0 and y == 0.0):
        ox[0] = 0.0
        oy[0] = 0.0
        return

    cdef double r = np.sqrt(x*x + y*y)
    ox[0] = (1.0 * x) / (1.0 - r)
    oy[0] = (1.0 * y) / (1.0 - r)
    return

cdef void euclidean_to_poincare(double x, double y, double* ox, double* oy):
    if (x == 0.0 and y == 0.0):
        ox[0] = 0.0
        oy[0] = 0.0
        return

    cdef double r = np.sqrt(x*x + y*y)
    ox[0] = (1.0 * x) / (1.0 + r)
    oy[0] = (1.0 * y) / (1.0 + r)
    return


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

cdef double max_distance_in_grid_square(int grid_x, int grid_y, double grid_width_x, double grid_width_y, double px, double py,
        double x_min, double width, double y_min, double height):
    cdef bl_x =     grid_x * grid_width_x + x_min
    cdef bl_y =     grid_y * grid_width_y + y_min
    cdef tr_x = (grid_x+1) * grid_width_x + x_min
    cdef tr_y = (grid_y+1) * grid_width_y + y_min

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


def reverse_reorder_array_inplace_py(original_array, result_indices):
    reverse_reorder_array_inplace(original_array, result_indices)
    #return

cdef void reverse_reorder_array_inplace(np.ndarray[np.float64_t, ndim=1] original_array, np.ndarray[np.int32_t, ndim=1] result_indices):
    cdef int num_elements = result_indices.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] temp_array = np.empty_like(original_array)

    cdef int i

    # Copy original array to temporary array
    for i in range(num_elements):
        temp_array[i*2+0] = original_array[i*2+0]
        temp_array[i*2+1] = original_array[i*2+1]

    # Reorder the original array according to result_indices
    for i in range(num_elements):
        original_array[result_indices[i]*2+0] = temp_array[i*2+0]
        original_array[result_indices[i]*2+1] = temp_array[i*2+1]



cdef void c_divide_points_over_grid(double[:,:] points, np.ndarray[np.float64_t, ndim=2] new_points, int n,
        np.ndarray[np.int32_t, ndim=1] grid_square_indices_per_point,
        np.ndarray[np.int32_t, ndim=1] result_indices,
        np.ndarray[np.int32_t, ndim=1] result_starts_counts,
        np.ndarray[np.float64_t, ndim=1] max_distances,
        np.ndarray[np.float64_t, ndim=1] square_positions,
        double* x_min, double* width, double* y_min, double* height):
    cdef int num_points = points.shape[0]
    cdef int grid_size = n * n
    cdef double grid_width_x, grid_width_y
    cdef int i, j, k, index
    cdef double x, y

    #print("[GRID] start")
    
    start_time = get_current_time()

    # Steps of this algorithm:
    # 1. Setup variables
    # 2. Find for each point the grid square it belongs to
    # 3. Count the number of members in each grid cell
    # 4. Setting the grid start and count values based on the number of points the grid cells hold
    # 5. Filling the array that holds the indices for each grid cell
    # 6. Calculating the mean of each grid cell


    # ======= Step 1: Setup variables =======
    
    cdef double dist = 0.0
    cdef double ex, ey

    x_min[0] = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min[0] = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    width[0] = x_max - x_min[0]
    height[0] = y_max - y_min[0]
    grid_width_x = width[0] / n
    grid_width_y = height[0] / n

    end_time = get_current_time()

    execution_time = end_time - start_time
    #print("[GRID] Init: ", execution_time, "ms")

    start_time = get_current_time()


    # ======= Step 2: Find for each point the grid square it belongs to =======

    for p in range(num_points):
        x = points[p, 0]
        y = points[p, 1]

        i = int((x - x_min[0]) / grid_width_x)
        j = int((y - y_min[0]) / grid_width_y)

        index = i * n + j

        if (index < 0):
            index = 0
        
        if (index >= grid_size):
            index = grid_size-1
        

        grid_square_indices_per_point[p] = index


    end_time = get_current_time()

    execution_time = end_time - start_time
    #print("[GRID] Indices: ", execution_time, "ms")


    # ======= Step 3: Count the number of members in each grid cell =======

    start_time = get_current_time()

    cdef int* grid_counts = <int*>malloc(sizeof(int) * grid_size)

    # Initialise the counts at zero
    for i in range(grid_size):
        grid_counts[i] = 0

    # Counting the members
    for p in range(num_points):
        index = grid_square_indices_per_point[p]

        grid_counts[index] += 1


    end_time = get_current_time()

    execution_time = end_time - start_time
    #print("[GRID] Counts: ", execution_time, "ms")


    # ======= Step 4: Setting the grid start and count values based on the number of points the grid cells hold =======

    start_time = get_current_time()

    cdef int* grid_start_indices = <int*>malloc(sizeof(int) * grid_size)
    cdef int current_start_index = 0

    for i in range(grid_size):
        # Saving the index and stride data per grid cell
        result_starts_counts[i*2+0] = current_start_index
        result_starts_counts[i*2+1] = grid_counts[i]

        grid_start_indices[i] = current_start_index
        current_start_index += grid_counts[i]

    end_time = get_current_time()

    execution_time = end_time - start_time
    #print("[GRID] Starts, counts: ", execution_time, "ms")


    # ======= Step 5: Filling the array that holds the indices for each grid cell =======

    start_time = get_current_time()

    for p in range(num_points):
        # Finding the grid square this point belongs to
        index = grid_square_indices_per_point[p]

        # Then finding the pointer into the result_indices array (holds the indices per grid square)
        result_index = grid_start_indices[index]

        # Moving the current pointer over
        result_indices[result_index] = p
        grid_start_indices[index] += 1


    end_time = get_current_time()

    execution_time = end_time - start_time
    #print("[GRID] Starts, counts: ", execution_time, "ms")


    # ======= Step 6: Calculating the mean of each grid cell =======

    start_time = get_current_time()

    cdef double point_poincare_x = 0.0
    cdef double point_poincare_y = 0.0

    mids, gs = calculate_average_grid_square_positions_gpu(points, num_points, n, grid_square_indices_per_point)

    end_time = get_current_time()

    execution_time = end_time - start_time
    #print("[GRID] Average positions p1: ", execution_time, "ms")
    
    start_time = get_current_time()

    # Calculating the mean position of each square
    for i in range(grid_size):
        if (grid_counts[i] == 0):
            continue

        klein_to_poincare(mids[i*2+0] / gs[i], mids[i*2+1] / gs[i], &point_poincare_x, &point_poincare_y)
        
        square_positions[i*2 + 0] = point_poincare_x
        square_positions[i*2 + 1] = point_poincare_y
    
    end_time = get_current_time()

    execution_time = end_time - start_time
    #print("[GRID] Average positions p2: ", execution_time, "ms")


    # Freeing memory for temporary arrays
    free(grid_counts)
    free(grid_start_indices)



