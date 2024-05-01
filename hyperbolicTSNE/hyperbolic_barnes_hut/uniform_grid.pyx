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
cdef double MACHINE_EPSILON = np.finfo(np.double).eps

cdef double clamp(double n, double lower, double upper) nogil:
    cdef double t = lower if n < lower else n
    return upper if t > upper else t

def py_divide_points_over_grid(points, n):
    cdef int num_points = points.shape[0]
    cdef int grid_size = n * n
    cdef np.ndarray[np.int32_t, ndim=1] result_indices = np.empty(num_points, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] grid_square_indices_per_point = np.empty(num_points, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] result_starts_counts = np.empty((grid_size * 2), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] max_distances = np.empty(grid_size, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] square_positions = np.zeros((grid_size * 2), dtype=np.float64)
    cdef double x_min, width, y_min, height
    c_divide_points_over_grid(points, n, grid_square_indices_per_point, result_indices, result_starts_counts, max_distances, square_positions, &x_min, &width, &y_min, &height)
    return grid_square_indices_per_point, result_indices, result_starts_counts, max_distances, square_positions, x_min, width, y_min, height

def py_poincare_to_euclidean(x, y):
    cdef double ex, ey
    poincare_to_euclidian(x, y, &ex, &ey)
    return ex, ey

def py_euclidean_to_poincare(x, y):
    cdef double px, py
    euclidean_to_poincare(x, y, &px, &py)
    return px, py

# r = (px^2 + py^2)
# ex = 2px / (1 - r)
# ey = 2py / (1 - r)
#
#
#
#
#
#

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

cdef void c_divide_points_over_grid(double[:,:] points, int n,
        np.ndarray[np.int32_t, ndim=1] grid_square_indices_per_point,
        np.ndarray[np.int32_t, ndim=1] result_indices,
        np.ndarray[np.int32_t, ndim=1] result_starts_counts,
        np.ndarray[np.float64_t, ndim=1] max_distances,
        np.ndarray[np.float64_t, ndim=1] square_positions,
        double* x_min, double* width, double* y_min, double* height):
    cdef int num_points = points.shape[0]
    cdef int grid_size = n * n
    cdef double grid_width_x, grid_width_y
    #cdef int[:, :] grid_indices = np.empty((n, n), dtype=np.int32)
    cdef int i, j, k, index
    cdef double x, y
    
    # Steps of this algorithm:
    # 1. decide for each point which grid square index it belongs to (saved into indices)
    # 2. find for each grid square how many members it has (grid_counts)
    # 3. using the member count to find the start index int he final array for each grid square
    # 4. using these values as the start values, fill the final result indices array

    # Initialise the max distances at 0
    for i in range(grid_size):
        max_distances[i] = 0.0

    # ======= STEP 1 =======
    
    #cdef int* indices = <int*>malloc(sizeof(int) * num_points) # = [0, grid_size-1]
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

    for p in range(num_points):
        x = points[p, 0] # valid
        y = points[p, 1] # valid

        i = int((x - x_min[0]) / grid_width_x)
        j = int((y - y_min[0]) / grid_width_y)

        index = i * n + j

        if (index < 0):
            index = 0
        
        if (index >= grid_size):
            index = grid_size-1
        

        grid_square_indices_per_point[p] = index # valid


        # TODO: calculate max distance in square from center
        #dist = max_distance_in_grid_square(i, j, grid_width_x, grid_width_y, x, y, x_min[0], width[0], y_min[0], height[0])
        #dist = distance(0.0, 0.0, x, y)
        #if (dist > max_distances[index]):
        #    max_distances[index] = dist

        #poincare_to_euclidian(x, y, &ex, &ey)
        #square_positions[index*2 + 0] += x
        #square_positions[index*2 + 1] += y


    # ======= STEP 2 =======

    cdef int* grid_counts = <int*>malloc(sizeof(int) * grid_size)

    # Initialise the counts at zero
    for i in range(grid_size):
        grid_counts[i] = 0

    # Counting the members
    for p in range(num_points):
        index = grid_square_indices_per_point[p]

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
        index = grid_square_indices_per_point[p] # valid

        # Moving the current pointer over
        result_index = grid_start_indices[index]

        result_indices[result_index] = p
        grid_start_indices[index] += 1


    # ======= STEP 5 =======

    cdef double mid_x = 0.0
    cdef double mid_y = 0.0
    cdef double point_klein_x = 0.0
    cdef double point_klein_y = 0.0
    cdef double point_poincare_x = 0.0
    cdef double point_poincare_y = 0.0
    cdef double gamma_sum = 0.0
    cdef double g = 0.0

    # Calculating the average position of each square
    for i in range(grid_size):
        if (grid_counts[i] == 0):
            continue

        mid_x = 0.0
        mid_y = 0.0
        gamma_sum = 0.0
        g = 0.0
        
        # Calculate the midpoint
        for j in range(result_starts_counts[i*2+0], result_starts_counts[i*2+0] + result_starts_counts[i*2+1]):
            index = result_indices[j]
            poincare_to_klein(points[index, 0], points[index, 1], &point_klein_x, &point_klein_y)
            g = gamma(point_klein_x, point_klein_y)
            mid_x += g * point_klein_x
            mid_y += g * point_klein_y
            gamma_sum += g
        
        mid_x /= gamma_sum
        mid_y /= gamma_sum

        klein_to_poincare(mid_x, mid_y, &point_poincare_x, &point_poincare_y)
        
        square_positions[i*2 + 0] = point_poincare_x
        square_positions[i*2 + 1] = point_poincare_y
    
    # ======= STEP 6 ========
    
    for p in range(num_points):
        x = points[p, 0] # valid
        y = points[p, 1] # valid

        i = int((x - x_min[0]) / grid_width_x)
        j = int((y - y_min[0]) / grid_width_y)

        index = i * n + j

        if (index < 0):
            index = 0
        
        if (index >= grid_size):
            index = grid_size-1

        # TODO: calculate max distance in square from center
        dist = distance(x, y, square_positions[index*2 + 0], square_positions[index*2 + 1])

        #dist = distance(0.0, 0.0, x, y)
        if (dist > max_distances[index]):
            max_distances[index] = dist


    # Free memory for temporary arrays
    free(grid_counts)
    free(grid_start_indices)



