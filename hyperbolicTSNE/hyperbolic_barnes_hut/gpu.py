import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
from . import uniform_grid

exact_compute_gradient_negative_gpu_func = None
exact_compute_gradient_positive_gpu_func = None
ugrid_compute_gradient_negative_gpu_func = None

def get_exact_compute_gradient_negative_gpu_func():
    global exact_compute_gradient_negative_gpu_func

    if (exact_compute_gradient_negative_gpu_func == None):
        with open("gpu_code\exact_negative_gradient.cu", "r") as file:
            cuda_kernel = file.read()

        # Compile the CUDA kernel
        mod = SourceModule(cuda_kernel)

        # Get the kernel function
        exact_compute_gradient_negative_gpu_func = mod.get_function("add")
    
    return exact_compute_gradient_negative_gpu_func

def get_exact_compute_gradient_positive_gpu_func():
    global exact_compute_gradient_positive_gpu_func

    if (exact_compute_gradient_positive_gpu_func == None):
        with open("gpu_code\exact_positive_gradient.cu", "r") as file:
            cuda_kernel = file.read()

        # Compile the CUDA kernel
        mod = SourceModule(cuda_kernel)

        # Get the kernel function
        exact_compute_gradient_positive_gpu_func = mod.get_function("add")
    
    return exact_compute_gradient_positive_gpu_func

def get_ugrid_compute_gradient_negative_gpu_func():
    global ugrid_compute_gradient_negative_gpu_func

    if (ugrid_compute_gradient_negative_gpu_func == None):
        with open("gpu_code/u_grid_negative_gradient.cu", "r") as file:
            cuda_kernel = file.read()

        # Compile the CUDA kernel
        mod = SourceModule(cuda_kernel)

        # Get the kernel function
        ugrid_compute_gradient_negative_gpu_func = mod.get_function("add")
    
    return ugrid_compute_gradient_negative_gpu_func

def compute_gradient_positive_gpu(start, pos_reference, n_dimensions, n_samples, sumQ, neighbours, indptr, val_P):

    # Allocate memory for neg_f and sumQs
    pos_f = np.zeros(n_samples * n_dimensions, dtype=np.double)
    C_values = np.zeros(n_samples, dtype=np.double)

    start_time = time.time()

    # Convert pos and neg_f to ctypes pointers
    pos_gpu = cuda.mem_alloc(pos_reference.nbytes)
    posf_gpu = cuda.mem_alloc(pos_f.nbytes)
    C_values_gpu = cuda.mem_alloc(C_values.nbytes)

    neighbours_gpu = cuda.mem_alloc(neighbours.nbytes)
    indptr_gpu = cuda.mem_alloc(indptr.nbytes)
    val_P_gpu = cuda.mem_alloc(val_P.nbytes)

    end_time = time.time()

    start_time = time.time()

    execution_time = end_time - start_time
    print("Mem alloc: ", execution_time, "seconds")

    cuda.memcpy_htod(pos_gpu, pos_reference)
    cuda.memcpy_htod(posf_gpu, pos_f)
    cuda.memcpy_htod(C_values_gpu, C_values)

    cuda.memcpy_htod(neighbours_gpu, neighbours)
    cuda.memcpy_htod(indptr_gpu, indptr)
    cuda.memcpy_htod(val_P_gpu, val_P)

    end_time = time.time()

    execution_time = end_time - start_time
    print("Mem copy: ", execution_time, "seconds")

    # optimisation: start at start instead of 0
    iterations = n_samples# - start

    block_size = 256
    num_blocks = (iterations + block_size - 1) // block_size


    start_time = time.time()

    cuda_func = get_exact_compute_gradient_positive_gpu_func()

    end_time = time.time()

    execution_time = end_time - start_time
    print("Get func: ", execution_time, "seconds")

    start_time = time.time()

    # Call the CUDA kernel
    # You would need to modify this part to fit your actual CUDA kernel invocation
    #          int start,       int n_samples,       int n_dimensions, double sum_Q, double *pos, double *pos_f, double *C_values, long *neighbors, long *indptr, double *val_P
    cuda_func(np.int32(start), np.int32(n_samples), np.int32(n_dimensions), np.double(sumQ),    pos_gpu,     posf_gpu,        C_values_gpu,    neighbours_gpu,  indptr_gpu, val_P_gpu,
              block=(block_size, 1, 1), grid=(num_blocks, 1))

    end_time = time.time()

    execution_time = end_time - start_time
    print("Run: ", execution_time, "seconds")

    start_time = time.time()

    # Transfer results back to CPU
    cuda.memcpy_dtoh(pos_f, posf_gpu)
    cuda.memcpy_dtoh(C_values, C_values_gpu)

    end_time = time.time()

    execution_time = end_time - start_time
    print("Copy back: ", execution_time, "seconds")

    error = np.sum(C_values)

    #print("negf: ", neg_f)  # Output: [5 7 9]
    #print("sq: ", sQ)  # Output: [5 7 9]
    return pos_f, error


def uniform_grid_compute_gradient_negative_gpu(start, pos_reference, n_dimensions, n_samples):

    total_start_time = time.time()

    # Allocate memory for neg_f and sumQs
    neg_f = np.zeros(n_samples * n_dimensions, dtype=np.double)
    sumQ = np.zeros(1, dtype=np.double)

    start_time = time.time()
    grid_n = 30
    grid_size = grid_n*grid_n
    result_indices, result_starts_counts, max_distances, square_positions, x_min, width, y_min, height = uniform_grid.py_divide_points_over_grid(pos_reference, grid_n)

    end_time = time.time()
    execution_time = end_time - start_time
    #print("Grid generation: ", execution_time, "seconds")

    start_time = time.time()

    # Convert pos and neg_f to ctypes pointers
    pos_gpu = cuda.mem_alloc(pos_reference.nbytes)
    negf_gpu = cuda.mem_alloc(neg_f.nbytes)
    sumQ_gpu = cuda.mem_alloc(sumQ.nbytes)
    result_indices_gpu = cuda.mem_alloc(result_indices.nbytes)
    result_starts_counts_gpu = cuda.mem_alloc(result_starts_counts.nbytes)
    max_distances_gpu = cuda.mem_alloc(max_distances.nbytes)
    square_positions_gpu = cuda.mem_alloc(square_positions.nbytes)

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Mem alloc: ", execution_time, "seconds")

    start_time = time.time()

    cuda.memcpy_htod(pos_gpu, pos_reference)
    cuda.memcpy_htod(negf_gpu, neg_f)
    cuda.memcpy_htod(sumQ_gpu, sumQ)
    cuda.memcpy_htod(result_indices_gpu, result_indices)
    cuda.memcpy_htod(result_starts_counts_gpu, result_starts_counts)
    cuda.memcpy_htod(max_distances_gpu, max_distances)
    cuda.memcpy_htod(square_positions_gpu, square_positions)

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Mem copy: ", execution_time, "seconds")

    block_size = 256
    num_blocks = (n_samples + block_size - 1) // block_size

    cuda_func = get_ugrid_compute_gradient_negative_gpu_func()

    start_time = time.time()

    # Call the CUDA kernel
    # You would need to modify this part to fit your actual CUDA kernel invocation
    cuda_func(np.int32(start),
              np.int32(n_samples),
              np.int32(n_dimensions),
              np.int32(grid_size),
              np.int32(grid_n),
              pos_gpu,
              negf_gpu,
              result_indices_gpu,
              result_starts_counts_gpu,
              max_distances_gpu,
              square_positions_gpu,
              sumQ_gpu,
              block=(block_size, 1, 1), grid=(num_blocks, 1))

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Run: ", execution_time, "seconds")

    start_time = time.time()

    # Transfer results back to CPU
    cuda.memcpy_dtoh(neg_f, negf_gpu)
    cuda.memcpy_dtoh(sumQ, sumQ_gpu)

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Copy back: ", execution_time, "seconds")
    
    end_time = time.time()
    execution_time = end_time - total_start_time
    #print("Total: ", execution_time, "seconds")

    #print("negf: ", neg_f)  # Output: [5 7 9]
    #print("sumQ: ", sumQ[0])
    return neg_f, sumQ[0]



def exact_compute_gradient_negative_gpu(start, pos_reference, n_dimensions, n_samples):

    total_start_time = time.time()

    # Allocate memory for neg_f and sumQs
    neg_f = np.zeros(n_samples * n_dimensions, dtype=np.double)
    sumQ = np.zeros(1, dtype=np.double)

    start_time = time.time()

    # Convert pos and neg_f to ctypes pointers
    pos_gpu = cuda.mem_alloc(pos_reference.nbytes)
    negf_gpu = cuda.mem_alloc(neg_f.nbytes)
    sumQ_gpu = cuda.mem_alloc(sumQ.nbytes)

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Mem alloc: ", execution_time, "seconds")

    start_time = time.time()

    cuda.memcpy_htod(pos_gpu, pos_reference)
    cuda.memcpy_htod(negf_gpu, neg_f)
    cuda.memcpy_htod(sumQ_gpu, sumQ)

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Mem copy: ", execution_time, "seconds")

    block_size = 256
    num_blocks = (n_samples + block_size - 1) // block_size

    cuda_func = get_exact_compute_gradient_negative_gpu_func()

    start_time = time.time()

    # Call the CUDA kernel
    # You would need to modify this part to fit your actual CUDA kernel invocation
    cuda_func(np.int32(start), np.int32(n_samples), np.int32(n_dimensions), pos_gpu, negf_gpu, sumQ_gpu, block=(block_size, 1, 1), grid=(num_blocks, 1))

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Run: ", execution_time, "seconds")

    start_time = time.time()

    # Transfer results back to CPU
    cuda.memcpy_dtoh(neg_f, negf_gpu)
    cuda.memcpy_dtoh(sumQ, sumQ_gpu)

    end_time = time.time()

    execution_time = end_time - start_time
    #print("Copy back: ", execution_time, "seconds")
    
    end_time = time.time()
    execution_time = end_time - total_start_time
    #print("Total: ", execution_time, "seconds")

    #print("negf: ", neg_f)  # Output: [5 7 9]
    #print("sumQ: ", sumQ[0])
    return neg_f, sumQ[0]
