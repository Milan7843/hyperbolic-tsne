import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time

calculate_average_grid_square_positions_gpu_func = None
pos_gpu = None
mids_gpu = None
gs_gpu = None
grid_square_indices_per_point_gpu = None
saved_num_points = None
saved_grid_n = None

def get_calculate_average_grid_square_positions_gpu_func():
    global calculate_average_grid_square_positions_gpu_func

    if (calculate_average_grid_square_positions_gpu_func == None):
        with open("gpu_code/calculate_average_grid_square_positions.cu", "r") as file:
            cuda_kernel = file.read()

        # Compile the CUDA kernel
        # TODO test options=["-fmad=false"]
        mod = SourceModule(cuda_kernel)

        # Get the kernel function
        calculate_average_grid_square_positions_gpu_func = mod.get_function("add")
    
    return calculate_average_grid_square_positions_gpu_func

def calculate_average_grid_square_positions_gpu(points, n_samples, grid_n, grid_square_indices_per_point):
    global pos_gpu
    global mids_gpu
    global gs_gpu
    global grid_square_indices_per_point_gpu
    global saved_num_points
    global saved_grid_n

    total_start_time = time.time()

    grid_size = grid_n*grid_n

    start_time = time.time()

    mids = np.zeros(grid_size*2, dtype=np.double)
    gs = np.zeros(grid_size, dtype=np.double)

    end_time = time.time()
    execution_time = end_time - start_time
    #print("[Grid AVG] Create mids, gs: ", execution_time, "seconds")

    start_time = time.time()

    if (saved_num_points == None or saved_num_points != n_samples or saved_grid_n != grid_n):
        pos_gpu = cuda.mem_alloc(points.nbytes)
        mids_gpu = cuda.mem_alloc(mids.nbytes)
        gs_gpu = cuda.mem_alloc(gs.nbytes)
        grid_square_indices_per_point_gpu = cuda.mem_alloc(grid_square_indices_per_point.nbytes)
        saved_num_points = n_samples
        saved_grid_n = grid_n
    
    end_time = time.time()
    execution_time = end_time - start_time
    #print("[Grid AVG] Memory allocation: ", execution_time, "seconds (", points.nbytes + mids.nbytes + gs.nbytes + grid_square_indices_per_point.nbytes," bytes)")

    start_time = time.time()

    #cuda.memcpy_htod(pos_gpu, reordered_points)
    cuda.memcpy_htod(pos_gpu, points)
    cuda.memcpy_htod(mids_gpu, mids)
    cuda.memcpy_htod(gs_gpu, gs)
    cuda.memcpy_htod(grid_square_indices_per_point_gpu, grid_square_indices_per_point)
    
    end_time = time.time()
    execution_time = end_time - start_time
    #print("[Grid AVG] Memory copy: ", execution_time, "seconds")


    block_size = 256
    num_blocks = (n_samples + block_size - 1) // block_size

    start_time = time.time()

    cuda_func = get_calculate_average_grid_square_positions_gpu_func()

    end_time = time.time()
    execution_time = end_time - start_time
    #print("[Grid AVG] Get function: ", execution_time, "seconds")


    start_time = time.time()

    # Call the CUDA kernel
    cuda_func(np.int32(n_samples), grid_square_indices_per_point_gpu, pos_gpu, mids_gpu, gs_gpu,
              block=(block_size, 1, 1), grid=(num_blocks, 1))

    end_time = time.time()
    execution_time = end_time - start_time
    #print("[Grid AVG] CUDA run: ", execution_time, "seconds")

    start_time = time.time()

    # Transfer results back to CPU
    cuda.memcpy_dtoh(mids, mids_gpu)
    cuda.memcpy_dtoh(gs, gs_gpu)

    end_time = time.time()
    execution_time = end_time - start_time
    #print("[Grid AVG] Retrieve results: ", execution_time, "seconds")

    end_time = time.time()
    execution_time = end_time - total_start_time
    #print("[Grid AVG] Total: ", execution_time, "seconds")

    #print("negf: ", neg_f)  # Output: [5 7 9]
    #print("sumQ: ", sumQ[0])
    return mids, gs


