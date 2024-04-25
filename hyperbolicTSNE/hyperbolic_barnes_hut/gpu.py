import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

exact_compute_gradient_negative_gpu_func = None

def gpu_init():
    with open("gpu_code\exact_negative_gradient.cu", "r") as file:
        cuda_kernel = file.read()

    # Compile the CUDA kernel
    mod = SourceModule(cuda_kernel)

    # Get the kernel function
    add_cuda = mod.get_function("add")

    # Example usage
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.zeros_like(a)

    # Allocate GPU memory
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    # Transfer data to GPU memory
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Launch the CUDA kernel
    print("c0: ", c)
    add_cuda(a_gpu, b_gpu, c_gpu, block=(3, 1, 1), grid=(1, 1))
    print("c1: ", c)
    # Transfer results back to CPU
    cuda.memcpy_dtoh(c, c_gpu)

    print("c:  ", c)  # Output: [5 7 9]

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

def exact_compute_gradient_negative_gpu(start, pos_reference, n_dimensions, n_samples):

    # Allocate memory for neg_f and sumQs
    neg_f = np.zeros(n_samples * n_dimensions, dtype=np.double)
    sumQs = np.zeros(n_samples, dtype=np.double)

    # Convert pos and neg_f to ctypes pointers
    pos_gpu = cuda.mem_alloc(pos_reference.nbytes)
    negf_gpu = cuda.mem_alloc(neg_f.nbytes)
    sumQs_gpu = cuda.mem_alloc(sumQs.nbytes)

    cuda.memcpy_htod(pos_gpu, pos_reference)
    cuda.memcpy_htod(negf_gpu, neg_f)
    cuda.memcpy_htod(sumQs_gpu, sumQs)

    block_size = 256
    num_blocks = (n_samples + block_size - 1) // block_size

    cuda_func = get_exact_compute_gradient_negative_gpu_func()

    # Call the CUDA kernel
    # You would need to modify this part to fit your actual CUDA kernel invocation
    cuda_func(np.int32(start), np.int32(n_samples), np.int32(n_dimensions), pos_gpu, negf_gpu, sumQs_gpu, block=(block_size, 1, 1), grid=(num_blocks, 1))



    # Transfer results back to CPU
    cuda.memcpy_dtoh(neg_f, negf_gpu)
    cuda.memcpy_dtoh(sumQs, sumQs_gpu)

    sQ = np.sum(sumQs)

    #print("negf: ", neg_f)  # Output: [5 7 9]
    #print("sq: ", sQ)  # Output: [5 7 9]
    return neg_f, sQ
