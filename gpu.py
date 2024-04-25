from cuda import cuda, nvrtc
import numpy as np
import os
import ctypes

exact_compute_gradient_negative_ptx = None
cuDevice = None
context = None
kernel = None

def gpu_reset():
    global cuDevice
    global context
    global kernel
    global exact_compute_gradient_negative_ptx

    exact_compute_gradient_negative_ptx = None
    cuDevice = None
    context = None
    kernel = None

def gpu_init():
    global cuDevice
    global context
    global kernel

    #if cuDevice != None:
        #return

    print("========== RUNNING GPU CODE ==========")

    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)

    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)

    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)

    # Load PTX as module data and retrieve function
    ptx = np.char.array(get_exact_compute_gradient_negative_ptx())
    # Note: Incompatible --gpu-architecture would be detected here
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err, "1")
    err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
    ASSERT_DRV(err, "2")

    NUM_THREADS = 512  # Threads per block
    NUM_BLOCKS = 16  # Blocks per grid

    a = np.array([2.0], dtype=np.float32)
    n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
    bufferSize = n * a.itemsize

    hX = np.random.rand(n).astype(dtype=np.float32)
    hY = np.random.rand(n).astype(dtype=np.float32)
    hOut = np.zeros(n).astype(dtype=np.float32)


    err, dXclass = cuda.cuMemAlloc(bufferSize)
    err, dYclass = cuda.cuMemAlloc(bufferSize)
    err, dOutclass = cuda.cuMemAlloc(bufferSize)

    err, stream = cuda.cuStreamCreate(0)

    err, = cuda.cuMemcpyHtoDAsync(
        dXclass, hX.ctypes.data, bufferSize, stream
    )
    err, = cuda.cuMemcpyHtoDAsync(
        dYclass, hY.ctypes.data, bufferSize, stream
    )

    # The following code example is not intuitive 
    # Subject to change in a future release
    dX = np.array([int(dXclass)], dtype=np.uint64)
    dY = np.array([int(dYclass)], dtype=np.uint64)
    dOut = np.array([int(dOutclass)], dtype=np.uint64)

    args = [a, dX, dY, dOut, n]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    err, = cuda.cuLaunchKernel(
        kernel,
        NUM_BLOCKS,  # grid x dim
        1,  # grid y dim
        1,  # grid z dim
        NUM_THREADS,  # block x dim
        1,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )

    err, = cuda.cuMemcpyDtoHAsync(
        hOut.ctypes.data, dOutclass, bufferSize, stream
    )
    err, = cuda.cuStreamSynchronize(stream)

    # Assert values are same after running kernel
    hZ = a * hX + hY
    if not np.allclose(hOut, hZ):
        raise ValueError("Error outside tolerance for host-device vectors")
    
    print("========== SUCCESS ==========")
    print(zip(hOut, hZ))
    
    err, = cuda.cuStreamDestroy(stream)
    err, = cuda.cuMemFree(dXclass)
    err, = cuda.cuMemFree(dYclass)
    err, = cuda.cuMemFree(dOutclass)
    err, = cuda.cuModuleUnload(module)
    err, = cuda.cuCtxDestroy(context)

def get_exact_compute_gradient_negative_ptx():
    global exact_compute_gradient_negative_ptx

    if exact_compute_gradient_negative_ptx == None:
        exact_compute_gradient_negative_ptx = compile_cuda_code("gpu_code\exact_negative_gradient.cu")

    return exact_compute_gradient_negative_ptx

def ASSERT_DRV(err, marker=""):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error ({}): {}".format(err, marker))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error ({}): {}".format(err, marker))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))
    
    
def compile_cuda_code(file_path):
    # Read CUDA code from file
    with open(file_path, 'r') as file:
        cuda_code = file.read()

    saxpy = """\
    extern "C" __global__
    void saxpy(float a, float *x, float *y, float *out, size_t n)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            out[tid] = a * x[tid] + y[tid];
        }
    }
    """

    print("code:")
    print(cuda_code)
    print("code 2:")
    print(saxpy)

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], [])
    ASSERT_DRV(err, "4")

    # Compile program
    opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
    err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)
    ASSERT_DRV(err, "5")

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err, "6")
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    ASSERT_DRV(err, "7")

    return ptx

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cuda_code), os.path.basename(file_path).encode(), 0, [], [])

    # Compile program
    opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    if err != 0:
        log = nvrtc.nvrtcGetProgramLogSize(prog)
        info = ctypes.create_string_buffer(log)
        nvrtc.nvrtcGetProgramLog(prog, info)
        raise RuntimeError("Compilation failed:\n{}".format(info.value.decode()))

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
