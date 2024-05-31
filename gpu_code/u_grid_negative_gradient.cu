/*
__device__ const double EPSILON = 1e-5;
__device__ const double BOUNDARY = 1.0 - EPSILON;
*/
__device__ double clamp(double x, double min, double max) {
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }

    return x;
}

__device__ double distance(double u0, double u1, double v0, double v1) {
    double EPSILON = 1e-5;
    double BOUNDARY = 1.0 - EPSILON;
    if ((fabs(u0 - v0) <= EPSILON) && (fabs(u1 - v1) <= EPSILON)) {
        return 0.0;
    }

    double uv2 = ((u0 - v0) * (u0 - v0)) + ((u1 - v1) * (u1 - v1));
    double u_sq = clamp(u0 * u0 + u1 * u1, 0, BOUNDARY);
    double v_sq = clamp(v0 * v0 + v1 * v1, 0, BOUNDARY);
    double alpha = 1.0 - u_sq;
    double beta = 1.0 - v_sq;
    double result = acosh( 1.0 + 2.0 * uv2 / ( alpha * beta ) );

    return result;
}

__device__ double distance_grad(double u0, double u1, double v0, double v1, int ax) {
    double EPSILON = 1e-5;
    double BOUNDARY = 1.0 - EPSILON;
    if ((fabs(u0 - v0) <= EPSILON) && (fabs(u1 - v1) <= EPSILON)) {
        return 0.0;
    }

    double MACHINE_EPSILON = 2.220446049250313e-16;

    double a = u0 - v0;
    double b = u1 - v1;
    double uv2 = a * a + b * b;

    double u_sq = clamp(u0 * u0 + u1 * u1, 0, BOUNDARY);
    double v_sq = clamp(v0 * v0 + v1 * v1, 0, BOUNDARY);
    double alpha = 1.0 - u_sq;
    double beta = 1.0 - v_sq;

    double gamma = 1.0 + (2.0 / (alpha * beta)) * uv2;
    double shared_scalar = 4.0 / fmax(beta * sqrt((gamma * gamma) - 1.0), MACHINE_EPSILON);

    double u_scalar = (v_sq - 2.0 * (u0 * v0 + u1 * v1) + 1.0) / (alpha * alpha);
    double v_scalar = 1.0 / alpha;

    if (ax == 0) {
        return shared_scalar * (u_scalar * u0 - v_scalar * v0);
    }

    return shared_scalar * (u_scalar * u1 - v_scalar * v1);
}

// Function that calculates the negative forces for a point using the Uniform Grid
__global__ void add(int start,
                    int n_samples,
                    int n_dimensions,
                    int grid_size,
                    int grid_n,
                    double *pos,
                    double *neg_f,
                    int* result_starts_counts,
                    double* square_positions,
                    double *sumQ) {

    // Finding the index of this CUDA run
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Due to rounding, there may be more CUDA runs than there are points
    // so we stop when we reach the number of points
    if (i >= n_samples) {
        return;
    }

    int point_index = i;

    // Defining some variables that will be set and read later
    double qij = 0.0;
    double dij = 0.0;
    double dij_sq = 0.0;
    double thread_sQ = 0.0;
    int point_count = 0;

    // Looping over all grid cells
    for (int k = 0; k < grid_size; k++) {
        point_count = result_starts_counts[k*2+1];

        // Not running if the grid cell is empty
        if (point_count == 0) {
            continue;
        }
        
        // Finding the multiplier for the negative force
        dij = distance(pos[point_index*2 + 0], pos[point_index*2 + 1], square_positions[k*2 + 0], square_positions[k*2 + 1]);
        dij_sq = dij * dij;

        qij = 1.0 / (1.0 + dij_sq);

        double mult = qij * qij;

        thread_sQ += qij * point_count;
        for (int ax = 0; ax < n_dimensions; ax++) {
            // Calculating the negative force for each axis
            neg_f[i * n_dimensions + ax] += point_count * mult * distance_grad(pos[point_index*2 + 0], pos[point_index*2 + 1], square_positions[k*2 + 0], square_positions[k*2 + 1], ax);
        }
    }
    
    // Adding the Q sum from this thread to the total using atomicAdd so that other threads don't interfere
    atomicAdd(sumQ, thread_sQ);
}