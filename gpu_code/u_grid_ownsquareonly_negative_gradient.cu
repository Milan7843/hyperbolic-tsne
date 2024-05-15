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

__global__ void add(int start,
                    int n_samples,
                    int n_dimensions,
                    int grid_size,
                    int grid_n,
                    double *pos,
                    double *neg_f,
                    int* grid_square_indices_per_point,
                    int* result_indices, 
                    int* result_starts_counts,
                    double* max_distances,
                    double* square_positions,
                    double *sumQ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    /*
    for (int k = 0; k < n_samples; k++) {
        neg_f[k] = 1.0;
    }
    atomicAdd(sumQ, 1.0);

    return;
    */
    if (i > n_samples) {
        return;
    }

    int point_index = i;

    int i_grid_index = grid_square_indices_per_point[point_index];
    int grid_x = i_grid_index % grid_n;
    int grid_y = int(i_grid_index / grid_n);

    double qij = 0.0;
    double dij = 0.0;
    double dij_sq = 0.0;
    double thread_sQ = 0.0;
    double max_square_dist = 0.0;
    double dist_to_square = 0.0;
    double theta = 0.5;
    double theta_sq = theta*theta;
    int point_count = 0;
    int range = 1;

    // Check all points in own square
    for (int m = result_starts_counts[i_grid_index*2+0]; m < result_starts_counts[i_grid_index*2+0] + result_starts_counts[i_grid_index*2+1]; m++) {
        //int j = result_indices[m];
        int j = m;
        if (j == point_index) {
            continue;
        }
        
        dij = distance(pos[point_index*2 + 0], pos[point_index*2 + 1], pos[j*2 + 0], pos[j*2 + 1]);
        dij_sq = dij * dij;

        qij = 1.0 / (1.0 + dij_sq);

        double mult = qij * qij;

        thread_sQ += qij;
        for (int ax = 0; ax < n_dimensions; ax++) {
            neg_f[point_index * n_dimensions + ax] += mult * distance_grad(pos[point_index*2 + 0], pos[point_index*2 + 1], pos[j*2 + 0], pos[j*2 + 1], ax);
            //neg_f[i * n_dimensions + ax] = distance_grad(pos[i*2 + 0], pos[i*2 + 1], pos[j*2 + 0], pos[j*2 + 1], ax);
            //neg_f[i * n_dimensions + ax] = mult;
            //neg_f[i * n_dimensions + ax] = distance(0.1, -0.1, 0.3, 0.5);
            //neg_f[i * n_dimensions + ax] = distance_grad(0.1, -0.1, 0.3, 0.5, 0);
        }
    }
    

    atomicAdd(sumQ, thread_sQ);
}