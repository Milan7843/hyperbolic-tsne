__device__ void poincare_to_klein(double px, double py, double* kx, double* ky) {
    double denominator = 1.0 + px*px + py*py;
    kx[0] = 2.0 * px / denominator;
    ky[0] = 2.0 * py / denominator;
    return;
}

__device__ double gamma(double v0, double v1) {
    double norm_v_sq = v0 * v0 + v1 * v1;
    return 1.0 / sqrt(1.0 - norm_v_sq);
}

__global__ void add(int n_samples,
                    int* grid_square_indices_per_point,
                    double* points,
                    double* mids,
                    double* gs) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_samples) {
        return;
    }

    int grid_index = grid_square_indices_per_point[i];
    double point_klein_x, point_klein_y;
    poincare_to_klein(points[i*2+ 0], points[i*2+ 1], &point_klein_x, &point_klein_y);
    double g = gamma(point_klein_x, point_klein_y);
    
    atomicAdd(&mids[grid_index*2+0], g * point_klein_x);
    atomicAdd(&mids[grid_index*2+1], g * point_klein_y);
    atomicAdd(&gs[grid_index], g);
}