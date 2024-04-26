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

__device__ void negative_gradient(int i, int n_dimensions, double *pos, double *neg_f, double *sumQs) {

    double qij = 0.0;
    double dij = 0.0;
    double dij_sq = 0.0;

    for (int j = start; j < n_samples; j++) {
        if (i == j) {
            continue;
        }

        dij = distance(pos[i*2 + 0], pos[i*2 + 1], pos[j*2 + 0], pos[j*2 + 1]);
        dij_sq = dij * dij;

        qij = 1.0 / (1.0 + dij_sq);

        double mult = qij * qij;

        /*
        if (true) {
            // New Fix
            mult = qij * qij * dij;
        }
        else {
            // Old Solution
            mult = qij * qij;
        }*/

        sumQs[i] += qij;
        for (int ax = 0; ax < n_dimensions; ax++) {
            neg_f[i * n_dimensions + ax] += mult * distance_grad(pos[i*2 + 0], pos[i*2 + 1], pos[j*2 + 0], pos[j*2 + 1], ax);
            //neg_f[i * n_dimensions + ax] = distance_grad(pos[i*2 + 0], pos[i*2 + 1], pos[j*2 + 0], pos[j*2 + 1], ax);
            //neg_f[i * n_dimensions + ax] = mult;
            //neg_f[i * n_dimensions + ax] = distance(0.1, -0.1, 0.3, 0.5);
            //neg_f[i * n_dimensions + ax] = distance_grad(0.1, -0.1, 0.3, 0.5, 0);
        }
    }
}

__device__ void positive_gradient() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > n_samples) {
        return;
    }

    double qij = 0.0;
    double dij = 0.0;
    double dij_sq = 0.0;

    for (int j = start; j < n_samples; j++) {
        if (i == j) {
            continue;
        }

        dij = distance(pos[i*2 + 0], pos[i*2 + 1], pos[j*2 + 0], pos[j*2 + 1]);
        dij_sq = dij * dij;

        qij = 1.0 / (1.0 + dij_sq);

        double mult = qij * qij;

        /*
        if (true) {
            // New Fix
            mult = qij * qij * dij;
        }
        else {
            // Old Solution
            mult = qij * qij;
        }*/

        sumQs[i] += qij;
        for (int ax = 0; ax < n_dimensions; ax++) {
            neg_f[i * n_dimensions + ax] += mult * distance_grad(pos[i*2 + 0], pos[i*2 + 1], pos[j*2 + 0], pos[j*2 + 1], ax);
            //neg_f[i * n_dimensions + ax] = distance_grad(pos[i*2 + 0], pos[i*2 + 1], pos[j*2 + 0], pos[j*2 + 1], ax);
            //neg_f[i * n_dimensions + ax] = mult;
            //neg_f[i * n_dimensions + ax] = distance(0.1, -0.1, 0.3, 0.5);
            //neg_f[i * n_dimensions + ax] = distance_grad(0.1, -0.1, 0.3, 0.5, 0);
        }
    }

    /*
    for i in prange(start, n_samples, schedule='static'):
        # Init the gradient vector
        for ax in range(n_dimensions):
            pos_f[i * n_dimensions + ax] = 0.0
        # Compute the positive interaction for the nearest neighbors
        for k in range(indptr[i], indptr[i+1]):
            j = neighbors[k]
            pij = val_P[k]

            dij = distance(pos_reference[i, 0], pos_reference[i, 1], pos_reference[j, 0], pos_reference[j, 1])
            dij_sq = dij * dij

            qij = 1. / (1. + dij_sq)

            if GRAD_FIX:
                # New Fix
                mult = pij * qij * dij
            else:
                # Old solution
                mult = pij * qij

            # only compute the error when needed
            if compute_error:
                qij = qij / sum_Q
                C += pij * log(max(pij, FLOAT32_TINY) / max(qij, FLOAT32_TINY))
            for ax in range(n_dimensions):
                pos_f[i * n_dimensions + ax] += mult * distance_grad(pos_reference[i, 0], pos_reference[i, 1], pos_reference[j, 0], pos_reference[j, 1], ax)
    
    */
}

__global__ void gradient(int start, int n_samples, int n_dimensions, double *pos, double *neg_f, double *sumQs) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + start;

    // i = [start, n_samples)
    if (i > n_samples) {
        return;
    }
}