
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cassert>
#include <chrono>
#include "read_mat_files.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4    // number of warps in bin1 kernel block
#define THREADS_BIN1 (WARPS_PER_BLOCK * WARP_SIZE)
#define MAX_ITERS 10000
#define TOL 1e-11f

#define PRECISION float

// Matrix times vector
void mat_times_vec(const std::vector<PRECISION>& sub_A_values, const std::vector<int>& sub_A_columns, const std::vector<int>& sub_A_rowptrs, const std::vector<PRECISION>& v, std::vector<PRECISION>& sub_result)
{
    PRECISION dot_prod;
    int start = 0, end = 0;

    for (int i = 0; i < sub_A_rowptrs.size() - 1; i++) { // Per row iteration
        dot_prod = 0;
        start = sub_A_rowptrs[i] - sub_A_rowptrs.front();
        end = sub_A_rowptrs[i + 1] - sub_A_rowptrs.front();
        for (; start < end; start++) {
            dot_prod += sub_A_values[start] * v[sub_A_columns[start]];
        }
        sub_result[i] = dot_prod;
    }

}

// --- Device Kernel Utilities ---
__global__ void vector_axpy(int n, PRECISION* y, const PRECISION* x, PRECISION alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

__global__ void vector_scale_add(int n, PRECISION* p, const PRECISION* r, PRECISION beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + beta * p[i];
}

__global__ void vector_dot(const PRECISION* a, const PRECISION* b, PRECISION* result, int n) {
    __shared__ PRECISION buf[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    PRECISION tmp = (idx < n) ? a[idx] * b[idx] : 0.0f;
    buf[tid] = tmp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, buf[0]);
}

// --- Row-Binned SpMV Kernels ---
__global__ void spmv_bin0_kernel(const int* row_ids, int num_rows,
    const int* row_ptr, const int* col_ind,
    const PRECISION* val, const PRECISION* x, PRECISION* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    int row = row_ids[idx];
    PRECISION sum = 0.0f;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j)
        sum += val[j] * x[col_ind[j]];
    y[row] = sum;
}

__global__ void spmv_bin1_kernel(const int* row_ids, int num_rows,
    const int* row_ptr, const int* col_ind,
    const PRECISION* val, const PRECISION* x, PRECISION* y) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id >= num_rows) return;
    int row = row_ids[warp_id];
    PRECISION sum = 0.0f;
    for (int j = row_ptr[row] + lane; j < row_ptr[row + 1]; j += WARP_SIZE)
        sum += val[j] * x[col_ind[j]];
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
}

// --- Conjugate Gradient Solver (GPU) ---
void conjugate_gradient_gpu(int n,
    const int* d_row_ptr, const int* d_col_ind, const PRECISION* d_val,
    PRECISION* d_x, const PRECISION* d_b,
    const int* d_bin0_rows, int bin0_size,
    const int* d_bin1_rows, int bin1_size) {
    // Allocate workspace: r, p, Ap
    PRECISION* d_r, * d_p, * d_Ap;
    cudaMalloc(&d_r, n * sizeof(PRECISION));
    cudaMalloc(&d_p, n * sizeof(PRECISION));
    cudaMalloc(&d_Ap, n * sizeof(PRECISION));

    // Allocate scalars
    PRECISION* d_rr_old, * d_dot_pAp;
    cudaMalloc(&d_rr_old, sizeof(PRECISION));
    cudaMalloc(&d_dot_pAp, sizeof(PRECISION));


    PRECISION dot_pAp;
    PRECISION rr_old;
    PRECISION alpha;
    PRECISION rr_new;
    PRECISION beta;

    // Create streams
    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    cudaMemcpyAsync(d_p, d_b, n * sizeof(PRECISION), cudaMemcpyDeviceToDevice, s0);

    cudaMemcpyAsync(d_r, d_b, n * sizeof(PRECISION), cudaMemcpyDeviceToDevice);

    int threads = 256;
    int vec_blocks = (n + threads - 1) / threads;
    int bin0_blocks = (bin0_size + threads - 1) / threads;
    int bin1_blocks = (bin1_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // rr_old = r·r
    cudaMemsetAsync(d_rr_old, 0, sizeof(PRECISION));
    vector_dot << <vec_blocks, threads >> > (d_r, d_r, d_rr_old, n);

    cudaDeviceSynchronize();

    cudaMemcpyAsync(&rr_old, d_rr_old, sizeof(PRECISION), cudaMemcpyDeviceToHost);
    
    int k = 0;
    // Main loop
    for (; k < MAX_ITERS; ++k) {
        // alpha = (r·r) / (p·(A*p))
        // compute Ap = A*p
        spmv_bin0_kernel << <bin0_blocks, threads, 0, s1 >> > (d_bin0_rows, bin0_size, d_row_ptr, d_col_ind, d_val, d_p, d_Ap);
        spmv_bin1_kernel << <bin1_blocks, THREADS_BIN1, 0, s0 >> > (d_bin1_rows, bin1_size, d_row_ptr, d_col_ind, d_val, d_p, d_Ap);
        cudaMemsetAsync(d_dot_pAp, 0, sizeof(PRECISION)); // dot_pAp

        cudaDeviceSynchronize();

        vector_dot << <vec_blocks, threads >> > (d_p, d_Ap, d_dot_pAp, n);

        cudaDeviceSynchronize();

        cudaMemcpy(&dot_pAp, d_dot_pAp, sizeof(PRECISION), cudaMemcpyDeviceToHost);

        alpha = rr_old / dot_pAp;


        // Check convergence sqrt(r·r) < TOL
        cudaMemsetAsync(d_rr_old, 0, sizeof(PRECISION), s0);

        // r = r - alpha * Ap
        vector_axpy << <vec_blocks, threads, 0, s0 >> > (n, d_r, d_Ap, -alpha);

        // x = x + alpha * p
        vector_axpy << <vec_blocks, threads, 0, s1 >> > (n, d_x, d_p, alpha);

        cudaStreamSynchronize(s0);

        vector_dot << <vec_blocks, threads, 256, s0 >> > (d_r, d_r, d_rr_old, n);

        cudaStreamSynchronize(s0);

        cudaMemcpy(&rr_new, d_rr_old, sizeof(PRECISION), cudaMemcpyDeviceToHost);
        if (sqrt(rr_new) < TOL) break;

        // beta = (r·r) / (rr_old)
        beta = rr_new / rr_old;

        // p = r + beta * p
        vector_scale_add << <vec_blocks, threads >> > (n, d_p, d_r, beta);

        // r_old = r·r
        rr_old = rr_new;

        cudaDeviceSynchronize();
    }

    // Destroy streams
    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);

    // Cleanup
    cudaFree(d_r); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_rr_old); cudaFree(d_dot_pAp);

    // Print iter
    std::cout << "\n\nGPU - ITER: " << k << std::endl;
}

// --- CPU-Side Setup and Main ---
void prepare_and_solve(const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<PRECISION>& val,
    const std::vector<PRECISION>& b) {
    int n = b.size();
    std::vector<PRECISION> x(n, 0.0f);
    // Row binning
    std::vector<int> bin0_rows, bin1_rows;
    for (int i = 0; i < n; ++i) {
        int nnz = row_ptr[i + 1] - row_ptr[i];
       if (nnz <= 16) bin0_rows.push_back(i);
       else bin1_rows.push_back(i);
    }

    // Print bin rows
    std::cout << "\n\nGPU - bin0_rows: " << bin0_rows.size() << std::endl;
    std::cout << "\n\nGPU - bin1_rows: " << bin1_rows.size() << std::endl;

    // Device allocations
    int* d_row_ptr, * d_col_ind, * d_bin0, * d_bin1;
    PRECISION* d_val, * d_b, * d_x;
    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, col_ind.size() * sizeof(int));
    cudaMalloc(&d_val, val.size() * sizeof(PRECISION));
    cudaMalloc(&d_b, b.size() * sizeof(PRECISION));
    cudaMalloc(&d_x, x.size() * sizeof(PRECISION));
    cudaMalloc(&d_bin0, bin0_rows.size() * sizeof(int));
    cudaMalloc(&d_bin1, bin1_rows.size() * sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr.data(),
        (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind.data(),
        col_ind.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val.data(), val.size() * sizeof(PRECISION), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b.size() * sizeof(PRECISION), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, x.size() * sizeof(PRECISION));
    cudaMemcpy(d_bin0, bin0_rows.data(),
        bin0_rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin1, bin1_rows.data(),
        bin1_rows.size() * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA event timers initialization
    cudaEvent_t startCUDA, stopCUDA;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);
    cudaEventRecord(startCUDA);

    // Solve
    conjugate_gradient_gpu(n, d_row_ptr, d_col_ind, d_val,
        d_x, d_b,
        d_bin0, bin0_rows.size(),
        d_bin1, bin1_rows.size());

    // CUDA timing events stop and synchronize
    cudaEventRecord(stopCUDA);
    cudaEventSynchronize(stopCUDA);

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startCUDA, stopCUDA);

    std::cout << "GPU Execution time: " << milliseconds << " ms.\n\n";

    // Fetch and print
    cudaMemcpy(x.data(), d_x, x.size() * sizeof(PRECISION), cudaMemcpyDeviceToHost);
    std::cout << "Solution x[:10]: ";
    for (int i = 0; i < std::min(10, n); ++i) std::cout << x[i] << " ";
    std::cout << std::endl;

    //------------------------- Verification Test ----------------------------------------------------------------------
    // we will compare the A*x result to the right hand side
    std::vector<PRECISION> A_times_x(x.size());
    std::vector<PRECISION> error(x.size());
    mat_times_vec(val, col_ind, row_ptr, x, A_times_x);

    for (int i = 0; i < x.size(); i++) {
        error[i] = abs(A_times_x[i] - b[i]);
    }

    auto max_error = *std::max_element(error.begin(), error.end());

    if (max_error > 2e-2)
        std::cout << "\n\nSEQ Error in solution is " << max_error << " and larger than " << 2e-2 << std::endl;
    // std::cout << "INFO - SEQ Error in solution is " << *std::max_element(error.begin(), error.end()) << std::endl;

    // Free
    cudaFree(d_row_ptr); cudaFree(d_col_ind);
    cudaFree(d_val); cudaFree(d_b); cudaFree(d_x);
    cudaFree(d_bin0); cudaFree(d_bin1);
}

// Linear combination of vectors (CPU)
void vector_axpy_cpu(PRECISION a, std::vector<PRECISION>& x, PRECISION b, const std::vector<PRECISION>& p)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = a * x[i] + b * p[i];
}

// Dot product (CPU)
PRECISION dot_product_cpu(const std::vector<PRECISION>& u, const std::vector<PRECISION>& v)
{
    int length = u.size();
    PRECISION prod = 0.0;
    for (int i = 0; i < length; i++) {
        prod += u[i] * v[i];
    }
    return prod;
}

// --- CPU Implementation of Sequential CG ---
void solveCG_cpu(const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<PRECISION>& val,
    const std::vector<PRECISION>& b) {

    // Start time
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<PRECISION> x(b.size(), 0.0);
    std::vector<PRECISION> p(b);
    std::vector<PRECISION> r(b);
    std::vector<PRECISION> Ap(b.size(), 0.0);

    PRECISION dot_r = dot_product_cpu(r, r);
    PRECISION dot_r_old = dot_r;

    PRECISION alpha, beta;

    int iter = 0;

    for (; iter < MAX_ITERS; iter++)
    {
        mat_times_vec(val, col_ind, row_ptr, p, Ap);
        alpha = dot_r_old / dot_product_cpu(Ap, p);

        vector_axpy_cpu(1.0, x, alpha, p);
        vector_axpy_cpu(1.0, r, -alpha, Ap);
        dot_r = dot_product_cpu(r, r);

        if (sqrt(dot_r) < TOL) {
            break;
        }
        beta = dot_r / dot_r_old;
        vector_axpy_cpu(beta, p, 1.0, r);
        dot_r_old = dot_r;
    }

    // End time
    auto end = std::chrono::high_resolution_clock::now();

    //------------------------- Verification Test ----------------------------------------------------------------------
    // we will compare the A*x result to the right hand side
    std::vector<PRECISION> A_times_x(x.size());
    std::vector<PRECISION> error(x.size());
    mat_times_vec(val, col_ind, row_ptr, x, A_times_x);

    for (int i = 0; i < x.size(); i++) {
        error[i] = abs(A_times_x[i] - b[i]);
    }

    auto max_error = *std::max_element(error.begin(), error.end());

    if (max_error > 2e-2)
        std::cout << "\n\nCPU - SEQ Error in solution is " << max_error << " and larger than " << 2e-2 << std::endl;
    // std::cout << "INFO - SEQ Error in solution is " << *std::max_element(error.begin(), error.end()) << std::endl;

    // Calculate duration
    std::chrono::duration<double> duration = end - start;

    // Output results
    std::cout << "\nCPU Execution time: " << duration.count() * 1e3 << " ms." << std::endl;
}

// Main Function
int main(int argc, char** argv) {

    std::string current_exec_name = argv[0];
    std::vector<std::string> all_args;

    if (argc > 1) {
        all_args.assign(argv + 1, argv + argc);
    }

    std::vector<PRECISION> val;
    std::vector<int> col_ind;
    std::vector<int> row_ptr;

    std::vector<PRECISION> b;
    //std::string dataset_name = all_args[0];
    std::string dataset_name = "Dubcova2_6";
    // Read the matrix and right hand side vector
    read_val_col_rowptrs_from_txts(val, col_ind, row_ptr, dataset_name);
    read_b_from_txt(b, dataset_name);

    prepare_and_solve(row_ptr, col_ind, val, b);
    solveCG_cpu(row_ptr, col_ind, val, b);

    return 0;
}
