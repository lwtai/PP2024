#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

const int INF = ((1 << 30) - 1);
int *Dist, *d_Dist[2];
int N, n, m;

__global__ void p1(int* d_Dist, int N, int B, int cur_round) {
    int tid_x = threadIdx.y, tid_y = threadIdx.x;
    
    __shared__ int s_Dist[64][64];
    s_Dist[tid_x]   [tid_y]    = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + tid_y)];
    s_Dist[tid_x]   [tid_y+32] = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + (tid_y+32))];
    s_Dist[tid_x+32][tid_y]    = d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + tid_y)];
    s_Dist[tid_x+32][tid_y+32] = d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + (tid_y+32))];

    #pragma unroll
    for (int k = 0; k < B; k++) {
        __syncthreads();
        s_Dist[tid_x]   [tid_y]    = min(s_Dist[tid_x]   [tid_y],    s_Dist[tid_x][k]    + s_Dist[k][tid_y]);
        s_Dist[tid_x]   [tid_y+32] = min(s_Dist[tid_x]   [tid_y+32], s_Dist[tid_x][k]    + s_Dist[k][tid_y+32]);
        s_Dist[tid_x+32][tid_y]    = min(s_Dist[tid_x+32][tid_y],    s_Dist[tid_x+32][k] + s_Dist[k][tid_y]);
        s_Dist[tid_x+32][tid_y+32] = min(s_Dist[tid_x+32][tid_y+32], s_Dist[tid_x+32][k] + s_Dist[k][tid_y+32]);
    }
    d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + tid_y)]      = s_Dist[tid_x][tid_y];
    d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + (tid_y+32))] = s_Dist[tid_x][tid_y+32];
    d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + tid_y)]      = s_Dist[tid_x+32][tid_y];
    d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + (tid_y+32))] = s_Dist[tid_x+32][tid_y+32];
}

__global__ void p2(int* d_Dist, int N, int B, int cur_round) {
    if (blockIdx.x == cur_round) return;
    int tid_x = threadIdx.y, tid_y = threadIdx.x;
    int bid_x = blockIdx.x, bid_y = blockIdx.y;

    __shared__ int s_pivot[64][64];
    __shared__ int s_Dist[64][64];
    int row_idx = (bid_y * bid_x + (!bid_y) * cur_round) * B;
    int col_idx = ((!bid_y) * bid_x + bid_y * cur_round) * B;
    s_Dist[tid_x]   [tid_y]    = d_Dist[(row_idx + tid_x) * N      + (col_idx + tid_y)];
    s_Dist[tid_x]   [tid_y+32] = d_Dist[(row_idx + tid_x) * N      + (col_idx + (tid_y+32))];
    s_Dist[tid_x+32][tid_y]    = d_Dist[(row_idx + (tid_x+32)) * N + (col_idx + tid_y)];
    s_Dist[tid_x+32][tid_y+32] = d_Dist[(row_idx + (tid_x+32)) * N + (col_idx + (tid_y+32))];

    s_pivot[tid_x]   [tid_y]    = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + tid_y)];
    s_pivot[tid_x]   [tid_y+32] = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + (tid_y+32))];
    s_pivot[tid_x+32][tid_y]    = d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + tid_y)];
    s_pivot[tid_x+32][tid_y+32] = d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + (tid_y+32))];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; k++) {
        s_Dist[tid_x]   [tid_y]    = min(s_Dist[tid_x]   [tid_y],    bid_y ? s_Dist[tid_x]   [k] + s_pivot[k][tid_y]    : s_pivot[tid_x]   [k] + s_Dist[k][tid_y]);
        s_Dist[tid_x]   [tid_y+32] = min(s_Dist[tid_x]   [tid_y+32], bid_y ? s_Dist[tid_x]   [k] + s_pivot[k][tid_y+32] : s_pivot[tid_x]   [k] + s_Dist[k][tid_y+32]);
        s_Dist[tid_x+32][tid_y]    = min(s_Dist[tid_x+32][tid_y],    bid_y ? s_Dist[tid_x+32][k] + s_pivot[k][tid_y]    : s_pivot[tid_x+32][k] + s_Dist[k][tid_y]);
        s_Dist[tid_x+32][tid_y+32] = min(s_Dist[tid_x+32][tid_y+32], bid_y ? s_Dist[tid_x+32][k] + s_pivot[k][tid_y+32] : s_pivot[tid_x+32][k] + s_Dist[k][tid_y+32]);
    }
    d_Dist[(row_idx + tid_x) * N      + (col_idx + tid_y)]      = s_Dist[tid_x]   [tid_y];
    d_Dist[(row_idx + tid_x) * N      + (col_idx + (tid_y+32))] = s_Dist[tid_x]   [tid_y+32];
    d_Dist[(row_idx + (tid_x+32)) * N + (col_idx + tid_y)]      = s_Dist[tid_x+32][tid_y];
    d_Dist[(row_idx + (tid_x+32)) * N + (col_idx + (tid_y+32))] = s_Dist[tid_x+32][tid_y+32];
    return;
}

__global__ void p3(int* d_Dist, int N, int B, int cur_round, int start, int size) {
    if (blockIdx.x == cur_round || blockIdx.y == cur_round) return;
    if (blockIdx.x < start || blockIdx.x >= start+size) return;

    int tid_x = threadIdx.y, tid_y = threadIdx.x;
    int bid_x = blockIdx.x, bid_y = blockIdx.y;
    int dist[4] = {
        d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + tid_y)],
        d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + (tid_y+32))],
        d_Dist[(bid_x * B + (tid_x+32)) * N + (bid_y * B + tid_y)],
        d_Dist[(bid_x * B + (tid_x+32)) * N + (bid_y * B + (tid_y+32))]
    };

    __shared__ int s_row[64][64];
    __shared__ int s_col[64][64];
    s_row[tid_x]   [tid_y]    = d_Dist[(bid_x * B + tid_x) * N      + (cur_round * B + tid_y)];
    s_row[tid_x]   [tid_y+32] = d_Dist[(bid_x * B + tid_x) * N      + (cur_round * B + (tid_y+32))];
    s_row[tid_x+32][tid_y]    = d_Dist[(bid_x * B + (tid_x+32)) * N + (cur_round * B + tid_y)];
    s_row[tid_x+32][tid_y+32] = d_Dist[(bid_x * B + (tid_x+32)) * N + (cur_round * B + (tid_y+32))];

    s_col[tid_x]   [tid_y]    = d_Dist[(cur_round * B + tid_x) * N      + (bid_y * B + tid_y)];
    s_col[tid_x]   [tid_y+32] = d_Dist[(cur_round * B + tid_x) * N      + (bid_y * B + (tid_y+32))];
    s_col[tid_x+32][tid_y]    = d_Dist[(cur_round * B + (tid_x+32)) * N + (bid_y * B + tid_y)];
    s_col[tid_x+32][tid_y+32] = d_Dist[(cur_round * B + (tid_x+32)) * N + (bid_y * B + (tid_y+32))];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; k++) {
        dist[0] = min(dist[0], s_row[tid_x]   [k] + s_col[k][tid_y]);
        dist[1] = min(dist[1], s_row[tid_x]   [k] + s_col[k][tid_y+32]);
        dist[2] = min(dist[2], s_row[tid_x+32][k] + s_col[k][tid_y]);
        dist[3] = min(dist[3], s_row[tid_x+32][k] + s_col[k][tid_y+32]);
    }
    d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + tid_y)]      = dist[0];
    d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + (tid_y+32))] = dist[1];
    d_Dist[(bid_x * B + (tid_x+32)) * N + (bid_y * B + tid_y)]      = dist[2];
    d_Dist[(bid_x * B + (tid_x+32)) * N + (bid_y * B + (tid_y+32))] = dist[3];
    return;
}

void input(char* infile) {
    FILE* fp_in = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, fp_in);
    fread(&m, sizeof(int), 1, fp_in);
    
    N = ((n + 64 - 1) / 64) * 64;
    Dist = (int*)malloc(N * N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Dist[i*N+j] = (i == j && i < n) ? 0 : INF;
        }
    }
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, fp_in);
        Dist[pair[0] * N + pair[1]] = pair[2];
    }
    fclose(fp_in);
    return;
}

void output(char* outFileName) {
    FILE* fp_out = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i*n+j] >= INF) Dist[i*n+j] = INF;
        }
        fwrite(&Dist[i*N], sizeof(int), n, fp_out);
    }
    fclose(fp_out);
    return;
}

void block_FW() {
#pragma omp parallel num_threads(2)
{
    int id = omp_get_thread_num();
    int R = N / 64;

    const int start = (R/2) * id;
    const int size = (R/2) + (R%2) * id;
    const size_t chunk_bytes = size*64*N*sizeof(int);
    const size_t offset = start*64*N;
    
    cudaSetDevice(id);
    cudaMalloc((void**)&d_Dist[id], N*N*sizeof(int));
    cudaMemcpy(d_Dist[id] + offset, Dist + offset, chunk_bytes, cudaMemcpyHostToDevice);

#pragma omp barrier
    
    dim3 thread_dim(32, 32);
    for (int r = 0; r < R; ++r) {
        size_t transfer_size = 0;
        if (r >= start) {
            if (r < start + size) {
                transfer_size = 64*N*sizeof(int);
            }
        }
        cudaMemcpyPeer(d_Dist[!id]+(r*64*N), !id, d_Dist[id]+(r*64*N), id, transfer_size);

#pragma omp barrier
        
        p1<<<1, thread_dim>>>(d_Dist[id], N, 64, r);
        p2<<<dim3(R, 2), thread_dim>>>(d_Dist[id], N, 64, r);
        p3<<<dim3(R, R), thread_dim>>>(d_Dist[id], N, 64, r, start, size);
    }
    cudaMemcpy(Dist + offset, d_Dist[id] + offset, chunk_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_Dist[id]);
}
}

int main(int argc, char* argv[]) {
    if (argc != 3) return 1;
    input(argv[1]);
    block_FW();
    output(argv[2]);
    free(Dist);
    return 0;
}