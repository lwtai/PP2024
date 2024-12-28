#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BD 64
#define TD 32

const int INF = ((1 << 30) - 1);
int *Dist, *d_Dist;
int N, n, m;

__global__ void p1(int* d_Dist, int N, int B, int cur_round) {
    int tid_x = threadIdx.y, tid_y = threadIdx.x;
    // int tid_y = threadIdx.y, tid_x = threadIdx.x;
    
    __shared__ int s_Dist[BD][BD];
    s_Dist[tid_x]   [tid_y]    = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + tid_y)];
    s_Dist[tid_x]   [tid_y+TD] = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + (tid_y+TD))];
    s_Dist[tid_x+TD][tid_y]    = d_Dist[(cur_round * B + (tid_x+TD)) * N + (cur_round * B + tid_y)];
    s_Dist[tid_x+TD][tid_y+TD] = d_Dist[(cur_round * B + (tid_x+TD)) * N + (cur_round * B + (tid_y+TD))];

    #pragma unroll
    for (int k = 0; k < B; k++) {
        __syncthreads();
        s_Dist[tid_x]   [tid_y]    = min(s_Dist[tid_x]   [tid_y],    s_Dist[tid_x][k]    + s_Dist[k][tid_y]);
        s_Dist[tid_x]   [tid_y+TD] = min(s_Dist[tid_x]   [tid_y+TD], s_Dist[tid_x][k]    + s_Dist[k][tid_y+TD]);
        s_Dist[tid_x+TD][tid_y]    = min(s_Dist[tid_x+TD][tid_y],    s_Dist[tid_x+TD][k] + s_Dist[k][tid_y]);
        s_Dist[tid_x+TD][tid_y+TD] = min(s_Dist[tid_x+TD][tid_y+TD], s_Dist[tid_x+TD][k] + s_Dist[k][tid_y+TD]);
    }
    d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + tid_y)]      = s_Dist[tid_x][tid_y];
    d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + (tid_y+TD))] = s_Dist[tid_x][tid_y+TD];
    d_Dist[(cur_round * B + (tid_x+TD)) * N + (cur_round * B + tid_y)]      = s_Dist[tid_x+TD][tid_y];
    d_Dist[(cur_round * B + (tid_x+TD)) * N + (cur_round * B + (tid_y+TD))] = s_Dist[tid_x+TD][tid_y+TD];
}

__global__ void p2(int* d_Dist, int N, int B, int cur_round) {
    if (blockIdx.x == cur_round) return;

    int tid_x = threadIdx.y, tid_y = threadIdx.x;
    // int tid_y = threadIdx.y, tid_x = threadIdx.x;
    int bid_x = blockIdx.x, bid_y = blockIdx.y;

    __shared__ int s_pivot[BD][BD];
    __shared__ int s_Dist[BD][BD];

    int row_idx = (bid_y * bid_x + (!bid_y) * cur_round) * B;
    int col_idx = ((!bid_y) * bid_x + bid_y * cur_round) * B;

    s_Dist[tid_x]   [tid_y]    = d_Dist[(row_idx + tid_x) * N      + (col_idx + tid_y)];
    s_Dist[tid_x]   [tid_y+TD] = d_Dist[(row_idx + tid_x) * N      + (col_idx + (tid_y+TD))];
    s_Dist[tid_x+TD][tid_y]    = d_Dist[(row_idx + (tid_x+TD)) * N + (col_idx + tid_y)];
    s_Dist[tid_x+TD][tid_y+TD] = d_Dist[(row_idx + (tid_x+TD)) * N + (col_idx + (tid_y+TD))];

    s_pivot[tid_x]   [tid_y]    = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + tid_y)];
    s_pivot[tid_x]   [tid_y+TD] = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + (tid_y+TD))];
    s_pivot[tid_x+TD][tid_y]    = d_Dist[(cur_round * B + (tid_x+TD)) * N + (cur_round * B + tid_y)];
    s_pivot[tid_x+TD][tid_y+TD] = d_Dist[(cur_round * B + (tid_x+TD)) * N + (cur_round * B + (tid_y+TD))];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; k++) {
        s_Dist[tid_x]   [tid_y]    = min(s_Dist[tid_x]   [tid_y],    bid_y ? s_Dist[tid_x]   [k] + s_pivot[k][tid_y]    : s_pivot[tid_x]   [k] + s_Dist[k][tid_y]);
        s_Dist[tid_x]   [tid_y+TD] = min(s_Dist[tid_x]   [tid_y+TD], bid_y ? s_Dist[tid_x]   [k] + s_pivot[k][tid_y+TD] : s_pivot[tid_x]   [k] + s_Dist[k][tid_y+TD]);
        s_Dist[tid_x+TD][tid_y]    = min(s_Dist[tid_x+TD][tid_y],    bid_y ? s_Dist[tid_x+TD][k] + s_pivot[k][tid_y]    : s_pivot[tid_x+TD][k] + s_Dist[k][tid_y]);
        s_Dist[tid_x+TD][tid_y+TD] = min(s_Dist[tid_x+TD][tid_y+TD], bid_y ? s_Dist[tid_x+TD][k] + s_pivot[k][tid_y+TD] : s_pivot[tid_x+TD][k] + s_Dist[k][tid_y+TD]);
    }
    d_Dist[(row_idx + tid_x) * N      + (col_idx + tid_y)]      = s_Dist[tid_x]   [tid_y];
    d_Dist[(row_idx + tid_x) * N      + (col_idx + (tid_y+TD))] = s_Dist[tid_x]   [tid_y+TD];
    d_Dist[(row_idx + (tid_x+TD)) * N + (col_idx + tid_y)]      = s_Dist[tid_x+TD][tid_y];
    d_Dist[(row_idx + (tid_x+TD)) * N + (col_idx + (tid_y+TD))] = s_Dist[tid_x+TD][tid_y+TD];
    return;
}

__global__ void p3(int* d_Dist, int N, int B, int cur_round) {
    if (blockIdx.x == cur_round || blockIdx.y == cur_round) return;

    int tid_x = threadIdx.y, tid_y = threadIdx.x;
    // int tid_y = threadIdx.y, tid_x = threadIdx.x;
    int bid_x = blockIdx.x, bid_y = blockIdx.y;
    
    int dist[4] = {
        d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + tid_y)],
        d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + (tid_y+TD))],
        d_Dist[(bid_x * B + (tid_x+TD)) * N + (bid_y * B + tid_y)],
        d_Dist[(bid_x * B + (tid_x+TD)) * N + (bid_y * B + (tid_y+TD))]
    };

    __shared__ int s_row[BD][BD];
    __shared__ int s_col[BD][BD];

    s_row[tid_x]   [tid_y]    = d_Dist[(bid_x * B + tid_x) * N      + (cur_round * B + tid_y)];
    s_row[tid_x]   [tid_y+TD] = d_Dist[(bid_x * B + tid_x) * N      + (cur_round * B + (tid_y+TD))];
    s_row[tid_x+TD][tid_y]    = d_Dist[(bid_x * B + (tid_x+TD)) * N + (cur_round * B + tid_y)];
    s_row[tid_x+TD][tid_y+TD] = d_Dist[(bid_x * B + (tid_x+TD)) * N + (cur_round * B + (tid_y+TD))];

    s_col[tid_x]   [tid_y]    = d_Dist[(cur_round * B + tid_x) * N      + (bid_y * B + tid_y)];
    s_col[tid_x]   [tid_y+TD] = d_Dist[(cur_round * B + tid_x) * N      + (bid_y * B + (tid_y+TD))];
    s_col[tid_x+TD][tid_y]    = d_Dist[(cur_round * B + (tid_x+TD)) * N + (bid_y * B + tid_y)];
    s_col[tid_x+TD][tid_y+TD] = d_Dist[(cur_round * B + (tid_x+TD)) * N + (bid_y * B + (tid_y+TD))];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; k++) {
        dist[0] = min(dist[0], s_row[tid_x]   [k] + s_col[k][tid_y]);
        dist[1] = min(dist[1], s_row[tid_x]   [k] + s_col[k][tid_y+TD]);
        dist[2] = min(dist[2], s_row[tid_x+TD][k] + s_col[k][tid_y]);
        dist[3] = min(dist[3], s_row[tid_x+TD][k] + s_col[k][tid_y+TD]);
    }
    d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + tid_y)]      = dist[0];
    d_Dist[(bid_x * B + tid_x) * N      + (bid_y * B + (tid_y+TD))] = dist[1];
    d_Dist[(bid_x * B + (tid_x+TD)) * N + (bid_y * B + tid_y)]      = dist[2];
    d_Dist[(bid_x * B + (tid_x+TD)) * N + (bid_y * B + (tid_y+TD))] = dist[3];
    return;
}

void input(char* infile) {
    FILE* fp_in = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, fp_in);
    fread(&m, sizeof(int), 1, fp_in);
    N = ((n + BD - 1) / BD) * BD;
    Dist = (int*)malloc(N * N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Dist[i * N + j] = (i == j && i < n) ? 0 : INF;
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
            if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
        }
        fwrite(&Dist[i * N], sizeof(int), n, fp_out);
    }
    fclose(fp_out);
    return;
}

void block_FW() {
    size_t size = N * N * sizeof(int);
    cudaMalloc((void**)&d_Dist, size);
    cudaMemcpy(d_Dist, Dist, size, cudaMemcpyHostToDevice);
    int R = N / BD;
    dim3 thread_dim(TD, TD);
    for (int r = 0; r < R; ++r) {
        p1<<<1, thread_dim>>>(d_Dist, N, BD, r);
        p2<<<dim3(R, 2), thread_dim>>>(d_Dist, N, BD, r);
        p3<<<dim3(R, R), thread_dim>>>(d_Dist, N, BD, r);
    }
    cudaMemcpy(Dist, d_Dist, size, cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);
    return;
}

int main(int argc, char* argv[]) {
    if (argc != 3) return 1;

    input(argv[1]);
    block_FW();
    output(argv[2]);
    
    free(Dist);
    return 0;
}