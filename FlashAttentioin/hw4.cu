#include <stdio.h>
#include <stdlib.h>

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o, int B, int N, int d);

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        exit(1);
    }
    input(argv[1]);
    flash_attention(Q, K, V, O, B, N, d);
    output(argv[2]);
    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    if (!file) {
        printf("Error opening input file\n");
        exit(1);
    }

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));
    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    if (!file) {
        printf("Error opening output file\n");
        exit(1);
    }
    fwrite(O, sizeof(float), B * N * d, file);
    free(Q);
    free(K);
    free(V);
    free(O);
    fclose(file);
}

__global__ void forward(
    const float* Q, const float* K, const float* V, 
    const int B, const int N, const int d,
    const float softmax_scale, float* l, float* m, float* O
) {
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    
    extern __shared__ float sram[];
    float* Q_i = sram;                           
    float* K_j = &sram[64*d];          
    float* V_j = &sram[128*d];      
    
    float4* Q_i4 = (float4*)Q_i;
    float4* K_j4 = (float4*)K_j;
    float4* V_j4 = (float4*)V_j;
    
    
    for (int j = 0; j < N/64; j++) {
        const float4* K_ptr4 = (float4*)(K + bx*N*d + j*64*d + tx*d);
        const float4* V_ptr4 = (float4*)(V + bx*N*d + j*64*d + tx*d);
        
        for (int k = 0; k < d/4; ++k) {
            K_j4[tx * (d/4) + k] = K_ptr4[k];
            V_j4[tx * (d/4) + k] = V_ptr4[k];
        }
        __syncthreads();
        
        for (int i = 0; i < N/64; i++) {
            const float4* Q_ptr4 = (float4*)(Q + bx*N*d + i*64*d + tx*d);
            for (int k = 0; k < d/4; ++k) {
                Q_i4[tx * (d/4) + k] = Q_ptr4[k];
            }
            __syncthreads();
            
            float m_i = m[bx*N + i*64 + tx];
            float l_i = l[bx*N + i*64 + tx];
            float m_ij = -INFINITY;
            float local_S[64];
            
            for (int col = 0; col < 64; ++col) {
                float qk = 0;
                const float4* q_row = (float4*)(Q_i + tx*d);
                const float4* k_row = (float4*)(K_j + col*d);
                
                // #pragma unroll 8
                for (int k = 0; k < d/4; ++k) {
                    float4 q4 = q_row[k];
                    float4 k4 = k_row[k];
                    qk += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                }
                
                qk *= softmax_scale;
                local_S[col] = qk;
                m_ij = max(m_ij, qk);
            }
            
            float l_ij = 0;
            for (int col = 0; col < 64; ++col) {
                local_S[col] = __expf(local_S[col] - m_ij);
                l_ij += local_S[col];
            }
            
            float m_new = max(m_i, m_ij);
            float l_new = __expf(m_i - m_new) * l_i + __expf(m_ij - m_new) * l_ij;
            
            // #pragma unroll 8
            for (int k = 0; k < d/4; ++k) {
                float4 o_curr4;
                float4* o_ptr4 = (float4*)(O + bx*N*d + i*64*d + tx*d);
                o_curr4 = o_ptr4[k];
                
                float4 pv4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int col = 0; col < 64; ++col) {
                    float4 v4 = ((float4*)(V_j + col*d))[k];
                    float s = local_S[col];
                    pv4.x += s * v4.x;
                    pv4.y += s * v4.y;
                    pv4.z += s * v4.z;
                    pv4.w += s * v4.w;
                }
                
                float scale = (1.0f / l_new);
                float exp_scale = __expf(m_i - m_new) * l_i;
                float exp_scale_ij = __expf(m_ij - m_new);
                
                o_ptr4[k] = make_float4(
                    scale * (exp_scale * o_curr4.x + exp_scale_ij * pv4.x),
                    scale * (exp_scale * o_curr4.y + exp_scale_ij * pv4.y),
                    scale * (exp_scale * o_curr4.z + exp_scale_ij * pv4.z),
                    scale * (exp_scale * o_curr4.w + exp_scale_ij * pv4.w)
                );
            }
            
            m[bx*N + i*64 + tx] = m_new;
            l[bx*N + i*64 + tx] = l_new;
            __syncthreads();
        }
    }
}

void flash_attention(float *q, float *k, float *v, float *o, int B, int N, int d) {

    float *d_q, *d_k, *d_v, *d_o, *d_l, *d_m;
    
    cudaMalloc(&d_q, B * N * d * sizeof(float));
    cudaMalloc(&d_k, B * N * d * sizeof(float));
    cudaMalloc(&d_v, B * N * d * sizeof(float));
    cudaMalloc(&d_o, B * N * d * sizeof(float));
    cudaMalloc(&d_l, B * N * sizeof(float));
    cudaMalloc(&d_m, B * N * sizeof(float));

    cudaMemcpy(d_q, q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, B * N * d * sizeof(float));
    cudaMemset(d_l, 0, B * N * sizeof(float));
    
    float *h_m = (float *)malloc(B * N * sizeof(float));
    for (int i = 0; i < B * N; i++) {
        h_m[i] = -INFINITY;
    }
    cudaMemcpy(d_m, h_m, B * N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_m);

    float softmax_scale = 1.0f / sqrt(d);
    const int sram_size = (192*d) * sizeof(float);
    forward<<<B, dim3(64), sram_size>>>(
        d_q, d_k, d_v, B, N, d, 
        softmax_scale, d_l, d_m, d_o
    );

    cudaMemcpy(o, d_o, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_l);
    cudaFree(d_m);
}
