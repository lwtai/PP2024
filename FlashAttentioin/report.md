# HW4 FlashAttension

#### ID: 112062698 / Name: 戴樂為

## Implementation

這次 FlashAttention 的實作是照著 paper 上面寫的 pseudo code 完成的：

**1. Set Block Size:**

原本和 paper 一樣 block size 是用動態調整的，但在後續 optimize 的時候是根據 GTX 1080 的規格，選擇寫死成 `Bc=Br=64` 這樣可以確保不管是在 `d=32` `d=64` 的情況下都不會超過 sram 容量。

```c
const int sram_size = (192*d) * sizeof(float);
forward<<<B, dim3(64), sram_size>>>(
    d_q, d_k, d_v, B, N, d, 
    softmax_scale, d_l, d_m, d_o
);
```

所有的 batch 會被同時處理，所以開了 `B` 個 block，sram 的部分留了 `Q` `K` `V` 的空間，並且因為 `Bc=Br` 所以大小都是一樣的，總共為 `192*d`，另外，thread 設定為 64 條，也就是說一根 thread 負責處理一整條的資料。

**2. Load `K` & `V`:**

根據 pseudo code 在外層迴圈要把 `K` `V` 搬到 sram，因為我的切法很簡單，所以一條 thread 需要做 `d` 次，把一整條的資料搬到 sram，其中 `bx*N*d` 是 batch 的 offset，`j*Bc*d` 是 tile 的 offset，`tx*d` 是 thread offset。 

```c
int Tc = N / Bc;
int Tr = N / Br;

for (int j = 0; j < Tc; j++) {
    
    for (int k = 0; k < d; ++k) {
        K_j[tx*d + k] = K[bx*N*d + j*Bc*d + tx*d + k];
        V_j[tx*d + k] = V[bx*N*d + j*Bc*d + tx*d + k];
    }
    __syncthreads();
    
    for (int i = 0; i < Tr; i++) { ... }
    ...
}
```

因為我用的 thread 數量比較少，這邊還有一個優化過的版本是用 `float4` 加速，利用 SIMD 的指令讓 thread 可以一次處理 4 個 value，在 `d=64` 的情況下，原本一條 thread 要跑 64 次迴圈，現在只需要跑 16 次。

```c
for (int j = 0; j < N/64; j++) {
    
    const float4* K_ptr4 = (float4*)(K + bx*N*d + j*64*d + tx*d);
    const float4* V_ptr4 = (float4*)(V + bx*N*d + j*64*d + tx*d);
    
    for (int k = 0; k < d/4; ++k) {
        K_j4[tx * (d/4) + k] = K_ptr4[k];
        V_j4[tx * (d/4) + k] = V_ptr4[k];
    }
    __syncthreads();

    for (int i = 0; i < N/64; i++) { ... }
    ...
}
```

**3. Load `Q` & Compute `QK`:**

接下來進入到內層迴圈，load `Q` 的地方完全一樣。接下來計算 `QK` 時要先進行設定一些變數，`bx*N` 是 `m_i` `l_i` 的 batch offset，`local_S[64]` 的大小必須和 `Bc` 一樣才夠放，矩陣乘法一樣一個 thread 要負責一整條。

```c
for (int i = 0; i < Tr; i++) {
    
    for (int k = 0; k < d; ++k) {
        Q_i[tx*d+k] = Q[bx*N*d + i*Br*dt + tx*d + k];
    }
    __syncthreads();

    float m_i = m[bx*N + i*Br + tx];
    float l_i = l[bx*N + i*Br + tx];
    float m_ij = -INFINITY;
    float local_S[64];

    for (int col = 0; col < Bc; ++col) {
        float qk = 0;
        for (int k = 0; k < d; ++k) {
            qk += Q_i[tx*d + k] * K_j[col*d + k];
        }
        qk *= softmax_scale;
        local_S[col] = qk;
        m_ij = max(m_ij, qk);
    }
    ...
}
```

可以看得出來在矩陣乘法的部分，一條 thread 的工作量滿大的，因此用 `float4` 可以加速不少，這邊一樣附上優化過後的版本。

```c
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

        for (int k = 0; k < d/4; ++k) {
            float4 q4 = q_row[k];
            float4 k4 = k_row[k];
            qk += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        }

        qk *= softmax_scale;
        local_S[col] = qk;
        m_ij = max(m_ij, qk);
    }
    ...
}
```
這邊有一個比較需要注意的地方，在 compile 的時候，因為 `local_S[64]` 算是一個滿大的 array，要藉由 ptxas 印出來的 log 判斷它被 compiler 放到哪裡，若是上圖的情況表示有成功塞進 register，下圖的話表示要再想辦法節省 register 的用量，因為 `local_S` 放的位置會造成很大的速度差異。
![image](https://hackmd.io/_uploads/BJyCJbdSJx.png)
![image](https://hackmd.io/_uploads/S1GA1b_Skl.png)

**4. Compute `O`:**

接著就是內層迴圈剩下的部分，主要就是計算 output 和更新 `l` `m`，indexing 的部分也是一樣的邏輯，以 `O[bx*N*d + i*Br*d + tx*d + k]` 來說的話，`bx*N*d` 是 batch offset，`i*Br*d` 是 row offset，`tx*d` 是 thread offset。

```c
for (int i = 0; i < Tr; i++) {    
    
    ...
    float l_ij = 0;
    for (int col = 0; col < Bc; ++col) {
        local_S[col] = __expf(local_S[col] - m_ij);
        l_ij += local_S[col];
    }

    float m_new = max(m_i, m_ij);
    float l_new = __expf(m_i - m_new) * l_i + __expf(m_ij - m_new) * l_ij;

    for (int k = 0; k < d; ++k) {
        float o_curr = O[bx*N*d + i*Br*d + tx*d + k];
        float pv = 0;

        for (int col = 0; col < Bc; ++col) {
            pv += local_S[col] * V_j[col*d + k];
        }

        O[bx*N*d + i*Br*d + tx*d + k] = (1.0f / l_new) * (
            __expf(m_i - m_new) * l_i * o_curr + 
            __expf(m_ij - m_new) * pv);
    }

    m[bx*N + i*Br + tx] = m_new;
    l[bx*N + i*Br + tx] = l_new;
    __syncthreads();
```
這邊一樣附上 `float4` 的版本。
```c
for (int j = 0; j < N/64; j++) {
            
    ...
    float l_ij = 0;
    for (int col = 0; col < 64; ++col) {
        local_S[col] = __expf(local_S[col] - m_ij);
        l_ij += local_S[col];
    }

    float m_new = max(m_i, m_ij);
    float l_new = __expf(m_i - m_new) * l_i + __expf(m_ij - m_new) * l_ij;

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
```
## Profiling Results

這邊的結果用 `t10` `t20` 兩種測資來其觀察差異性，其中 `t10`: `(B=10, N=2048, D=64)`，而 `t20`: `(B=100, N=2048, D=64)`，固定使用 `Bc=Br=64`。profiling 使用的是 `nvprof` 加上不同的 `--metrics` option 來得到想要的結果。

![image](https://hackmd.io/_uploads/rkcZNHdB1g.png)

從結果可以發現在相同的 `N` 及 `D` 之下，batch size 比較大的測資可以得到更好的結果，這說明了我的實作在比較大的 batch size 可以發揮比較好的效率。

接著嘗試用 `t23` `t24` 兩種測資來測試，其中 `t23`: `(B=100, N=4096, D=32)`，而 `t24`: `(B=100, N=4096, D=64)`，固定使用 `Bc=Br=64`。

![image](https://hackmd.io/_uploads/Skw6wBdr1e.png)

可以看出在幾乎所有的指標上，一樣是較多 data 的測資比較有優勢。

## Experiment & Analysis

這邊的測試都是使用 Apollo GPU 平台進行。

**1. Blocking Factor:**

這裡用 `t25` 試一下不同設定。

![image](https://hackmd.io/_uploads/Bk_Xe8uryx.png)

理論上越大的 `Bc` `Br` 就會有越好的加速，之所以選擇全部設為 64 是因為 128\*128 太大，會超出 sram 容量。而雖然 128\*64 可能放得下，但隨著 block size 加大，`local_S` 所佔的空間也要加大，所以採用 `Bc=Br` 的方式簡化實作，節省 register 使用量。

**2. Optimization:**

在優化的部分，我主要是靠 `float4` 的使用來縮短時間，所以這邊使用 `t10` `t20` `t25` `t30` 來測試一下他的效果，可以發現在越大的測資上面效果越明顯。

![image](https://hackmd.io/_uploads/HJdkuI_Hkg.png)

## Experience & conclusion

這次的作業讓我深入理解了 FlashAttention 的實作，並且 HW3 及 HW4 這兩次對 CUDA 的練習，也讓我對這種語言更加熟悉。這次的實作在 thread 配置方式上比較保守，只使用了 64 條 thread，相較於上限 1024 還有很大的操作空間，這也導致了最後 2 筆較大的測資沒辦法通過。

不過這樣的設計也帶來了一些優點，例如程式的可讀性相當好，thread 的工作分配非常簡單易懂，負責處理一整條完整的資料，這種設計也讓中間產生的計算結果可以存放在 thread 自己的 register 中。這種設計方式也讓後續的擴展變得較為容易，理論上要是有規格更好的硬體，可以直接透過增加 block size 和 thread 數量來提升效能。由於時間限制，沒有完成更多版本的實作，這是未來可以繼續改善的方向。