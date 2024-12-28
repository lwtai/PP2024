---
title: HW3 All-Pairs Shortest Pair

---


# HW3 All-Pairs Shortest Pair

#### ID: 112062698 / Name: 戴樂為

## Implementation

### Q&A
(a) 3 個版本都是用 Blocked Floyd-Warshall的演算法。
(b) 將整個矩陣切成邊長為 64 的正方形小塊。
\(c) HW3-2/3 中採用的是同樣的設定: 使用最多的 32\*32 的 thread，以及 64\*64 的 block 大小已經接近 GPU 記憶體的上限，這樣的設定組合讓 GPU 有最佳的使用率，並且 4 的倍數關係讓實作比較方便。
(d) HW3-3 中兩張卡溝通的地方是採用 `cudaMemcpyPeer` 來交換 pivot row。

### 1. HW3-1

作業 3-1 只要修改 sequential 版本，將內層迴圈加上 omp 就可以得到很顯著的優化，也可以跑過所有的測資，由此可見 omp 的確是一個很方便的解法，雖然它比起 pthread 常常會慢一些。

```c
#pragma omp parallel for schedule(dynamic)
for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
        if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
            Dist[i][j] = Dist[i][k] + Dist[k][j];
        }
    }
}
```

### 2. HW3-2
**V1**: 最初的版本是將矩陣切成 32\*32 並且 load 到 shared memory 處理，thread 的數量也是32\*32，`p2` kernel 是切成兩份的，但這種設定沒辦法跑過比較大的測資。
```c
for (int r = 0; r < R; ++r) {
    p1<<<1, dim3(B, B)>>>(d_Dist, n, N, B, r);
    p2_row<<<dim3(R, 1), dim3(B, B)>>>(d_Dist, n, N, B, r);
    p2_col<<<dim3(R, 1), dim3(B, B)>>>(d_Dist, n, N, B, r);
    p3<<<dim3(R, R), dim3(B, B)>>>(d_Dist, n, N, B, r);
}
```

**V2**: 在這個版本做了很多優化，首先我把 `p2_row` `p2_col` 簡化成一個，另外利用將輸入矩陣 padding 的方式刪掉了很多 kernel 裡面的 `if` statement，另外也增加 block size 到 64\*64，一個 thread 做 4 個 pixels。
```c
dim3 thread_dim(32, 32);
for (int r = 0; r < R; ++r) {
    p1<<<1, thread_dim>>>(d_Dist, N, B, r);
    p2<<<dim3(R, 2), thread_dim>>>(d_Dist, N, B, r);
    p3<<<dim3(R, R), thread_dim>>>(d_Dist, N, B, r);
}
```
除此之外，也觀察到 coalesced memory access 與否表現差異巨大，並且考慮到 bank conflict 的問題，所以讓每個 thread 抓取不相鄰的 pixel 來處理。
```c
__shared__ int s_Dist[BLOCK_DIM][BLOCK_DIM];
s_Dist[tid_x]   [tid_y]    = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + tid_y)];
s_Dist[tid_x]   [tid_y+32] = d_Dist[(cur_round * B + tid_x) * N      + (cur_round * B + (tid_y+32))];
s_Dist[tid_x+32][tid_y]    = d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + tid_y)];
s_Dist[tid_x+32][tid_y+32] = d_Dist[(cur_round * B + (tid_x+32)) * N + (cur_round * B + (tid_y+32))];
```

### 2. HW3-3

**V1**: 一開始直接拿 HW2-2 中，`V1`分開的 `p2` 來用，在 phase 2 的時候將 `p2_col` `p2_row` 分散到兩張 GPU 上面做，因為這樣不用寫新的 code，沒想到跑的時間變很慢，雖然答案是正確的，但是時間過不了門檻。
```c
void block_FW() {
#pragma omp parallel num_threads(2)
{
    size_t size = N * N * sizeof(int);
    int id = omp_get_thread_num();
    cudaSetDevice(id);
    cudaMalloc((void**)&d_Dist[id], size);

#pragma omp barrier
    cudaMemcpy(d_Dist[id], Dist, size, cudaMemcpyHostToDevice);
    int R = N / BLOCK_DIM;
    dim3 thread_dim(32, 32);
    for (int r = 0; r < R; ++r) {
        p1<<<1, thread_dim>>>(d_Dist[id], N, BLOCK_DIM, r);
        if (id == 0) {
            p2_row<<<dim3(R, 1), thread_dim>>>(d_Dist[id], N, BLOCK_DIM, r);
        } else {
            p2_col<<<dim3(R, 1), thread_dim>>>(d_Dist[id], N, BLOCK_DIM, r);
        }
#pragma omp barrier
        if (id == 1) {
            cudaMemcpyPeer(d_Dist[id]+(r*64*N), id, d_Dist[!id]+(r*64*N), id, 64*N*sizeof(int));
            p3<<<dim3(R, R), thread_dim>>>(d_Dist[id], N, BLOCK_DIM, r);
            cudaMemcpyPeer(d_Dist[!id], !id, d_Dist[id], id, N*N*sizeof(int));
        }
#pragma omp barrier
    }
    if (id == 0) {
        cudaMemcpy(Dist, d_Dist[id], size, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_Dist[id]);
}
    return;
}
```
表現很差的原因也很明顯，每個 round 裡面要做兩次 `cudaMemcpyPeer`，並且傳輸的資料量也不小，另外就是最大的 `p3` 並沒有被分散。

**V2**: 這個版本中則是選擇分散 `p3`，只交換有 pivot block 的那條 row，這樣的好處是每個 round 只需要呼叫 `cudaMemcpyPeer` 一次，交換的資料量不會隨著測資變大。

```c
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
```

## Profiling Results

這邊的結果用 `c10.1` `c20.1` 兩種測資來其觀察差異性，其中 `c10.1` 的資料量為 512\*512，而 `c20.1` 的資料量為 5056\*5056，固定使用 64\*64 的 block size。profiling 使用的是 `nvprof` 加上不同的 `--metrics` option 來得到想要的結果。

![image](https://hackmd.io/_uploads/HylUa4FYE1e.png)
![image](https://hackmd.io/_uploads/r1hySYKNJl.png)

**`P1` Kernel**

在 phase 1 中對於兩筆大小不同的測資沒有明顯變化，因為要做的範圍只有一個 block，另外 occupancy 大約都是 50% 左右，其他指標相對於另外兩個 kernel 也都偏低，這是由於資料數量較少的關係。

**`P2` Kernel**

相較於 phase 1 只做一個 block，phase 2 的計算量受到測資的影響程度比較大，在 `c10.1` `c20.1` 中，occupancy 從 50% 增加到 96%，SM efficiency 從 58% 增加到 92%，其他的指標也有類似的趨勢，這說明一直到 `c20.1` 的大小才有比較好的 scalability。

**`P3` Kernel**

Phase 3 作為計算量最大的一個 kernel，可以觀察到在 `c10.1` 中的使用率就已經有滿高的，而到了 `c20.1` 中其 occupancy: 94%, SM efficiency: 99.6% 說明這樣的 `p3` kernel 對於這樣的資料大小，可以讓 GPU 跑到幾乎滿載，發揮整張卡的算力。

## Experiment & Analysis

這邊的測試都是使用 Apollo 平台進行。

### 1. Blocking Factor (HW3-2)

使用 `c10.1` 對 `p3` kernel 進行測試，記錄不同 blocking factor 對各種指標產生的影響。

**GOPS**: 先用 `nvprof` 得到執行時間，integer 指令數量則是用 `--metrics inst_integer` 的結果，最後將 `total instruction count` / `total executioni time`。
**Global**: 計算 `--metrics gld_throughput,gst_throughput` 得到的結果總和。
**Shared**: 計算 `--metrics shared_load_throughput,shared_store_throughput` 結果總和。

![image](https://hackmd.io/_uploads/r1A0ac2VJg.png)

由 3 種 metrics 的圖表可以看出，除了 global memory 之外，大致和 blocking factor 呈現正相關，這也代表在 global memory 的存取方面還有一些潛在的優化空間。

### 2. Optimization (HW3-2)

這邊使用 `p11k1` 來測試各種 optimization 的效果。

![image](https://hackmd.io/_uploads/H1t4RUT41l.png)

由圖表可以發現記憶體 access pattern 以及 shared memory 的使用對表現有很大的影響，再次印證了 GPU的效能幾乎都是 memory bound 的這個說法。

### 3. Weak Scalability (HW3-3)
這邊用 HW3-3 提供的 `c05.1` 進行測試，可以發現在比較小的測資上，使用兩張 GPU 不一定具有優勢，例如在這筆測資上兩種設定是時間是差不多的。

![image](https://hackmd.io/_uploads/r1TR1dTNkg.png)

### 4. Time Distribution (HW3-2)
接下來使用 `p15k1` `p20k1` `p25k1` `p30k1`，測試在不同的測資大小下 time distributioin 對應的變化，畫成以下圖表。

![image](https://hackmd.io/_uploads/rkRcr_aN1e.png)

隨著 data size 的增加，所花的總體時間也隨之增加，但可以觀察到仍以計算時間為主，這也符合 Floyd-Warshall 時間複雜度的特性。而 Memory Copy 所花的時間就大致和 n 的規模呈現線性增長，所佔百分比也越來越低。

## Experiment on AMD GPU

**HW3-2**: 我利用一樣的測資 `p15k1` `p20k1` `p25k1` `p30k1` 做了 time distribution 的實驗，發現分佈非常類似，但是在計算所花的時間減少許多。

![image](https://hackmd.io/_uploads/SJpJ7YaEkl.png)

**HW3-3**: 多卡在跑 judge 的時候有出現錯誤，但速度上快很多，尤其在 `c07.1`的這筆測資上。

![image](https://hackmd.io/_uploads/HJYHNFaNJg.png)

由以上的結果我們可以發現，GPU 運作的 pipeline 其實都是十分類似的，甚至在 single GPU 的測試中只需要透過 `hipify-clang` 進行API的轉換就可以通用。但因為 AMD MI210 和 NVIDIA GTX 1080 之間的世代差異，導致所花的時間也有顯著不同。

## Experience & conclusion

HW3 是第一個要寫 `.cu` 的作業，整體經驗十分具有挑戰性。特別是在處理 indexing 時，資料搬移的 mapping 以及 access pattern 的考慮，這部分花了很多時間來調整和優化。這讓我體會到，要寫出高效能的 CUDA 程式，必須對 GPU 的硬體架構有深入的理解，包括 memory hierarchy，warp 執行方式等等。

從實驗結果中，我們發現記憶體存取模式對效能的影響最為顯著。使用 shared memory 和調整 memory access pattern 能帶來明顯的效能提升。另外在選擇 block size 時，64\*64 的設定在各項效能指標上能達到較好的平衡。而在雙 GPU 的實作中，我們發現只有在較大的測資才能展現加速優勢，這是因為較小的測資可能受到 communication overhead 的影響而無法發揮加速效果。

這次的 Floyd-Warshall 實作也讓我理解到，GPU 的應用並不限於圖形處理或機器學習。只要是 computation intensive 的 task，都有機會透過 GPU 獲得可觀的效能提升。這個作業讓我對 CUDA programming 有初步的了解，也加深了我對平行運算的認識，是一個寶貴的學習經驗。