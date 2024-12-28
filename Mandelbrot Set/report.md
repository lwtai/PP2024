# HW2 Mandelbrot set

#### ID: 112062698 / Name: 戴樂為

## HW2a

### 1. Only Pthread

最初的版本主要是去改作業本身提供的 sequential code，把內層的迴圈拉出來變成 pthread 的函數，這邊比較需要注意的是用 `mutex_lock` 保護 critical section 的部分。

```c
void* calculate_mandelbrot(void* arg) {
    thread_arg_t* args = (thread_arg_t*)arg;
    while (1) {
        int j;
        pthread_mutex_lock(args->mutex);
        j = *(args->next_row);
        *(args->next_row) = j + 1;
        pthread_mutex_unlock(args->mutex);
        if (j >= args->height) break;
        double y0 = j * args->y0_step + args->lower;
        for (int i = 0; i < args->width; ++i) {
            double x0 = i * args->x0_step + args->left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < args->iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            args->image[j * args->width + i] = repeats;
        }
    }
    return NULL;
}
```

### 2. Pthread & Vectorization

**AVX-512**

在向量化的時候選擇用 `avx-512` 來做，因為照理來說，比較長的指令集可以放更多的資料，應該會得到更快的 speed up。會用到的型別有 `__m512d` `__m512i` `__mmask8` 等等，主要改寫的部分是 thread 裡面計算每個像素的地方。

**Vecotrized Function**

下面是 thread 函數向量化之後的樣子，可以看到說 512 bits 的寬度優勢在於可以放得下 8 個 `float`，以平行度來說是大大提升。另外因為每個 pixel 的結束時間並不同，所以需要用 mask 追蹤像素的計算情況，並以 mask 的資訊決定什麼時候可以進行下一個 pass，或是用來做寫入的檢查，以免錯誤結果。

```c
void* calculate_mandelbrot(void* arg) {
    thread_arg_t* args = (thread_arg_t*)arg;
    while (true) {
        int j;
        pthread_mutex_lock(args->mutex);

        j = *(args->next_row);
        *(args->next_row) = j + 1;

        pthread_mutex_unlock(args->mutex);
        if (j >= height) break;
        double y0 = j * y0_step + lower;
        for (int i = 0; i < width; i += 8) {
            __mmask8 mask = (i + 7 < width) ? 0xFF : (1 << (width - i)) - 1;
            __m512d x0 = _mm512_set_pd(
                (i+7) * x0_step + left,
                (i+6) * x0_step + left,
                (i+5) * x0_step + left,
                (i+4) * x0_step + left,
                (i+3) * x0_step + left,
                (i+2) * x0_step + left,
                (i+1) * x0_step + left,
                (i) * x0_step + left
            );
            __m512d y0_vec = _mm512_set1_pd(y0);
            __m512d x = _mm512_setzero_pd();
            __m512d y = _mm512_setzero_pd();
            __m512i repeats = _mm512_setzero_si512();
            __mmask8 active_thread = mask;   
            for (int k = 0; k < iters; k++) {
                __m512d xx = _mm512_mul_pd(x, x);
                __m512d yy = _mm512_mul_pd(y, y);
                __m512d xy = _mm512_mul_pd(x, y);
                __m512d length_squared = _mm512_add_pd(xx, yy);
                __mmask8 status = _mm512_cmp_pd_mask(length_squared, _mm512_set1_pd(4.0), _CMP_LT_OS);
                active_thread &= status;

                if (active_thread == 0) break;

                y = _mm512_fmadd_pd(_mm512_set1_pd(2.0), xy, y0_vec);
                x = _mm512_add_pd(_mm512_sub_pd(xx, yy), x0);
                repeats = _mm512_mask_add_epi64(repeats, active_thread, repeats, _mm512_set1_epi64(1));
            }
            _mm512_mask_cvtepi64_storeu_epi32(&args->image[j * width + i], mask, repeats);
        }
    }
    return NULL;
}
```

**Pitfall**

在這部分花了滿多時間 debug，最後發現是錯在型別的使用上。由於 SIMD 的特性，計算時需要去對齊 vector 裡面要計算的值，而為了要和 8 個 64bits 的浮點數對齊，應該要使用 64bits 的整數，也就是有 `epi64` 後綴的指令來寫才會對，除此之外，寫回去之前還需要再轉回 `epi32`。

### 3. 16bits Mask Vectorization

在翻找 Intel 指令集文件的時候，意外發現除了 `__mmask8` 之外，還有另一種 `__mmask16`，代表說可以一次追蹤 16 個浮點數，因此在原先 8bits mask 版本的架構下，寫了 16bits 的版本，速度上的確是快了不少。
```c
// 16 bits mask
__mmask16 mask = (i + 15 < width) ? 0xFFFF : (1 << (width - i)) - 1;

// load 16 pixels at once
__m512d x0_low = _mm512_set_pd(
    (i+7) * x0_step + left,
    (i+6) * x0_step + left,
    (i+5) * x0_step + left,
    (i+4) * x0_step + left,
    (i+3) * x0_step + left,
    (i+2) * x0_step + left,
    (i+1) * x0_step + left,
    (i) * x0_step + left
);
__m512d x0_high = _mm512_set_pd(
    (i+15) * x0_step + left,
    (i+14) * x0_step + left,
    (i+13) * x0_step + left,
    (i+12) * x0_step + left,
    (i+11) * x0_step + left,
    (i+10) * x0_step + left,
    (i+9) * x0_step + left,
    (i+8) * x0_step + left
);
```


## HW2b MPI & OpenMP

### 1. Task Distribution

**Continuous Chunk**

在這個版本中用最簡單的想法去寫，把前面提到 16bits mask 的版本進一步改寫，先確定有幾個 process 之後，以 row 為單位再把 task 分給不同 process，原本的 thread 函數就包成 `compute_mandelbrot` 讓每個 process 去呼叫。

```c
// continuous chunk
int rows_per_process = height / size;
int remainder = height % size;
int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
int local_height = end_row - start_row;

// allocate memory
int* local_image = (int*)malloc(local_height * width * sizeof(int));
assert(local_image);

// compute local image
compute_mandelbrot(start_row, end_row, local_image, num_threads);
```

**Scatter Chunk**

測試之後發現 continuous chunk 的結果非常慢，顯示說這不是一種好的分法，所以後來又試了另一種切法，它會將每個 row 輪流交給不同的 process，鄰近的 row 會分給不同的 process，經過實測之後發現時間進步非常多。
```c
// scatter chunk
#pragma omp parallel for schedule(dynamic)
for (int row = rank; row < height; row += size) {
    compute_mandelbrot(image, row);
}
```

### 2. Result Aggregation

**MPI_GATHERV**

在最後蒐集結果的時候，我先嘗試了 `MPI_Gatherv` 這個 API，之所以選擇他是因為他很彈性，可以分開指定要從每個 process 收幾個 elements，寫入位置的 offset 等等，但實際用起來發現不太方便，需要 maintain 很多額外的資料才能運作，尤其牽涉到越多 process 就會需要越長的 array 來準備和儲存資料。

```c
// allocate memory
int local_height = end_row - start_row;
int* local_image = (int*)malloc((end_row - start_row) * width * sizeof(int));
assert(local_image);

/* calculate mandelbrot set */
/* prepare information for gathering */

// gather image from all processes
MPI_Gatherv(local_image, local_size, MPI_INT, full_image, recv_counts, offsets, MPI_INT, 0, MPI_COMM_WORLD);
```

**MPI_REDUCE**

後來用了一個比較簡潔的方式，在每個 process 只需要兩塊一樣大的 memory，並且在初始化的時候要記得設為 0，這樣比較保險，因為計算結束之後，`MPI_Reduce` 會用 `MPI_BOR` 去搜集所有結果。這邊利用的是 process 之間寫入的 memory 位置不重疊，以及初始值為 0 的假設。
```c
// allocate memory
int *image = (int*)malloc(width * height * sizeof(int));
int *ans = (int*)malloc(width * height * sizeof(int));

// zero out memory
memset(image, 0, width * height * sizeof(int));
memset(ans, 0, width * height * sizeof(int));

/* calculate mandelbrot set */

// reduce all values
MPI_Reduce(image, ans, height*width, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
```

## Experiment & Analysis

### 1. Scalability Test

這邊除了 scalability 的測試之外，同時也會看一下程式的 profiling 和 speedup。測試資料在 `2a` `2b` 皆會採用 `strict34.txt`，比較大的測資應該可以有比較明顯的效果。

**- HW2a**

在 `hw2a` 的測試中，我們用 `srun -n1 -c {#cpu} ./wrapper.sh ./hw2a ./out.png 10000 -0.5506164691618783 -0.5506164628264113 0.6273445437118131 0.6273445403522527 7680 4320` 跑最快的 16bits mask 版本。

| #CPU |  IO Time (sec) | Calculation Time (sec)  | Total Time (sec) |
| :---: | :---: | :---: | :---:  |
|  1   | 1.578 | 52.193 | 53.771 |
|  2   | 1.527 | 25.880 | 27.417 |
|  3   | 1.538 | 17.161 | 18.699 |
|  4   | 1.538 | 12.854 | 14.393 |
|  5   | 1.536 | 10.291 | 11.827 | 
|  6   | 1.532 | 8.550  | 10.082 |
|  7   | 1.529 | 7.336  | 8.874  | 
|  8   | 1.539 | 6.425  | 7.970  |
|  9   | 1.532 | 5.700  | 7.240  | 
|  10  | 1.533 | 5.143  | 6.685  | 
|  20  | 1.539 | 2.573  | 4.113  | 
|  40  | 1.542 | 1.631  | 3.172  | 
|  80  | 1.541 | 0.896  | 2.437  | 
|  96  | 1.548 | 0.724  | 2.273  | 

上表是針對不同 thread 數量所算出來的時間，以及用 `nvtx` 切的區段來看時間分佈，時間組成主要只有兩大項，其他時間可以忽略不計，`IO Time` 對應的會是 `write png` 的部分，接下來我們把它畫成長條圖以及 speedup 折線圖。

![Screenshot 2024-10-24 at 02.03.30](https://hackmd.io/_uploads/Hkh95hIl1e.png)
![Screenshot 2024-10-24 at 02.03.41](https://hackmd.io/_uploads/r1Vj5hLlkl.png)


**Observation**

由圖可以看出 `hw2a` 的 scalability 非常強，在 1~10 幾乎是完美的線性關係，甚至在 thread 數量等於 10~80 的區間也能保持近乎線性的成長。這個結果算是合理，因為 `IO` 所花的時間很固定，另外單個 process 比較沒有 communication 的成本，並且像素之間並沒有 dependency，因此 Mandelbrot set 的計算本身就很適合平行化。

**- HW2b**

既然單個 process 有很優秀的 scalability，在 `hw2b` 的測試中，我們用 `srun -n {#process} -c 1 ./wrapper.sh ./hw2b ./out.png 10000 -0.5506164691618783 -0.5506164628264113 0.6273445437118131 0.6273445403522527 7680 4320` 來測試 MPI 的表現。

| #process |  IO & MPI Time (sec)  | Calculation Time (sec) | Total Time (sec) |
| :---: | :---: |:---:| :---: |
|  1   | 2.228 | 52.803 | 55.031 |
|  2   | 2.314 | 26.580 | 28.894 |
|  3   | 2.424 | 17.734 | 20.158 |
|  4   | 2.336 | 13.364 | 15.700 |
|  5   | 2.332 | 10.683 | 13.015 | 
|  6   | 2.374 | 8.860  | 11.234 |
|  7   | 2.388 | 7.611  | 9.999  | 
|  8   | 2.397 | 6.664  | 9.061  |
|  9   | 2.456 | 5.898  | 8.354  | 
|  10  | 2.536 | 5.337  | 7.873  | 

這邊一樣 `IO` 對應的是 `write png`，因為它時間跟 `hw2a` 中的結果一樣，花的時間都穩定的在 1.5 秒左右，所以我將 `IO` 和 `MPI` 合併一起計算。

![Screenshot 2024-10-24 at 03.10.03](https://hackmd.io/_uploads/SJU75T8ekx.png)
![Screenshot 2024-10-24 at 03.14.21](https://hackmd.io/_uploads/HkkNjT8e1x.png)

**Observation**

其實 `hw2b` 加入 MPI 的 scalability 也算是不錯，但可以發現因為 MPI overhead 的關係，使得曲線比起 `hw2a` 較為平緩，可以預期的是若繼續增加 process 數目，speedup 的成長速度會趨緩。總體來說，`hw2b` 每個測資會比 `hw2a` 多出 1.5 秒左右的時間，MPI 佔 1 秒，calculation 多 0.5 秒左右。


### 2. Load Banlancing Test

**- HW2a**

在這邊的測試中，一樣採用 `strict34.txt` 來進行，並且用 5 個 thread 的設定 `srun -n1 -c5`，另外藉由 `nvtx` 在 thread function 裡面設定計算時間的區間，可以看到每一個 row 精確的執行時間，結果整理在下面。

- Total calculation time: $51.387$ sec
- Longest calculation per row: $25.970$ ms
- Shortest calculation per row: $9.421$ ms
- Total iterations: $4320$

由詳細的數字知道，每個 row 確實會有執行時間上的差異，但由於所有的 task 都集中在一個 process 裡面，並且又是 thread pool 的設計，所以這些小差異並不會造成太大的影響，也因此 `hw2a` 才能有比較優異的時間表現。

**- HW2b**

接下來我們會對比 `hw2b` 的兩種實作方式，也就是講前面提到的 **continuous chunk** 和 **scatter chunk** 做比較，希望可以藉由負載的觀察，更了解時間差異的來源。我們用 `srun -n5 -c1` 的設定來跑 `strict34.txt`，並且用 `nvtx` 紀錄結果。

| Distribution    |  Rank 0 | Rank 1 | Rank 2 | Rank 3 | Rank 4 | diff (max, min) |
| :---: | :---:   | :---: | :---: | :---: | :---: | :---: |
|  **Scatter**    | 10.588| 10.590 | 10.584 | 10.589 | 10.582 | 0.008 (sec) |
|  **Continuous** | 9.502 | 10.268 | 11.659 | 11.260 | 10.601 | 2.157 (sec) |

在上面的表格分別記錄了 5 個 processes 做計算的時間。


![Screenshot 2024-10-24 at 20.48.33](https://hackmd.io/_uploads/H1JdGTPlyg.jpg)


從很多的圖中都可以觀察到 Mandelbrot set 有聚集的特性，所以連續的切法很容易將相同的色塊分給同一個程式，造成負載不平衡。尤其計算完成之後所有程式需要同步化，比較快的 process 就要等最慢的 process 完成，而在 `strict34.txt` 這筆測資上，continuous 的分法使得最快和最慢的 process 差到了有 2 秒的時間。

採用 scatter 的方式可以有效的避免這種情況，將相似的 work load 平均的分給每個程式。這也說明了即便都是靜態的 distribution，方法還是有好與壞的差異。另外，即便是 scatter 的方式，在做了 profiling 之後，也可以發現它不平衡的地方。

| Process Rank |  MPI (sec) | Write PNG (sec)  | Calculation (sec)  | Total Time (sec) |
| :---: | :---: | :---: | :---: | :---: |
|  0   |0.875 | 1.584 | 10.588 | 13.047 |
|  1   |2.451 | 0.0	  | 10.590 | 13.041 |
|  2   |2.453 | 0.0	  | 10.584 | 13.037 |
|  3   |2.458 | 0.0	  | 10.589 | 13.047 |
|  4   |2.456 | 0.0	  | 10.582 | 13.047 |

![Screenshot 2024-10-24 at 19.44.06](https://hackmd.io/_uploads/BkFGQhwxyl.png)

由於只有 rank 為 0 的 process 負責輸出圖檔，這個 `write png` 就變成一個多出來的 1.5 秒，其他 proces 看似有在做事，但事實上只是在 `MPI_Finalize` 等它寫完而已，不過這個步驟需要在所有計算完成之後才能進行，確實是比較難平行化。


## Experience & Conclusion

寫完這次作業之後，學到了很多原本不會接觸到的新知識。首先遇到的難關在於向量化程式的部分，因為之前沒有使用過，在 debug 的時候滿有挑戰性的，而現在對指令集有比較具體的了解，學到了如何利用指令集操控 SIMD 的硬體，用來在計算密集的場景下提供巨大的加速，將原本需要 30 秒左右的測資加速到 1 秒以內。

再來就是負載平衡對於平行程式的影響，跟作業 2a 只有 pthread 非常理想的版本比較起來，作業 2b 的 continuous chunk 真的慢了滿多，那這一點的話又可以經由 scatter chunk 的設計稍微彌補回來。另外 profiler 真的是滿重要的工具，例如我在 `hw2b` 其實也有試過 process pool 的版本，當時很疑惑為什麼沒有比較快，後來用 `nsys` 跑過之後才發現原來每個程式的 work load 已經非常平衡，剩下的是 MPI 的 overhead，因此沒有進步也是理所當然。

那講到 MPI 其實也有一點小小遺憾，不管是作業 1 還是作業 2 寫出來的 MPI 似乎都沒辦法有很好的效能，尤其在 `hw2a` 的排名算是不錯，到了 `hw2b` 直接被拖垮。還是希望之後課堂上可以稍微講解要如何寫出效能很好的 MPI 程式。
