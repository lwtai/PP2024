---
title: HW1 odd-even sort

---


# HW1 odd-even sort

#### ID: 112062698 / Name: 戴樂為

## Implementation

### - `main`函數架構

**Initialize**

一開始會先大略計算出 `chunk` 和 `remain`，接下來 `get_elem_num` 函數會利用這兩個數值計算出精確的 chunk size，分配的方式是若餘數不為零，剩下的 element 會被平均分配到前面的 process，避免全部塞給最後一個 process 的情況。在此同時，process 溝通對象的 element 數量也會被記錄下來，方便以後溝通使用。

```c++
int chunk = n / size;
int remain = modulo(n, size);

int arr_elem = get_elem_num(rank, chunk, remain, size);
int left_elem = get_elem_num(rank - 1, chunk, remain, size);
int right_elem = get_elem_num(rank + 1, chunk, remain, size);
```

**Read File**

讀檔的時候先計算好每個 process 的 `offset`，接著使用 `MPI_File_read_at` 抓取每個 process 需要的部分，這邊因為不是每個 process 都需要 data，所以若使用 `MPI_File_read_at_all` 會有一些問題。

```c++
MPI_File FIN;
MPI_Offset offset = ((chunk * rank) + (rank < remain ? rank : remain)) * sizeof(float);

MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &FIN);
MPI_File_read_at(FIN, offset, arr, arr_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
MPI_File_close(&FIN);
```

**Sorting**

利用前面計算的 element 數量來決定 process 需不需要 sorting，`arr_elem` 小於等於零則跳過此步驟。
```c++
if (arr_elem > 0) {
    boost::sort::spreadsort::float_sort(arr, arr + arr_elem);
}
```

**Communication**

這邊的重點主要在這兩個 `if`，`phase` 的值會在 0 1 之間 toggle，當`phase`等於 0 的時候，偶數 `rank` 的 process 會進到 `(rank & 1) != phase` 的條件，而當 `phase` 等於 1 的時候，偶數 `rank` 的 process 會進到 `(rank & 1) == phase` 的條件。

```c++
if (((rank & 1) == phase) && (right_elem > 0)) { /* do something... */ }
if (((rank & 1) != phase) && (left_elem > 0)) { /* do something... */ }
```


**Write File**

寫檔的時候和讀檔大同小異，用前面計算好的`offset`去寫入每個process負責的部分。
```c++
MPI_File FOUT;

MPI_File_open(MPI_COMM_WORLD, output_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &FOUT);
MPI_File_write_at(FOUT, offset, arr, arr_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
MPI_File_close(&FOUT);
```

### - Key Optimization

**Sorting**

有觀察到排序是很花時間的步驟，所以上網找了一下有沒有比較快速的排序，直接用現成的非常簡單，但時間上差距滿大。

```c++
// before
std::sort(arr, arr + arr_elem);

// after
boost::sort::spreadsort::float_sort(arr, arr + arr_elem);
```

**Custom Merge**

程式收到隔壁的資料之後，會把兩筆小的資料 merge 成一組大的 sorted array，因為每個 process 只會用到其中一半的 merge 結果，所以做完整的 merge 很浪費時間，於是我寫了兩個小 function，分別可以 merge 從左邊及從右邊來的資料，並且當做到需要的數量就停止。

```c++
// merge left
void merge_from_left(float* arr1, int n1, float* arr2, int n2, float* result) {
    int i = 0, j = 0, r = 0;
    while (r < n2 && i < n1 && j < n2) {
        if (arr1[i] <= arr2[j]) {
            result[r++] = arr1[i++];
        } else {
            result[r++] = arr2[j++];
        }
    }
    while (r < n2 && i < n1) result[r++] = arr1[i++];
    while (r < n2 && j < n2) result[r++] = arr2[j++];
}

// merge right
void merge_from_right(float* arr1, int n1, float* arr2, int n2, float* result) {
    int i = n1 - 1, j = n2 - 1, r = n2 - 1;
    while (r >= 0 && i >= 0 && j >= 0) {
        if (arr1[i] >= arr2[j]) {
            result[r--] = arr1[i--];
        } else {
            result[r--] = arr2[j--];
        }
    }
    while (r >= 0 && i >= 0) result[r--] = arr1[i--];
    while (r >= 0 && j >= 0) result[r--] = arr2[j--];
}
```

**Memory**

記憶體管理部分，在迴圈外面就把所有 memory space 準備好，會比放在迴圈裡面快上許多時間，另外 `arr` 和 `merge_arr` 可以使用同樣的大小，這樣的話方便在 merge 之後使用 `std::swap`，比起用 `std::copy` 會省下滿多時間。
```c++
// allocate outside the loop
float* arr = (float*)malloc(arr_elem * sizeof(float));
float* right_arr = (float*)malloc(right_elem * sizeof(float));
float* left_arr = (float*)malloc(left_elem * sizeof(float));
float* merge_arr = (float*)malloc(arr_elem * sizeof(float));
```

### - Version Variation: 幾種主要的版本如下
**Version 0**

只寫了單純的 `while` 迴圈，做的次數等於 worst case 的次數，在測資小的時候速度甚至更快，但有觀察到在比較大的 test case 會出現 performance 差異比較大的情況。(113.27 sec)
```c++
// version 0
int phase = 0, round = 0;
while (round <= size + 1) {
    if (arr_elem > 0)
    {
        MPI_Status status;
        if (((rank & 1) == phase) && (right_elem > 0)) {
            MPI_Sendrecv(arr, arr_elem, MPI_FLOAT, rank + 1, 0, 
                        right_arr, right_elem, MPI_FLOAT, rank + 1, 1,
                        MPI_COMM_WORLD, &status);

            if (right_arr[0] < arr[arr_elem - 1]) {
                merge_from_left(right_arr, right_elem, arr, arr_elem, merge_right_arr);
            }
        }
        if (((rank & 1) != phase) && (left_elem > 0)) { // (rank & 1) != phase
            MPI_Sendrecv(arr, arr_elem, MPI_FLOAT, rank - 1, 1, 
                        left_arr, left_elem, MPI_FLOAT, rank - 1, 0,
                        MPI_COMM_WORLD, &status);

            if (left_arr[left_elem - 1] > arr[0]) {
                merge_from_right(left_arr, left_elem, arr, arr_elem, merge_left_arr + left_elem);
            }
        }
    }
    phase ^= 1;
    round += 1;
}
```

**Version 1**

這個版本是用 `MPI_Allreduce` 去做的，每一回合結束用 `MPI_OR` 整合所有 process 的 `local_swap`，並且利用 `global_swap` 來決定要不要跳出迴圈，但是效果並沒有顯著的提升，原因的話後面會分析。(114.56 sec)
```c++
// version 1
bool global_swap;
bool local_swap;
for (int round = 0; round < size + 1; ++round) {
    int phase = round & 1;
    if (arr_elem <= 0)
    {
        MPI_Allreduce(&local_swap, &global_swap, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    }
    else
    {
        local_swap = false;
        if (((rank & 1) == phase) && (right_elem > 0)) {
            MPI_Status status;
            MPI_Sendrecv(arr, arr_elem, MPI_FLOAT, rank + 1, 0, 
                        right_arr, right_elem, MPI_FLOAT, rank + 1, 1,
                        MPI_COMM_WORLD, &status);

            if (right_arr[0] < arr[arr_elem - 1]) {
                merge_from_left(right_arr, right_elem, arr, arr_elem, merge_arr);
                std::swap(arr, merge_arr);
                local_swap = true;
            }

        }
        if (((rank & 1) != phase) && (left_elem > 0)) { // (rank & 1) != phase
            MPI_Status status;
            MPI_Sendrecv(arr, arr_elem, MPI_FLOAT, rank - 1, 1, 
                        left_arr, left_elem, MPI_FLOAT, rank - 1, 0,
                        MPI_COMM_WORLD, &status);

            if (left_arr[left_elem - 1] > arr[0]) {
                merge_from_right(left_arr, left_elem, arr, arr_elem, merge_arr);
                std::swap(arr, merge_arr);
                local_swap = true;
            }
        }
        MPI_Allreduce(&local_swap, &global_swap, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    }
    if (!global_swap && round >= 2) break;
}
```
**Version 2**

在最終的版本中，我選擇讓 process 在交換資料之前檢查需不需要交換，若不需要則可以快速略過這次迴圈，假設要和右邊的程式溝通，藉由多一個 `MPI_Sendrecv`，送出 local array 的最大值，同時收到右邊 array 的最小值並做一個比較，確定交換的必要性。(105.43 sec)

```c++
// version 2
while (round <= size + 1) {
    if (arr_elem > 0)
    {
        float tmp;
        MPI_Status status;
        if (((rank & 1) == phase) && (right_elem > 0)) {
            MPI_Sendrecv(&arr[arr_elem - 1], 1, MPI_FLOAT, rank + 1, 0, 
                        &tmp, 1, MPI_FLOAT, rank + 1, 1,
                        MPI_COMM_WORLD, &status);

            if (tmp < arr[arr_elem - 1]) {
                MPI_Sendrecv(arr, arr_elem, MPI_FLOAT, rank + 1, 0, 
                            right_arr, right_elem, MPI_FLOAT, rank + 1, 1,
                            MPI_COMM_WORLD, &status);

                merge_from_left(right_arr, right_elem, arr, arr_elem, merge_arr);
                std::swap(arr, merge_arr);
            }
        }
        if (((rank & 1) != phase) && (left_elem > 0)) {
            MPI_Sendrecv(&arr[0], 1, MPI_FLOAT, rank - 1, 1, 
                        &tmp, 1, MPI_FLOAT, rank - 1, 0,
                        MPI_COMM_WORLD, &status);

            if (tmp > arr[0]) {
                MPI_Sendrecv(arr, arr_elem, MPI_FLOAT, rank - 1, 1, 
                            left_arr, left_elem, MPI_FLOAT, rank - 1, 0,
                            MPI_COMM_WORLD, &status);

                merge_from_right(left_arr, left_elem, arr, arr_elem, merge_arr);
                std::swap(arr, merge_arr);
            }
        }
    }
    phase ^= 1;
    round += 1;
}
```

**Time comparsion**

根據 `hw1-judge` 的結果，選擇 `37.txt` 以及 `38.txt` 當作接下來 `nsys` 的測試對象。`37.txt` 在不同版本間有比較明顯的時間差異，所以實驗 1 我會用 `37.txt` 來觀察 early stopping 的效果。另外 `38.txt` 有比較長的執行時間，因此實驗 2 的 single-node 和實驗 3 的 multi-node 比較中，皆採用 `38.txt` 來做測試。

![versions](https://hackmd.io/_uploads/Hk1B8UMxyl.jpg)



## Experiment & Analysis

### - Experiment 1

首先對前面提到的 3 種主要版本，在課堂提供的 Apollo cluster 上進行測試，測資使用的是 `37.txt`，執行的指令也還原測資設定是 `srun -N 3 -n 24 ./wrapper.sh ./hw1 536869888 ../testcases/37.in ./37.out`。

**MPI Time**

利用 `nsys` 的 `MPI Event Summary`，搜集了一些主要的 MPI operation，並且以 `(second - instance - percent)` 的格式整理如下表，追蹤的 process 為 `rank = 4`。

| Operations \ Version            | v0  | v1 | v2 |
| ----------------------------- | :---: | :---: | :---: |
|  **MPI_Sendrecv**        | 4.724 - 52 - 26.4% | 2.358 - 24 - 13.8% | 1.846 - 58 - 11.6% |
|  **MPI_Allreduce**       | 0.000 - 0 - 0% | 0.976 - 12 - 5.7% | 0.000 - 0 - 0% |
|  **MPI_File_write_at**   | 6.569 - 1 - 36.7% | 0.601 - 1 - 3.5% | 11.056 - 1 - 69.4% |
|  **MPI_File_close**      | 3.985 - 2 - 22.2% | 10.271 - 2 - 60.0% | 0.001 - 2 - 0%  |
|  **Total Time** (37.txt) | 7.28 | 5.43 | 4.88 |
|  **Total Time** (all)    | 113.27 | 114.56 | 105.43 |

**Observation**

由上表可以看出一些端倪，以及 `v1` 比起 `v0` 進步不多的原因。

`v1` 藉由一些額外的判斷，在某些測資上確實比較吃香，例如 `37.txt` 把 `MPI_Sendrecv` 次數從 52 次降為 24 次，時間上也大約省了一半，但同時 `MPI_Allreduce` 的使用也會增加一些成本(0.976秒)，這樣導致在沒辦法 early stop 的測試資料上，`v1` 反而會慢一些些，於是以所有時間加總來看反而稍慢。

`v2` 則是利用 `MPI_Sendrecv` 先交換一個 float 的資料量，來決定要不要交換 array，可以發現雖然 `MPI_Sendrecv` 次數從 52 次變為 58 次，但是時間上卻只花了 1.846 秒(v0 花了 4.724 秒)，顯示說在 `MPI_Sendrecv` 所花的時間其實跟交換的資料量有很大的關係，也因此利用這種方法做條件判斷十分划算，也不必擔心在其他測資上會增加太多成本。

最後可以發現，其實每個 process 在 `MPI_File_write_at` 及 `MPI_File_close` 上面所花的時間幾乎都超過 50%，這也顯示了 IO 時間其實非常耗時，程式不是在排 write file，就是在等別人寫完要 close file。



### - Experiment 2

**Singe-node Test**

做完版本比較之後，在這個實驗中皆採用最快的 `v2` 來進行測試，測資使用的是 `38.txt`，執行的指令為 `srun -N 1 -n {#process} ./wrapper.sh ./hw1 536869888 ../testcases/38.in ./38.out`，進行 single node 的測試。


| #Process |  IO Time (sec) | Communication Time (sec) | CPU Time (sec)  | Total Time (sec) |
| :---: | :---: | :---: | :---: | :---: |
|  1   | 21.762 | 0.0   | 28.037| 49.799 |
|  2   | 12.344 | 0.887 | 19.304| 32.535 |
|  3   | 14.158 | 2.187 | 11.878| 28.223 |
|  4   | 12.243 | 1.417 | 12.539| 26.199 |
|  5   | 12.497 | 1.993 | 10.563| 25.053 | 
|  6   | 12.707 | 1.817 | 9.642 | 24.166 |
|  7   | 14.826 | 2.115 | 7.362 | 24.303 | 
|  8   | 12.661 | 2.071 | 7.179 | 21.911 |
|  9   | 13.198 | 2.070 | 8.793 | 24.061 | 
|  10  | 14.169 | 2.296 | 6.075 | 22.540 | 

在表格中執行時間被分成 `IO` `communication` `cpu` 三個大類，計算方式是先加總所有 prcoess 的時間，再除以 process 數量得到平均，接著把表格整理成下圖。

![Screenshot 2024-10-22 at 14.53.22](https://hackmd.io/_uploads/BJYm36Ne1e.png)


**Observation** 

首先以總執行時間來說，確實是隨著 process 數量增加而漸漸下降，但可以發現他並不是線性的下降，進步的幅度會漸漸趨緩。

進步趨緩的原因由直方圖可以明顯地觀察到，`IO` 會是一個主要的瓶頸，在 process 數量超過 5 之後佔的時間都超過了一半以上。另一方面，雖然 `communication` 時間成本有微幅的增加，但跟原本預期會隨著 process 數量增加大幅上升的情況不同。

**Speedup**

若以 1 個 process 的執行時間為基準，可以得到下面的 speedup 折線圖。

![Screenshot 2024-10-22 at 17.17.56](https://hackmd.io/_uploads/ryKfAkHgyg.png)



### - Experiment 3

**Multi-node Test**

這邊一樣採用最快的 `v2` 來進行測試，測資使用 `38.txt` 並且固定 process 數量為 10，執行的指令為 `srun -N {#node} -n 10 ./wrapper.sh ./hw1 536869888 ../testcases/38.in ./38.out`，來測試不同 node 數量的時間差異。

| #Node |  IO Time | Communication Time | MPI Init / Finalize| CPU Time  | Total Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  1   | 11.894 | 2.142 | 3.634 | 6.039 | 23.709 |
|  2   |  7.487 | 2.338 | 7.432 | 5.923 | 23.180 |
|  3   |  4.409 | 2.328 | 10.179| 5.860 | 22.776 |
|  4   | 13.891 | 2.306 | 0.745 | 5.803 | 22.745 |
|  5   | 15.575 | 2.050 | 0.698 | 5.640 | 23.963 | 
|  6   | 13.993 | 2.303 | 0.714 | 5.561 | 22.571 |
|  7   | 12.559 | 2.416 | 2.924 | 5.628 | 23.527 | 
|  8   | 15.274 | 2.452 | 0.684 | 5.603 | 24.013 |

把時間整理成直方圖，比較容易觀察，資料一樣是所有 process 的平均。

![Screenshot 2024-10-22 at 16.33.41](https://hackmd.io/_uploads/S1UYSkBgJg.png)


**Observation**

這次發現 `MPI Init / Finalize` 的時間變化也非常大，所以把它設為一個新類別。以總共花的時間來說，不管 `#Node` 的數量設定為多少，結果都非常相似。在 `communication` 的部分結果也非常穩定，皆在 2.0~2.5 秒之間浮動，並沒有隨著 node 數上升的趨勢，或是說影響很小。

`IO` 時間的浮動令人困惑，和 `MPI Init / Finalize` 加總起來大約是恆定的狀態，應該可以視為 MPI 的 overhead，總之，可以發現 process 數量才是影響執行時間的關鍵。

**Speedup**

同樣以 1 個 Node 的時間為基準計算 speedup。

![Screenshot 2024-10-22 at 17.24.28](https://hackmd.io/_uploads/ByPq1lBeJe.png)


## Discussion & Conclusion

### - Bottleneck

**IO** 

由前面的實驗可以知道，程式潛在的平行度是被 `IO` 限制住了，尤其在 single node 的測試中非常明顯。另外在 `nsys` 的分析報告中也顯示了讀寫資料的時候，會有許多 OS 層面的 operation 加入，會比較花費時間，同時也能看到每隻process不斷 polling 等待著要讀寫檔案，造成很多時間的浪費。但換句話說，這也是最有 speedup potential 的部分。

**Communication** 

MPI 之間的溝通成本令人意外的低，並且也很穩定，我認為 `v2` 版本的實作方式確實地減少了不必要的溝通成本，顯示了用小體積的通訊做檢查是一個有效率的方式，解決了潛在的通訊瓶頸。


### - Scalability

**Single-node** 

在 single-node 的測試中有算是很好的 scalability，在 8 個 process 時可以有 2.3x 的加速，由 1 個 process 慢慢增加到 8 個 process 的過程中可以得到越來越高的 speedup。但同時也觀察到如果再增加 process 數量，則需要解決前述 IO 的瓶頸。

**Multi-node** 

沒有 scalability，不管 node 的數量為多少，speedup 皆在 1x 附近浮動。

### - Conclusion

從實驗結果來看，平行化的 odd-even sort 雖然能達到一定程度的加速，但在 scalability 方面仍面臨不少挑戰，主要是受到 IO 效能瓶頸的限制。在三個版本中，`v2` 的表現最好，在 single-node 上使用 8 個 process 時可以達到約 2.3 倍的加速。不過在 multi-node 的情況 scalability 較為受限，這也說明了對於這個問題來說，與其增加 node 的數量，更應該著重在 single-node 的優化以及改善 IO 效能。

由於第一次寫平行程式，發現最大的問題在 debug 上面，並沒有辦法像寫 python / c++ 時，利用 `pdb` `lldb`等 debugger 快速地抓到錯誤，或是一行一行 trace 等等，尤其當資料數量太多時，只要有錯基本上無從找起，後來才發現在寫平行程式的時候要格外謹慎，新寫的東西一定要充分測試，確定沒問題之後再進行下一步。

另外 profiler 的使用也讓我獲益良多，與其自己通靈猜哪裡是效能瓶頸，用 Nsight 跑一次就會有很詳細的資訊，最後由 leaderboard 來看，我寫的版本跟第一名差了30秒左右，應該有什麼可以優化 IO 的方式，但我想不出來。這部分希望在作業結束之後，可以由助教替我們解惑，不用公布 code 沒關係，只要大概告訴我們要怎麼做，要是可以知道箇中的秘密，我會非常感激。


