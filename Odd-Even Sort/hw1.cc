#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <mpi.h>
#include <string.h>
#define DEBUG (rank == 19) && 0

int modulo(int X, int Y) {
    if (X < Y) return X;
    int M = modulo(X, Y << 1);
    if (M >= Y) M -= Y;
    return M;
}

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

int get_elem_num(int rank, int chunk, int remain, int size) {
    if (rank < 0 || rank >= size) return 0;
    if (remain != 0) return chunk + (rank < remain ? 1 : 0);
    return chunk;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);
    char *input_file = argv[2];
    char *output_file = argv[3];

    int chunk = n / size;
    int remain = modulo(n, size);

    int arr_elem = get_elem_num(rank, chunk, remain, size);
    int left_elem = get_elem_num(rank - 1, chunk, remain, size);
    int right_elem = get_elem_num(rank + 1, chunk, remain, size);

    float* arr = (float*)malloc(arr_elem * sizeof(float));
    float* right_arr = (float*)malloc(right_elem * sizeof(float));
    float* left_arr = (float*)malloc(left_elem * sizeof(float));
    float* merge_arr = (float*)malloc(arr_elem * sizeof(float));
    
    MPI_File FIN;
    MPI_Offset offset = ((chunk * rank) + (rank < remain ? rank : remain)) * sizeof(float);

    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &FIN);
    MPI_File_read_at(FIN, offset, arr, arr_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&FIN);

    if (arr_elem > 0) {
        boost::sort::spreadsort::float_sort(arr, arr + arr_elem);
    }

    int phase = 0, round = 0;
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

    MPI_File FOUT;
    MPI_File_open(MPI_COMM_WORLD, output_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &FOUT);
    MPI_File_write_at(FOUT, offset, arr, arr_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&FOUT);

    MPI_Finalize();
    return 0;
}
