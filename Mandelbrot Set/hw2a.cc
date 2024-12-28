#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <immintrin.h>

int num_threads;
int iters;
double left;
double right;
double lower;
double upper;
double y0_step;
double x0_step;
int width;
int height;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

typedef struct {
    int tid;
    int* image;
    int* next_row;
    pthread_mutex_t* mutex;
} thread_arg_t;

void* thread_func(void* arg) {
    thread_arg_t* args = (thread_arg_t*)arg;
    
    while (true) {
        int j;
        pthread_mutex_lock(args->mutex);
        j = *(args->next_row);
        *(args->next_row) = j + 1;
        pthread_mutex_unlock(args->mutex);
        if (j >= height) break;
        
        double y0 = j * y0_step + lower;
        for (int i = 0; i < width; i += 16) {
            __mmask16 mask = (i + 15 < width) ? 0xFFFF : (1 << (width - i)) - 1;
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

            __m512d y0_vec = _mm512_set1_pd(y0);
            __m512d x_low = _mm512_setzero_pd();
            __m512d y_low = _mm512_setzero_pd();
            __m512d x_high = _mm512_setzero_pd();
            __m512d y_high = _mm512_setzero_pd();
            __m512i repeats_low = _mm512_setzero_si512();
            __m512i repeats_high = _mm512_setzero_si512();
            __mmask16 active_thread = mask;
            
            for (int k = 0; k < iters; k++) {
                __m512d xx_low = _mm512_mul_pd(x_low, x_low);
                __m512d yy_low = _mm512_mul_pd(y_low, y_low);
                __m512d xy_low = _mm512_mul_pd(x_low, y_low);
                __m512d xx_high = _mm512_mul_pd(x_high, x_high);
                __m512d yy_high = _mm512_mul_pd(y_high, y_high);
                __m512d xy_high = _mm512_mul_pd(x_high, y_high);

                __m512d length_squared_low = _mm512_add_pd(xx_low, yy_low);
                __m512d length_squared_high = _mm512_add_pd(xx_high, yy_high);
                __mmask8 status_low = _mm512_cmp_pd_mask(length_squared_low, _mm512_set1_pd(4.0), _CMP_LT_OS);
                __mmask8 status_high = _mm512_cmp_pd_mask(length_squared_high, _mm512_set1_pd(4.0), _CMP_LT_OS);
                __mmask16 status = (__mmask16)status_low | ((__mmask16)status_high << 8);
                active_thread &= status;
                
                if (active_thread == 0) break;
                
                y_low = _mm512_fmadd_pd(_mm512_set1_pd(2.0), xy_low, y0_vec);
                x_low = _mm512_add_pd(_mm512_sub_pd(xx_low, yy_low), x0_low);
                y_high = _mm512_fmadd_pd(_mm512_set1_pd(2.0), xy_high, y0_vec);
                x_high = _mm512_add_pd(_mm512_sub_pd(xx_high, yy_high), x0_high);
                
                repeats_low = _mm512_mask_add_epi64(repeats_low, status_low, repeats_low, _mm512_set1_epi64(1));
                repeats_high = _mm512_mask_add_epi64(repeats_high, status_high, repeats_high, _mm512_set1_epi64(1));
            }
            
            _mm512_mask_cvtepi64_storeu_epi32(&args->image[j * width + i], mask & 0xFF, repeats_low);
            _mm512_mask_cvtepi64_storeu_epi32(&args->image[j * width + i + 8], mask >> 8, repeats_high);
        }
    }
    
    return NULL;
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* precompute step size */
    y0_step = (upper - lower) / height;
    x0_step = (right - left) / width;

    /* thread related */
    pthread_t threads[num_threads];
    thread_arg_t thread_args[num_threads];
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    int next_row = 0;
    
    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    
    for (int tid = 0; tid < num_threads; ++tid) {
        thread_args[tid] = (thread_arg_t){
            .tid = tid,
            .image = image,
            .next_row = &next_row,
            .mutex = &mutex,
        };
        pthread_create(&threads[tid], NULL, thread_func, &thread_args[tid]);
    }
    
    for (int tid = 0; tid < num_threads; ++tid) {
        pthread_join(threads[tid], NULL);
    }
    
    write_png(filename, iters, width, height, image);
    free(image);
    
    return 0;
}