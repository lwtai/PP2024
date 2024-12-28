#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <time.h>  // For CPU timing
#include <sys/time.h>  // For microsecond precision
#define CHUNK_SIZE 10

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, NUM_THREADS;
static int Dist[V][V];

// Function to get current time in microseconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char* argv[]) {
    double total_start = get_time();
    double io_time = 0;
    double computation_time = 0;

    // Time input operation
    double input_start = get_time();
    input(argv[1]);
    double input_end = get_time();
    io_time += (input_end - input_start);
    printf("Input time: %.6f seconds\n", input_end - input_start);

    // Get CPU affinity
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);
    printf("Number of threads: %d\n", NUM_THREADS);

    // Time the main computation
    double compute_start = get_time();
    int B = 512;
    block_FW(B);
    double compute_end = get_time();
    computation_time = compute_end - compute_start;
    printf("Computation time: %.6f seconds\n", computation_time);

    // Time output operation
    double output_start = get_time();
    output(argv[2]);
    double output_end = get_time();
    io_time += (output_end - output_start);
    printf("Output time: %.6f seconds\n", output_end - output_start);

    // Print total timing summary
    double total_time = get_time() - total_start;
    printf("\nTiming Summary:\n");
    printf("Total I/O time: %.6f seconds (%.2f%%)\n", io_time, (io_time/total_time)*100);
    printf("Computation time: %.6f seconds (%.2f%%)\n", computation_time, (computation_time/total_time)*100);
    printf("Total execution time: %.6f seconds\n", total_time);
    
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n: %d, m: %d\n", n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    double phase_times[3] = {0, 0, 0};  // Track time for each phase
    
    for (int r = 0; r < round; ++r) {
        printf("Round %d of %d\n", r + 1, round);
        fflush(stdout);

        /* Phase 1*/
        double phase1_start = get_time();
        cal(B, r, r, r, 1, 1);
        phase_times[0] += get_time() - phase1_start;

        /* Phase 2*/
        double phase2_start = get_time();
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);
        phase_times[1] += get_time() - phase2_start;

        /* Phase 3*/
        double phase3_start = get_time();
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
        phase_times[2] += get_time() - phase3_start;
    }
    
    // Print phase timing summary
    printf("\nPhase Timing Summary:\n");
    printf("Phase 1 total time: %.6f seconds\n", phase_times[0]);
    printf("Phase 2 total time: %.6f seconds\n", phase_times[1]);
    printf("Phase 3 total time: %.6f seconds\n", phase_times[2]);
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                #pragma omp parallel for schedule(dynamic)
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
                            Dist[i][j] = Dist[i][k] + Dist[k][j];
                        }
                    }
                }
            }
        }
    }
}