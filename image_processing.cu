// =============================================================================
// CUDA Image Processing
// Applies grayscale conversion, Gaussian blur, and Sobel edge detection
// to one or more PPM images using GPU-accelerated kernels.
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d  %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Simple PPM image structure
// ---------------------------------------------------------------------------
typedef struct {
  int width;
  int height;
  unsigned char *data;  // interleaved RGB, row-major
} Image;

// ---------------------------------------------------------------------------
// PPM I/O helpers
// ---------------------------------------------------------------------------
Image *read_ppm(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }

  char magic[3];
  if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
    fprintf(stderr, "Not a binary PPM (P6): %s\n", filename);
    fclose(fp); return NULL;
  }

  Image *img = (Image *)malloc(sizeof(Image));
  int maxval;
  // Skip comments
  int c = fgetc(fp);
  while (c == '\n' || c == ' ' || c == '\r') c = fgetc(fp);
  while (c == '#') { while (fgetc(fp) != '\n'); c = fgetc(fp); }
  ungetc(c, fp);

  if (fscanf(fp, "%d %d %d", &img->width, &img->height, &maxval) != 3) {
    fprintf(stderr, "Malformed PPM header: %s\n", filename);
    free(img); fclose(fp); return NULL;
  }
  fgetc(fp);  // consume single whitespace after maxval

  size_t npixels = (size_t)img->width * img->height * 3;
  img->data = (unsigned char *)malloc(npixels);
  if (fread(img->data, 1, npixels, fp) != npixels) {
    fprintf(stderr, "Truncated PPM data: %s\n", filename);
    free(img->data); free(img); fclose(fp); return NULL;
  }
  fclose(fp);
  return img;
}

void write_ppm(const char *filename, const Image *img) {
  FILE *fp = fopen(filename, "wb");
  if (!fp) { fprintf(stderr, "Cannot write %s\n", filename); return; }
  fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
  fwrite(img->data, 1, (size_t)img->width * img->height * 3, fp);
  fclose(fp);
}

void write_pgm(const char *filename, const unsigned char *data,
               int width, int height) {
  FILE *fp = fopen(filename, "wb");
  if (!fp) { fprintf(stderr, "Cannot write %s\n", filename); return; }
  fprintf(fp, "P5\n%d %d\n255\n", width, height);
  fwrite(data, 1, (size_t)width * height, fp);
  fclose(fp);
}

void free_image(Image *img) {
  if (img) { free(img->data); free(img); }
}

// ---------------------------------------------------------------------------
// Kernel 1: RGB -> Grayscale  (luminance formula)
// ---------------------------------------------------------------------------
__global__ void kernel_grayscale(const unsigned char *rgb,
                                  unsigned char *gray,
                                  int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int idx = (y * width + x) * 3;
  float r = rgb[idx];
  float g = rgb[idx + 1];
  float b = rgb[idx + 2];
  gray[y * width + x] = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);
}

// ---------------------------------------------------------------------------
// Kernel 2: 5x5 Gaussian blur on a grayscale image
// ---------------------------------------------------------------------------
__constant__ float d_gauss[25] = {
  1,  4,  7,  4, 1,
  4, 16, 26, 16, 4,
  7, 26, 41, 26, 7,
  4, 16, 26, 16, 4,
  1,  4,  7,  4, 1
};
__constant__ float d_gauss_sum = 273.0f;

__global__ void kernel_gaussian_blur(const unsigned char *in,
                                      unsigned char *out,
                                      int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float acc = 0.0f;
  for (int ky = -2; ky <= 2; ++ky) {
    for (int kx = -2; kx <= 2; ++kx) {
      int nx = min(max(x + kx, 0), width - 1);
      int ny = min(max(y + ky, 0), height - 1);
      float w = d_gauss[(ky + 2) * 5 + (kx + 2)];
      acc += w * in[ny * width + nx];
    }
  }
  out[y * width + x] = (unsigned char)(acc / d_gauss_sum);
}

// ---------------------------------------------------------------------------
// Kernel 3: Sobel edge detection on a grayscale image
// ---------------------------------------------------------------------------
__global__ void kernel_sobel(const unsigned char *in,
                               unsigned char *out,
                               int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  // Clamp to border
  auto px = [&](int dx, int dy) -> float {
    int nx = min(max(x + dx, 0), width - 1);
    int ny = min(max(y + dy, 0), height - 1);
    return (float)in[ny * width + nx];
  };

  float gx = -px(-1,-1) - 2*px(0,-1) - px(1,-1)
             + px(-1, 1) + 2*px(0, 1) + px(1, 1);
  float gy = -px(-1,-1) - 2*px(-1,0) - px(-1, 1)
             + px( 1,-1) + 2*px( 1,0) + px( 1, 1);

  float mag = sqrtf(gx*gx + gy*gy);
  out[y * width + x] = (unsigned char)min(mag, 255.0f);
}

// ---------------------------------------------------------------------------
// Host helper: process one image through all three stages
// ---------------------------------------------------------------------------
typedef struct {
  double grayscale_ms;
  double blur_ms;
  double sobel_ms;
} Timings;

Timings process_image(const char *input_path,
                      const char *out_gray_path,
                      const char *out_blur_path,
                      const char *out_edge_path) {
  Timings t = {0};

  // ---- Load image ---------------------------------------------------------
  Image *img = read_ppm(input_path);
  if (!img) { return t; }

  int W = img->width, H = img->height;
  size_t rgb_bytes  = (size_t)W * H * 3;
  size_t gray_bytes = (size_t)W * H;

  printf("  Loaded: %s  (%d x %d)\n", input_path, W, H);

  // ---- Allocate device memory ---------------------------------------------
  unsigned char *d_rgb, *d_gray, *d_blur, *d_edge;
  CUDA_CHECK(cudaMalloc(&d_rgb,  rgb_bytes));
  CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
  CUDA_CHECK(cudaMalloc(&d_blur, gray_bytes));
  CUDA_CHECK(cudaMalloc(&d_edge, gray_bytes));

  CUDA_CHECK(cudaMemcpy(d_rgb, img->data, rgb_bytes, cudaMemcpyHostToDevice));

  // ---- Launch config ------------------------------------------------------
  dim3 block(16, 16);
  dim3 grid((W + block.x - 1) / block.x,
            (H + block.y - 1) / block.y);

  // ---- CUDA events for timing ---------------------------------------------
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  float ms;

  // Stage 1: Grayscale
  CUDA_CHECK(cudaEventRecord(ev0));
  kernel_grayscale<<<grid, block>>>(d_rgb, d_gray, W, H);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  t.grayscale_ms = ms;

  // Stage 2: Gaussian blur
  CUDA_CHECK(cudaEventRecord(ev0));
  kernel_gaussian_blur<<<grid, block>>>(d_gray, d_blur, W, H);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  t.blur_ms = ms;

  // Stage 3: Sobel edge detection
  CUDA_CHECK(cudaEventRecord(ev0));
  kernel_sobel<<<grid, block>>>(d_blur, d_edge, W, H);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  t.sobel_ms = ms;

  // ---- Copy results back --------------------------------------------------
  unsigned char *h_gray = (unsigned char *)malloc(gray_bytes);
  unsigned char *h_blur = (unsigned char *)malloc(gray_bytes);
  unsigned char *h_edge = (unsigned char *)malloc(gray_bytes);

  CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_blur, d_blur, gray_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_edge, d_edge, gray_bytes, cudaMemcpyDeviceToHost));

  // ---- Save outputs -------------------------------------------------------
  write_pgm(out_gray_path, h_gray, W, H);
  write_pgm(out_blur_path, h_blur, W, H);
  write_pgm(out_edge_path, h_edge, W, H);

  printf("    -> grayscale : %s  (%.3f ms)\n", out_gray_path, t.grayscale_ms);
  printf("    -> blurred   : %s  (%.3f ms)\n", out_blur_path, t.blur_ms);
  printf("    -> edges     : %s  (%.3f ms)\n", out_edge_path, t.sobel_ms);

  // ---- Cleanup ------------------------------------------------------------
  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_gray));
  CUDA_CHECK(cudaFree(d_blur));
  CUDA_CHECK(cudaFree(d_edge));
  free(h_gray); free(h_blur); free(h_edge);
  free_image(img);

  return t;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static void print_usage(const char *prog) {
  fprintf(stderr,
    "Usage:\n"
    "  Single image : %s <input.ppm> <output_dir>\n"
    "  Directory    : %s --dir <input_dir> <output_dir>\n",
    prog, prog);
}

int main(int argc, char *argv[]) {
  if (argc < 3) { print_usage(argv[0]); return EXIT_FAILURE; }

  // Print CUDA device info
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("=== CUDA Image Processor ===\n");
  printf("GPU: %s  (SM %d.%d, %d MPs, %.1f GB)\n\n",
         prop.name, prop.major, prop.minor,
         prop.multiProcessorCount,
         prop.totalGlobalMem / 1e9);

  int dir_mode = (strcmp(argv[1], "--dir") == 0);
  const char *output_dir = argv[dir_mode ? 3 : 2];

  // Ensure output directory exists
  mkdir(output_dir, 0755);

  double total_gray = 0, total_blur = 0, total_sobel = 0;
  int count = 0;

  if (!dir_mode) {
    // ---- Single file mode -------------------------------------------------
    const char *input = argv[1];
    char base[256], out_gray[512], out_blur[512], out_edge[512];

    // Strip path and extension for base name
    const char *slash = strrchr(input, '/');
    strncpy(base, slash ? slash + 1 : input, sizeof(base) - 1);
    base[sizeof(base)-1] = '\0';
    char *dot = strrchr(base, '.'); if (dot) *dot = '\0';

    snprintf(out_gray, sizeof(out_gray), "%s/%s_gray.pgm",  output_dir, base);
    snprintf(out_blur, sizeof(out_blur), "%s/%s_blur.pgm",  output_dir, base);
    snprintf(out_edge, sizeof(out_edge), "%s/%s_edges.pgm", output_dir, base);

    Timings t = process_image(input, out_gray, out_blur, out_edge);
    total_gray  += t.grayscale_ms;
    total_blur  += t.blur_ms;
    total_sobel += t.sobel_ms;
    count = 1;

  } else {
    // ---- Directory mode ---------------------------------------------------
    if (argc < 4) { print_usage(argv[0]); return EXIT_FAILURE; }
    const char *input_dir = argv[2];

    DIR *dp = opendir(input_dir);
    if (!dp) {
      fprintf(stderr, "Cannot open directory: %s\n", input_dir);
      return EXIT_FAILURE;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
      size_t len = strlen(entry->d_name);
      if (len < 5) continue;
      // Accept .ppm files
      if (strcasecmp(entry->d_name + len - 4, ".ppm") != 0) continue;

      char input_path[512];
      snprintf(input_path, sizeof(input_path), "%s/%s", input_dir, entry->d_name);

      char base[256];
      strncpy(base, entry->d_name, sizeof(base) - 1);
      base[sizeof(base)-1] = '\0';
      char *dot = strrchr(base, '.'); if (dot) *dot = '\0';

      char out_gray[512], out_blur[512], out_edge[512];
      snprintf(out_gray, sizeof(out_gray), "%s/%s_gray.pgm",  output_dir, base);
      snprintf(out_blur, sizeof(out_blur), "%s/%s_blur.pgm",  output_dir, base);
      snprintf(out_edge, sizeof(out_edge), "%s/%s_edges.pgm", output_dir, base);

      printf("[%d] Processing: %s\n", count + 1, entry->d_name);
      Timings t = process_image(input_path, out_gray, out_blur, out_edge);
      total_gray  += t.grayscale_ms;
      total_blur  += t.blur_ms;
      total_sobel += t.sobel_ms;
      ++count;
    }
    closedir(dp);
  }

  // ---- Summary ------------------------------------------------------------
  printf("\n=== Summary (%d image%s) ===\n", count, count == 1 ? "" : "s");
  printf("  Grayscale kernel total : %.3f ms\n", total_gray);
  printf("  Gaussian blur total    : %.3f ms\n", total_blur);
  printf("  Sobel edge total       : %.3f ms\n", total_sobel);
  printf("  Grand total (kernels)  : %.3f ms\n",
         total_gray + total_blur + total_sobel);
  printf("Output files written to : %s/\n", output_dir);

  return EXIT_SUCCESS;
}
