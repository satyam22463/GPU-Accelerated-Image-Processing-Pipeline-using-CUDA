# CUDA Image Processing

A GPU-accelerated image processing pipeline written in CUDA C++.  
The program reads one or more PPM images and applies three successive
stages entirely on the GPU:

1. **Grayscale conversion** – luminance-weighted RGB → single-channel
2. **Gaussian blur** – 5×5 kernel to suppress noise before edge detection
3. **Sobel edge detection** – horizontal + vertical gradient magnitude

All three CUDA kernels are timed independently with `cudaEvent` so you can
see exactly how long each stage takes per image.

---

## Directory layout

```
cuda_image_processing/
├── src/
│   └── image_processing.cu   # All CUDA kernels + host code
├── scripts/
│   └── gen_test_image.py     # Generate synthetic PPM test images
├── input/                    # Place your .ppm images here
├── output/                   # Results written here (created automatically)
├── Makefile
├── run.sh
└── README.md
```

---

## Requirements

| Tool | Version |
|------|---------|
| CUDA Toolkit | 11.0 + |
| NVCC | bundled with CUDA Toolkit |
| Python 3 | for test-image generation (stdlib only) |
| GNU Make | any modern version |

> The project targets `sm_60` (Pascal) by default. Edit `NVCC_FLAGS` in
> `Makefile` to match your GPU architecture (e.g. `sm_75` for Turing,
> `sm_86` for Ampere).

---

## Build

```bash
make
# or
make all
```

The binary is placed in `build/image_processing`.

---

## Run

### Option A – single image

```bash
./build/image_processing <input.ppm> <output_dir>

# example
./build/image_processing input/photo.ppm output/
```

### Option B – entire directory of PPM files

```bash
./build/image_processing --dir <input_dir> <output_dir>

# example
./build/image_processing --dir input/ output/
```

### Option C – convenience shell script

```bash
# Build + run all PPMs in input/
./run.sh

# Build + run a single image
./run.sh input/photo.ppm

# Generate two synthetic test images (512×512, 1024×1024) then run
./run.sh --generate-test
```

### Option D – Make targets

```bash
make test              # generate a 512×512 test image and run the pipeline
make run INPUT=input/photo.ppm
make run-dir           # process all PPMs in input/
```

---

## Output files

For every input file `<name>.ppm` the program writes three PGM (grayscale)
images to the output directory:

| File | Content |
|------|---------|
| `<name>_gray.pgm` | Grayscale image |
| `<name>_blur.pgm` | Gaussian-blurred grayscale |
| `<name>_edges.pgm` | Sobel edge map |

PGM files can be viewed with any standard image viewer that supports
NetPBM formats (e.g. GIMP, IrfanView, `eog`, `display` from ImageMagick).

---

## CUDA kernels

### `kernel_grayscale`
Converts each RGB pixel to luminance using the ITU-R BT.709 coefficients:

```
Y = 0.2126·R + 0.7152·G + 0.0722·B
```

Each thread handles one pixel. The thread grid is 2-D with 16×16 blocks.

### `kernel_gaussian_blur`
Applies a 5×5 Gaussian kernel (stored in CUDA constant memory) with
border clamping. Sum of weights = 273.

### `kernel_sobel`
Computes the Sobel gradient magnitude:

```
Gx = [-1 -2 -1; 0 0 0; 1 2 1]
Gy = [-1 0 1; -2 0 2; -1 0 1]
|G| = sqrt(Gx² + Gy²), clamped to [0,255]
```

---

## Example output (terminal)

```
=== CUDA Image Processor ===
GPU: NVIDIA GeForce RTX 3070  (SM 8.6, 46 MPs, 8.0 GB)

[1] Processing: test_512.ppm
  Loaded: input/test_512.ppm  (512 x 512)
    -> grayscale : output/test_512_gray.pgm  (0.041 ms)
    -> blurred   : output/test_512_blur.pgm  (0.053 ms)
    -> edges     : output/test_512_edges.pgm (0.047 ms)

[2] Processing: test_1024.ppm
  Loaded: input/test_1024.ppm  (1024 x 1024)
    -> grayscale : output/test_1024_gray.pgm  (0.149 ms)
    -> blurred   : output/test_1024_blur.pgm  (0.187 ms)
    -> edges     : output/test_1024_edges.pgm (0.171 ms)

=== Summary (2 images) ===
  Grayscale kernel total : 0.190 ms
  Gaussian blur total    : 0.240 ms
  Sobel edge total       : 0.218 ms
  Grand total (kernels)  : 0.648 ms
Output files written to : output/
```

---

## Lessons learned

- **Constant memory** is well-suited for small, read-only data shared across
  all threads (the Gaussian kernel weights).  Storing the 5×5 weights in
  `__constant__` memory avoids redundant global reads.
- **Border handling** with `min/max` clamping inside the kernel is simple
  and correct; for large blur radii, shared-memory tiling would be more
  efficient.
- **CUDA events** provide high-resolution, device-side timers that are far
  more accurate than host-side `clock()` for measuring kernel duration.
- **PPM/PGM** is the simplest binary image format for CUDA projects—no
  external library needed, making the build completely self-contained.

---

## License

MIT – free to use and modify.
