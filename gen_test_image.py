#!/usr/bin/env python3
"""
gen_test_image.py  –  Generate a synthetic PPM test image.

Creates a colorful image with geometric shapes (gradient background,
circles, rectangles, a diagonal stripe) so that edge-detection output
is visually meaningful.

Usage:
    python3 scripts/gen_test_image.py <output.ppm> [width] [height]
"""

import sys
import struct
import math


def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(v)))


def generate_ppm(path, width=512, height=512):
    data = bytearray(width * height * 3)

    for y in range(height):
        for x in range(width):
            # Background: diagonal gradient
            r = clamp((x / width) * 200 + 30)
            g = clamp((y / height) * 200 + 30)
            b = clamp(((x + y) / (width + height)) * 200 + 55)

            # White circle (centre)
            cx, cy, radius = width // 2, height // 2, min(width, height) // 5
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                r, g, b = 255, 255, 255

            # Dark rectangle (top-left quadrant)
            if width // 8 <= x < width * 3 // 8 and height // 8 <= y < height * 3 // 8:
                r, g, b = 20, 20, 80

            # Bright rectangle (bottom-right quadrant)
            if width * 5 // 8 <= x < width * 7 // 8 and height * 5 // 8 <= y < height * 7 // 8:
                r, g, b = 230, 180, 30

            # Diagonal stripe
            stripe = abs(x - y)
            if stripe < 8:
                r, g, b = 255, 50, 50

            idx = (y * width + x) * 3
            data[idx]     = r
            data[idx + 1] = g
            data[idx + 2] = b

    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(bytes(data))

    print(f"Generated: {path}  ({width}x{height})")


if __name__ == "__main__":
    out  = sys.argv[1] if len(sys.argv) > 1 else "input/test_512.ppm"
    w    = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    h    = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    generate_ppm(out, w, h)
