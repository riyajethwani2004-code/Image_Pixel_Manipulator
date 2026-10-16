"""
PixelForge: NumPy-only image pixel manipulator.

Treats an image as a 3D array (height x width x RGB) and creates, crops,
recolours and filters synthetic images using only NumPy. Matplotlib is used
purely to display / save results.

Usage
-----
    python pixel_toolkit.py                 # new random art every run
    python pixel_toolkit.py --seed 42       # reproducible run
    python pixel_toolkit.py --out my_dir    # choose output folder

As a module
-----------
    from pixel_toolkit import make_rng, blocks_and_stripes, FILTERS
    seed, rng = make_rng(42)
    img = blocks_and_stripes(rng)
    edges = FILTERS["edges"](img, rng)
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

H = W = 200  # default image size


# --------------------------------------------------------------------------- #
# Randomness
# --------------------------------------------------------------------------- #
def make_rng(seed=None):
    """Return (seed, Generator). seed=None picks a fresh random seed."""
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 10**8)
    return seed, np.random.default_rng(seed)


def rand_color(rng):
    """Random RGB colour as a uint8 array of shape (3,)."""
    return rng.integers(0, 256, 3).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Core operations (the project brief)
# --------------------------------------------------------------------------- #
def blank_canvas(h=H, w=W):
    return np.zeros((h, w, 3), dtype=np.uint8)


def white_canvas(h=H, w=W):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def solid_canvas(color, h=H, w=W):
    return np.full((h, w, 3), color, dtype=np.uint8)


def blocks_and_stripes(rng, h=H, w=W):
    """Random colour blocks, stripes and a random-noise patch via slicing."""
    img = solid_canvas(rand_color(rng), h, w)
    img[10:60, 20:90, 0] = 255                       # max out RED in one block
    for _ in range(5):
        y, x = rng.integers(0, 150, 2)
        bh, bw = rng.integers(20, 55, 2)
        img[y:y + bh, x:x + bw] = rand_color(rng)
    stripe = rand_color(rng)
    for y in range(0, h, 40):
        img[y:y + 6, :] = stripe
    img[:, ::25] = rand_color(rng)                   # step slicing = pinstripes
    img[130:190, 110:190] = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)  # noise patch
    return img


def only_channel(a, c):
    """Keep one colour channel (0=R, 1=G, 2=B), zero the others."""
    out = np.zeros_like(a)
    out[..., c] = a[..., c]
    return out


def brightness(a, factor):
    """uint8 -> float32 -> multiply -> clip to 0-255 -> back to uint8."""
    return np.clip(a.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def bright_mask(a, percentile=80):
    """Boolean mask of the brightest pixels (adaptive threshold)."""
    gray = a.mean(axis=2)
    return gray > np.percentile(gray, percentile)


def memory_report(a):
    """nbytes of the same image stored as uint8 / float32 / float64."""
    return {"uint8": a.astype(np.uint8).nbytes,
            "float32": a.astype(np.float32).nbytes,
            "float64": a.astype(np.float64).nbytes}


# --------------------------------------------------------------------------- #
# Filter library (all NumPy, no PIL / OpenCV / SciPy)
# --------------------------------------------------------------------------- #
def to_gray(a):
    return a.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _stack3(g):
    return np.clip(np.stack([g] * 3, axis=-1), 0, 255).astype(np.uint8)


def f_grayscale(a, rng):
    return _stack3(to_gray(a))


def f_invert(a, rng):
    return 255 - a


def f_sepia(a, rng):
    M = np.array([[.393, .769, .189], [.349, .686, .168], [.272, .534, .131]], np.float32)
    return np.clip(a.astype(np.float32) @ M.T, 0, 255).astype(np.uint8)


def f_edges(a, rng):
    """Sobel operator using shifted slices."""
    g = to_gray(a)
    gx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    gy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:])
    mag = np.pad(np.hypot(gx, gy), 1)
    return _stack3(mag / (mag.max() + 1e-6) * 255)


def f_blur(a, rng, k=7):
    """Box blur = average of k x k shifted copies."""
    p = np.pad(a.astype(np.float32), ((k // 2,) * 2, (k // 2,) * 2, (0, 0)), mode="edge")
    out = sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(k) for j in range(k)) / k ** 2
    return out.astype(np.uint8)


def f_pixelate(a, rng):
    """Reshape trick: average each block, then repeat it."""
    b = int(rng.choice([5, 8, 10, 20]))
    h, w, _ = a.shape
    h2, w2 = h // b * b, w // b * b
    small = a[:h2, :w2].reshape(h2 // b, b, w2 // b, b, 3).mean(axis=(1, 3))
    return np.repeat(np.repeat(small, b, 0), b, 1).astype(np.uint8)


def f_vignette(a, rng):
    h, w, _ = a.shape
    yy, xx = np.ogrid[:h, :w]
    d = np.hypot(yy - h / 2, xx - w / 2) / np.hypot(h / 2, w / 2)
    return (a * (1 - 0.8 * d ** 2)[..., None]).astype(np.uint8)


def f_noise(a, rng):
    return np.clip(a + rng.normal(0, 30, a.shape), 0, 255).astype(np.uint8)


def f_posterize(a, rng):
    return ((a // 64) * 64 + 32).astype(np.uint8)


def f_swap(a, rng):
    return a[..., rng.permutation(3)]


def f_glitch(a, rng):
    out = a.copy()
    for c in range(3):                               # RGB split
        out[..., c] = np.roll(a[..., c], int(rng.integers(-8, 9)), axis=1)
    for _ in range(6):                               # shifted row bands
        y = int(rng.integers(0, a.shape[0] - 10))
        hh = int(rng.integers(3, 12))
        out[y:y + hh] = np.roll(out[y:y + hh], int(rng.integers(-30, 31)), axis=1)
    return out


FILTERS = {
    "grayscale": f_grayscale, "invert": f_invert, "sepia": f_sepia, "edges": f_edges,
    "blur": f_blur, "pixelate": f_pixelate, "vignette": f_vignette, "noise": f_noise,
    "posterize": f_posterize, "channel-swap": f_swap, "glitch": f_glitch,
}


# --------------------------------------------------------------------------- #
# Display / demo
# --------------------------------------------------------------------------- #
def grid(images, titles=None, cols=None, size=3.2, suptitle=None, path=None):
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 1.1))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles:
                ax.set_title(titles[i], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def run_demo(seed=None, out_dir="outputs", show=False):
    """Run the full pipeline and save figures to out_dir."""
    seed, rng = make_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)
    print(f"Seed: {seed}  (run `python pixel_toolkit.py --seed {seed}` to reproduce)")

    # 1-2: zeros, full, blocks, stripes, noise
    img = blocks_and_stripes(rng)
    grid([blank_canvas(), white_canvas(), solid_canvas(rand_color(rng)), img],
         ["np.zeros", "np.ones x 255", "np.full", "blocks + stripes + noise"],
         suptitle=f"Creating images (seed {seed})", path=p("01_basics.png"))

    # 3: crop / flip / channels
    grid([img[50:150, 50:150], img[:, ::-1], img[::-1], only_channel(img, 0),
          only_channel(img, 1), only_channel(img, 2)],
         ["crop [50:150, 50:150]", "flip L-R [:, ::-1]", "flip U-D [::-1]",
          "red only", "green only", "blue only"], cols=3,
         suptitle="Crop, flip, channels", path=p("02_crop_flip_channels.png"))

    # 4: brightness + boolean mask
    hl = img.copy()
    hl[bright_mask(img)] = [255, 0, 255]
    grid([brightness(img, 0.4), img, brightness(img, 1.6), hl],
         ["darker x0.4", "original", "brighter x1.6", "bright-pixel mask"],
         suptitle="float32 round-trip + boolean mask", path=p("03_brightness_mask.png"))

    # 5: memory (uint8 vs float32 vs float64) + plt.imshow of uint8 and float64
    mem = memory_report(img)
    f64 = img.astype(np.float64) / 255
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].imshow(img); ax[0].axis("off")
    ax[0].set_title(f"uint8\n{img.nbytes:,} bytes")
    ax[1].imshow(f64); ax[1].axis("off")
    ax[1].set_title(f"float64 (values 0-1)\n{f64.nbytes:,} bytes")
    bars = ax[2].bar(mem.keys(), [v / 1024 for v in mem.values()],
                     color=["#4FA6B8", "#E8A04C", "#202A44"])
    ax[2].bar_label(bars, fmt="%.0f KB")
    ax[2].set_ylabel("KB")
    ax[2].set_title("Memory of one 200x200 RGB image")
    fig.tight_layout()
    fig.savefig(p("04_memory.png"), dpi=110)
    print("Memory (bytes):", mem)

    # 6: filter showcase
    names = list(FILTERS)
    grid([img] + [FILTERS[n](img, rng) for n in names], ["original"] + names,
         cols=4, size=3, suptitle="Filter library", path=p("05_filters.png"))

    if show:
        plt.show()
    plt.close("all")
    print("Saved figures to", os.path.abspath(out_dir))
    return seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PixelForge: NumPy-only image manipulator")
    ap.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    ap.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    ap.add_argument("--show", action="store_true", help="open figure windows")
    args = ap.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    run_demo(args.seed, args.out, args.show)
