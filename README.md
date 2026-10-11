# Tiny inference engine in C++

A small CNN is trained in PyTorch, exported as raw weights, and run by a C++ engine built only on
hand-written kernels. The engine's output is checked against PyTorch's for every test image.

## Results

MNIST CNN, 206,922 parameters (conv 1->16, pool, conv 16->32, pool, fc 1568->128, fc 128->10):

| Check | Result |
|---|---|
| Largest difference in any logit, all 10,000 test images | 8.6e-06 |
| Images classified differently from PyTorch | 0 of 10,000 |
| Test accuracy (engine and PyTorch) | 98.66% |

| Configuration | Median latency, batch 1, 1 thread |
|---|---:|
| Engine, auto (direct conv1 + Winograd conv2) | 44.5 us |
| Engine, Winograd everywhere | 52.8 us |
| Engine, im2col + GEMM everywhere | 69.8 us |
| Engine, direct everywhere | 152.0 us |
| PyTorch 2.8, eager, 1 thread | 107.3 us |

Throughput with 11 threads (one image per thread): 152,058 images per second.

**Reading the PyTorch comparison.** On a model this small, PyTorch's time is probably dominated by
per-call framework overhead rather than arithmetic (its time was not profiled). The result shows
that a purpose-built engine has far less overhead on a tiny model; it does not show that these
kernels are twice as fast as PyTorch's.

Measured on an Apple M3 Pro (5 performance + 6 efficiency cores), Apple clang 21, `-O3 -mcpu=native`, no `-ffast-math`. Raw output is in `results/`.

## How it works

- `train_export.py` trains the model and writes `model/`: a layer list (`model.txt`), raw float32
  tensors (`weights.bin`), the normalised test set, and PyTorch's own logits (`ref_logits.bin`).
- `engine.h` reads the layer list, looks tensors up by name, checks shapes, pre-transposes fully
  connected weights and pre-transforms Winograd kernels. `forward()` walks the layers between two
  reusable buffers, so inference allocates nothing after the first call.
- The convolution algorithm is chosen per layer: the direct kernel for fewer than 8 input
  channels, Winograd otherwise.

## Requirements

- macOS on Apple Silicon (the SIMD code is ARM NEON) with Apple clang
- Homebrew `libomp`

```sh
brew install libomp
```

## Build and run

The trained model and test set are included, so no Python is needed to run the engine.

```sh
make          # build run_engine
make run      # verify against PyTorch and measure latency -> results/
```

To retrain and re-export (downloads MNIST, about a minute):

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make export PYTHON=.venv/bin/python
```

## Files

| File | Contents |
|---|---|
| `engine.h` | Weight loader, layer-list parser, `Model::forward` |
| `run_engine.cpp` | Verification, latency, per-layer times, throughput |
| `train_export.py` | PyTorch training and export |
| `model/` | Exported model, test set (images gzip-compressed) and PyTorch reference outputs |
| `deps/` | Kernels from [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) and [conv2d-kernels-cpp](https://github.com/SritarunLakkaraju/conv2d-kernels-cpp) |

## Limitations

- Supported operations: 3x3 convolution with padding 1, 2x2 max pooling, flatten, fully connected, ReLU.
- A straight sequence of layers, not a general graph; its own file format rather than ONNX.
- Batch size 1; throughput comes from running images on separate threads.

## Part of a series

Five projects that build on each other:

1. [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) - matrix multiply, the kernel everything else uses
2. [conv2d-kernels-cpp](https://github.com/SritarunLakkaraju/conv2d-kernels-cpp) - convolution on top of that GEMM
3. [tiny-inference-engine-cpp](https://github.com/SritarunLakkaraju/tiny-inference-engine-cpp) - a small inference engine on top of both
4. [int8-quantized-kernels-cpp](https://github.com/SritarunLakkaraju/int8-quantized-kernels-cpp) - the same ideas in 8-bit integers
5. [transformer-kernels-cpp](https://github.com/SritarunLakkaraju/transformer-kernels-cpp) - softmax, LayerNorm and attention

Each repository is self-contained: the headers it needs from the earlier projects are copied into `deps/`.
