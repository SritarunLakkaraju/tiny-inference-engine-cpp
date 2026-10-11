include deps/flags.mk
PYTHON ?= python3

all: run_engine

run_engine: run_engine.cpp engine.h deps/conv.h deps/gemm.h deps/util.h
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $< -o $@ $(OMP_LDFLAGS)

# The test images ship gzip-compressed; unpack them on first use.
model/test_images.bin: model/test_images.bin.gz
	gunzip -k -c $< > $@

run: run_engine model/test_images.bin
	mkdir -p results
	./run_engine model results/engine.json | tee results/engine.txt

# Optional: retrain in PyTorch and re-export everything in model/ (pip install -r requirements.txt).
export:
	$(PYTHON) train_export.py

clean:
	rm -f run_engine

.PHONY: all run export clean
