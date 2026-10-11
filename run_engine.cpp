// run_engine.cpp - loads the exported model, checks every output against
// PyTorch's and reports accuracy, latency and throughput.
//
// usage: run_engine [model_dir] [results.json]
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "engine.h"

struct Verify {
    double max_diff = 0.0;
    int argmax_mismatch = 0, correct = 0;
};

struct Latency {
    double median_us, mean_us, p99_us;
};

static int argmax(const float* v, int n) { return (int)(std::max_element(v, v + n) - v); }

int main(int argc, char** argv) try {
    const std::string dir = argc > 1 ? argv[1] : "model";
    const char* json_path = argc > 2 ? argv[2] : nullptr;

    const auto images = engine::read_file<float>(dir + "/test_images.bin");
    const auto labels = engine::read_file<uint8_t>(dir + "/test_labels.bin");
    const auto ref = engine::read_file<float>(dir + "/ref_logits.bin");
    const int N = (int)labels.size();

    struct Config {
        const char* name;
        engine::ConvAlgo algo;
    };
    const Config configs[] = {
        {"auto", engine::ConvAlgo::Auto},
        {"direct", engine::ConvAlgo::Direct},
        {"im2col_gemm", engine::ConvAlgo::Im2col},
        {"winograd", engine::ConvAlgo::Winograd},
    };

    FILE* js = json_path ? std::fopen(json_path, "w") : nullptr;
    if (js) std::fprintf(js, "{\n  \"images\": %d,\n  \"configs\": {\n", N);

    for (size_t ci = 0; ci < 4; ++ci) {
        engine::Model model;
        model.load(dir, configs[ci].algo);
        const size_t in_n = model.input_size();
        const int classes = (int)model.output_size();
        engine::Workspace ws;
        std::vector<float> logits((size_t)N * classes);

        // --- correctness: every logit of every test image against PyTorch ---
        Verify v;
        for (int i = 0; i < N; ++i) {
            float* o = logits.data() + (size_t)i * classes;
            model.forward(images.data() + i * in_n, o, ws);
            const float* r = ref.data() + (size_t)i * classes;
            v.max_diff = std::max(v.max_diff, util::max_abs_diff(o, r, classes));
            v.argmax_mismatch += argmax(o, classes) != argmax(r, classes);
            v.correct += argmax(o, classes) == labels[i];
        }

        // --- latency: batch 1, one thread, best of 3 passes per image ---------
        std::vector<double> lat(N, 1e30);
        float out[64];
        for (int pass = 0; pass < 3; ++pass)
            for (int i = 0; i < N; ++i) {
                const double t0 = util::now();
                model.forward(images.data() + i * in_n, out, ws);
                lat[i] = std::min(lat[i], util::now() - t0);
            }
        std::sort(lat.begin(), lat.end());
        double sum = 0.0;
        for (double x : lat) sum += x;
        const Latency L{lat[N / 2] * 1e6, sum / N * 1e6, lat[(size_t)(N * 0.99)] * 1e6};

        // --- per-layer breakdown ---------------------------------------------
        std::vector<double> layer_s(model.layers().size(), 0.0);
        for (int i = 0; i < N; ++i) model.forward(images.data() + i * in_n, out, ws, layer_s.data());

        // --- throughput: images are independent, so parallelise across them ---
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            const double t0 = util::now();
#pragma omp parallel
            {
                engine::Workspace tws;
                float tout[64];
#pragma omp for schedule(dynamic, 64)
                for (int i = 0; i < N; ++i) model.forward(images.data() + i * in_n, tout, tws);
            }
            best = std::min(best, util::now() - t0);
        }
        const double throughput = N / best;

        std::printf("[conv = %s]\n", configs[ci].name);
        std::printf("  max |logit - pytorch| = %.3e   argmax mismatches = %d / %d   accuracy = %.2f%%\n", v.max_diff,
                    v.argmax_mismatch, N, 100.0 * v.correct / N);
        std::printf("  latency (batch 1, 1 thread): median %.1f us, mean %.1f us, p99 %.1f us\n", L.median_us, L.mean_us,
                    L.p99_us);
        std::printf("  throughput (%d threads): %.0f images/s\n", omp_get_max_threads(), throughput);
        std::printf("  per layer:");
        for (size_t li = 0; li < layer_s.size(); ++li)
            if (model.layers()[li].kind != engine::Layer::Flatten)
                std::printf("  %s %.1f us", model.layers()[li].name.c_str(), layer_s[li] / N * 1e6);
        std::printf("\n");

        if (js) {
            std::fprintf(js,
                         "    \"%s\": {\"max_abs_diff\": %.4e, \"argmax_mismatch\": %d, \"accuracy\": %.4f, "
                         "\"latency_us_median\": %.2f, \"latency_us_mean\": %.2f, \"latency_us_p99\": %.2f, "
                         "\"throughput_images_per_s\": %.0f, \"threads\": %d, \"layers_us\": {",
                         configs[ci].name, v.max_diff, v.argmax_mismatch, (double)v.correct / N, L.median_us, L.mean_us,
                         L.p99_us, throughput, omp_get_max_threads());
            bool first = true;
            for (size_t li = 0; li < layer_s.size(); ++li) {
                if (model.layers()[li].kind == engine::Layer::Flatten) continue;
                std::fprintf(js, "%s\"%zu_%s\": %.2f", first ? "" : ", ", li, model.layers()[li].name.c_str(),
                             layer_s[li] / N * 1e6);
                first = false;
            }
            std::fprintf(js, "}}%s\n", ci + 1 < 4 ? "," : "");
        }
    }
    if (js) {
        std::fprintf(js, "  }\n}\n");
        std::fclose(js);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
