#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

struct State {
    int n, k, window;
    std::vector<double> sums, history, cusum, work, exponent, invsqrt;
    State(int n_, int k_, int window_)
        : n(n_), k(k_), window(window_),
          sums(n * k, 0), history(n * (window + 1) * k, 0), cusum(n * k, 0),
          work(window * k), exponent(window * k), invsqrt(window + 1) {
        for (int l = 1; l <= window; ++l)
            invsqrt[l] = 1 / std::sqrt(double(l));
    }
};

extern "C" {

void* gb_create(int n, int k, int window) {
    return new State(n, k, window);
}

void gb_destroy(void* p) {
    delete static_cast<State*>(p);
}

void gb_step(void* p, const double* x, const uint8_t* active, int t, double* out) {
    State& s = *static_cast<State*>(p);
    const int K = s.k, W = s.window, B = W + 1, L = std::min(t, W);
    for (int i = 0; i < s.n; ++i) {
        out[2 * i] = out[2 * i + 1] = -INFINITY;
        if (active[i] & 1) {
            double* sum = &s.sums[i * K];
            double* hist = &s.history[i * B * K];
            for (int k = 0; k < K; ++k)
                sum[k] += x[i * K + k];
            for (int l = 1; l <= L; ++l) {
                const double* old = hist + ((t - l) % B) * K;
                double* a = s.exponent.data() + (l - 1) * K;
                for (int k = 0; k < K; ++k) {
                    double u = std::max(0.0, (sum[k] - old[k]) * s.invsqrt[l]);
                    a[k] = u * u / 2;
                }
            }
            int count = L * K;
#ifdef __APPLE__
            vvexp(s.work.data(), s.exponent.data(), &count);
            for (int j = 0; j < count; ++j)
                s.work[j] = .9 + .1 * s.work[j];
            vvlog(s.exponent.data(), s.work.data(), &count);
#else
            for (int j = 0; j < count; ++j)
                s.exponent[j] = std::log1p(.1 * std::expm1(s.exponent[j]));
#endif
            double best = 0;
            for (int l = 0; l < L; ++l) {
                double v = 0;
                for (int k = 0; k < K; ++k)
                    v += s.exponent[l * K + k];
                best = std::max(best, v);
            }
            out[2 * i] = best;
            std::copy(sum, sum + K, hist + (t % B) * K);
        }
        if (active[i] & 2) {
            double v = 0;
            for (int k = 0; k < K; ++k) {
                double& r = s.cusum[i * K + k];
                r = std::max(0.0, r + x[i * K + k] - .5);
                v += std::log(.9 + .064 * std::exp(r / 2));
            }
            out[2 * i + 1] = v;
        }
    }
}

}
