#include <benchmark/benchmark.h>
#include <random>

#include "cache_conscious_blas.hpp"

static void BENCHMARK_axpy_cache_conscious(benchmark::State& state) {
	long n = state.range(0);
	int prefetch_x = state.range(1);
	int prefetch_y = state.range(1);

	std::vector<double> x(n), y(n);
	int incx = 1, incy = 1;
	double alpha = 0.5;

	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	for (long i = 0; i < n; i++) {
		x[i] = dist(rng);
		y[i] = dist(rng);
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * n,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		ccblas::axpy(n, alpha, x.data(), incx, y.data(), incy, prefetch_x, prefetch_y);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_axpy_cache_conscious)
	->Args({16384, ccblas::CACHELINE * 1})
	->Args({16384, ccblas::CACHELINE * 2})
	->Args({16384, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 2})
	->Args({262144, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 4})
	->Args({1048576, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 16})
	->Args({1048576, ccblas::CACHELINE * 32});

static void BENCHMARK_scal_cache_conscious(benchmark::State& state) {
	long n = state.range(0);
	int prefetch_y = state.range(1);
	std::vector<double> y(n);
	int incy = 1;
	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double alpha = dist(rng);

	for (long i = 0; i < n; i++) {
		y[i] = dist(rng);
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * n,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		ccblas::scal(n, alpha, y.data(), incy, prefetch_y);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_scal_cache_conscious)
	->Args({16384, ccblas::CACHELINE * 1})
	->Args({16384, ccblas::CACHELINE * 2})
	->Args({16384, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 2})
	->Args({262144, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 4})
	->Args({1048576, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 16})
	->Args({1048576, ccblas::CACHELINE * 32});

static void BENCHMARK_dot_cache_conscious(benchmark::State& state) {
	long n = state.range(0);
	int prefetch_x = state.range(1);
	int prefetch_y = state.range(1);
	std::vector<double> x(n), y(n);
	int incx = 1, incy = 1;
	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double alpha = dist(rng);

	for (long i = 0; i < n; i++) {
		x[i] = dist(rng);
		y[i] = dist(rng);
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * n,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		double res = ccblas::dot(n, x.data(), incx, y.data(), incy, prefetch_x, prefetch_y);

		benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_dot_cache_conscious)
	->Args({16384, ccblas::CACHELINE * 1})
	->Args({16384, ccblas::CACHELINE * 2})
	->Args({16384, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 2})
	->Args({262144, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 4})
	->Args({1048576, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 16})
	->Args({1048576, ccblas::CACHELINE * 32});

static void BENCHMARK_nrm2_cache_conscious(benchmark::State& state) {
	long n = state.range(0);
	int prefetch_y = state.range(1);
	std::vector<double> y(n);
	int incy = 1;
	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double alpha = dist(rng);

	for (long i = 0; i < n; i++) {
		y[i] = dist(rng);
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * n,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		double res = ccblas::nrm2(n, y.data(), incy, prefetch_y);

		benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_nrm2_cache_conscious)
	->Args({16384, ccblas::CACHELINE * 1})
	->Args({16384, ccblas::CACHELINE * 2})
	->Args({16384, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 2})
	->Args({262144, ccblas::CACHELINE * 4})
	->Args({262144, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 4})
	->Args({1048576, ccblas::CACHELINE * 8})
	->Args({1048576, ccblas::CACHELINE * 16})
	->Args({1048576, ccblas::CACHELINE * 32});