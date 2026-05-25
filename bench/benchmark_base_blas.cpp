#include <benchmark/benchmark.h>
#include <random>

#include "blas_core.hpp"

static void BENCHMARK_axpy_baseline(benchmark::State& state) {
	long n = state.range(0);
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
		mblas::axpy(n, alpha, x.data(), incx, y.data(), incy);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_axpy_baseline)
	->Args({16384})
	->Args({262144})
	->Args({1048576});

static void BENCHMARK_scal_baseline(benchmark::State& state) {
	long n = state.range(0);
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
		mblas::scal(n, alpha, y.data(), incy);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_scal_baseline)
	->Args({16384})
	->Args({262144})
	->Args({1048576});

static void BENCHMARK_dot_baseline(benchmark::State& state) {
	long n = state.range(0);
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
		double res = mblas::dot(n, x.data(), incx, y.data(), incy);

		benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_dot_baseline)
	->Args({16384})
	->Args({262144})
	->Args({1048576});

static void BENCHMARK_nrm2_baseline(benchmark::State& state) {
	long n = state.range(0);
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
		double res = mblas::nrm2(n, y.data(), incy);

		benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_nrm2_baseline)
	->Args({16384})
	->Args({262144})
	->Args({1048576});

static void BENCHMARK_gemv_baseline(benchmark::State& state) {
	char trans = 'N';

	int m = state.range(0), n = state.range(0);
	int lda = n;
	int incx = 1, incy = 1;

	int len_x = n;
	int len_y = m;

	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	std::vector<double> A(m * n), x(len_x), y(len_y);

	double alpha = dist(rng);
	double beta  = dist(rng);

	for (int j = 0; j < len_x; j++)
		x[j] = dist(rng);

	for (int i = 0; i < len_y; i++) {
		y[i * incy] = dist(rng);
	}

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			A[row * lda + col] = dist(rng);
		} 
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * m * n,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		mblas::gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_gemv_baseline)
	->Args({256})
	->Args({512})
	->Args({1024});

static void BENCHMARK_ger_baseline(benchmark::State& state) {
	int m = state.range(0), n = state.range(0);
	int lda = n;
	int incx = 1, incy = 1;

	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double alpha = dist(rng);

	std::vector<double> A(m * n), x(m), y(n);

	for (int col = 0; col < n; col++) {
		y[col] = dist(rng);
	}

	for (int row = 0; row < m; row++) {
		x[row] = dist(rng);
		for (int col = 0; col < n; col++) {
		int idx = row * lda + col;
		A[idx] = dist(rng);
		}
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * m * n,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		mblas::ger(m, n, alpha, x.data(), incx, y.data(), incy, A.data(), lda);

		benchmark::DoNotOptimize(A.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_ger_baseline)
	->Args({256})
	->Args({512})
	->Args({1024});

static void BENCHMARK_gemm_baseline(benchmark::State& state) {
	int transa = 'N', transb = 'N';
	int m = state.range(0), n = state.range(0), k = state.range(0);
	int lda = n, ldb = k, ldc = k;

	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double alpha = dist(rng), beta = dist(rng);
	std::vector<double> A(m * n), B(n * k), C(m * k);

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int idxA = row * lda + col;
			A[idxA] = dist(rng);
		}
	}

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < k; col++) {
			int idxB = row * ldb + col;
			B[idxB] = dist(rng);
		}
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * m * n * k,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		mblas::gemm(transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

		benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_gemm_baseline)
	->Args({256})
	->Args({512})
	->Args({1024});

BENCHMARK_MAIN();