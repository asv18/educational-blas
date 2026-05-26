#include <benchmark/benchmark.h>
#include <random>

#include "par_blas.hpp"

static void BENCHMARK_axpy_parallel(benchmark::State& state) {
	long n = state.range(0);
	int nthreads = state.range(1);
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
		pblas::axpy(n, alpha, x.data(), incx, y.data(), incy, nthreads);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_axpy_parallel)
	->Args({16384, 2})
	->Args({16384, 4})
	->Args({262144, 4})
	->Args({262144, 8})
	->Args({1048576, 8})
	->Args({1048576, 16});

static void BENCHMARK_scal_parallel(benchmark::State& state) {
	long n = state.range(0);
	int nthreads = state.range(1);
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
		pblas::scal(n, alpha, y.data(), incy, nthreads);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_scal_parallel)
	->Args({16384, 2})
	->Args({16384, 4})
	->Args({262144, 4})
	->Args({262144, 8})
	->Args({1048576, 8})
	->Args({1048576, 16});

static void BENCHMARK_dot_parallel(benchmark::State& state) {
	long n = state.range(0);
	int nthreads = state.range(1);
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
		double res = pblas::dot(n, x.data(), incx, y.data(), incy, nthreads);

		benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_dot_parallel)
	->Args({16384, 2})
	->Args({16384, 4})
	->Args({262144, 4})
	->Args({262144, 8})
	->Args({1048576, 8})
	->Args({1048576, 16});

static void BENCHMARK_nrm2_parallel(benchmark::State& state) {
	long n = state.range(0);
	int nthreads = state.range(1);
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
		double res = pblas::nrm2(n, y.data(), incy, nthreads);

		benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_nrm2_parallel)
	->Args({16384, 2})
	->Args({16384, 4})
	->Args({262144, 4})
	->Args({262144, 8})
	->Args({1048576, 8})
	->Args({1048576, 16});

static void BENCHMARK_gemv_parallel(benchmark::State& state) {
	char trans = 'N';

	int m = state.range(0), n = state.range(0);
	int nthreads = state.range(1);
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
		pblas::gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy, nthreads);

		benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_gemv_parallel)
	->Args({256, 2})
	->Args({256, 4})
	->Args({512, 4})
	->Args({512, 8})
	->Args({1024, 8})
	->Args({1024, 16});

static void BENCHMARK_gemm_parallel(benchmark::State& state) {
	int transa = 'N', transb = 'N';
	int m = state.range(0), n = state.range(0), k = state.range(0);
	int nthreads = state.range(1);
	int lda = k, ldb = n, ldc = n;

	std::mt19937 rng(0);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double alpha = dist(rng), beta = dist(rng);
	std::vector<double> A(m * k), B(k * n), C(m * n);

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < k; col++) {
			int idxA = row * lda + col;
			A[idxA] = dist(rng);
		}
	}

	for (int row = 0; row < k; row++) {
		for (int col = 0; col < n; col++) {
			int idxB = row * ldb + col;
			B[idxB] = dist(rng);
		}
	}

	state.counters["FLOPS"] = benchmark::Counter(
        2.0 * m * n * k,
        benchmark::Counter::kIsIterationInvariantRate
	);

    for (auto _ : state) {
		pblas::gemm(transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc, nthreads);

		benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BENCHMARK_gemm_parallel)
	->Args({256, 2})
	->Args({256, 4})
	->Args({512, 4})
	->Args({512, 8})
	->Args({1024, 8})
	->Args({1024, 16});