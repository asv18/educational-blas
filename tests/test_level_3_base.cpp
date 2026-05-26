#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

#include "blas_core.hpp"

using namespace testing;
using namespace mblas;

// gemm

TEST(Level2_gemm_baseline, basic_gemm1_baseline) {
	int transa = 'N', transb = 'N';
	int m = 2, n = 2, k = 2;
	int lda = 2, ldb = 2, ldc = 2;
	double alpha = 1.0, beta = 1.0;
	double const A[4] = {
		0.0, 1.0,
		1.0, 0.0
	};
	double const B[4] = {
		0.0, 1.0,
		1.0, 0.0
	};
	double C[4] = {
		0.0, 0.0,
		0.0, 0.0
	};
	double rC[4] = {
		1.0, 0.0,
		0.0, 1.0
	};

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-10), rC));
}

TEST(Level2_gemm_baseline, basic_gemm2_baseline) {
	int transa = 'T', transb = 'T';
	int m = 2, n = 2, k = 2;
	int lda = 2, ldb = 2, ldc = 2;
	double alpha = 1.0, beta = 1.0;
	double const A[4] = {
		0.0, 1.0,
		1.0, 0.0
	};
	double const B[4] = {
		0.0, 1.0,
		1.0, 0.0
	};
	double C[4] = {
		0.0, 0.0,
		0.0, 0.0
	};
	double rC[4] = {
		1.0, 0.0,
		0.0, 1.0
	};

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-10), rC));
}

TEST(Level2_gemm_baseline, basic_gemm3_baseline) {
	int transa = 'N', transb = 'N';
	int m = 2, n = 2, k = 2;
	int lda = 2, ldb = 2, ldc = 2;
	double alpha = 1.0, beta = 1.0;
	double const A[4] = {
		1.0, 2.0,
		3.0, 4.0
	};
	double const B[4] = {
		5.0, 6.0,
		7.0, 8.0
	};
	double C[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	double rC[4] = {
		20.0, 22.0,
		43.0, 51.0
	};

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-10), rC));
}

TEST(Level2_gemm_baseline, hard_gemm_baseline) {
	int transa = 'N', transb = 'N';
	int m = 10, n = 20, k = 30;
	int lda = n, ldb = k, ldc = n;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX, beta = static_cast<double>(std::rand()) / RAND_MAX;
	std::vector<double> A(m * k), B(k * n), C(m * n), rC(m * n);

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < k; col++) {
			int idxA = row * lda + col;
			A[idxA] = static_cast<double>(std::rand()) / RAND_MAX;
		}
	}

	for (int row = 0; row < k; row++) {
		for (int col = 0; col < n; col++) {
			int idxB = row * ldb + col;
			B[idxB] = static_cast<double>(std::rand()) / RAND_MAX;
		}
	}

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int idxC = row * ldc + col;
			C[idxC] = 0;
		}
	}

	for (int i = 0; i < m ; i++) {
		for (int l = 0; l < n; l++) {
			int idxC = i * ldc + l;
			rC[idxC] *= beta;

			for (int j = 0; j < k; j++) {
				int idxA = i * lda + j;
				int idxB = j * ldb + l;
				
				rC[idxC] += alpha * A[idxA] * B[idxB];
			}

		}
	}

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-10), rC));
}