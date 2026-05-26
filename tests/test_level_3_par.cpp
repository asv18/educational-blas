#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

#include "par_blas.hpp"

using namespace testing;
using namespace pblas;

// gemm

TEST(Level2_gemm_parallel, basic_gemm1_parallel_2_threads) {
	int nthreads = 2;
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

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}

TEST(Level2_gemm_parallel, basic_gemm1_parallel_4_threads) {
	int nthreads = 4;
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

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}

TEST(Level2_gemm_parallel, basic_gemm2_parallel_2_threads) {
	int nthreads = 2;
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

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}

TEST(Level2_gemm_parallel, basic_gemm2_parallel_4_threads) {
	int nthreads = 4;
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

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}

TEST(Level2_gemm_parallel, basic_gemm3_parallel_2_threads) {
	int nthreads = 2;
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

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}

TEST(Level2_gemm_parallel, basic_gemm3_parallel_4_threads) {
	int nthreads = 4;
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

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}

TEST(Level2_gemm_parallel, hard_gemm_parallel_2_threads) {
	int nthreads = 2;
	int transa = 'N', transb = 'N';
	int m = 10, n = 20, k = 30;
	int lda = k, ldb = n, ldc = n;
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

	for (int i = 0; i < m; i++) {
    for (int l = 0; l < n; l++) {
        int idxC = i * ldc + l;
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            sum += A[i * lda + j] * B[j * ldb + l];
        }
        rC[idxC] = alpha * sum + beta * rC[idxC];
    }
}

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc, nthreads), 1);
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int idxC = row * ldc + col;
			ASSERT_NEAR(C[idxC], rC[idxC], 1e-9);
		}
	}

}

TEST(Level2_gemm_parallel, hard_gemm_parallel_4_threads) {
	int nthreads = 4;
	int transa = 'N', transb = 'N';
	int m = 10, n = 20, k = 30;
	int lda = k, ldb = n, ldc = n;
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

	for (int i = 0; i < m; i++) {
    for (int l = 0; l < n; l++) {
        int idxC = i * ldc + l;
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            sum += A[i * lda + j] * B[j * ldb + l];
        }
        rC[idxC] = alpha * sum + beta * rC[idxC];
    }
}

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}

TEST(Level2_gemm_parallel, hard_gemm_parallel_8_threads) {
	int nthreads = 8;
	int transa = 'N', transb = 'N';
	int m = 10, n = 20, k = 30;
	int lda = k, ldb = n, ldc = n;
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

	for (int i = 0; i < m; i++) {
    for (int l = 0; l < n; l++) {
        int idxC = i * ldc + l;
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            sum += A[i * lda + j] * B[j * ldb + l];
        }
        rC[idxC] = alpha * sum + beta * rC[idxC];
    }
}

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc, nthreads), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-9), rC));
}