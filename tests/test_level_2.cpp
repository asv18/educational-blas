#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

#include "blas_core.hpp"

using namespace testing;
using namespace mblas;

// gemv

TEST(Level2_gemv, basic_gemv1) {
	char trans = 'N';
	int m = 2, n = 2;
	double A[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 1.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy), 1);

	ASSERT_DOUBLE_EQ(x[0], y[0]);
	ASSERT_DOUBLE_EQ(x[1], y[1]);
}

TEST(Level2_gemv, basic_gemv2) {
	char trans = 'T';
	int m = 2, n = 2;
	double A[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 1.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy), 1);

	ASSERT_DOUBLE_EQ(x[0], y[0]);
	ASSERT_DOUBLE_EQ(x[1], y[1]);
}

TEST(Level2_gemv, basic_gemv3) {
	char trans = 'T';
	int m = 2, n = 2;
	double A[4] = {
		0.0, 1.0,
		1.0, 0.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 0.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy), 1);

	ASSERT_NEAR(x[1], y[0], 1e-10);
	ASSERT_NEAR(x[0], y[1], 1e-10);
}

TEST(Level2_gemv, hard_gemv) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int m = 100, n = 200;
    int lda = n;
    int incx = 1, incy = 1;

    int len_x = (trans == 'N') ? n : m;
    int len_y = (trans == 'N') ? m : n;

    std::vector<double> A(m * n), x(len_x), y(len_y), z(len_y);

    double alpha = static_cast<double>(std::rand()) / RAND_MAX;
    double beta  = static_cast<double>(std::rand()) / RAND_MAX;

    for (int j = 0; j < len_x; j++)
        x[j] = static_cast<double>(std::rand()) / RAND_MAX;

    for (int i = 0; i < len_y; i++) {
        y[i * incy] = static_cast<double>(std::rand()) / RAND_MAX;
        z[i * incy] = y[i * incy] * beta;
    }

    for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			A[row * lda + col] = static_cast<double>(std::rand()) / RAND_MAX;
		} 
	}

    if (trans == 'N') {
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                z[row * incy] += alpha * A[row * lda + col] * x[col * incx];
			}
		}
    } else {
		for (int col = 0; col < n; col++) {
            for (int row = 0; row < m; row++) {
                z[col * incy] += alpha * A[row * lda + col] * x[row * incx];
			}
		}
    }

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-10);
	}
}

TEST(Level2_gemv, skips_gemv) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int incx = 10, incy = 20;

	int m, n;
	if (trans == 'N') {
		m = incx * 100;
		n = incy * 100;
	} else {
		m = incy * 100;
		n = incx * 100;
	}
	int lda = n;

    int len_x = (trans == 'N') ? n : m;
    int len_y = (trans == 'N') ? m : n;

    std::vector<double> A(m * n), x(len_x), y(len_y), z(len_y);

    double alpha = static_cast<double>(std::rand()) / RAND_MAX;
    double beta  = static_cast<double>(std::rand()) / RAND_MAX;

    for (int j = 0; j < len_x; j++)
        x[j] = static_cast<double>(std::rand()) / RAND_MAX;

    for (int i = 0; i < len_y; i++) {
        y[i * incy] = static_cast<double>(std::rand()) / RAND_MAX;
        z[i * incy] = y[i * incy] * beta;
    }

    for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			A[row * lda + col] = static_cast<double>(std::rand()) / RAND_MAX;
		} 
	}

    if (trans == 'N') {
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                z[row * incy] += alpha * A[row * lda + col] * x[col * incx];
			}
		}
    } else {
		for (int col = 0; col < n; col++) {
            for (int row = 0; row < m; row++) {
                z[col * incy] += alpha * A[row * lda + col] * x[row * incx];
			}
		}
    }

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-10);
	}
}

// ger

TEST(Level2_ger, basic_ger1) {
	int m = 2, n = 2, lda = 2, incx = 1, incy = 1;
	double alpha = 1.0;
	double x[2] = {1.0, 0.0}, y[2] = {0.0, 1.0};
	double A[4] = {
		1.0, 0.0,
		0.0, 1.0
	};

	double rA[4] = {
		1.0, 1.0,
		0.0, 1.0
	};

	ASSERT_EQ(ger(m, n, alpha, x, incx, y, incy, A, lda), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-10), rA));
}

TEST(Level2_ger, basic_ger2) {
	int m = 2, n = 2, lda = 2, incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;
	double x[2] = {
		static_cast<double>(std::rand()) / RAND_MAX,
		static_cast<double>(std::rand()) / RAND_MAX
	}, y[2] = {static_cast<double>(std::rand()) / RAND_MAX, static_cast<double>(std::rand()) / RAND_MAX};
	double A[4] = {
		static_cast<double>(std::rand()) / RAND_MAX, static_cast<double>(std::rand()) / RAND_MAX,
		static_cast<double>(std::rand()) / RAND_MAX, static_cast<double>(std::rand()) / RAND_MAX
	};

	double rA[4] = {
		alpha * x[0] * y[0] + A[0], alpha * x[0] * y[1] + A[1],
		alpha * x[1] * y[0] + A[2], alpha * x[1] * y[1] + A[3]
	};

	ASSERT_EQ(ger(m, n, alpha, x, incx, y, incy, A, lda), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-10), rA));
}

TEST(Level2_ger, hard_ger) {
	int m = 100, n = 200;
	int lda = n;
	int incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	std::vector<double> A(m * n), x(m), y(n), rA(m * n);

	for (int col = 0; col < n; col++) {
		y[col] = static_cast<double>(std::rand()) / RAND_MAX;
	}

	for (int row = 0; row < m; row++) {
		x[row] = static_cast<double>(std::rand()) / RAND_MAX;
		for (int col = 0; col < n; col++) {
			int idx = row * lda + col;
			A[idx] = static_cast<double>(std::rand()) / RAND_MAX;
			rA[idx] = A[idx] + alpha * x[row * incx] * y[col * incy];
		}
	}

	ASSERT_EQ(ger(m, n, alpha, x.data(), incx, y.data(), incy, A.data(), lda), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-10), rA));
}

TEST(Level2_ger, skips_ger) {
	int incx = 10, incy = 20;
	int m_steps = 200, n_steps = 100;
	int m = m_steps * incx, n = n_steps * incy;
	int lda = n;

	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	std::vector<double> A(m * n), x(m), y(n), rA(m * n);

	for (int col = 0; col < n; col++) {
		y[col] = static_cast<double>(std::rand()) / RAND_MAX;
	}

	for (int row = 0; row < m; row++) {
		x[row] = static_cast<double>(std::rand()) / RAND_MAX;
	}

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int idx = row * lda + col;

			A[idx] = static_cast<double>(std::rand()) / RAND_MAX;
			rA[idx] = A[idx];
		}
	}

	for (int row = 0; row < m_steps; row++) {
		for (int col = 0; col < n_steps; col++) {
			int idx = row * lda + col;

			rA[idx] = A[idx] + alpha * x[row * incx] * y[col * incy];
		}
	}

	ASSERT_EQ(ger(m_steps, n_steps, alpha, x.data(), incx, y.data(), incy, A.data(), lda), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-10), rA));
}

// gemm

TEST(Level2_gemm, basic_gemm1) {
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

TEST(Level2_gemm, basic_gemm2) {
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

TEST(Level2_gemm, basic_gemm3) {
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

TEST(Level2_gemm, hard_gemm) {
	int transa = 'N', transb = 'N';
	int m = 10, n = 20, k = 30;
	int lda = n, ldb = k, ldc = k;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX, beta = static_cast<double>(std::rand()) / RAND_MAX;
	std::vector<double> A(m * n), B(n * k), C(m * k), rC(m * k);

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int idxA = row * lda + col;
			A[idxA] = static_cast<double>(std::rand()) / RAND_MAX;
		}
	}

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < k; col++) {
			int idxB = row * ldb + col;
			B[idxB] = static_cast<double>(std::rand()) / RAND_MAX;
		}
	}

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < k; col++) {
			int idxC = row * ldc + col;
			C[idxC] = 0;
		}
	}

	for (int i = 0; i < m ; i++) {
		for (int l = 0; l < k; l++) {
			int idxC = i * ldc + l;
			rC[idxC] *= beta;

			for (int j = 0; j < n; j++) {
				int idxA = i * lda + j;
				int idxB = j * ldb + l;
				
				rC[idxC] += alpha * A[idxA] * B[idxB];
			}

		}
	}

	ASSERT_EQ(gemm(transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc), 1);
	ASSERT_THAT(C, Pointwise(DoubleNear(1e-10), rC));
}


