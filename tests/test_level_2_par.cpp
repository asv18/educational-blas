#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

#include "par_blas.hpp"

using namespace testing;
using namespace pblas;

// gemv

TEST(Level2_gemv_parallel, basic_gemv1_parallel_2_threads) {
	char trans = 'N';
	int m = 2, n = 2;
	int nthreads = 2;
	double A[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 1.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, nthreads), 1);

	ASSERT_NEAR(x[0], y[0], 1e-9);
	ASSERT_NEAR(x[1], y[1], 1e-9);
}

TEST(Level2_gemv_parallel, basic_gemv1_parallel_4_threads) {
	char trans = 'N';
	int m = 2, n = 2;
	int nthreads = 4;
	double A[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 1.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, nthreads), 1);

	ASSERT_NEAR(x[0], y[0], 1e-9);
	ASSERT_NEAR(x[1], y[1], 1e-9);
}

TEST(Level2_gemv_parallel, basic_gemv2_parallel_2_threads) {
	char trans = 'T';
	int m = 2, n = 2;
	int nthreads = 2;
	double A[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 1.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, nthreads), 1);

	ASSERT_NEAR(x[0], y[0], 1e-9);
	ASSERT_NEAR(x[1], y[1], 1e-9);
}

TEST(Level2_gemv_parallel, basic_gemv2_parallel_4_threads) {
	char trans = 'T';
	int m = 2, n = 2;
	int nthreads = 4;
	double A[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 1.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, nthreads), 1);

	ASSERT_NEAR(x[0], y[0], 1e-9);
	ASSERT_NEAR(x[1], y[1], 1e-9);
}

TEST(Level2_gemv_parallel, basic_gemv3_parallel_2_threads) {
	char trans = 'T';
	int m = 2, n = 2;
	int nthreads = 2;
	double A[4] = {
		0.0, 1.0,
		1.0, 0.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 0.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, nthreads), 1);

	ASSERT_NEAR(x[1], y[0], 1e-9);
	ASSERT_NEAR(x[0], y[1], 1e-9);
}

TEST(Level2_gemv_parallel, basic_gemv3_parallel_4_threads) {
	char trans = 'T';
	int m = 2, n = 2;
	int nthreads = 4;
	double A[4] = {
		0.0, 1.0,
		1.0, 0.0
	};
	int lda = m;

	double alpha = 1.0, beta = 0.0;
	double y[2], x[2] = {1.0, 0.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, nthreads), 1);

	ASSERT_NEAR(x[1], y[0], 1e-9);
	ASSERT_NEAR(x[0], y[1], 1e-9);
}

TEST(Level2_gemv_parallel, hard_gemv_parallel_2_threads) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int m = 100, n = 200;
	int nthreads = 2;
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

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy, nthreads), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-9);
	}
}

TEST(Level2_gemv_parallel, hard_gemv_parallel_4_threads) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int m = 100, n = 200;
	int nthreads = 4;
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

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy, nthreads), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-9);
	}
}

TEST(Level2_gemv_parallel, hard_gemv_parallel_8_threads) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int m = 100, n = 200;
	int nthreads = 8;
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

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy, nthreads), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-9);
	}
}

TEST(Level2_gemv_parallel, skips_gemv_parallel_2_threads) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int incx = 10, incy = 20;

	int m, n;
	int nthreads = 2;
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

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy, nthreads), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-9);
	}
}

TEST(Level2_gemv_parallel, skips_gemv_parallel_4_threads) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int incx = 10, incy = 20;

	int m, n;
	int nthreads = 4;
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

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy, nthreads), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-9);
	}
}

TEST(Level2_gemv_parallel, skips_gemv_parallel_8_threads) {
    char trans = (static_cast<double>(std::rand()) / RAND_MAX) > 0.5 ? 'T' : 'N';

    int incx = 10, incy = 20;

	int m, n;
	int nthreads = 8;
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

    ASSERT_EQ(gemv(trans, m, n, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy, nthreads), 1);

    for (int i = 0; i < len_y; i++) {
        ASSERT_NEAR(y[i * incy], z[i * incy], 1e-9);
	}
}

// ger

TEST(Level2_ger_parallel, basic_ger1_parallel_2_threads) {
	int m = 2, n = 2, lda = 2, incx = 1, incy = 1;
	int nthreads = 2;
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

	ASSERT_EQ(ger(m, n, alpha, x, incx, y, incy, A, lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, basic_ger1_parallel_4_threads) {
	int m = 2, n = 2, lda = 2, incx = 1, incy = 1;
	int nthreads = 4;
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

	ASSERT_EQ(ger(m, n, alpha, x, incx, y, incy, A, lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, basic_ger2_parallel_2_threads) {
	int m = 2, n = 2, lda = 2, incx = 1, incy = 1;
	int nthreads = 2;
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

	ASSERT_EQ(ger(m, n, alpha, x, incx, y, incy, A, lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, basic_ger2_parallel_4_threads) {
	int m = 2, n = 2, lda = 2, incx = 1, incy = 1;
	int nthreads = 4;
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

	ASSERT_EQ(ger(m, n, alpha, x, incx, y, incy, A, lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, hard_ger_parallel_2_threads) {
	int nthreads = 2;
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

	ASSERT_EQ(ger(m, n, alpha, x.data(), incx, y.data(), incy, A.data(), lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, hard_ger_parallel_4_threads) {
	int nthreads = 4;
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

	ASSERT_EQ(ger(m, n, alpha, x.data(), incx, y.data(), incy, A.data(), lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, hard_ger_parallel_8_threads) {
	int nthreads = 8;
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

	ASSERT_EQ(ger(m, n, alpha, x.data(), incx, y.data(), incy, A.data(), lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, skips_ger_parallel_2_threads) {
	int nthreads = 2;
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

	ASSERT_EQ(ger(m_steps, n_steps, alpha, x.data(), incx, y.data(), incy, A.data(), lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, skips_ger_parallel_4_threads) {
	int nthreads = 4;
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

	ASSERT_EQ(ger(m_steps, n_steps, alpha, x.data(), incx, y.data(), incy, A.data(), lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}

TEST(Level2_ger_parallel, skips_ger_parallel_8_threads) {
	int nthreads = 8;
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

	ASSERT_EQ(ger(m_steps, n_steps, alpha, x.data(), incx, y.data(), incy, A.data(), lda, nthreads), 1);
	ASSERT_THAT(A, Pointwise(DoubleNear(1e-9), rA));
}
