#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

#include "par_blas.hpp"

using namespace testing;
using namespace pblas;

// test axpy

TEST(Level1_axpy_parallel, basic_axpy_parallel_2_threads) {
	long n = 3;
	int nthreads = 2;
	double x[3] = {1.0, 2.0, 3.0};
	double y[3] = {9.0, 8.0, 7.0};
	int incx = 1, incy = 1;
	double alpha = 1.0;

	ASSERT_EQ(axpy(n, alpha, x, incx, y, incy, nthreads), 1);

	ASSERT_THAT(y, Each(DoubleNear(10.0, 1e-9)));
}

TEST(Level1_axpy_parallel, basic_axpy_parallel_4_threads) {
	long n = 3;
	int nthreads = 4;
	double x[3] = {1.0, 2.0, 3.0};
	double y[3] = {9.0, 8.0, 7.0};
	int incx = 1, incy = 1;
	double alpha = 1.0;

	ASSERT_EQ(axpy(n, alpha, x, incx, y, incy, nthreads), 1);

	ASSERT_THAT(y, Each(DoubleNear(10.0, 1e-9)));
}

TEST(Level1_axpy_parallel, hard_axpy_parallel_2_threads) {
	long n = 100000;
	int nthreads = 2;
	std::vector<double> x(n), y(n), z(n);
	int incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * x[i] + y[i];
	}

	ASSERT_EQ(axpy(n, alpha, x.data(), incx, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

TEST(Level1_axpy_parallel, hard_axpy_parallel_4_threads) {
	long n = 100000;
	int nthreads = 4;
	std::vector<double> x(n), y(n), z(n);
	int incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * x[i] + y[i];
	}

	ASSERT_EQ(axpy(n, alpha, x.data(), incx, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

TEST(Level1_axpy_parallel, hard_axpy_parallel_8_threads) {
	long n = 100000;
	int nthreads = 8;
	std::vector<double> x(n), y(n), z(n);
	int incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * x[i] + y[i];
	}

	ASSERT_EQ(axpy(n, alpha, x.data(), incx, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

TEST(Level1_axpy_parallel, skips_axpy_parallel_2_threads) {
	long nx = 50, ny = 10;
	int nthreads = 2;
	std::vector<double> x(nx), y(ny), z(ny);
	int incx = 5, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < nx; i++) {
		x[i] = 1.0;		
	}

	for (long i = 0; i < ny; i++) {
		y[i] = 1.0;
		z[i] = alpha * x[i * incx] + y[i];
	}

	ASSERT_EQ(axpy(ny, alpha, x.data(), incx, y.data(), incy, nthreads), 1);
	for (long i = 0; i < ny; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

TEST(Level1_axpy_parallel, skips_axpy_parallel_4_threads) {
	long nx = 50, ny = 10;
	int nthreads = 4;
	std::vector<double> x(nx), y(ny), z(ny);
	int incx = 5, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < nx; i++) {
		x[i] = 1.0;		
	}

	for (long i = 0; i < ny; i++) {
		y[i] = 1.0;
		z[i] = alpha * x[i * incx] + y[i];
	}

	ASSERT_EQ(axpy(ny, alpha, x.data(), incx, y.data(), incy, nthreads), 1);
	for (long i = 0; i < ny; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

// test scal

TEST(Level1_scal_parallel, basic_scal_parallel_2_threads) {
	long n = 3;
	int nthreads = 2;
	double y[3] = {1.0, 1.0, 1.0};
	int incy = 1;
	double alpha = 10.0;

	ASSERT_EQ(scal(n, alpha, y, incy, nthreads), 1);

	ASSERT_THAT(y, Each(DoubleNear(10.0, 1e-9)));
}

TEST(Level1_scal_parallel, basic_scal_parallel_4_threads) {
	long n = 3;
	int nthreads = 4;
	double y[3] = {1.0, 1.0, 1.0};
	int incy = 1;
	double alpha = 10.0;

	ASSERT_EQ(scal(n, alpha, y, incy, nthreads), 1);

	ASSERT_THAT(y, Each(DoubleNear(10.0, 1e-9)));
}

TEST(Level1_scal_parallel, hard_scal_parallel_2_threads) {
	long n = 100000;
	int nthreads = 2;
	std::vector<double> y(n), z(n);
	int incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * y[i];
	}

	ASSERT_EQ(scal(n, alpha, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

TEST(Level1_scal_parallel, hard_scal_parallel_4_threads) {
	long n = 100000;
	int nthreads = 4;
	std::vector<double> y(n), z(n);
	int incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * y[i];
	}

	ASSERT_EQ(scal(n, alpha, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

TEST(Level1_scal_parallel, hard_scal_parallel_8_threads) {
	long n = 100000;
	int nthreads = 8;
	std::vector<double> y(n), z(n);
	int incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * y[i];
	}

	ASSERT_EQ(scal(n, alpha, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_NEAR(y[i], z[i], 1e-9);
	}
}

TEST(Level1_scal_parallel, skips_scal_parallel_2_threads) {
	long n = 100000;
	int nthreads = 2;
	int incy = 100;
	std::vector<double> y(n), z(n / incy);
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;

		if (i % incy == 0) {
			z[i / incy] = alpha * y[i];
		}
	}

	ASSERT_EQ(scal(n / incy, alpha, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n / incy; i++) {
    	ASSERT_NEAR(y[i * incy], z[i], 1e-9);
	}
}

TEST(Level1_scal_parallel, skips_scal_parallel_4_threads) {
	long n = 100000;
	int nthreads = 4;
	int incy = 100;
	std::vector<double> y(n), z(n / incy);
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;

		if (i % incy == 0) {
			z[i / incy] = alpha * y[i];
		}
	}

	ASSERT_EQ(scal(n / incy, alpha, y.data(), incy, nthreads), 1);
	for (long i = 0; i < n / incy; i++) {
    	ASSERT_NEAR(y[i * incy], z[i], 1e-9);
	}
}

// test dot product

TEST(Level1_dot_parallel, basic_dot_parallel_2_threads) {
	long n = 2;
	int nthreads = 2;
	double x[2] = {std::sqrt(0.5), std::sqrt(0.5)}, y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incx = 1, incy = 1;

	ASSERT_NEAR(dot(n, x, incx, y, incy, nthreads), 1.0, 1e-9);
}

TEST(Level1_dot_parallel, basic_dot_parallel_4_threads) {
	long n = 2;
	int nthreads = 4;
	double x[2] = {std::sqrt(0.5), std::sqrt(0.5)}, y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incx = 1, incy = 1;

	ASSERT_NEAR(dot(n, x, incx, y, incy, nthreads), 1.0, 1e-9);
}

TEST(Level1_dot_parallel, skips_dot_parallel_2_threads) {
	long nx = 100000, ny = nx * 5;
	int nthreads = 2;
    std::vector<double> x(nx), y(ny);
	double z = 0.0;
    int incx = 5, incy = 2;

    for (long i = 0; i < nx; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
        
    for (long i = 0; i < ny; i++) {
        y[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }

    for (long i = 0; i < nx; i++) {
        z += x[i * incx] * y[i * incy];
	}

    ASSERT_NEAR(dot(nx, x.data(), incx, y.data(), incy, nthreads), z, 1e-9);
}

TEST(Level1_dot_parallel, skips_dot_parallel_4_threads) {
	long nx = 100000, ny = nx * 5;
	int nthreads = 4;
    std::vector<double> x(nx), y(ny);
	double z = 0.0;
    int incx = 5, incy = 2;

    for (long i = 0; i < nx; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
        
    for (long i = 0; i < ny; i++) {
        y[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }

    for (long i = 0; i < nx; i++) {
        z += x[i * incx] * y[i * incy];
	}

    ASSERT_NEAR(dot(nx, x.data(), incx, y.data(), incy, nthreads), z, 1e-9);
}

TEST(Level1_dot_parallel, skips_dot_parallel_8_threads) {
	long nx = 100000, ny = nx * 5;
	int nthreads = 8;
    std::vector<double> x(nx), y(ny);
	double z = 0.0;
    int incx = 5, incy = 2;

    for (long i = 0; i < nx; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
        
    for (long i = 0; i < ny; i++) {
        y[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }

    for (long i = 0; i < nx; i++) {
        z += x[i * incx] * y[i * incy];
	}

    ASSERT_NEAR(dot(nx, x.data(), incx, y.data(), incy, nthreads), z, 1e-9);
}

// test nrm2

TEST(Level1_nrm2_parallel, basic_nrm2_parallel_2_threads) {
	long n = 3;
	int nthreads = 2;
	double y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incy = 1;

	ASSERT_NEAR(nrm2(n, y, incy, nthreads), 1.0, 1e-9);
}

TEST(Level1_nrm2_parallel, basic_nrm2_parallel_4_threads) {
	long n = 3;
	int nthreads = 4;
	double y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incy = 1;

	ASSERT_NEAR(nrm2(n, y, incy, nthreads), 1.0, 1e-9);
}

TEST(Level1_nrm2_parallel, skips_nrm2_parallel_2_threads) {
	long n = 100000;
	int nthreads = 2;
	std::vector<double> y(n);
	int incy = 100;
	double z = 0.0;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		
		if (i % incy == 0) {
			z += y[i] * y[i];
		}
	}

	z = std::sqrt(z);

	ASSERT_NEAR(nrm2(n / incy, y.data(), incy, nthreads), z, 1e-9);
}

TEST(Level1_nrm2_parallel, skips_nrm2_parallel_4_threads) {
	long n = 100000;
	int nthreads = 4;
	std::vector<double> y(n);
	int incy = 100;
	double z = 0.0;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		
		if (i % incy == 0) {
			z += y[i] * y[i];
		}
	}

	z = std::sqrt(z);

	ASSERT_NEAR(nrm2(n / incy, y.data(), incy, nthreads), z, 1e-9);
}

TEST(Level1_nrm2_parallel, skips_nrm2_parallel_8_threads) {
	long n = 100000;
	int nthreads = 8;
	std::vector<double> y(n);
	int incy = 100;
	double z = 0.0;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		
		if (i % incy == 0) {
			z += y[i] * y[i];
		}
	}

	z = std::sqrt(z);

	ASSERT_NEAR(nrm2(n / incy, y.data(), incy, nthreads), z, 1e-9);
}
