#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

#include "cache_conscious_blas.hpp"

using namespace testing;
using namespace ccblas;

// test axpy

TEST(Level1_axpy_cache_conscious, basic_axpy_cache_conscious) {
	int prefetch_x = 1;
	int prefetch_y = 1;
	long n = 3;
	double x[3] = {1.0, 2.0, 3.0};
	double y[3] = {9.0, 8.0, 7.0};
	int incx = 1, incy = 1;
	double alpha = 1.0;

	ASSERT_EQ(axpy(n, alpha, x, incx, y, incy, prefetch_x, prefetch_y), 1);

	ASSERT_THAT(y, Each(DoubleEq(10.0)));
}

TEST(Level1_axpy_cache_conscious, hard_axpy_cache_conscious) {
	int prefetch_x = 100;
	int prefetch_y = 100;
	long n = 100000;
	std::vector<double> x(n), y(n), z(n);
	int incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * x[i] + y[i];
	}

	ASSERT_EQ(axpy(n, alpha, x.data(), incx, y.data(), incy, prefetch_x, prefetch_y), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

TEST(Level1_axpy_cache_conscious, skips_axpy_cache_conscious) {
	int prefetch_x = 10;
	int prefetch_y = 10;
	long nx = 50, ny = 10;
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

	ASSERT_EQ(axpy(ny, alpha, x.data(), incx, y.data(), incy, prefetch_x, prefetch_y), 1);
	for (long i = 0; i < ny; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

// test scal

TEST(Level1_scal_cache_conscious, basic_scal_cache_conscious) {
	int prefetch_y = 1;
	long n = 3;
	double y[3] = {1.0, 1.0, 1.0};
	int incy = 1;
	double alpha = 10.0;

	ASSERT_EQ(scal(n, alpha, y, incy, prefetch_y), 1);

	ASSERT_THAT(y, Each(DoubleEq(10.0)));
}

TEST(Level1_scal_cache_conscious, hard_scal_cache_conscious) {
	int prefetch_y = 100;
	long n = 100000;
	std::vector<double> y(n), z(n);
	int incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * y[i];
	}

	ASSERT_EQ(scal(n, alpha, y.data(), incy, prefetch_y), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

TEST(Level1_scal_cache_conscious, skips_scal_cache_conscious) {
	int prefetch_y = 100;
	long n = 100000;
	int incy = 100;
	std::vector<double> y(n), z(n / incy);
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;

		if (i % incy == 0) {
			z[i / incy] = alpha * y[i];
		}
	}

	ASSERT_EQ(scal(n / incy, alpha, y.data(), incy, prefetch_y), 1);
	for (long i = 0; i < n / incy; i++) {
    	ASSERT_DOUBLE_EQ(y[i * incy], z[i]);
	}
}

// test dot product

TEST(Level1_dot_cache_conscious, basic_dot_cache_conscious) {
	int prefetch_x = 1;
	int prefetch_y = 1;
	long n = 2;
	double x[2] = {std::sqrt(0.5), std::sqrt(0.5)}, y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incx = 1, incy = 1;

	ASSERT_DOUBLE_EQ(dot(n, x, incx, y, incy, prefetch_x, prefetch_y), 1.0);
}

TEST(Level1_dot_cache_conscious, skips_dot_cache_conscious) {
	int prefetch_x = 100;
	int prefetch_y = 100;
	long nx = 100000, ny = nx * 5;
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

    ASSERT_DOUBLE_EQ(dot(nx, x.data(), incx, y.data(), incy, prefetch_x, prefetch_y), z);
}

// test nrm2

TEST(Level1_nrm2_cache_conscious, basic_nrm2_cache_conscious) {
	int prefetch_y = 1;
	long n = 2;
	double y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incy = 1;

	ASSERT_DOUBLE_EQ(nrm2(n, y, incy, prefetch_y), 1.0);
}

TEST(Level1_nrm2_cache_conscious, skips_nrm2_cache_conscious) {
	int prefetch_y = 100;
	long n = 100000;
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

	ASSERT_DOUBLE_EQ(nrm2(n / incy, y.data(), incy, prefetch_y), z);
}
