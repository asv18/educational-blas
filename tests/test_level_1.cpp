#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>

#include "blas.hpp"

using namespace testing;
using namespace mblas;

// test axpy

TEST(Level1_axpy, basic_axpy) {
	long n = 3;
	double x[3] = {1.0, 2.0, 3.0};
	double y[3] = {9.0, 8.0, 7.0};
	int incx = 1, incy = 1;
	double alpha = 1.0;

	ASSERT_EQ(axpy(n, alpha, x, incx, y, incy), 1);

	ASSERT_THAT(y, Each(DoubleEq(10.0)));
}

TEST(Level1_axpy, hard_axpy) {
	long n = 100000;
	std::vector<double> x(n), y(n), z(n);
	int incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * x[i] + y[i];
	}

	ASSERT_EQ(axpy(n, alpha, x.data(), incx, y.data(), incy), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

TEST(Level1_axpy, skips_axpy) {
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

	ASSERT_EQ(axpy(ny, alpha, x.data(), incx, y.data(), incy), 1);
	for (long i = 0; i < ny; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

TEST(Level1_axpy, hard_skips_axpy) {
    long nx = 100000, ny = nx * 5;
    std::vector<double> x(nx), y(ny), z(ny);
    int incx = 5, incy = 2;
    double alpha = static_cast<double>(std::rand()) / RAND_MAX;

    for (long i = 0; i < nx; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
        
    for (long i = 0; i < ny; i++) {
        y[i] = static_cast<double>(std::rand()) / RAND_MAX;
        z[i] = y[i];
    }

    for (long i = 0; i < nx; i++) {
        z[i * incy] = alpha * x[i * incx] + y[i * incy];
	}

    ASSERT_EQ(axpy(nx, alpha, x.data(), incx, y.data(), incy), 1);
    for (long i = 0; i < ny; i++) {
        ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

TEST(Level1_axpy, long_axpy) {
	long n = 1000000;
	std::vector<double> x(n), y(n), z(n);
	int incx = 1, incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * x[i] + y[i];
	}

	ASSERT_EQ(axpy(n, alpha, x.data(), incx, y.data(), incy), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

// test scal

TEST(Level1_scal, basic_scal) {
	long n = 3;
	double y[3] = {1.0, 1.0, 1.0};
	int incy = 1;
	double alpha = 10.0;

	ASSERT_EQ(scal(n, alpha, y, incy), 1);

	ASSERT_THAT(y, Each(DoubleEq(10.0)));
}

TEST(Level1_scal, hard_scal) {
	long n = 100000;
	std::vector<double> y(n), z(n);
	int incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * y[i];
	}

	ASSERT_EQ(scal(n, alpha, y.data(), incy), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

TEST(Level1_scal, skips_scal) {
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

	ASSERT_EQ(scal(n / incy, alpha, y.data(), incy), 1);
	for (long i = 0; i < n / incy; i++) {
    	ASSERT_DOUBLE_EQ(y[i * incy], z[i]);
	}
}

TEST(Level1_scal, long_scal) {
	long n = 1000000;
	std::vector<double> y(n), z(n);
	int incy = 1;
	double alpha = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;
		z[i] = alpha * y[i];
	}

	ASSERT_EQ(scal(n, alpha, y.data(), incy), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

// test copy
TEST(Level1_copy, basic_copy) {
	long n = 3;
	double x[3] = {1.0, 2.0, 3.0}, y[3] = {4.0, 5.0, 6.0}, z[3] = {1.0, 2.0, 3.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(copy(n, x, incx, y, incy), 1);
	for (long i = 0; i < n; i++) {
		ASSERT_EQ(z[i], y[i]);
	}
}

TEST(Level1_copy, skips_copy) {
	long nx = 100000, ny = nx * 5;
    std::vector<double> x(nx), y(ny), z(ny);
    int incx = 5, incy = 2;

    for (long i = 0; i < nx; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
        
    for (long i = 0; i < ny; i++) {
        y[i] = static_cast<double>(std::rand()) / RAND_MAX;
        z[i] = y[i];
    }

    for (long i = 0; i < nx; i++) {
        z[i * incy] = x[i * incx];
	}

    ASSERT_EQ(copy(nx, x.data(), incx, y.data(), incy), 1);
    for (long i = 0; i < ny; i++) {
        ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

// test swap

TEST(Level1_swap, basic_swap) {
	long n = 3;
	double x[3] = {1.0, 2.0, 3.0}, y[3] = {4.0, 5.0, 6.0}, z[3] = {1.0, 2.0, 3.0};
	int incx = 1, incy = 1;

	ASSERT_EQ(swap(n, x, incx, y, incy), 1);
	for (long i = 0; i < n; i++) {
		ASSERT_EQ(z[i], y[i]);
	}
}

TEST(Level1_swap, skips_swap) {
	long nx = 100000, ny = nx * 5;
    std::vector<double> x(nx), y(ny), z(ny);
    int incx = 5, incy = 2;

    for (long i = 0; i < nx; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
        
    for (long i = 0; i < ny; i++) {
        y[i] = static_cast<double>(std::rand()) / RAND_MAX;
        z[i] = y[i];
    }

    for (long i = 0; i < nx; i++) {
        z[i * incy] = x[i * incx];
	}

    ASSERT_EQ(swap(nx, x.data(), incx, y.data(), incy), 1);
    for (long i = 0; i < ny; i++) {
        ASSERT_DOUBLE_EQ(y[i], z[i]);
	}
}

// test dot product

TEST(Level1_dot, basic_dot) {
	long n = 2;
	double x[2] = {std::sqrt(0.5), std::sqrt(0.5)}, y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incx = 1, incy = 1;

	ASSERT_DOUBLE_EQ(dot(n, x, incx, y, incy), 1.0);
}

TEST(Level1_dot, skips_dot) {
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

    ASSERT_DOUBLE_EQ(dot(nx, x.data(), incx, y.data(), incy), z);
}

// test nrm2

TEST(Level1_nrm2, basic_nrm2) {
	long n = 3;
	double y[2] = {std::sqrt(0.5), std::sqrt(0.5)};
	int incy = 1;

	ASSERT_DOUBLE_EQ(nrm2(n, y, incy), 1.0);
}

TEST(Level1_nrm2, skips_nrm2) {
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

	ASSERT_DOUBLE_EQ(nrm2(n / incy, y.data(), incy), z);
}

// test asum

TEST(Level1_asum, basic_asum) {
	long n = 2;
	double y[2] = { -1.0, 2.0 };
	int incy = 1;

	ASSERT_DOUBLE_EQ(asum(n, y, incy), 3.0);
}

TEST(Level1_asum, skips_asum) {
	long n = 100000;
	std::vector<double> y(n);
	int incy = 100;
	double z = 0.0;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;
		
		if (i % incy == 0) {
			z += fabs(y[i]);
		}
	}

	ASSERT_DOUBLE_EQ(asum(n / incy, y.data(), incy), z);
}

// test iamax

TEST(Level1_iamax, basic_iamax) {
	long n = 3;
	double y[3] = {-5.0, 6.0, 10.0};
	int incy = 1;

	ASSERT_DOUBLE_EQ(iamax(n, y, incy), 2);
}

TEST(Level1_iamax, skips_iamax) {
	long n = 100000;
	std::vector<double> y(n);
	int incy = 40;
	double z = DBL_MIN;
	long idx = 0;

	for (long i = 0; i < n; i++) {
		y[i] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;
		
		if (i % incy == 0 && fabs(y[i]) > z) {
			z = fabs(y[i]);
			idx = i;
		}
	}

	ASSERT_DOUBLE_EQ(iamax(n / incy, y.data(), incy), idx);
}

// test rotg

TEST(Level1_rotg, basic_rotg) {
	double a = 3.0, b = 4.0, c, s;

	double r = 5.0;
	double res_c = 0.6;
	double res_s = 0.8;

	ASSERT_EQ(rotg(&a, &b, &c, &s), 1);

	ASSERT_DOUBLE_EQ(a, r);
	ASSERT_DOUBLE_EQ(b, 0.0);

	ASSERT_DOUBLE_EQ(c, res_c);
	ASSERT_DOUBLE_EQ(s, res_s);
}

TEST(Level1_rotg, hard_rotg) {
	double a, b, c, s;

	long n = 100000;
	for (long i = 0; i < n; i++) {
		a = static_cast<double>(std::rand()) / RAND_MAX - .5;
		b = static_cast<double>(std::rand()) / RAND_MAX - .5;

		double orig_a = a;
		double orig_b = b;
		double r = std::hypot(a, b);

		ASSERT_EQ(rotg(&a, &b, &c, &s), 1);

		ASSERT_NEAR(a, r, 1e-10);
		ASSERT_NEAR(-s * orig_a + c * orig_b, 0.0, 1e-10); // allow for floating point tolerance
	}
}

// test rot

TEST(Level1_rot, basic_rot) {
	long n = 3;
	double x[3] = {1.0, 2.0, 3.0}, y[3] = {4.0, 5.0, 6.0};
	int incx = 1, incy = 1;
	double c = 1.0, s = 1.0;

	ASSERT_EQ(rot(n, x, incx, y, incy, c, s), 1);

	double r_x[3] = {5.0, 7.0, 9.0};
	double r_y[3] = {3.0, 3.0, 3.0};
	
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(r_x[i], x[i]);
		ASSERT_DOUBLE_EQ(r_y[i], y[i]);
	}
}

TEST(Level1_rot, hard_rot) {
	long n = 100000;
	std::vector<double> x(n), y(n), r_x(n), r_y(n);
	int incx = 1, incy = 1;
	double c = static_cast<double>(std::rand()) / RAND_MAX, s = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;

		r_x[i] = c * x[i] + s * y[i];
		r_y[i] = c * y[i] - s * x[i];
	}

	ASSERT_EQ(rot(n, x.data(), incx, y.data(), incy, c, s), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(r_x[i], x[i]);
		ASSERT_DOUBLE_EQ(r_y[i], y[i]);
	}
}

TEST(Level1_rot, skips_rot) {
	long steps = 1000;
	int incx = 5, incy = 2;
	long n = steps * incx * incy;
	std::vector<double> x(n), y(n), r_x(n), r_y(n);
	double c = static_cast<double>(std::rand()) / RAND_MAX, s = static_cast<double>(std::rand()) / RAND_MAX;

	for (long i = 0; i < n; i++) {
		x[i] = static_cast<double>(std::rand()) / RAND_MAX;
		y[i] = static_cast<double>(std::rand()) / RAND_MAX;

		r_x[i] = x[i];
		r_y[i] = y[i];
	}

	for (long i = 0; i < steps; i++) {
		r_x[i * incx] = c * x[i * incx] + s * y[i * incy];
		r_y[i * incy] = c * y[i * incy] - s * x[i * incx];
	}

	ASSERT_EQ(rot(steps, x.data(), incx, y.data(), incy, c, s), 1);
	for (long i = 0; i < n; i++) {
    	ASSERT_DOUBLE_EQ(r_x[i], x[i]);
		ASSERT_DOUBLE_EQ(r_y[i], y[i]);
	}
}

// test rotm

TEST(Level1_rotm, basic_rotm) {
    long n = 3;
    double x[3] = {1.0, 2.0, 3.0};
    double y[3] = {4.0, 5.0, 6.0};
    int incx = 1, incy = 1;
    double param[5] = {-1.0, 2.0, 3.0, 4.0, 5.0};

    ASSERT_EQ(rotm(n, x, incx, y, incy, param), 1.0);

    double r_x[3] = {18.0, 24.0, 30.0};
    double r_y[3] = {23.0, 31.0, 39.0};

    for (long i = 0; i < n; i++) {
        ASSERT_DOUBLE_EQ(r_x[i], x[i]);
        ASSERT_DOUBLE_EQ(r_y[i], y[i]);
    }
}

TEST(Level1_rotm, no_op_rotm) {
    long n = 4;
    double x[4] = {1.0, -2.0, 3.5, -4.5};
    double y[4] = {4.0, -1.0, 2.5,  0.5};
    int incx = 1, incy = 1;
    double param[5] = {-2.0, 0.0, 0.0, 0.0, 0.0};

    double orig_x[4] = {1.0, -2.0, 3.5, -4.5};
    double orig_y[4] = {4.0, -1.0, 2.5,  0.5};

    ASSERT_EQ(rotm(n, x, incx, y, incy, param), 1.0);
    for (long i = 0; i < n; i++) {
        ASSERT_DOUBLE_EQ(orig_x[i], x[i]);
        ASSERT_DOUBLE_EQ(orig_y[i], y[i]);
    }
}

TEST(Level1_rotm, skips_rotm) {
    long steps = 1000;
    int incx = 5, incy = 2;
    long n = steps * incx;
    long m = steps * incy;
    std::vector<double> x(n), y(m), r_x(n), r_y(m);
    double param[5] = {-1.0, 1.5, -2.0, 0.5, -1.0};

    for (long i = 0; i < n; i++) {
        x[i] = static_cast<double>(std::rand()) / RAND_MAX;
        r_x[i] = x[i];
    }

    for (long i = 0; i < m; i++) {
        y[i] = static_cast<double>(std::rand()) / RAND_MAX;
        r_y[i] = y[i];
    }

    for (long i = 0; i < steps; i++) {
        long xi = i * incx;
        long yi = i * incy;
        r_x[xi] = param[1] * x[xi] + param[3] * y[yi];
        r_y[yi] = param[2] * x[xi] + param[4] * y[yi];
    }

    ASSERT_EQ(rotm(steps, x.data(), incx, y.data(), incy, param), 1.0);
    for (long i = 0; i < n; i++) {
        ASSERT_DOUBLE_EQ(r_x[i], x[i]);
    }
    for (long i = 0; i < m; i++) {
        ASSERT_DOUBLE_EQ(r_y[i], y[i]);
    }
}


