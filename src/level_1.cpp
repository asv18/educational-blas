#include "blas.hpp"

namespace mblas {
	int axpy(long n, double alpha, const double* x, int incx, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			y[i * incy] = y[i * incy] + alpha * x[i * incx];
		}

		return 1;
	}

	int scal(long n, double alpha, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			y[i * incy] = alpha * y[i * incy];
		}

		return 1;
	}
	
	int copy(long n, const double* x, int incx, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			y[i * incy] = x[i * incx];
		}

		return 1;
	}

	int swap(long n, double* x, int incx, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			double t = y[i * incy];
			y[i * incy] = x[i * incx];
			x[i * incx]  = t;
		}

		return 1;
	}

	double dot(long n, const double* x, int incx, const double* y, int incy) {
		double z = 0.0;
		for (long i = 0; i < n; i++) {
			z += x[i * incx] * y[i * incy];
		}

		return z;
	}

	double nrm2(long n, double* y, int incy) {
		double z = 0;
		for (long i = 0; i < n; i++) {
			z += y[i * incy] * y[i * incy];
		}

		return std::sqrt(z);
	}

	double asum(long n, double* y, int incy) {
		double z = 0;
		for (long i = 0; i < n; i++) {
			z += fabs(y[i * incy]);
		}

		return z;
	}

	long iamax(long n, double* y, int incy) {
		double z = DBL_MIN;
		long idx = 0;
		for (long i = 0; i < n; i++) {
			if (fabs(y[i * incy]) > z) {
				z = fabs(y[i * incy]);
				idx = i * incy;
			}
		}

		return idx;
	}

	int rotg(double* a, double* b, double* c, double* s) {
		double r = std::hypot(*a, *b);

		if (r == 0.0) {
			*c = 1.0;
			*s = 0.0;
		} else {
			*c = *a / r;
			*s = *b / r;
		}

		*a = r;
		*b = 0.0;

		return 1;
	}
}