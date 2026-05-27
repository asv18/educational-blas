#include "cache_conscious_blas.hpp"

// Since these are all linear O(n) operations, not much we can do outside of warming up the cache with prefetching
// Interesting thing to note - could there potentially be further speed ups
// if we used different step sizes? Something to explore later

namespace ccblas {
	int axpy(long n, double alpha, const double* x, int incx, double* y, int incy, int prefetch_amount_x, int prefetch_amount_y) {
		__builtin_prefetch(x + prefetch_amount_x * sizeof(double));
		__builtin_prefetch(y + prefetch_amount_y * sizeof(double));

		for (long i = 0; i < n; i++) {
			y[i * incy] += alpha * x[i * incx];
		}

		return 1;
	}

	int scal(long n, double alpha, double* y, int incy, int prefetch_amount_y) {
		__builtin_prefetch(y + prefetch_amount_y * sizeof(double));

		for (long i = 0; i < n; i++) {
			y[i * incy] = alpha * y[i * incy];
		}

		return 1;
	}

	double dot(long n, const double* x, int incx, const double* y, int incy, int prefetch_amount_x, int prefetch_amount_y) {
		__builtin_prefetch(x + prefetch_amount_x * sizeof(double));
		__builtin_prefetch(y + prefetch_amount_y * sizeof(double));

		double z = 0.0;
		for (long i = 0; i < n; i++) {
			z += x[i * incx] * y[i * incy];
		}

		return z;
	}

	double nrm2(long n, double* y, int incy, int prefetch_amount_y) {
		__builtin_prefetch(y + prefetch_amount_y * sizeof(double));
		
		double z = 0;
		for (long i = 0; i < n; i++) {
			z += y[i * incy] * y[i * incy];
		}

		return std::sqrt(z);
	}
}