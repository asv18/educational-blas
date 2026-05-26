#include "par_blas.hpp"

namespace pblas {
	int axpy(long n, double alpha, const double* x, int incx, double* y, int incy, int nthreads) {
		int mthreads = omp_get_max_threads();
		if (mthreads < nthreads) {
			nthreads = mthreads;
		}
		
		#pragma omp parallel for num_threads(nthreads)
		for (long i = 0; i < n; i++) {
			y[i * incy] += alpha * x[i * incx];
		}

		return 1;
	}

	int scal(long n, double alpha, double* y, int incy, int nthreads) {
		int mthreads = omp_get_max_threads();
		if (mthreads < nthreads) {
			nthreads = mthreads;
		}

		#pragma omp parallel for num_threads(nthreads)
		for (long i = 0; i < n; i++) {
			y[i * incy] *= alpha;
		}

		return 1;
	}

	double dot(long n, const double* x, int incx, const double* y, int incy, int nthreads) {
		int mthreads = omp_get_max_threads();
		if (mthreads < nthreads) {
			nthreads = mthreads;
		}

		double z = 0.0;

		#pragma omp parallel for reduction(+ : z) num_threads(nthreads)
		for (long i = 0; i < n; i++) {
			z += x[i * incx] * y[i * incy];
		}

		return z;
	}

	double nrm2(long n, double* y, int incy, int nthreads) {
		int mthreads = omp_get_max_threads();
		if (mthreads < nthreads) {
			nthreads = mthreads;
		}

		double z = 0.0;

		#pragma omp parallel for reduction(+ : z) num_threads(nthreads)
		for (long i = 0; i < n; i++) {
			z += y[i * incy] * y[i * incy];
		}

		return std::sqrt(z);
	}
}