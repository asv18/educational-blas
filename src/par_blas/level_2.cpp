#include "par_blas.hpp"

namespace pblas {
	int gemv(
		char trans,
		int m, int n,
		double alpha, double const* A, int lda,
		double const* x, int incx,
		double beta, double* y, int incy,
		int nthreads
	) {
		int mthreads = omp_get_max_threads();
		if (mthreads < nthreads) {
			nthreads = mthreads;
		}

		int rows = m, cols = n;
		int row_stride = lda, col_stride = 1;
		if (trans == 'T' || trans == 't') {
			rows = n, cols = m;
			row_stride = 1, col_stride = lda;
		} else if (trans != 'N' && trans != 'n') {
			return 0;
		}

		#pragma omp parallel for num_threads(nthreads) schedule(static)
		for (int row = 0; row < rows; row++) {
			double sum = 0.0;
			for (int col = 0; col < cols; col++) {
				int idx = row * row_stride + col * col_stride;

				sum += A[idx] * x[col * incx];
			}

			y[row * incy] *= beta;
			y[row * incy] += alpha * sum;
		}

		return 1;
	}

	int ger(
		int m, int n,
		double alpha,
		double const* x, int incx,
		double const* y, int incy,
		double* A, int lda,
		int nthreads
	) {
		#pragma omp parallel for num_threads(nthreads) schedule(static)
		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				int idx = row * lda + col;
				A[idx] += alpha * x[row * incx] * y[col * incy];
			}
		}

		return 1;
	}
}