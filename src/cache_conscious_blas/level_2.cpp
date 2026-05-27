#include "cache_conscious_blas.hpp"

// Here, we have O(nm) operations - this means we can both warm up the cache, and implement tiling

namespace ccblas {
	int gemv(
		char trans,
		int m, int n,
		double alpha, double const* A, int lda,
		double const* x, int incx,
		double beta, double* y, int incy,
		int prefetch_amount_x, int prefetch_amount_y, int prefetch_amount_A
	) {
		int rows = m, cols = n;
		int row_stride = lda, col_stride = 1;
		if (trans == 'T' || trans == 't') {
			rows = n, cols = m;
			row_stride = 1, col_stride = lda;
		} else if (trans != 'N' && trans != 'n') {
			return 0;
		}

		for (int row = 0; row < rows; row++) {
			double sum = 0.0;
			for (int col = 0; col < cols; col++) {
				int idx = row * row_stride + col * col_stride;

				sum += alpha * A[idx] * x[col * incx];
			}

			y[row * incy] *= beta;
			y[row * incy] += sum;
		}

		return 1;
	}

	int ger(
		int m, int n,
		double alpha,
		double const* x, int incx,
		double const* y, int incy,
		double* A, int lda,
		int prefetch_amount_x, int prefetch_amount_y, int prefetch_amount_A
	) {
		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				int idx = row * lda + col;
				A[idx] += alpha * x[row * incx] * y[col * incy];
			}
		}

		return 1;
	}
}