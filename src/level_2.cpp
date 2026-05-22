#include "blas_core.hpp"

namespace mblas {
	int gemv(
		char trans,
		int m, int n,
		double alpha, double const* A, int lda,
		double const* x, int incx,
		double beta, double* y, int incy
	) {
		int rows = m, cols = n;
		int row_stride = lda, col_stride = 1;
		if (trans == 'T' || trans == 't') {
			rows = n, cols = m;
			row_stride = 1, col_stride = lda;
		} else if (trans != 'N' && trans != 'n') {
			return 0;
		}

		for (int row = 0; row < m; row++) {
			y[row * incy] *= beta;
			for (int col = 0; col < n; col++) {
				int idx = row * row_stride + col * col_stride;

				y[row * incy] += alpha * A[idx] * x[col * incx];
			}
		}

		return 1;
	}

	int ger(
		int m, int n,
		double alpha,
		double const* x, int incx,
		double const* y, int incy,
		double* A, int lda
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