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

	}

	int ger(
		int m, int n,
		double alpha,
		double const* x, int incx,
		double const* y, int incy,
		double* A, int lda,
		int nthreads
	) {

	}
}