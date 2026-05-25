#include "par_blas.hpp"

namespace pblas {
	int gemm(
		int transa, int transb,
		int m, int n, int k,
		double alpha, double const* A, int lda,
		double const* B, int ldb,
		double beta, double* C, int ldc,
		int nthreads
	) {

	}
}