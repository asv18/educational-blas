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
		int mthreads = omp_get_max_threads();
		if (mthreads < nthreads) {
			nthreads = mthreads;
		}

		#pragma omp parallel for num_threads(nthreads) schedule(static)
		// implementing: (m x k) * (k x n) = (m x n)
		for (int i = 0; i < m ; i++) {
			for (int l = 0; l < n; l++) {
				int idxC = i * ldc + l;

				double sum = 0.0;
				for (int j = 0; j < k; j++) {
					int idxA = i * lda + j;
					int idxB = j * ldb + l;
					
					sum += A[idxA] * B[idxB];
				}

				C[idxC] *= beta;
				C[idxC] += alpha * sum;
			}
		}

		return 1;
	}
}