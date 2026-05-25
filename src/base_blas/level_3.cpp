#include "blas_core.hpp"

namespace mblas {
	int gemm(
		int transa, int transb,
		int m, int n, int k,
		double alpha, double const* A, int lda,
		double const* B, int ldb,
		double beta, double* C, int ldc
	) {
		// (m x n) * (n x k) = (m x k)
		for (int i = 0; i < m ; i++) {
			for (int l = 0; l < k; l++) {
				int idxC = i * ldc + l;
				C[idxC] *= beta;

				for (int j = 0; j < n; j++) {
					int idxA = i * lda + j;
					int idxB = j * ldb + l;
					
					C[idxC] += alpha * A[idxA] * B[idxB];
				}

			}
		}

		return 1;
	}
}