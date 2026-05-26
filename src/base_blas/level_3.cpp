#include "blas_core.hpp"

namespace mblas {
	int gemm(
		int transa, int transb,
		int m, int n, int k,
		double alpha, double const* A, int lda,
		double const* B, int ldb,
		double beta, double* C, int ldc
	) {
		// (m x k) * (k x n) = (m x n)
		for (int i = 0; i < m ; i++) {
			for (int l = 0; l < n; l++) {
				int idxC = i * ldc + l;

				double sum = 0.0;
				for (int j = 0; j < k; j++) {
					int idxA = i * lda + j;
					int idxB = j * ldb + l;
					
					sum += alpha * A[idxA] * B[idxB];
				}

				C[idxC] *= beta;
				C[idxC] += sum;
			}
		}

		return 1;
	}
}