#pragma once

#include <cmath>
#include <cfloat>

namespace mblas {
	/**
	 * Computes `y = y + ax`
	 * 
	 * @param n number of steps to take
	 * @param alpha scalar to multiply to `x`
	 * @param x vector to update `y` with
	 * @param incx step-size of `x`
	 * @param y vector to update
	 * @param incy step-size of `y`
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int axpy(long n, double alpha, const double* x, int incx, double* y, int incy);

	/**
	 * Computes `y = ay`
	 * 
	 * @param n number of steps to take
	 * @param alpha scaling factor
	 * @param y vector to scale
	 * @param incy step-size of `y`
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int scal(long n, double alpha, double* y, int incy);

	/**
	 * Computes and returns `x^T * y`
	 * 
	 * @param n number of steps to take
	 * @param x vector to dot `y` with
	 * @param incx step-size of `x`
	 * @param y vector to dot
	 * @param incy step-size of `y`
	 * 
	 * @return `x^T * y`
	 */
	double dot(long n, const double* x, int incx, const double* y, int incy);

	/**
	 * Computes the L-2 norm of `y`
	 * 
	 * @param n number of steps to take
	 * @param y vector to take norm of
	 * @param incy step-size of `y`
	 * 
	 * @return ||`y`||_2
	 */
	double nrm2(long n, double* y, int incy);

	/**
	 * Computes a matrix-vector product
	 * 
	 * @param trans character representing whether to take the transpose of A; 
	 * 'N' or 'n' -> no transpose
	 * 'T' or 't' -> transpose
	 * @param m, n dimensions of our m-by-n matrix A
	 * @param alpha scalar alpha multiplied to A
	 * @param A flat m-by-n matrix A
	 * @param lda specifies leading dimension of A (at least max(1, m))
	 * @param x vector multipled by A
	 * @param incx step-size of x
	 * @param beta scalar multiplied to y
	 * @param y array that contains the updated value; y := alpha * trans(A) * x + beta * y
	 * @param incy step-size of y
	 * 
	 * @return status code - 1 for success, 0 for failure
	 * 
	 */
	int gemv(
		char trans,
		int m, int n,
		double alpha, double const* A, int lda,
		double const* x, int incx,
		double beta, double* y, int incy
	);

	/**
	 * Perform rank-1 update of a general matrix
	 * 
	 * @param m, n dimensions of our m-by-n matrix A
	 * @param alpha scalar alpha multiplied to A
	 * @param x first vector that updates A
	 * @param incx step-size of x
	 * @param y second vector that updates A
	 * @param incy step-size of y
	 * @param A flat m-by-n matrix A; updated with A := alpha * x * y' + A
	 * @param lda specifies leading dimension of A (at least max(1, m))
	 * 
	 * @return status code - 1 for success, 0 for failure
	 * 
	 */
	int ger(
		int m, int n,
		double alpha,
		double const* x, int incx,
		double const* y, int incy,
		double* A, int lda
	);

	/**
	 * Perform rank-1 update of a general matrix
	 * 
	 * @param transa, transb character representing whether to take the transpose of A or B; 
	 * 'N' or 'n' -> no transpose
	 * 'T' or 't' -> transpose
	 * @param m, n, k dimensions of our matrix A and B, and resulting matrix C
	 * @param alpha scalar alpha multiplied to transa(A) * transb(B)
	 * @param A flat m-by-n matrix A
	 * @param lda specifies leading dimension of A (at least max(1, m))
	 * @param B flat n-by-k matrix B
	 * @param ldb specifies leading dimension of B (at least max(1, n))
	 * @param beta scalar beta multiplied to C
	 * @param C flat m-by-k matrix C; updated by C := alpha * transa(A) * transb(B) + beta * C
	 * @param ldc specifies leading dimension of C (at least max(1, m))
	 * 
	 * @return status code - 1 for success, 0 for failure
	 * 
	 */
	int gemm(
		int transa, int transb,
		int m, int n, int k,
		double alpha, double const* A, int lda,
		double const* B, int ldb,
		double beta, double* C, int ldc
	);
}