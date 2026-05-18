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
	 * Set `y` equal to `x`
	 * 
	 * @param n number of steps to take
	 * @param x vector to set `y` with
	 * @param incx step-size of `x`
	 * @param y vector to update
	 * @param incy step-size of `y`
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int copy(long n, const double* x, int incx, double* y, int incy);

	/**
	 * Swap elements of `y` with elements of `x`
	 * 
	 * @param n number of steps to take
	 * @param x vector to swap `y` with
	 * @param incx step-size of `x`
	 * @param y vector to update
	 * @param incy step-size of `y`
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int swap(long n, double* x, int incx, double* y, int incy);

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
	 * Computes the L-1 norm of `y`
	 * 
	 * @param n number of steps to take
	 * @param y vector to take norm of
	 * @param incy step-size of `y`
	 * 
	 * @return ||`y`||_1
	 */
	double asum(long n, double* y, int incy);

	/**
	 * Computes the L-infty norm of `y`
	 * 
	 * @param n number of steps to take
	 * @param y vector to take norm of
	 * @param incy step-size of `y`
	 * 
	 * @return argmax_i(|`y[i]`|)
	 */
	long iamax(long n, double* y, int incy);

	/**
	 * Constructs the Given's rotation and zero out `b` and update `a` with `r`
	 * 
	 * @param a first component of the vector
	 * @param b second component of the vector
	 * @param c magnitude of the diagonal component of the Givens rotation
	 * @param s magnitude of the off-diagonal component of the Givens rotation
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int rotg(double* a, double* b, double* c, double* s);

	/**
	 * Applies a plane rotation
	 * 
	 * @param n number of steps to take
	 * @param b second component of the vector
	 * @param c magnitude of the diagonal component of the Givens rotation
	 * @param s magnitude of the off-diagonal component of the Givens rotation
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int rot(long n, double* x, int incx, double*y, int incy, double c, double s);
}