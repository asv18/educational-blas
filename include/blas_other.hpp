#pragma once

namespace mblas {
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
	 * @param a scalar a
	 * @param b scalar b
	 * @param c cosine of rotation
	 * @param s sine of rotation
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int rotg(double* a, double* b, double* c, double* s);

	/**
	 * Applies a plane rotation to `x` and `y`
	 * 
	 * @param n number of steps to take
	 * @param x vector `x`
	 * @param incx step-size of `x`
	 * @param y vector `y`
	 * @param incy step-size of `y`
	 * @param c cosine of rotation
	 * @param s sine of rotation
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int rot(long n, double* x, int incx, double* y, int incy, double c, double s);

	/**
	 * Applies a modified plane rotation to `x` and `y`
	 * 
	 * @param n number of steps to take
	 * @param x vector `x`
	 * @param incx step-size of `x`
	 * @param y vector `y`
	 * @param incy step-size of `y`
	 * @param param length 5 array giving parameters of modified plane rotation H - 
	 * `param[1-4]` contain the matrix i.e. h11, h21, h12, h22
	 * `param[0]` contains the flag indicating how the component's of H are set:
	 * 
	 * - -1.0 -> leave H unmodified
	 * - 0.0 -> replace h11, h22 with 1.0
	 * - 1.0 -> replace h21 and h12 with -1.0 and 1.0 respectively
	 * - -2.0 -> replace H with the identity matrix
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int rotm(long n, double* x, int incx, double* y, int incy, double param[5]);

	/**
	 * Generates a modified plane rotation
	 * 
	 * @param d1 scaling factor for x-component
	 * @param d2 scaling factor for y-component
	 * @param x1 x-component of vector
	 * @param y1 y-component of vector
	 * @param param length 5 array giving parameters of the generated modified plane rotation H - 
	 * `param[1-4]` contain the matrix i.e. h11, h21, h12, h22.
	 * 
	 * `param[0]` contains the flag indicating how the component's of H are set:
	 * 
	 * - -1.0 -> leave H unmodified
	 * 
	 * - 0.0 -> replace h11, h22 with 1.0
	 * 
	 * - 1.0 -> replace h21 and h12 with -1.0 and 1.0 respectively
	 * 
	 * - -2.0 -> replace H with the identity matrix
	 * 
	 * @return status code - 1 for success, 0 for failure
	 */
	int rotmg(double* d1, double* d2, double* x1, const double y1, double param[5]);
}