#include "blas_core.hpp"
#include "blas_other.hpp"

namespace mblas {
	int axpy(long n, double alpha, const double* x, int incx, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			y[i * incy] = y[i * incy] + alpha * x[i * incx];
		}

		return 1;
	}

	int scal(long n, double alpha, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			y[i * incy] = alpha * y[i * incy];
		}

		return 1;
	}

	double dot(long n, const double* x, int incx, const double* y, int incy) {
		double z = 0.0;
		for (long i = 0; i < n; i++) {
			z += x[i * incx] * y[i * incy];
		}

		return z;
	}

	double nrm2(long n, double* y, int incy) {
		double z = 0;
		for (long i = 0; i < n; i++) {
			z += y[i * incy] * y[i * incy];
		}

		return std::sqrt(z);
	}


	// ================================ UNECESSARY ================================
	
	int copy(long n, const double* x, int incx, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			y[i * incy] = x[i * incx];
		}

		return 1;
	}

	int swap(long n, double* x, int incx, double* y, int incy) {
		for (long i = 0; i < n; i++) {
			double t = y[i * incy];
			y[i * incy] = x[i * incx];
			x[i * incx]  = t;
		}

		return 1;
	}

	double asum(long n, double* y, int incy) {
		double z = 0;
		for (long i = 0; i < n; i++) {
			z += fabs(y[i * incy]);
		}

		return z;
	}

	long iamax(long n, double* y, int incy) {
		double z = DBL_MIN;
		long idx = 0;
		for (long i = 0; i < n; i++) {
			if (fabs(y[i * incy]) > z) {
				z = fabs(y[i * incy]);
				idx = i * incy;
			}
		}

		return idx;
	}

	int rotg(double* a, double* b, double* c, double* s) {
		double r = std::hypot(*a, *b);

		if (r == 0.0) {
			*c = 1.0;
			*s = 0.0;
		} else {
			*c = *a / r;
			*s = *b / r;
		}

		*a = r;
		*b = 0.0;

		return 1;
	}

	int rot(long n, double* x, int incx, double* y, int incy, double c, double s) {
		for (long i = 0; i < n; i++) {
			double temp_x = x[i * incx];
			double temp_y = y[i * incy];

			x[i * incx] = c * temp_x + s * temp_y;
			y[i * incy] = c * temp_y - s * temp_x;
		}

		return 1;
	}

	int rotm(long n, double* x, int incx, double* y, int incy, double param[5]) {
		double H[4] = {
			param[1], param[3],
			param[2], param[4]
		};

		int flag = static_cast<int>(param[0]);
		if (flag == 0) {
			H[0] = 1.0;
			H[3] = 1.0;
		} else if (flag == 1) {
			H[1] = 1.0;
			H[2] = -1.0;
		} else if (flag == -2)  {
			return 1;
		} else if (flag != -1) {
			return 0;
		}

		for (long i = 0; i < n; i++) {
			double temp_x = x[i * incx];
			double temp_y = y[i * incy];

			x[i * incx] = H[0] * temp_x + H[1] * temp_y;
			y[i * incy] = H[2] * temp_x + H[3] * temp_y;
		}

		return 1;
	}
	
	int rotmg(double* d1, double* d2, double* x1, const double y1, double param[5]) {
		// TODO: no need to implement rotmg for educational implementation,
		// not sure why i implemented a lot of this stuff in the first place
		return -1;
	}
}