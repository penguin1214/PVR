#include "interpolate.h"

__host__ __device__ double interpolate(double t, double f1, double f2) {
	return (double)(f1 + (f2 - f1)*t);
}

__host__ __device__ double linearInterpolate(double *g, int i, int j, int k, double x, double y, double z, int xdim, int ydim, int zdim) {
	auto indexAt = [xdim, ydim, zdim](int i, int j, int k) -> int {
		return i + xdim*j + xdim*ydim*k;
	};
	double t1 = interpolate(x, g[indexAt(i, j, k)], g[indexAt(i + 1, j, k)]);
	double t2 = interpolate(x, g[indexAt(i, j + 1, k)], g[indexAt(i + 1, j + 1, k)]);
	double t3 = interpolate(x, g[indexAt(i, j, k + 1)], g[indexAt(i + 1, j, k + 1)]);
	double t4 = interpolate(x, g[indexAt(i, j + 1, k + 1)], g[indexAt(i + 1, j + 1, k + 1)]);

	double t5 = interpolate(y, t1, t2);
	double t6 = interpolate(y, t3, t4);

	double t7 = interpolate(z, t5, t6);
	return t7;
}