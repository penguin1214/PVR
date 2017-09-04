#include "util.h"

__host__ __device__ double& d_max(double& left, double& right) {
	if (left < right)
		return right;
	else
		return left;
}

__host__ __device__ double& d_min(double& left, double& right) {
	if (left > right)
		return right;
	else
		return left;
}

__host__ __device__ void binsort(double *max, double *min) {
	if (*max < *min) {
		double temp = *max;
		*max = *min;
		*min = temp;
	}
}