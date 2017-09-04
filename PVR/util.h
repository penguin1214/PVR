#ifndef UTIL_H
#define UTIL_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>

double diffclock(clock_t clock1, clock_t clock2) {
	double diffticks = clock1 - clock2;
	double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
	return diffms / 1000.0;
}

__host__ __device__ double& d_max(double& left, double& right);

__host__ __device__ double& d_min(double& left, double& right);

__host__ __device__ void binsort(double *max, double *min);

#endif
