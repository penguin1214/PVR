#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ double linearInterpolate(double *g, int i, int j, int k, double x, double y, double z, int xdim, int ydim, int zdim);

#endif
