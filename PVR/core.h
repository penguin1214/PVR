#ifndef PVR_CORE_H
#define PVR_CORE_H

#define _USE_MATH_DEFINES
#include <math.h>

#define G_NUM_PHOTON 200000
#define G_MAX_DEPTH 1
#define G_TMIN -1000.0f
#define G_TMAX 1000.0f
#define G_RADIANCE_ESTIMATE_R	1.0f

struct Presets {
	// sampling
	static const int NUM_RAY_SAMPLES = 500;
	static const int NUM_LIGHT_RAY_SAMPLES = 500;
	static const int SAMPLE_STEP_QUALITY = 2;

	// resolution
	static const int RESOLUTION_X = 400;
	static const int RESOLUTION_Y = 400;

};


#endif PVR_CORE_H

