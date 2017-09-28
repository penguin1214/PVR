#ifndef PVR_CORE_H
#define PVR_CORE_H

#define _USE_MATH_DEFINES
#include <math.h>

#define G_NUM_PHOTON 20000
#define G_MAX_DEPTH 5

struct Presets {
	// sampling
	static const int NUM_RAY_SAMPLES = 500;
	static const int NUM_LIGHT_RAY_SAMPLES = 500;
	static const int SAMPLE_STEP_QUALITY = 2;

	// resolution
	static const int RESOLUTION_X = 1500;
	static const int RESOLUTION_Y = 1200;

};


#endif PVR_CORE_H

