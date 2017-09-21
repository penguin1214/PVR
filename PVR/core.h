#ifndef PVR_CORE_H
#define PVR_CORE_H

#define _USE_MATH_DEFINES
#include <math.h>

struct Presets {
	// sampling
	static const int NUM_RAY_SAMPLES = 100;
	static const int NUM_LIGHT_RAY_SAMPLES = 50;
	static const int SAMPLE_STEP_QUALITY = 2;

	// resolution
	static const int RESOLUTION_X = 600;
	static const int RESOLUTION_Y = 400;
};


#endif PVR_CORE_H

