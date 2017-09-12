#ifndef RENDERER_H
#define RENDERER_H 

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>

#include "Vector3.h"
#include "camera.h"
#include "volume.h"
#include "interpolate.h"


#define kCIELEN 95

extern __constant__ int dev_xsize;
extern __constant__ int dev_ysize;
extern __constant__ int dev_gridx;
extern __constant__ int dev_gridy;
extern __constant__ int dev_gridz;
extern __constant__ double dev_oa;
extern __constant__ double dev_os;
extern __constant__ double dev_ot;

extern __constant__ double dev_aspect_ratio;
extern __constant__ double dev_wds;
extern __constant__ double dev_PI;
extern __constant__ double dev_C;
extern __constant__ double dev_CIE_X[kCIELEN];
extern __constant__ double dev_CIE_Y[kCIELEN];
extern __constant__ double dev_CIE_Z[kCIELEN];
extern __constant__ double dev_fast_trans[16];
extern __constant__ double dev_fast_itrans[16];


class Renderer {
public:
	int _x; int _y;
	Camera *_cam;
	/* TODO : array of volume? */
	BlackBody *_volume;

	const int SPEC_SAMPLE_STEP = 15;
	const int SPEC_TOTAL_SAMPLES = 89 / SPEC_SAMPLE_STEP;
	const int CHROMA = 100;
	bool QUALITY_ROOM = false;

	int num_file;


	Renderer() {
		_x = 100; _y = 100;
		_cam = new Camera();
		_volume = new BlackBody();
	}

	Renderer(const int x_pixels, const int y_pixels) {
		_x = x_pixels; _y = y_pixels;
		_cam = new Camera();
		_volume = new BlackBody();
	}

	~Renderer() {}

	void setCamera(Vector3 eyep, Vector3 at, Vector3 r, Vector3 f, Vector3 u, double ang) {
		_cam->_eyepos = eyep;
		_cam->_look_at = at;
		_cam->_up = u;
		_cam->_right = r;
		_cam->_forward = f;
		_cam->_angle = ang;
	}

	cudaError_t loadConstantMem();
	cudaError_t loadMem(Vector3 *deyepos, Vector3 *dforward, Vector3 *dright, Vector3 *dup, Vector3 *devmincorrd, Vector3 *devmaxcoord,
		float *dimg, double *devT, double *dle, double *dl, double *dlemean, double *dxm, double *dym, double *dzm,
		double *Le, double *L, double *LeMean, double *xm, double *ym, double *zm,
		double *temperature_grid);
	float* drawFire(float *temperatureGrid, float *image);
	void saveImage(float *image);
	//cudaError_t renderWithCuda(double *T);
};


__global__ void colorKernel(int SPEC_TOTAL_SAMPLES, bool QUALITY_ROOM, double LeScale, int SPEC_SAMPLE_STEP, int CHROMA,
	double *dev_le, double* dev_l, double *dev_le_mean, double *dev_temperature_grid, float *dev_image, double *dev_xm, double *dev_ym, double *dev_zm, Vector3 *dev_eyepos, Vector3 *dev_forward, Vector3 *dev_right, Vector3 *dev_up, Vector3 *dev_minCoord, Vector3 *dev_maxCoord);

#endif