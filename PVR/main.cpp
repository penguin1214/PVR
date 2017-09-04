#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>

#include "util.h"
#include "Renderer.h"
#include "volume.h"


struct Presets {
	// sampling
	static const int TOTAL_SAMPLES;
	static const int SAMPLE_STEP;
	static const int SAMPLE_STEP_QUALITY = 2;

	// grid
	static const int GRID_X = 30;
	static const int GRID_Y = 60;
	static const int GRID_Z = 30;

	// frame
	static const int FRAME_X = 300;
	static const int FRAME_Y = 600;

	static double dt;
	static const bool QUALITY_ROOM = false; //def how the walls radiance should be calculated
};

const int Presets::SAMPLE_STEP = 15;
const int Presets::TOTAL_SAMPLES = 89 / Presets::SAMPLE_STEP;
double Presets::dt = 1.0 / 100.0;


void loadFile() {}

int main() {
	int grid_size = Presets::GRID_X * Presets::GRID_Y * Presets::GRID_Z;

	Renderer renderer(Presets::FRAME_X, Presets::FRAME_Y);
	renderer._volume->setCoefficent(0.05, 0.0, 0.05);
	renderer._volume->_wds = double(renderer._volume->_grid->getSize()) / double(std::max(renderer._volume->_grid->_xdim, std::max(renderer._volume->_grid->_ydim, renderer._volume->_grid->_zdim)) * Presets::SAMPLE_STEP_QUALITY);


	Vector3 lookAt(2, 4, 2);
	Vector3 eyepos(2, 2, 7.5);
	double angle = 0;
	// rotate camera according to angle
	eyepos -= lookAt;
	eyepos.rotY(6.28 * angle);
	eyepos += lookAt;

	angle += Presets::dt*5.0;	// ?

	Vector3 up(0.0, 1.0, 0.0);
	Vector3 forward = lookAt - eyepos;
	forward.normalize();
	Vector3 right = forward.cross(&up);
	right.normalize();
	up = right.cross(&forward);
	up.normalize();

	renderer.setCamera(eyepos, lookAt, right, forward, up, angle);

	renderer._cam->_film->_nearPlaneDistance = 0.1;
	const double aspect_ratio = double(renderer._x) / double(renderer._y);
	const double fovy = PI*0.25;	// 90 deg
	renderer._cam->_film->_vfov = fovy;
	renderer._cam->_film->_w = renderer._cam->_film->_nearPlaneDistance * tan(renderer._cam->_film->_vfov);
	renderer._cam->_film->_h = renderer._cam->_film->_w / aspect_ratio;


	double *T = new double[grid_size];


	float *image = new float[renderer._x * renderer._y * 3];

	cudaError_t cudaStatus;
	cudaStatus = renderer.loadConstantMem();
	renderer.drawFire(T, image);
	renderer.saveImage(image);

	/*int num_file = 0;
	std::string filename = "fire" + std::to_string(num_file) + ".bin";

	clock_t start = clock();

	std::ifstream f(filename, std::ios::binary);
	if (f) {
	f.read(reinterpret_cast<char*>(T), grid_size * sizeof(double));
	}

	num_file++;

	std::cout << "Read file " << filename << "  done." << std::endl;


	clock_t end = clock();
	std::cout << "\n" << diffclock(end, start) << std::endl;*/


	int t;
	std::cin >> t;

	return 0;
}