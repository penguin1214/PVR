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

	static const double PI;;
};

const int Presets::SAMPLE_STEP = 15;
const int Presets::TOTAL_SAMPLES = 89 / Presets::SAMPLE_STEP;
double Presets::dt = 1.0 / 100.0;
const double PI = 3.14159265359;


template<class T>
T* loadBinary(const char * fn, int &gridx, int &gridy, int &gridz) {
	if (sizeof(T) != sizeof(float))return nullptr;
	FILE *fp = fopen(fn, "rb");
	fread(&gridx, 1, sizeof(int), fp);
	fread(&gridy, 1, sizeof(int), fp);
	fread(&gridz, 1, sizeof(int), fp);

	//N = Nx, NN = Nx*Ny, total = Nx*Ny*Nz;
	//_MLCuMallocHost((void **)&data, total * sizeof(T));
	int total = gridx * gridy * gridz;
	T *data = new T[total];

	//fread(&data[0], sizeof(T), total, fp);
	fread(data, sizeof(T), total, fp);
	for (int i = 0; i < total; i++) {
		std::cout << data[i] << ", ";
	}
	fclose(fp);
	printf("loaded %s <%d,%d,%d>\n", fn, gridx, gridy, gridz);
	return data;
}


int main() {
	int num_file = 0;
	int grid_x, grid_y, grid_z;
	/*===========================================Read File================================================*/
	//std::string str_filename = "fire" + std::to_string(num_file) + ".bin";
	std::string str_filename = "car_data0200.bin";
	const char* filename = str_filename.c_str();
	num_file++;

	float *T = loadBinary<float>(filename, grid_x, grid_y, grid_z);
	std::cout << "Read file " << filename << "  done." << std::endl;

	int grid_size = grid_x * grid_y * grid_z;
	double trans[] = { 4,0,0,0,0,4,0,0,0,0,4,0,0,0,0,1 };

	/*===========================================Renderer Set================================================*/

	Renderer renderer(Presets::FRAME_X, Presets::FRAME_Y);

	renderer.num_file = 0;   
	renderer._volume->setCoefficent(0.05, 0.0, 0.05);
	renderer._volume->_grid->setSize(grid_x, grid_y, grid_z);
	renderer._volume->_grid->setTransform(trans);
	renderer._volume->_wds = double(renderer._volume->_grid->GRID_SIZE) / double(std::max(renderer._volume->_grid->_xdim, std::max(renderer._volume->_grid->_ydim, renderer._volume->_grid->_zdim)) * Presets::SAMPLE_STEP_QUALITY);

	/*===========================================Set Camera================================================*/
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

	double near_plane_distance = 0.1;
	const double aspect_ratio = double(renderer._x) / double(renderer._y);
	const double fovy = PI*0.25;	// 90 deg

	renderer._cam->_film->setFovAndDis(fovy, near_plane_distance, aspect_ratio);

	/*===========================================Set Camera================================================*/
	float *image = new float[renderer._x * renderer._y * 3];

	cudaError_t cudaStatus;
	cudaStatus = renderer.loadConstantMem();

	
		std::cout << "Start rendering..." << std::endl;
		renderer.drawFire(T, image);
		renderer.saveImage(image);

	int t;
	std::cin >> t;

	return 0;
}