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
	static const int FRAME_Y = 300;

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
	/*for (int i = 0; i < total; i++) {
		std::cout << data[i] << ", ";
	}*/
	fclose(fp);
	printf("loaded %s <%d,%d,%d>\n", fn, gridx, gridy, gridz);
	return data;
}


int main() {
	int num_file = 0;
	int grid_x, grid_y, grid_z;
	/*===========================================Read File================================================*/
	std::string str_filename = "car_data0200.bin";
	const char* filename = str_filename.c_str();

	float *D = loadBinary<float>(filename, grid_x, grid_y, grid_z);	// 199, 399, 199
	std::cout << "Read file " << filename << "  done." << std::endl;

	int grid_size = grid_x * grid_y * grid_z;
	/*===========================================Renderer Set================================================*/

	Renderer renderer(Presets::FRAME_X, Presets::FRAME_Y);

	renderer.num_file = 0;
	renderer._volume->_grid->setSize(grid_x, grid_y, grid_z);
	/*===========================================Set Camera================================================*/
	//Vector3 lookAt(2, 4, 2);
	//Vector3 eyepos(2, 2, 7.5);
	Vector3 lookAt(0, 0, 1);
	Vector3 eyepos(-100, 300, -50.0);
	double angle = 0;
	// rotate camera according to angle
	eyepos -= lookAt;
	eyepos.rotY(6.28 * angle);
	eyepos += lookAt;

	Vector3 up(0.0, 1.0, 0.0);
	Vector3 forward = lookAt - eyepos;
	forward.normalize();
	//Vector3 right = forward.cross(&up);
	Vector3 right = Vector3(1, 0, 0);
	right.normalize();
	//up = right.cross(&forward);
	up.normalize();

	std::cout << "lookat: " << lookAt << std::endl;
	std::cout << "eyepos: " << eyepos << std::endl;
	std::cout << "up: " << up << std::endl;
	std::cout << "forward: " << forward << std::endl;
	std::cout << "right : " << right << std::endl;

	renderer.setCamera(eyepos, lookAt, right, forward, up, angle);

	//const double fovy = PI*0.25;	// 90 deg
	const double dis = 50;
	renderer._cam->_film->setDis(dis);


	/*=========================================== Render ================================================*/
	float *image = new float[renderer._cam->_film->_w * renderer._cam->_film->_h * 3];

	// bbox
	renderer._volume->_grid->_min_coord = Vector3(0.0);
	renderer._volume->_grid->_max_coord = Vector3(grid_x - 1, grid_y - 1, grid_z - 1);

	// ray marching
	int num_samples = 10;
	float u, v;
	float tmin, tmax;
	float stride;
	Vector3 sample_pos;
	float density;
	Vector3 col;

	for (int y = 0; y < renderer._cam->_film->_h; y++) {
		v = 0.5 - y / renderer._cam->_film->_h;
		for (int x = 0; x < renderer._cam->_film->_w; x++) {
			u = -0.5 + x / renderer._cam->_film->_w;

			col = Vector3(0.1, 0.3, 0.0);
			// pixel pos
			Vector3 cursor = renderer._cam->_eyepos + renderer._cam->_forward*renderer._cam->_film->_nearPlaneDistance + renderer._cam->_right*u*(renderer._cam->_film->_w/2) + renderer._cam->_up*v*(renderer._cam->_film->_h/2);
			//std::cout << cursor << std::endl;

			Vector3 ray_dir = cursor - renderer._cam->_eyepos;
			ray_dir.normalize();

			if (renderer.rayBBoxIntersection(renderer._volume->_grid->_min_coord, renderer._volume->_grid->_max_coord, cursor, ray_dir, tmin, tmax)) {
				if (tmin > 0) {
					//std::cout << "tmin: " << tmin << ", tmax: " << tmax << std::endl;
					stride = (tmax - tmin) / (num_samples + 1);
					for (int n = 0; n < num_samples; n++) {
						sample_pos = cursor + ray_dir*tmin + (n + 1)*stride*ray_dir;
						int coordx = (int)sample_pos.x; int coordy = (int)sample_pos.y; int coordz = (int)sample_pos.z;
						// interpolate
						int index = sample_pos.x + sample_pos.y*renderer._volume->_grid->_xdim + sample_pos.z*renderer._volume->_grid->_xdim*renderer._volume->_grid->_ydim;
						density = D[index];
						col += Vector3(density);
					}
				}
			}

			image[3*(y*(int)renderer._cam->_film->_w + x) + 0] = col.x;
			image[3*(y*(int)renderer._cam->_film->_w + x) + 1] = col.y;
			image[3*(y*(int)renderer._cam->_film->_w + x) + 2] = col.z;
		}
	}
	/*for (int z = 0; z < grid_z; z++) {
		for (int y = 0; y < grid_y; y++) {
			for (int x = 0; x < grid_x; x++) {
			}
		}
	}*/

	std::cout << "Start rendering..." << std::endl;
	renderer.saveImage(image);

	int t;
	std::cin >> t;

	return 0;
}