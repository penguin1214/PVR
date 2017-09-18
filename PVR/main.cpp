#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <cmath>

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
	static const int FRAME_X = 600;
	static const int FRAME_Y = 400;

	static double dt;
	static const bool QUALITY_ROOM = false; //def how the walls radiance should be calculated

	static const double PI;;
};

const int Presets::SAMPLE_STEP = 15;
const int Presets::TOTAL_SAMPLES = 89 / Presets::SAMPLE_STEP;
double Presets::dt = 1.0 / 100.0;
const double PI = 3.14159265359;

struct hit_record {
	float t;
	Vector3 p;
	Vector3 norm;
	//material* mat_ptr;
};


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
		if (data[i] > 1) {
			std::cout << data[i] << ", ";
		}
	}*/
	fclose(fp);
	printf("loaded %s <%d,%d,%d>\n", fn, gridx, gridy, gridz);
	return data;
}

// i,j,k are floor()
int getIndex(int gridx, int gridy, int gridz, int i, int j, int k) {
	return k*gridx*gridy + j*gridx + i;
}

float trilinearInterp(float *data, int gridx, int gridy, int gridz, float x, float y, float z) {
	int i = floor(x); int j = floor(y); int k = floor(z);
	//std::cout << i << ", " << j << ", " << k << std::endl;
	float xd = (x - i); float yd = y - j; float zd = z - k;

	float c00 = data[getIndex(gridx, gridy, gridz, i, j, k)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j, k)] * xd;
	float c01 = data[getIndex(gridx, gridy, gridz, i, j, k + 1)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j, k + 1)] * xd;
	float c10 = data[getIndex(gridx, gridy, gridz, i, j + 1, k)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j + 1, k)] * xd;
	float c11 = data[getIndex(gridx, gridy, gridz, i, j + 1, k + 1)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j + 1, k + 1)] * xd;

	float c0 = c00*(1 - yd) + c10*yd;
	float c1 = c01*(1 - yd) + c11*yd;

	float c = c0*(1 - zd) + c1*zd;
	if (c > 1) {
		std::cout << c << std::endl;
	}
	return c;
}

bool sphereHit(Vector3 center, float radius, Ray &r, float t_min, float t_max, hit_record &rec) {
	Vector3 oc = r.origin - center;
	float a = r.direction.dot(r.direction);
	float b = 2 * oc.dot(r.direction);
	float c = oc.dot(oc) - radius*radius;
	float disc = b*b - 4 * a*c;
	if ((sqrt(disc)) >= 0) {
		// calculate hit point
		float temp = (-b - sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.origin + r.direction*temp;
			rec.norm = rec.p - center;
			rec.norm.normalize();
			return true;
		}
	}
	return false;
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
	renderer._volume->setCoefficent(0.3, 1, 1);

	PointLight *pl = new PointLight();
	renderer._lights.push_back(pl);
	/*===========================================Set Camera================================================*/
	Vector3 lookAt(0, 50, 1);
	Vector3 eyepos(0, 50, -100);
	double angle = 0;
	// rotate camera according to angle
	eyepos -= lookAt;
	eyepos.rotY(6.28 * angle);
	eyepos += lookAt;

	Vector3 up(0.0, 1.0, 0.0);
	Vector3 forward = lookAt - eyepos;
	forward.normalize();
	Vector3 right = up.cross(&forward);
	right.normalize();
	up = forward.cross(&right);
	up.normalize();

	std::cout << "lookat: " << lookAt << std::endl;
	std::cout << "eyepos: " << eyepos << std::endl;
	std::cout << "up: " << up << std::endl;
	std::cout << "forward: " << forward << std::endl;
	std::cout << "right : " << right << std::endl;

	renderer.setCamera(eyepos, lookAt, right, forward, up, angle);

	//const double fovy = PI*0.25;	// 90 deg
	const double dis = 80;
	renderer._cam->_film->setDis(dis);
	/*=========================================== Set Sphere ================================================*/
	Vector3 sphere_o(0, 0, 0);
	float sphere_r = 10;
	Vector3 sphere_color(1.0, 1.0, 0.0);
	/*=========================================== Render ================================================*/
	float *image = new float[renderer._cam->_film->_w * renderer._cam->_film->_h * 3];

	// bbox
	renderer._volume->_grid->_min_coord = Vector3(0.0);
	renderer._volume->_grid->_max_coord = Vector3(grid_x - 1, grid_y - 1, grid_z - 1);

	// ray marching
	int num_samples = 50;
	int num_light_samples = 50;
	float u, v;
	float tmin, tmax;
	float stride;
	Vector3 sample_pos;
	float density;
	Vector3 col;

	int xdim = grid_x; int ydim = grid_y; int zdim = grid_z;

	for (int y = 0; y < renderer._cam->_film->_h; y++) {
		v = 0.5 - y / renderer._cam->_film->_h;
		std::cout << (float)y / float(renderer._cam->_film->_h) << std::endl;
		for (int x = 0; x < renderer._cam->_film->_w; x++) {
			u = -0.5 + x / renderer._cam->_film->_w;

			float T = 1.0;	// transparency
			float Lo = 0;

			// pixel pos
			Vector3 cursor = renderer._cam->_eyepos + renderer._cam->_forward*renderer._cam->_film->_nearPlaneDistance + renderer._cam->_right*u*renderer._cam->_film->_w + renderer._cam->_up*v*renderer._cam->_film->_h;

			Vector3 ray_dir = cursor - renderer._cam->_eyepos;
			ray_dir.normalize();

			if (renderer.rayBBoxIntersection(renderer._volume->_grid->_min_coord, renderer._volume->_grid->_max_coord, cursor, ray_dir, tmin, tmax)) {
				if (tmin > 0) {
					stride = (tmax - tmin) / (num_samples + 1);
					for (int n = 0; n < num_samples; n++) {
						/* Use back-to-front compositing */
						sample_pos = cursor + ray_dir*tmax - (n + 1)*stride*ray_dir;
						float l_T = 1.0;

						if (sample_pos.x > 0 && sample_pos.x < xdim && sample_pos.y > 0 && sample_pos.y < ydim && sample_pos.z < xdim && sample_pos.z > 0) {
							density = trilinearInterp(D, xdim, ydim, zdim, sample_pos.x, sample_pos.y, sample_pos.z);

							if (density > 0) {
								//T *= 1.0 - density * stride *renderer._volume->_oa;
								T *= std::exp(-renderer._volume->_oa*density*stride);

								Vector3 light_ray = sample_pos - renderer._lights[0]->pos;
								float l_stride = light_ray.length() / (num_light_samples + 1);
								light_ray.normalize();

								for (int j = 0; j < num_light_samples; j++) {
									// compute the radiance at the sample point
									Vector3 l_sample_pos = sample_pos - j*l_stride*light_ray;
									if (l_sample_pos.x > 0 && l_sample_pos.x < grid_x && l_sample_pos.y > 0 && l_sample_pos.y < grid_y && l_sample_pos.z > 0 && l_sample_pos.z < grid_z) {
										int l_index = l_sample_pos.x + l_sample_pos.y*renderer._volume->_grid->_xdim + l_sample_pos.z*renderer._volume->_grid->_xdim*renderer._volume->_grid->_ydim;
										float l_density = trilinearInterp(D, xdim, ydim, zdim, l_sample_pos.x, l_sample_pos.y, l_sample_pos.z);
										l_T *= std::exp(-renderer._volume->_oa*density*l_stride); // T = e^(-k(t)*dx)
									}
								}
								float Li = renderer._lights[0]->intensity * (1.0 - l_T);	// light radiance at sampled position
								Lo += Li*(1.0 - T);
							}
						}
					}
				}
			}
			else {
				Lo = 0;
			}
			Lo /= 255;
			image[3 * (y*(int)renderer._cam->_film->_w + x) + 0] = Lo;
			image[3 * (y*(int)renderer._cam->_film->_w + x) + 1] = Lo;
			image[3 * (y*(int)renderer._cam->_film->_w + x) + 2] = Lo;
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