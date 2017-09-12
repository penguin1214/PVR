#include "Renderer.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "util.h"

struct Presets;

#define PI 3.14159265359;
#define kC_1 3.7418e-16
#define kC_2 1.4388e-2

__constant__ int dev_xsize;
__constant__ int dev_ysize;
__constant__ int dev_gridx;
__constant__ int dev_gridy;
__constant__ int dev_gridz;
__constant__ double dev_oa;
__constant__ double dev_os;
__constant__ double dev_ot;

__constant__ double dev_w;
__constant__ double dev_h;
__constant__ double dev_near_plane_distance;
__constant__ double dev_wds;
__constant__ double dev_C;
__constant__ double dev_CIE_X[kCIELEN];
__constant__ double dev_CIE_Y[kCIELEN];
__constant__ double dev_CIE_Z[kCIELEN];
__constant__ double dev_fast_trans[16];
__constant__ double dev_fast_itrans[16];


const double CIE_X[] = {
	1.299000e-04, 2.321000e-04, 4.149000e-04, 7.416000e-04, 1.368000e-03,
	2.236000e-03, 4.243000e-03, 7.650000e-03, 1.431000e-02, 2.319000e-02,
	4.351000e-02, 7.763000e-02, 1.343800e-01, 2.147700e-01, 2.839000e-01,
	3.285000e-01, 3.482800e-01, 3.480600e-01, 3.362000e-01, 3.187000e-01,
	2.908000e-01, 2.511000e-01, 1.953600e-01, 1.421000e-01, 9.564000e-02,
	5.795001e-02, 3.201000e-02, 1.470000e-02, 4.900000e-03, 2.400000e-03,
	9.300000e-03, 2.910000e-02, 6.327000e-02, 1.096000e-01, 1.655000e-01,
	2.257499e-01, 2.904000e-01, 3.597000e-01, 4.334499e-01, 5.120501e-01,
	5.945000e-01, 6.784000e-01, 7.621000e-01, 8.425000e-01, 9.163000e-01,
	9.786000e-01, 1.026300e+00, 1.056700e+00, 1.062200e+00, 1.045600e+00,
	1.002600e+00, 9.384000e-01, 8.544499e-01, 7.514000e-01, 6.424000e-01,
	5.419000e-01, 4.479000e-01, 3.608000e-01, 2.835000e-01, 2.187000e-01,
	1.649000e-01, 1.212000e-01, 8.740000e-02, 6.360000e-02, 4.677000e-02,
	3.290000e-02, 2.270000e-02, 1.584000e-02, 1.135916e-02, 8.110916e-03,
	5.790346e-03, 4.106457e-03, 2.899327e-03, 2.049190e-03, 1.439971e-03,
	9.999493e-04, 6.900786e-04, 4.760213e-04, 3.323011e-04, 2.348261e-04,
	1.661505e-04, 1.174130e-04, 8.307527e-05, 5.870652e-05, 4.150994e-05,
	2.935326e-05, 2.067383e-05, 1.455977e-05, 1.025398e-05, 7.221456e-06,
	5.085868e-06, 3.581652e-06, 2.522525e-06, 1.776509e-06, 1.251141e-06
};
const double CIE_Y[] = { 3.917000e-06, 6.965000e-06, 1.239000e-05, 2.202000e-05, 3.900000e-05,
6.400000e-05, 1.200000e-04, 2.170000e-04, 3.960000e-04, 6.400000e-04,
1.210000e-03, 2.180000e-03, 4.000000e-03, 7.300000e-03, 1.160000e-02,
1.684000e-02, 2.300000e-02, 2.980000e-02, 3.800000e-02, 4.800000e-02,
6.000000e-02, 7.390000e-02, 9.098000e-02, 1.126000e-01, 1.390200e-01,
1.693000e-01, 2.080200e-01, 2.586000e-01, 3.230000e-01, 4.073000e-01,
5.030000e-01, 6.082000e-01, 7.100000e-01, 7.932000e-01, 8.620000e-01,
9.148501e-01, 9.540000e-01, 9.803000e-01, 9.949501e-01, 1.000000e+00,
9.950000e-01, 9.786000e-01, 9.520000e-01, 9.154000e-01, 8.700000e-01,
8.163000e-01, 7.570000e-01, 6.949000e-01, 6.310000e-01, 5.668000e-01,
5.030000e-01, 4.412000e-01, 3.810000e-01, 3.210000e-01, 2.650000e-01,
2.170000e-01, 1.750000e-01, 1.382000e-01, 1.070000e-01, 8.160000e-02,
6.100000e-02, 4.458000e-02, 3.200000e-02, 2.320000e-02, 1.700000e-02,
1.192000e-02, 8.210000e-03, 5.723000e-03, 4.102000e-03, 2.929000e-03,
2.091000e-03, 1.484000e-03, 1.047000e-03, 7.400000e-04, 5.200000e-04,
3.611000e-04, 2.492000e-04, 1.719000e-04, 1.200000e-04, 8.480000e-05,
6.000000e-05, 4.240000e-05, 3.000000e-05, 2.120000e-05, 1.499000e-05,
1.060000e-05, 7.465700e-06, 5.257800e-06, 3.702900e-06, 2.607800e-06,
1.836600e-06, 1.293400e-06, 9.109300e-07, 6.415300e-07, 4.518100e-07 };
const double CIE_Z[] = { 6.061000e-04, 1.086000e-03, 1.946000e-03, 3.486000e-03, 6.450001e-03,
1.054999e-02, 2.005001e-02, 3.621000e-02, 6.785001e-02, 1.102000e-01,
2.074000e-01, 3.713000e-01, 6.456000e-01, 1.039050e+00, 1.385600e+00,
1.622960e+00, 1.747060e+00, 1.782600e+00, 1.772110e+00, 1.744100e+00,
1.669200e+00, 1.528100e+00, 1.287640e+00, 1.041900e+00, 8.129501e-01,
6.162000e-01, 4.651800e-01, 3.533000e-01, 2.720000e-01, 2.123000e-01,
1.582000e-01, 1.117000e-01, 7.824999e-02, 5.725001e-02, 4.216000e-02,
2.984000e-02, 2.030000e-02, 1.340000e-02, 8.749999e-03, 5.749999e-03,
3.900000e-03, 2.749999e-03, 2.100000e-03, 1.800000e-03, 1.650001e-03,
1.400000e-03, 1.100000e-03, 1.000000e-03, 8.000000e-04, 6.000000e-04,
3.400000e-04, 2.400000e-04, 1.900000e-04, 1.000000e-04, 4.999999e-05,
3.000000e-05, 2.000000e-05, 1.000000e-05, 0.000000e+00, 0.000000e+00,
0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00 };

__device__ void indexToLocal(int i, int j, int k, double &l_x, double &l_y, double &l_z) {
	l_x = double(i) / double(dev_gridx - 1);
	l_y = double(j) / double(dev_gridy - 1);
	l_z = double(k) / double(dev_gridz - 1);
}

__device__ void localToWorld(double *trans, double l_x, double l_y, double l_z, double &w_x, double &w_y, double &w_z) {
	w_x = l_x*trans[0] + l_y*trans[1] + l_z*trans[2] + trans[3];
	w_y = l_x*trans[4] + l_y*trans[5] + l_z*trans[6] + trans[7];
	w_z = l_x*trans[8] + l_y*trans[9] + l_z*trans[10] + trans[11];
}

__device__ void indexToWorld(double *trans, int i, int j, int k, double &w_x, double &w_y, double &w_z) {
	double l_x, l_y, l_z;
	indexToLocal(i, j, k, l_x, l_y, l_z);
	localToWorld(trans, l_x, l_y, l_z, w_x, w_y, w_z);
}

__device__ void worldToLocal(double *itrans, const double w_x, const double w_y, const double w_z, double &l_x, double &l_y, double &l_z) {
	l_x = w_x*itrans[0] + w_y*itrans[1] + w_z*itrans[2] + itrans[3];
	l_y = w_x*itrans[4] + w_y*itrans[5] + w_z*itrans[6] + itrans[7];
	l_z = w_x*itrans[8] + w_y*itrans[9] + w_z*itrans[10] + itrans[11];
}

__device__ void localToUpperLeftIndex(const double l_x, const double l_y, const double l_z, int &i, int &j, int &k) {
	i = (int)floor(l_x*double(dev_gridx - 1));
	j = (int)floor(l_y*double(dev_gridy - 1));
	k = (int)floor(l_z*double(dev_gridz - 1));
}

__device__ void worldToUpperLeftIndex(double *itrans, const double w_x, const double w_y, const double w_z, int &i, int &j, int &k) {
	double l_x, l_y, l_z;
	worldToLocal(itrans, w_x, w_y, w_z, l_x, l_y, l_z);
	localToUpperLeftIndex(l_x, l_y, l_z, i, j, k);
}

__device__ bool localIsValid(double l_x, double l_y, double l_z) {
	return l_x >= 0.0 && l_x < 1.0 && l_y >= 0.0 && l_y < 1.0 && l_z >= 0.0 && l_z < 1.0;
}

__device__ bool worldIsValid(double *itrans, const double w_x, const double w_y, const double w_z) {
	double l_x, l_y, l_z;
	worldToLocal(itrans, w_x, w_y, w_z, l_x, l_y, l_z);
	return localIsValid(l_x, l_y, l_z);
}

__device__ double valueAtWorld(double *g, double *itrans, double* trans, double w_x, double w_y, double w_z) {

	int i, j, k;
	double w_x0, w_y0, w_z0;
	double w_x1, w_y1, w_z1;
	double x, y, z;

	//Konvertera v�rldskoordinater till cellkoordinater
	worldToUpperLeftIndex(itrans, w_x, w_y, w_z, i, j, k);
	indexToWorld(trans, i, j, k, w_x0, w_y0, w_z0);
	indexToWorld(trans, i + 1, j + 1, k + 1, w_x1, w_y1, w_z1);

	x = (w_x - w_x0) / (w_x1 - w_x0);
	y = (w_y - w_y0) / (w_y1 - w_y0);
	z = (w_z - w_z0) / (w_z1 - w_z0);

	return linearInterpolate(g, i, j, k, x, y, z, dev_gridx, dev_gridy, dev_gridz);
}
// minbox is the corner of AABB with minimal coordinates - left bottom, maxbox is maximal corner
//code originally from http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
__device__ bool rayBoxIntersection(Vector3 minbox, Vector3 maxbox, const Vector3 &lineOrigin, const Vector3 &lineDirection, double *tmin, double *tmax) {
	//avoid division with zero
	Vector3 dir = lineDirection;
	/*dir.x = sgn(dir.x)*std::max(fabs(dir.x), DBL_MIN);
	dir.y = sgn(dir.y)*std::max(fabs(dir.y), DBL_MIN);
	dir.z = sgn(dir.z)*std::max(fabs(dir.z), DBL_MIN);*/

	Vector3 dirfrac;
	dirfrac.x = 1.0f / dir.x;
	dirfrac.y = 1.0f / dir.y;
	dirfrac.z = 1.0f / dir.z;

	double xmin = (minbox.x - lineOrigin.x)*dirfrac.x;
	double xmax = (maxbox.x - lineOrigin.x)*dirfrac.x;
	double ymin = (minbox.y - lineOrigin.y)*dirfrac.y;
	double ymax = (maxbox.y - lineOrigin.y)*dirfrac.y;
	double zmin = (minbox.z - lineOrigin.z)*dirfrac.z;
	double zmax = (maxbox.z - lineOrigin.z)*dirfrac.z;

	binsort(&xmax, &xmin);
	binsort(&ymax, &ymin);
	binsort(&zmax, &zmin);

	*tmin = d_max(d_max(xmin, ymin), zmin);
	*tmax = d_min(d_min(xmax, ymax), zmax);

	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behind us (if tmin is < 0 i think we start inside the box //axel)
	if (*tmax < 0)
		return false;

	// if tmin > tmax, ray doesn't intersect AABB
	if (*tmin > *tmax)
		return false;

	return true;
}

__device__ Vector3 XYZtoLMS(const Vector3 &xyz) {
	Vector3 lms(0.0, 0.0, 0.0);

	// D65
	/*lms.x = xyz.x * 0.400	+ xyz.y * 0.708		+ xyz.z * -0.081;
	lms.y = xyz.x * -0.226	+ xyz.y * 1.165		+ xyz.z * 0.046;
	lms.z = xyz.x * 0		+ xyz.y * 0			+ xyz.z * 0.918;*/

	//CAT02
	lms.x = 0.7328*xyz.x + 0.4296*xyz.y - 0.1624*xyz.z;
	lms.y = -0.7036*xyz.x + 1.6975*xyz.y + 0.0061*xyz.z;
	lms.z = 0.0030*xyz.x + 0.0136*xyz.y + 0.9834*xyz.z;

	return lms;
}

__device__ Vector3 XYZtoRGB(const Vector3 &xyz) {
	Vector3 rgb(0.0, 0.0, 0.0);

	// sRGB
	rgb.x = xyz.x * 3.2410 + xyz.y * -1.5374 + xyz.z * -0.4986;
	rgb.y = xyz.x * -0.9692 + xyz.y * 1.8760 + xyz.z * 0.0416;
	rgb.z = xyz.x * 0.0556 + xyz.y * -0.2040 + xyz.z * 1.0570;

	return rgb;
}

__device__ Vector3 LMStoXYZ(const Vector3 &lms) {
	Vector3 xyz(0.0, 0.0, 0.0);

	// D65
	/*xyz.x = lms.x * 1.861	+ lms.y * -1.131	+ lms.z * 2.209;
	xyz.y = lms.x * 0.361	+ lms.y * 0.639		+ lms.z * -0.002;
	xyz.z = lms.x * 0		+ lms.y * 0			+ lms.z * 10.89;*/

	//CAT02
	xyz.x = 1.09612	*lms.x - 0.278869*lms.y + 0.182745  *lms.z;
	xyz.y = 0.454369	*lms.x + 0.473533	*lms.y + 0.0720978 *lms.z;
	xyz.z = -0.00962761	*lms.x - 0.00569803	*lms.y + 1.01533   *lms.z;

	return xyz;
}

__device__ double radiance(double lambda, double T) {
	double C_1 = kC_1;	// 0.00000000000000037417999999999999806568434161
	double C_2 = kC_2;	// 0.014387999999999999747868

	return (2.0*C_1) / (pow(lambda, 5.0) * (exp(C_2 / (lambda*T)) - 1.0));
}

__global__ void colorKernel(int SPEC_TOTAL_SAMPLES, bool QUALITY_ROOM, double LeScale, int SPEC_SAMPLE_STEP, int CHROMA,
	double *dev_le, double* dev_l, double *dev_le_mean, double *dev_temperature_grid, float *dev_image, double *dev_xm, double *dev_ym, double *dev_zm,
	Vector3 *dev_eyepos, Vector3 *dev_forward, Vector3 *dev_right, Vector3 *dev_up, Vector3 *dev_minCoord, Vector3 *dev_maxCoord) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double du = 2.0 / double(dev_xsize - 1);
	double dv = 2.0 / double(dev_ysize - 1);

	while (!(x < 0 || x >= dev_xsize || y < 0 || y >= dev_ysize)) {
		double u = -1.0 + double(x) * du;
		double v = -1.0 + double(y) * dv;

		Vector3 nearPlanePos = *dev_eyepos + *dev_forward * dev_near_plane_distance + (*dev_right) * u * (dev_w / 2) + (*dev_up) * v * (dev_h / 2);
		Vector3 direction = nearPlanePos - *dev_eyepos;
		direction.normalize();

		double tmin, tmax;
		//Vector3 normal;

		int index_p = y * (dev_xsize)+x;

		//double *local_L = new double[SPEC_TOTAL_SAMPLES];
		double local_L[5];


		if (rayBoxIntersection(*dev_minCoord, *dev_maxCoord, nearPlanePos, direction, &tmin, &tmax)) {
			if (tmax > 0.0) {
				Vector3 endPoint = nearPlanePos + direction * tmax;
				Vector3 startPoint;
				if (tmin > 0.0) // set startpoint at first intersection with box if camera is outside the box
					startPoint = nearPlanePos + direction*tmin;
				else
					startPoint = nearPlanePos;

				for (size_t i = 0; i < SPEC_TOTAL_SAMPLES; i++) {

					local_L[i] = 0.0;

					if (QUALITY_ROOM) {
						for (int a = 0; a < dev_gridx; a++) {
							for (int b = 0; b < dev_gridy; b++) {
								for (int c = 0; c < dev_gridz; c++) {
									int index = (a * dev_gridy * dev_gridz +
										b * dev_gridz +
										c)*SPEC_TOTAL_SAMPLES + i;

									double xw2, yw2, zw2;
									indexToWorld(dev_fast_trans, a, b, c, xw2, yw2, zw2);
									Vector3 p1(xw2, yw2, zw2);

									Vector3 diff = p1 - endPoint;
									double dist = diff.norm();
									/**diffuse*/
									local_L[i] += pow(dist, -2.0)*dev_le[index] * LeScale;
								}
							}
						}
					}
					else {
						if (dev_le_mean[i] > 0.0) {
							double xw2, yw2, zw2;
							indexToWorld(dev_fast_trans, dev_xm[i], dev_ym[i], dev_zm[i], xw2, yw2, zw2);
							Vector3 p1(xw2, yw2, zw2);

							Vector3 diff = p1 - endPoint;
							double dist = diff.norm();
							//diff.normalize(); // we dont have the normal and therefor we dont use the diffuse term here
							//double diffuse = std::max(Vector3::dot(diff, normal), 0.0); 
							local_L[i] += pow(dist, -2.0)*dev_le_mean[i] * LeScale/**diffuse*/;
						}
					}
				}

				// ray casting
				double length = (endPoint - startPoint).norm();
				double steps = length / dev_wds;
				int intsteps = int(steps);
				Vector3 pos;
				for (int z = 0; z < intsteps; z += 1) {
					pos = endPoint - direction*double(z)*dev_wds; //From end to start, we traverse the ray backwards here.

					if (worldIsValid(dev_fast_itrans, pos.x, pos.y, pos.z)) // check if pos is inside the grid
					{
						double T = valueAtWorld(dev_temperature_grid, dev_fast_itrans, dev_fast_trans, pos.x, pos.y, pos.z);

						// calculate the radiance for each wave length sample
						for (int i = 0; i < SPEC_TOTAL_SAMPLES; i += 1) {
							const double lambda = (360.0 + double(i*SPEC_SAMPLE_STEP) * 5)*1e-9;
							local_L[i] = dev_C * local_L[i] + dev_oa * radiance(lambda, T) * dev_wds;
						}
					}
				}
				double restLength = (pos - nearPlanePos).norm();
				double Crest = exp(-dev_ot * restLength);
				for (int i = 0; i < SPEC_TOTAL_SAMPLES; i += 1) // correct the intensity if startpoint is not at nearplane
					local_L[i] = Crest*local_L[i];

				// Calc XYZ-color from the radiance L
				Vector3 XYZ = Vector3(0.0);
				for (int i = 0; i < SPEC_TOTAL_SAMPLES; i += 1) {
					int j = SPEC_SAMPLE_STEP*i;
					XYZ.x += local_L[i] * dev_CIE_X[j];
					XYZ.y += local_L[i] * dev_CIE_Y[j];
					XYZ.z += local_L[i] * dev_CIE_Z[j];
				}

				double SAMPLE_DL = 5e-9*double(SPEC_SAMPLE_STEP);
				XYZ *= SAMPLE_DL;

				// Chromatic adaption of the light, high CHROMA value decreases the intensity of the light
				Vector3 LMS = XYZtoLMS(XYZ);
				LMS.x = LMS.x / (LMS.x + CHROMA);
				LMS.y = LMS.y / (LMS.y + CHROMA);
				LMS.z = LMS.z / (LMS.z + CHROMA);
				XYZ = LMStoXYZ(LMS);

				Vector3 rgb = XYZtoRGB(XYZ);
				dev_image[index_p * 3 + 0] = rgb.x; //R
				dev_image[index_p * 3 + 1] = rgb.y; //G
				dev_image[index_p * 3 + 2] = rgb.z; //B
			}
			else {
				dev_image[index_p * 3 + 0] = 0.0; //R
				dev_image[index_p * 3 + 1] = 0.0; //G
				dev_image[index_p * 3 + 2] = 0.0; //B
			}
		}
		else {
			dev_image[index_p * 3 + 0] = 0.0; //R
			dev_image[index_p * 3 + 1] = 0.0; //G
			dev_image[index_p * 3 + 2] = 0.0; //B
		}
		// print progress

		// update index
		x += blockDim.x * gridDim.x;
		y += blockDim.y * gridDim.y;
	}
}

cudaError_t Renderer::loadConstantMem() {
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_xsize, &_x, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpyToSymbol(dev_ysize, &_y, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpyToSymbol(dev_gridx, &_volume->_grid->_xdim, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! gridx");
		;
	}

	cudaStatus = cudaMemcpyToSymbol(dev_gridy, &_volume->_grid->_ydim, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpyToSymbol(dev_gridz, &_volume->_grid->_zdim, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpyToSymbol(dev_oa, &_volume->_oa, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! oa");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_os, &_volume->_os, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_ot, &_volume->_ot, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_wds, &_volume->_wds, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_w, &_cam->_film->_w, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_h, &_cam->_film->_h, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_near_plane_distance, &_cam->_film->_nearPlaneDistance, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	double tr = _volume->transparency();
	cudaStatus = cudaMemcpyToSymbol(dev_C, &tr, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}


	cudaStatus = cudaMemcpyToSymbol(dev_CIE_X, &CIE_X, kCIELEN * sizeof(double));

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpyToSymbol(dev_CIE_Y, &CIE_Y, kCIELEN * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpyToSymbol(dev_CIE_Z, &CIE_Z, kCIELEN * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	/* TODO: why can't use trans*/
	cudaStatus = cudaMemcpyToSymbol(dev_fast_trans, &(_volume->_grid->_trans[0]), 16 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpyToSymbol(dev_fast_itrans, &(_volume->_grid->_itrans[0]), 16 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus = cudaSuccess) {
		std::cout << "Constant memory load done." << std::endl;
	}

	return cudaStatus;
}

cudaError_t Renderer::loadMem(Vector3 *deyepos, Vector3 *dforward, Vector3 *dright, Vector3 *dup, Vector3 *devmincoord, Vector3 *devmaxcoord,
	float *dimg, double *devT, double *dle, double *dl, double *dlemean, double *dxm, double *dym, double *dzm,
	double *Le, double *L, double *LeMean, double *xm, double *ym, double *zm,
	double *temperature_grid) {

	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&deyepos, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dforward, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dright, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dup, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(deyepos, &_cam->_eyepos, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(dforward, &_cam->_forward, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(dright, &_cam->_right, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(dup, &_cam->_up, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&devmincoord, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&devmaxcoord, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(devmincoord, &_volume->_grid->_min_coord, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(devmaxcoord, &_volume->_grid->_max_coord, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&dxm, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dym, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dzm, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	int imsize = _x * _y * 3;
	cudaStatus = cudaMalloc((void**)&dimg, imsize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	cudaStatus = cudaMalloc((void**)&devT, _volume->_grid->getSize() * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	int LeSize = _volume->_grid->_xdim * _volume->_grid->_ydim * _volume->_grid->_zdim * SPEC_TOTAL_SAMPLES;
	cudaStatus = cudaMalloc((void**)&dle, LeSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	cudaStatus = cudaMalloc((void**)&dl, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	cudaStatus = cudaMalloc((void**)&dlemean, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(devT, temperature_grid, _volume->_grid->getSize() * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dlemean, LeMean, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dle, Le, LeSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dl, L, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dxm, xm, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dym, ym, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dzm, zm, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	return cudaStatus;
}

float* Renderer::drawFire(float* temperature_grid, float *image) {
	// transform
	// grid index to word coordinate
	_volume->_grid->indexToWorld(_volume->_grid->_trans, 0, 0, 0, _volume->_grid->_min_coord.x, _volume->_grid->_min_coord.y, _volume->_grid->_min_coord.z);
	_volume->_grid->indexToWorld(_volume->_grid->_trans, _volume->_grid->_xdim - 1, _volume->_grid->_ydim - 1, _volume->_grid->_zdim - 1, _volume->_grid->_max_coord.x, _volume->_grid->_max_coord.y, _volume->_grid->_max_coord.z);

	const int IMGSIZE = _x * _y * 3;
	int LeSize = _volume->_grid->_xdim * _volume->_grid->_ydim * _volume->_grid->_zdim * SPEC_TOTAL_SAMPLES;

	double *Le = new double[LeSize];
	double *L = new double[SPEC_TOTAL_SAMPLES];
	double * LeMean = new double[SPEC_TOTAL_SAMPLES];

	double *xm = new double[SPEC_TOTAL_SAMPLES];
	double *ym = new double[SPEC_TOTAL_SAMPLES];
	double *zm = new double[SPEC_TOTAL_SAMPLES];

	for (int i = 0; i < SPEC_TOTAL_SAMPLES; i++) {
		xm[i] = 0; ym[i] = 0; zm[i] = 0;
	}

	/* Start rendering */
	float startTime = omp_get_wtime();
	int n = 0;

#pragma omp parallel
	{
		if (!QUALITY_ROOM) {
			// reset the mean radiance value
#pragma omp for
			for (int i = 0; i < SPEC_TOTAL_SAMPLES; i += 1) {
				LeMean[i] = 0.0;
			}
		}

#pragma omp for
		for (int x = 0; x < _volume->_grid->_xdim; ++x) {
			for (int y = 0; y < _volume->_grid->_ydim; ++y) {
				for (int z = 0; z < _volume->_grid->_zdim; ++z) {
					//const double T = valueAtIndex(x, y, z);
					int idx = x + _volume->_grid->_xdim*y + _volume->_grid->_xdim*_volume->_grid->_ydim*z;
					const double T = temperature_grid[idx];

					//std::cout << std::endl << T << " | ";

					for (int i = 0; i < SPEC_TOTAL_SAMPLES; i += 1) {
						int index = (x*_volume->_grid->_ydim*_volume->_grid->_zdim +
							y*_volume->_grid->_zdim +
							z)*SPEC_TOTAL_SAMPLES + i;

						const double lambda = (360.0 + double(i*SPEC_SAMPLE_STEP) * 5)*1e-9;
						Le[index] = _volume->radiance(lambda, T);
						//std::cout << Le[index] << std::endl;
						if (Le[index] < 0.0) Le[index] == 0.0;

						Le[index] = _volume->_oa*Le[index] * _volume->_grid->dx()*_volume->_grid->dy()*_volume->_grid->dz();

						if (!QUALITY_ROOM) {
							LeMean[i] += Le[index];
							xm[i] += double(x)*Le[index];
							ym[i] += double(y)*Le[index];
							zm[i] += double(z)*Le[index];

							//std::cout << x << "," << y << "," << z << "," << i << " |  " << LeMean[i] << " | " << xm[i] << std::endl;

						}
						//std::cout << LeMean[i] << ", " << "|" << index;
					}
				}
			}
		}

		if (!QUALITY_ROOM) {
#pragma omp for
			for (int i = 0; i < SPEC_TOTAL_SAMPLES; i += 1) {
				if (LeMean[i] > 0.0) {
					xm[i] /= LeMean[i];
					ym[i] /= LeMean[i];
					zm[i] /= LeMean[i];
				}
			}
		}

		double *local_L = &L[omp_get_thread_num()*SPEC_TOTAL_SAMPLES];
		Vector3 normal = Vector3(0.0, 0.0, 1.0); //Not used yet, since we dont have a way to find the normal with the box yet.
												 //#pragma omp for
	}

	//copy value 
	Vector3 *d_eyepos;
	Vector3 *d_forward;
	Vector3 *d_right;
	Vector3 *d_up;
	Vector3 *dev_minCoord;
	Vector3 *dev_maxCoord;

	float *dev_img = new float[IMGSIZE];
	double *dev_T = new double[_volume->_grid->getSize()];
	double *dev_le = new double[LeSize];
	double *dev_l = new double[SPEC_TOTAL_SAMPLES];
	double *dev_le_mean = new double[SPEC_TOTAL_SAMPLES];

	double *dev_xm = new double[SPEC_TOTAL_SAMPLES];
	double *dev_ym = new double[SPEC_TOTAL_SAMPLES];
	double *dev_zm = new double[SPEC_TOTAL_SAMPLES];

	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSetCacheConfig failed!");
	}


	cudaStatus = cudaMalloc((void**)&d_eyepos, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	
	cudaStatus = cudaMalloc((void**)&d_forward, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_right, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&d_up, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_eyepos, &_cam->_eyepos, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_forward, &_cam->_forward, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_right, &_cam->_right, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_up, &_cam->_up, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_minCoord, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_maxCoord, sizeof(Vector3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_minCoord, &_volume->_grid->_min_coord, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(dev_maxCoord, &_volume->_grid->_max_coord, sizeof(Vector3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_xm, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_ym, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_zm, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	int imsize = _x * _y * 3;
	cudaStatus = cudaMalloc((void**)&dev_img, imsize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	cudaStatus = cudaMalloc((void**)&dev_T, _volume->_grid->getSize() * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	cudaStatus = cudaMalloc((void**)&dev_le, LeSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	cudaStatus = cudaMalloc((void**)&dev_l, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	cudaStatus = cudaMalloc((void**)&dev_le_mean, SPEC_TOTAL_SAMPLES * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_T, temperature_grid, _volume->_grid->getSize() * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dev_le_mean, LeMean, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dev_le, Le, LeSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dev_l, L, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dev_xm, xm, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dev_ym, ym, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}

	cudaStatus = cudaMemcpy(dev_zm, zm, SPEC_TOTAL_SAMPLES * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		;
	}


	// call kernel
	std::cout << "start kernel... " << std::endl;
	dim3 color_grid(32, 32);
	//dim3 color_block(SPEC_TOTAL_SAMPLES);
	dim3 color_block(16, 16);

	cudaEvent_t start, stop;
	float elapse_time;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	colorKernel << <color_grid, color_block >> > (SPEC_TOTAL_SAMPLES, QUALITY_ROOM, _volume->LeScale, SPEC_SAMPLE_STEP, CHROMA, dev_le, dev_l, dev_le_mean, dev_T, dev_img, dev_xm, dev_ym, dev_zm, d_eyepos, d_forward, d_right, d_up, dev_minCoord, dev_maxCoord);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapse_time, start, stop);
	printf("Elapsed time : %f ms\n", elapse_time);

	// cudaError_t cudaStatus;
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "colorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
	/*	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching colorKernel!\n", cudaStatus);
		;
	}*/

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(image, dev_img, IMGSIZE * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	/*cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}*/

	return image;
}



