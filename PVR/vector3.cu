#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Vector3.h"

//__host__ __device__ double& d_max(double& left, double& right) {
//	if (left < right)
//		return right;
//	else
//		return left;
//}
//
//__host__ __device__ double& d_min(double& left, double& right) {
//	if (left > right)
//		return right;
//	else
//		return left;
//}

// Constructors
__host__ __device__ Vector3::Vector3() {
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
}

__host__ __device__ Vector3::Vector3(double xf, double yf, double zf) {
	x = xf;
	y = yf;
	z = zf;
}

__host__ __device__ Vector3::Vector3(double v) {
	x = v;
	y = v;
	z = v;
}

__host__ __device__ Vector3::~Vector3() {}

// Dot product between two vectors, returns float
__host__ __device__ double Vector3::dot(const Vector3 &vTemp) {
	double dotProduct = x*vTemp.x + y*vTemp.y + z*vTemp.z;
	return dotProduct;
}

__host__ __device__ double Vector3::dot(const Vector3 &v1, const Vector3 &v2) {
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

// Multiplikation by f to all elements, void
__host__ __device__ void Vector3::mult(double f) {
	x *= f;
	y *= f;
	z *= f;
}

__host__ __device__ void Vector3::substract(Vector3 *v) {

	x -= v->x;
	y -= v->y;
	z -= v->z;

}

__host__ __device__ Vector3 Vector3::from(Vector3 *v) {
	double dx = x - v->x;
	double dy = y - v->y;
	double dz = z - v->z;
	Vector3 vec = Vector3(dx, dy, dz);

	return vec;
}

__host__ __device__ void Vector3::translate(Vector3 *v) {
	x += v->x;
	y += v->y;
	z += v->z;
}

__host__ __device__ Vector3 Vector3::add(Vector3 *v) {
	double dx = x + v->x;
	double dy = y + v->y;
	double dz = z + v->z;
	Vector3 vec = Vector3(dx, dy, dz);

	return vec;
}


// Norm (length) of the vector, returns float
__host__ __device__ double Vector3::norm() const {
	double norm = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	return norm;
}

__host__ __device__ double Vector3::length() {
	return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
}

// Normalize the vector
__host__ __device__ void Vector3::normalize() {
	double size = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	double invSize = 1 / size;
	x *= invSize;
	y *= invSize;
	z *= invSize;
}

// stolen from : http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ 
//Rotate around vector axisRotate 
__host__ __device__ void Vector3::rotate(Vector3 *axisRotate, double *angle, double a, double b, double c) {

	//En smula snabbare
	if ((a == 0) && (b == 0) && (c == 0)) {
		double u, v, w;
		u = axisRotate->x;
		v = axisRotate->y;
		w = axisRotate->z;

		// Set some intermediate values.
		double cosT = cos(*angle);
		double oneMinusCosT = 1 - cosT;
		double sinT = sin(*angle);

		// Use the formula in the paper.
		x = ((-u*(-u*x - v*y - w*z))*oneMinusCosT + x*cosT + (-w*y + v*z)*sinT);

		y = (-v*(-u*x - v*y - w*z)) * oneMinusCosT
			+ y*cosT
			+ (w*x - u*z)*sinT;

		z = ((-w*(-u*x - v*y - w*z)) * oneMinusCosT
			+ z*cosT
			+ (-v*x + u*y)*sinT);
	}
	else {

		double u, v, w;
		u = axisRotate->x;
		v = axisRotate->y;
		w = axisRotate->z;

		// Set some intermediate values.
		double u2 = u*u;
		double v2 = v*v;
		double w2 = w*w;
		double cosT = cos(*angle);
		double oneMinusCosT = 1 - cosT;
		double sinT = sin(*angle);


		// Use the formula in the paper.
		x = ((a*(v2 + w2) - u*(b*v + c*w - u*x - v*y - w*z))*oneMinusCosT + x*cosT + (-c*v + b*w - w*y + v*z)*sinT);

		y = ((b*(u2 + w2) - v*(a*u + c*w - u*x - v*y - w*z)) * oneMinusCosT
			+ y*cosT
			+ (c*u - a*w + w*x - u*z)*sinT);

		z = ((c*(u2 + v2) - w*(a*u + b*v - u*x - v*y - w*z)) * oneMinusCosT
			+ z*cosT
			+ (-b*u + a*v - v*x + u*y)*sinT);
	}
}


// Smallest angle between two vectors, returns float
__host__ __device__ double Vector3::angle(Vector3 *vTemp) {
	if (*this == *vTemp)
		return 0;
	else if (*vTemp == Vector3(0., 0., 0))
		return 0;
	else {
		double angle = acos(dot(*vTemp) / (norm()*vTemp->norm()));
		return angle;
	}
}

// Smallest angle between two vectors, returns float
__host__ __device__ double Vector3::cosA(Vector3 *vTemp) {
	double cosA = dot(*vTemp) / (norm()*vTemp->norm());

	return cosA;
}



// Cross product, returns a Vector3
__host__ __device__ Vector3 Vector3::cross(Vector3 *vTemp) {
	double xTemp = y*vTemp->z - z*vTemp->y;
	double yTemp = -(x*vTemp->z - z*vTemp->x);
	double zTemp = x*vTemp->y - y*vTemp->x;

	return Vector3(xTemp, yTemp, zTemp);
}

//Projection http://en.wikipedia.org/wiki/Vector_projection
__host__ __device__ Vector3 Vector3::projectionOnVector(Vector3 v) {
	double lenght = norm()*cosA(&v);

	v.normalize();

	v.x *= lenght;
	v.y *= lenght;
	v.z *= lenght;


	return v;
}



//Overloaded operators :)
__host__ __device__ void Vector3::operator+= (const Vector3 &v) {
	x += v.x;
	y += v.y;
	z += v.z;
}

__host__ __device__ void Vector3::operator-= (const Vector3 &v) {
	x -= v.x;
	y -= v.y;
	z -= v.z;
}

__host__ __device__ void Vector3::operator*= (const Vector3 &v) {

	x *= v.x;
	y *= v.y;
	z *= v.z;
}

__host__ __device__ Vector3 Vector3::operator+ (const Vector3 &v) const {

	return Vector3(x + v.x, y + v.y, z + v.z);
}

__host__ __device__ Vector3 Vector3::operator- (const Vector3 &v) const {

	return Vector3(x - v.x, y - v.y, z - v.z);
}

__host__ __device__ Vector3 Vector3::operator* (const Vector3 &v) const {

	return Vector3(x*v.x, y*v.y, z*v.z);
}

__host__ __device__ bool Vector3::operator==(const Vector3 &v) const {
	return (x == v.x) && (y == v.y) && (z == v.z);
}


__host__ __device__ void Vector3::operator*= (const double f) {

	x *= f;
	y *= f;
	z *= f;
}

__host__ __device__ Vector3 Vector3::operator+ (const double f) const {
	return Vector3(x + f, y + f, z + f);
}

__host__ __device__ Vector3 Vector3::operator- (const double f) const {
	return Vector3(x - f, y - f, z - f);
}

__host__ __device__ Vector3 Vector3::operator* (const double f) const {
	return Vector3(x*f, y*f, z*f);
}

__host__ __device__ Vector3 Vector3::operator/ (const double f) const {
	// if (f == 0)
	// throw;
	if (f == 0)
		return Vector3(0, 0, 0);
	return Vector3(x / f, y / f, z / f);
}

__host__ __device__ Vector3 operator+ (const double f, const Vector3 &v) {
	return v + f;
}

__host__ __device__ Vector3 operator- (const double f, const Vector3 &v) {
	return v - f;
}

__host__ __device__ Vector3 operator* (const double f, const Vector3 &v) {
	return v*f;
}

__host__ __device__ Vector3 Vector3::operator- () const {
	return Vector3(-x, -y, -z);
}

__host__ __device__ bool Vector3::operator> (const Vector3 &v) const {
	return norm() > v.norm();
}


__host__ __device__ void Vector3::rotY(double angle) {
	double temp = x;
	x = x*cos(angle) + z*sin(angle);
	z = z*cos(angle) - temp*sin(angle);
}


void Vector3::description() {

	std::cout << "(" << x << " " << y << " " << z << ")";

}
//Hj�lpfunktioner endast
template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

//__host__ __device__ void binsort(double *max, double *min)
//{
//	if (*max < *min)
//	{
//		double temp = *max;
//		*max = *min;
//		*min = temp;
//	}
//}


