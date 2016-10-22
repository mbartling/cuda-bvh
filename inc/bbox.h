#pragma once
#include "common.h"

class BoundingBox {
	
	bool bEmpty;
	bool dirty;
	Vec3f bmin;
	Vec3f bmax;
	float bArea;
	float bVolume;

public:

    __device__
	BoundingBox() : bEmpty(true) {}
    
    __device__
	BoundingBox(Vec3f bMin, Vec3f bMax) : bmin(bMin), bmax(bMax), bEmpty(false), dirty(true) {}

    __device__ __inline__
	Vec3f getMin() const { return bmin; }
    __device__ __inline__
	Vec3f getMax() const { return bmax; }
    __device__
	bool isEmpty() { return bEmpty; }

    __device__
	void setMin(Vec3f bMin) {
		bmin = bMin;
		dirty = true;
		bEmpty = false;
	}
    __device__
	void setMax(Vec3f bMax) {
		bmax = bMax;
		dirty = true;
		bEmpty = false;
	}
    __device__
	void setMin(int i, float val) {
		if (i == 0) { bmin.x = val; dirty = true; bEmpty = false; }
		else if (i == 1) { bmin.y = val; dirty = true; bEmpty = false; }
			else if (i == 2) { bmin.z = val; dirty = true; bEmpty = false; }
	}
    __device__
	void setMax(int i, float val) {
		if (i == 0) { bmax.x = val; dirty = true; bEmpty = false; }
		else if (i == 1) { bmax.y = val; dirty = true; bEmpty = false; }
			else if (i == 2) { bmax.z = val; dirty = true; bEmpty = false; }
	}
    __device__
	void setEmpty() {
		bEmpty = true;
	}

	// Does this bounding box intersect the target?
    __device__
		bool intersects(const BoundingBox &target) const {
			return ((target.getMin().x - RAY_EPSILON <= bmax.x) && (target.getMax().x + RAY_EPSILON >= bmin.x) &&
				(target.getMin().y - RAY_EPSILON <= bmax.y) && (target.getMax().y + RAY_EPSILON >= bmin.y) &&
				(target.getMin().z - RAY_EPSILON <= bmax.z) && (target.getMax().z + RAY_EPSILON >= bmin.z));
	}

	// does the box contain this point?
    __device__
	bool intersects(const Vec3f& point) const {
		return ((point.x + RAY_EPSILON >= bmin.x) && (point.y + RAY_EPSILON >= bmin.y) && (point.z + RAY_EPSILON >= bmin.z) &&
			(point.x - RAY_EPSILON <= bmax.x) && (point.y - RAY_EPSILON <= bmax.y) && (point.z - RAY_EPSILON <= bmax.z));
	}

	// if the ray hits the box, put the "t" value of the intersection
	// closest to the origin in tMin and the "t" value of the far intersection
	// in tMax and return true, else return false.
	// Using Kay/Kajiya algorithm.
    __device__
	bool intersect(const ray& r, float& tMin, float& tMax) const {
		Vec3f R0 = r.getPosition();
		Vec3f Rd = r.getDirection();
		tMin = -1.0e308; // 1.0e308 is close to infinity... close enough for us!
		tMax = 1.0e308;
		float ttemp;
	
		for (int currentaxis = 0; currentaxis < 3; currentaxis++) {
			float vd = Rd[currentaxis];
			// if the ray is parallel to the face's plane (=0.0)
			if( vd == 0.0 ) continue;
			float v1 = bmin[currentaxis] - R0[currentaxis];
			float v2 = bmax[currentaxis] - R0[currentaxis];
			// two slab intersections
			float t1 = v1/vd;
			float t2 = v2/vd;
			if ( t1 > t2 ) { // swap t1 & t2
				ttemp = t1;
				t1 = t2;
				t2 = ttemp;
			}
			if (t1 > tMin) tMin = t1;
			if (t2 < tMax) tMax = t2;
			if (tMin > tMax) return false; // box is missed
			if (tMax < RAY_EPSILON) return false; // box is behind ray
		}
		 return true; // it made it past all 3 axes.
	}

    __device__
	void operator=(const BoundingBox& target) {
		bmin = target.bmin;
		bmax = target.bmax;
		bArea = target.bArea;
		bVolume = target.bVolume;
		dirty = target.dirty;
		bEmpty = target.bEmpty;
	}

    __device__
	float area() {
		if (bEmpty) return 0.0;
		else if (dirty) {
			bArea = 2.0 * ((bmax.x - bmin.x) * (bmax.y - bmin.y) + (bmax.y - bmin.y) * (bmax.z - bmin.z) + (bmax.z - bmin.z) * (bmax.x - bmin.x));
			dirty = false;
		}
		return bArea;
	}

    __device__
	float volume() {
		if (bEmpty) return 0.0;
		else if (dirty) {
			bVolume = ((bmax.x - bmin.x) * (bmax.y - bmin.y) * (bmax.z - bmin.z));
			dirty = false;
		}
		return bVolume;
	}

    __device__
	void merge(const BoundingBox& bBox)	{
		if (bBox.bEmpty) return;
		for (int axis = 0; axis < 3; axis++) {
			if (bEmpty || bBox.bmin[axis] < bmin[axis]) bmin[axis] = bBox.bmin[axis];
			if (bEmpty || bBox.bmax[axis] > bmax[axis]) bmax[axis] = bBox.bmax[axis];
		}
		dirty = true;
		bEmpty = false;
	}
};
