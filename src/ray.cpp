#include "ray.h"
#include <cmath>

Vec3d CosWeightedRandomHemiDir2(Vec3d n){
  double Xi1 = (double)rand()/(double)RAND_MAX;
  double Xi2 = (double)rand()/(double)RAND_MAX;

  double theta = acos(sqrt(1.0-Xi1));
  double phi = 2.0*3.1415926535897932384626433832795 * Xi2;

  double xs = sin(theta) * cos(phi);
  double ys = cos(theta);
  double zs = sin(theta)*sin(phi);

  Vec3d h = n;
  if(fabs(h[0]) <= fabs(h[1]) && fabs(h[0]) <= fabs(h[2]))
    h[0] = 1.0;
  else if(fabs(h[1]) <= fabs(h[0]) && fabs(h[1]) <= fabs(h[2]))
    h[1] = 1.0;
  else
    h[2] = 1.0;

  Vec3d x = (h ^ n); x.normalize();
  Vec3d z = (x ^ n); z.normalize();
  Vec3d dir = xs*x + ys*n + zs*z;
  dir.normalize();
  return dir;
}
