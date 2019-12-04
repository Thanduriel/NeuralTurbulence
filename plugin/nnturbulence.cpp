#include "vectorbase.h"
#include "shapes.h"
#include "commonkernels.h"
#include "kernel.h"

namespace Manta {
/*
KERNEL(bnd = 1)
	void CurlOpMAC(const MACGrid& grid, MACGrid& dst) {
	Vec3 v = Vec3(0., 0.,
		0.5 * ((grid(i, j, k).y - grid(i - 1, j, k).y) - (grid(i, j, k).x - grid(i, j - 1, k).x)));
	if (dst.is3D()) {
		v[0] = 0.5 * ((grid(i, j, k).z - grid(i, j - 1, k).z) - (grid(i, j, k + 1).y - grid(i, j, k - 1).y));
		v[1] = 0.5 * ((grid(i, j, k).x - grid(i, j, k - 1).x) - (grid(i, j, k).z - grid(i - 1, j, k).z));
	}
	dst(i, j, k) = v;
};*/

KERNEL(idx) void SwizzleSub(Grid<Vec3>& grid, int unused) 
{ 
	const Vec3 v = grid[idx];
	grid[idx] = Vec3(v.y, -v.x, 0.0);
}

PYTHON() void getCodifferential(MACGrid& vel, const Grid<Real>& vort) {
	Grid<Vec3> velCenter(vel.getParent());
	GradientOp(velCenter, vort);
	SwizzleSub(velCenter, 0);

	GetMAC(vel, velCenter);
}

PYTHON() void getDivergence(Grid<Real>& div,const MACGrid& vel){
	DivergenceOpMAC(div, vel);
}
/*
PYTHON() void getCurlMAC(const MACGrid& vel, Grid<Real>& vort, int comp) {
	MACGrid curl(vel.getParent());

CurlOpMAC(velCenter, curl);
GetComponent(curl, vort, comp);
}*/

}