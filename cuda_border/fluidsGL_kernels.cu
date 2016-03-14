/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>    // Helper functions for CUDA Error handling

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// FluidsGL CUDA kernel definitions
#include "fluidsGL_kernels.cuh"

// Texture reference for reading velocity field
texture<float2, 2> texref;
static cudaArray *array = NULL;

// Particle data
extern GLuint vbo;                 // OpenGL vertex buffer object
extern struct cudaGraphicsResource *cuda_vbo_resource; // handles OpenGL-CUDA exchange

// Texture pitch
extern size_t tPitch;

void setupTexture(int x, int y)
{
    // Wrap mode appears to be the new default
    texref.filterMode = cudaFilterModeLinear;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

    cudaMallocArray(&array, &desc, y, x);
    getLastCudaError("cudaMalloc failed");
}

void bindTexture(void)
{
    cudaBindTextureToArray(texref, array);
    getLastCudaError("cudaBindTexture failed");
}

void unbindTexture(void)
{
    cudaUnbindTexture(texref);
}

void updateTexture(cData* data, size_t wib, size_t h, size_t pitch)
{
    cudaMemcpy2DToArray(array, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice);
    getLastCudaError("cudaMemcpy failed");
}

void deleteTexture(void)
{
    cudaFreeArray(array);
}

// Note that these kernels are designed to work with arbitrary
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06


__host__ __device__ int  IX(int i, int j) {
	return i + DIM*j;
}

void set_bnd(int b, cData* x, int c)
{
	int i;
	if (c == 1){
		for (i = 1; i <= N; i++) {
			x[IX(0, i)].x = b == 1 ? (-x[IX(1, i)].x) : x[IX(1, i)].x;
			x[IX(N + 1, i)].x = b == 1 ? -x[IX(N, i)].x : x[IX(N, i)].x;
			x[IX(i, 0)].x = b == 2 ? -x[IX(i, 1)].x : x[IX(i, 1)].x;
			x[IX(i, N + 1)].x = b == 2 ? -x[IX(i, N)].x : x[IX(i, N)].x;
		}
		x[IX(0, 0)].x = 0.5*(x[IX(1, 0)].x + x[IX(0, 1)].x);
		x[IX(0, N + 1)].x = 0.5*(x[IX(1, N + 1)].x + x[IX(0, N)].x);
		x[IX(N + 1, 0)].x = 0.5*(x[IX(N, 0)].x + x[IX(N + 1, 1)].x);
		x[IX(N + 1, N + 1)].x = 0.5*(x[IX(N, N + 1)].x + x[IX(N + 1, N)].x);
	}
	else{
		for (i = 1; i <= N; i++) {
			x[IX(0, i)].y = b == 1 ? -x[IX(1, i)].y : x[IX(1, i)].y;
				x[IX(N + 1, i)].y = b == 1 ? -x[IX(N, i)].y : x[IX(N, i)].y;
				x[IX(i, 0)].y = b == 2 ? -x[IX(i, 1)].y : x[IX(i, 1)].y;
				x[IX(i, N + 1)].y = b == 2 ? -x[IX(i, N)].y : x[IX(i, N)].y;
		}
		x[IX(0, 0)].y = 0.5*(x[IX(1, 0)].y + x[IX(0, 1)].y);
		x[IX(0, N + 1)].y = 0.5*(x[IX(1, N + 1)].y + x[IX(0, N)].y);
		x[IX(N + 1, 0)].y = 0.5*(x[IX(N, 0)].y + x[IX(N + 1, 1)].y);
		x[IX(N + 1, N + 1)].y = 0.5*(x[IX(N, N + 1)].y + x[IX(N + 1, N)].y);
	}
}


// This method adds constant force vectors to the velocity field
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void
addforces_k( int dx, int dy,int n, cData* v,int spx, int spy,float fx, float fy, int r, size_t pitch)
{
	// spx,spy : position de la souris
	// thread dans laquelle on est
	// tx is the thread location in x
	// ty is the thread location in y
	// fj is the global thread position in th velocity array
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	//printf("tx: %d, ty: %d \n", tx,ty);
	//printf("spx: %d, spy: %d \n", spx,spy);
	// passage sur tous les threads 8x8
    cData *fj = &v[IX((ty + spy) , tx + spx)];
    cData vterm = *fj;
    tx -= r;
    ty -= r;
	//s : coeff à appliquer à la force (génère de l'aléa) pourquoi?
    float s = 1.f / (1.f + tx*tx*tx*tx + ty*ty*ty*ty);
	//s = 0.1;
    vterm.x += s * fx;
    vterm.y += s * fy;//+0.1 ou + gravity?
	//if (tx == -1 && ty == -2)	printf("fj avant : %f,%f \n", fj->x,fj->y);
	//printf("(ty + spy) * pitch : %d , tx + spx : %d \n", (ty + spy) * pitch, tx + spx);

    *fj = vterm;
	//if (tx == 0 && ty == -2) printf("fj apres : %f,%f \n", fj->x, fj->y);
}




__global__ void
advect_k(int dx, int dy, int n, cData* v, cData* v_prev, float dt, int lb){

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;
	float x, y, s1, s0, t1, t0;
	float dt0 = dt*n;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				// traduction de (gtidx,fi) dans un tableau de gtidx*fi
				int i = gtidx;
				int j = fi;
				x = i - dt0*v[IX(i, j)].x;
				y = j - dt0*v[IX(i, j)].y;
				if (x < 0.5) x = 0.5;
				if (x > N + 0.5) x = N + 0.5;
				if (y < 0.5) y = 0.5;
				if (y > N + 0.5) y = N + 0.5;
				int i0 = int(x);
				int i1 = i0 + 1;
				int j0 = int(y);
				int j1 = j0 + 1;
				s1 = x - i0;
				s0 = 1 - s1;
				t1 = y - j0;
				t0 = 1 - t1;
				v[IX(i, j)].x = s0*(t0*v_prev[IX(i0, j0)].x + t1*v_prev[IX(i0, j1)].x) +
					s1*(t0*v_prev[IX(i1, j0)].x + t1*v_prev[IX(i1, j1)].x);
				v[IX(i, j)].y = s0*(t0*v_prev[IX(i0, j0)].y + t1*v_prev[IX(i0, j1)].y) +
					s1*(t0*v_prev[IX(i1, j0)].y + t1*v_prev[IX(i1, j1)].y);
			}
		}
	}
}

// This method performs velocity diffusion and forces mass conservation
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void
diffuse_k(int dx,int dy, int n, cData *v, cData *v0, float dt,
                 float diff, int lb)
{

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;
	float a = dt*diff*N*N;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx)
    {
        for (p = 0; p < lb; p++)
        {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;

            if (fi < dy)
            {
                int fj = fi * dx + gtidx;
				if (fj%(N+2) != 0 && fj % (N+1) != 0){
					v[fj].x = (v0[fj].x + a*(v[fi * dx + gtidx - 1].x + v[fi * dx + gtidx + 1].x +
						v[(fi + 1) * dx + gtidx].x + v[(fi - 1) * dx + gtidx].x)) / (1 + 4 * a);
					v[fj].y = (v0[fj].y + a*(v[fi * dx + gtidx - 1].y + v[fi * dx + gtidx + 1].y +
						v[(fi + 1) * dx + gtidx].y + v[(fi - 1) * dx + gtidx].y)) / (1 + 4 * a);

					//v[fj] = (v0[fj] + a*(v[fi * dx + gtidx - 1] + v[fi * dx + gtidx + 1] +
						//v[(fi + 1) * dx + gtidx] + v[(fi - 1) * dx + gtidx])) / (1 + 4 * a);
				}
            }
        }
    }
}

// This method updates the velocity field 'v' using the two complex
// arrays from the previous step: 'vx' and 'vy'. Here we scale the
// real components by 1/(dx*dy) to account for an unnormalized FFT.
__global__ void
updateVelocity_k(cData *v, float *vx, float *vy,
int dx, int pdx, int dy, int lb, size_t pitch)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	float vxterm, vyterm;
	cData nvterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fjr = fi * pdx + gtidx;
				

			}
		} // If this thread is inside the domain in Y
	}
}

__global__ void
advectParticles_k(cData *part, cData *v, int dx, int dy,
float dt, int lb, size_t pitch)
{

	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;

	// gtidx is the domain location in x for this thread
	cData pterm, vterm;

	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fj = fi * dx + gtidx;
				pterm = part[fj];

				int xvi = ((int)(pterm.x * dx));
				int yvi = ((int)(pterm.y * dy));
				vterm = *((cData *)((char *)v + yvi * pitch) + xvi);

				pterm.x += dt * vterm.x;
				pterm.x = pterm.x - (int)pterm.x;
				pterm.x += 1.f;
				pterm.x = pterm.x - (int)pterm.x;
				pterm.y += dt * vterm.y;
				pterm.y = pterm.y - (int)pterm.y;
				pterm.y += 1.f;
				pterm.y = pterm.y - (int)pterm.y;

				part[fj] = pterm;
			}
		} // If this thread is inside the domain in Y
	} // If this thread is inside the domain in X
}




// These are the external function calls necessary for launching fluid simulation
extern "C"
void addforce(int dx, int dy,int n,cData* v, int spx, int spy, float fx, float fy, int r)
{

    dim3 tids(2*r+1, 2*r+1);
    addforces_k<<<1, tids>>>(dx,dy,n,v,spx,spy,fx, fy, r, tPitch);
    getLastCudaError("addForces_k failed.");
}

extern "C"
void advect(int dx, int dy,int n, int bx, int by, cData* v, cData* v_prev, float dt)
{
    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));

    dim3 tids(TIDSX, TIDSY);

    updateTexture(v, DIM*sizeof(cData), DIM, tPitch);
    advect_k<<<grid, tids>>>(dx,dy,n,v, v_prev, dt, TILEY/TIDSY);
	set_bnd(bx, v,1); // 1 sur x
	set_bnd(by, v, 2); // 2 sur y
    getLastCudaError("advectVelocity_k failed.");
}

extern "C"
void diffuse(int dx, int dy, int n, int bx, int by,cData* v, cData* v_prev, float visc, float dt)
{

    uint3 grid = make_uint3((dx/TILEX)+(!(dx%TILEX)?0:1),
                            (dy/TILEY)+(!(dy%TILEY)?0:1), 1);
    uint3 tids = make_uint3(TIDSX, TIDSY, 1);
	for (int k = 0; k < 20; k++){
		diffuse_k <<<grid, tids >> >(dx,dy,n,v,v_prev,visc,dt, TILEY / TIDSY);
		set_bnd(bx, v,1);
		set_bnd(by, v, 2);
	}
    getLastCudaError("diffuse_k failed.");

}


extern "C"
void advectParticles(GLuint vbo, cData* v, int dx, int dy, float dt)
{
    dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1), (dy/TILEY)+(!(dy%TILEY)?0:1));
    dim3 tids(TIDSX, TIDSY);

    cData *p;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    getLastCudaError("cudaGraphicsMapResources failed");

    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes,
                                         cuda_vbo_resource);
    getLastCudaError("cudaGraphicsResourceGetMappedPointer failed");

    advectParticles_k<<<grid, tids>>>(p, v, dx, dy, dt, TILEY/TIDSY, tPitch);
    getLastCudaError("advectParticles_k failed.");
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    getLastCudaError("cudaGraphicsUnmapResources failed");
}
