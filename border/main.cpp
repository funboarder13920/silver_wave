#include <iostream>
#include<string>
#include <time.h>
#include <cstdlib>
using namespace std;
#include <Imagine/Images.h>
#include <Imagine/Graphics.h>
using namespace Imagine;

#define MIN(a, b) ((a > b) ? b : a)
#define MAX(a, b) ((a < b) ? b : a)

const int N = 512;
const float h = 1 / float(N);
const float dt = 0.005;
const int size = (N + 2)*(N + 2);
const float diff = 0.01;
const float visc = 0.00025;

float* u = new float[size];
float* u_prev = new float[size];
float* v = new float[size];
float* v_prev = new float[size];
float* dens = new float[size];
float* dens_prev = new float[size];

int IX(int i, int j){ return ((i)+(N + 2)*j); }

void add_source(float* x, float* s){
	for (int i = 0; i < size; i++) x[i] += dt*s[i];
}

void set_bnd(int b, float * x)// on peut s'arranger pour b
{
	int i;
	for (i = 1; i <= N; i++) {
		x[IX(0, i)] = b == 1 ? (- x[IX(1, i)]) : x[IX(1, i)];
		x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
		x[IX(i, 0)] = b == 2 ? - x[IX(i, 1)] : x[IX(i, 1)];
		x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
	}
	x[IX(0, 0)] = 0.5*(x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N + 1)] = 0.5*(x[IX(1, N + 1)] + x[IX(0, N)]);
	x[IX(N + 1, 0)] = 0.5*(x[IX(N, 0)] + x[IX(N + 1, 1)]);
	x[IX(N + 1, N + 1)] = 0.5*(x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

void diffuse(int b, float* x, float* x0){
	float a = dt*diff*N*N;
	for (int k = 0; k < 20; k++){
		for (int i = 1; i < N; i++){
			for (int j = 1; j < N; j++){
				x[IX(i, j)] = (x0[IX(i, j)] + a*(x[IX(i - 1, j)] + x[IX(i + 1, j)] +
					x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1 + 4 * a);
			}
		}
		set_bnd(b, x);
	}
}

void advect(int b, float* d, float* d0, float* u, float* v){
	float x, y, s0, t0, s1, t1, dt0;
	dt0 = dt*N;
	for (int i = 1; i < N; i++){
		for (int j = 1; j < N; j++) {
			x = i - dt0*u[IX(i, j)];
			y = j - dt0*v[IX(i, j)];
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
			d[IX(i, j)] = s0*(t0*d0[IX(i0, j0)] + t1*d0[IX(i0, j1)]) +
				s1*(t0*d0[IX(i1, j0)] + t1*d0[IX(i1, j1)]);
		}
	}
	set_bnd(b, d);
}

void dens_step(float* x, float* x0, float* u, float* v){
	add_source(x, x0);
	float* temp = x0; x0 = x; x = temp;
	diffuse(0, x, x0);
	temp = x0; x0 = x; x = temp;
	advect(0, x, x0, u, v);
}

void project(float * u, float * v, float * p, float * div)
{
	int i, j, k;
	float h;
	h = 1.0 / N;
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			div[IX(i, j)] = -0.5*h*(u[IX(i + 1, j)] - u[IX(i - 1, j)] +
				v[IX(i, j + 1)] - v[IX(i, j - 1)]);
			p[IX(i, j)] = 0;
		}
	}
	set_bnd(0, div); set_bnd( 0, p);
	for (k = 0; k<20; k++) {
		for (i = 1; i <= N; i++) {
			for (j = 1; j <= N; j++) {
				p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
					p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
			}
		}
		set_bnd(0, p);
	}
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			u[IX(i, j)] -= 0.5*(p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
			v[IX(i, j)] -= 0.5*(p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
		}
	}
	set_bnd(1, u); set_bnd(2, v);
}


void vel_step(float* u, float* v, float* u0, float* v0){
	add_source(u, u0); add_source(v, v0);
	float* temp = u0; u0 = u; u = temp;
	diffuse(1, u, u0);
	temp = v0; v0 = v; v = temp;
	diffuse(2, v, v0);
	project(u, v, u0, v0);
	temp = v0; v0 = v; v = temp;
	temp = u0; u0 = u; u = temp;
	advect(1, u, u0, u0, v0);
	advect(2, v, v0, u0, v0);
	project(u, v, u0, v0);
}



void maxmin(float &min, float &max, float* dens){
	for (int absc = 1; absc < N+1; absc++) {
		for (int ord = 1; ord < N+1; ord++){
			min = MIN(dens[(1 + ord)*(N + 2) + (1 + absc)], min);
			max = MAX(dens[(1 + ord)*(N + 2) + (1 + absc)], max);
		}
	}
}

void draw_dens(float* dens, Image<byte> image) {
	float t, dens_min, dens_max;
	maxmin(dens_min, dens_max, dens);
	dens_max = 5 * dens_max / 100;
	cout << "min : " << dens_min << endl;
	cout << "max : " << dens_max << endl;
	if (dens_min == dens_max) {
		cout << " min = max" << endl;
		return;
	}
	for (int absc = 0; absc < image.height(); absc++) {
		for (int ord = 0; ord < image.width(); ord++){
			image(absc, ord) = int(MIN(255,255 *((dens[(1 + ord)*(N + 2) + (1 + absc)] - dens_max) / (dens_min - dens_max))));
		}
	}
	display(image);
}

void init_u_v(float *u, float *v)
{
	for (int absc = 4*N / 10; absc < 5 * N / 10; absc++) {
		for (int ord = 7*N / 10; ord < 9 * N / 10; ord++) {
			u[ord * (N + 2) + absc] = 190;
			v[ord * (N + 2) + absc] = 0;
		}
	}
}

int main()
{
	srand(time(NULL));
	for (int i = 0; i < size; i++){
		dens[i] = rand() % 1000;
	}
	Image<byte> image = Image<byte>(N, N);
	openWindow(image.width(), image.height(),"Fluide");
	init_u_v(u_prev, v_prev);
	bool simulating = true;
	int i = 0;
	while (simulating)
	{
		i++;
		//get_from_UI(dens_prev, u_prev, v_prev);

		init_u_v(u_prev, v_prev);
		
		vel_step(u, v, u_prev, v_prev);
		dens_step(dens, dens_prev, u, v);
		if (i % 5 == 0){
			draw_dens(dens,image);
		}
		cout << i << endl;
	}
	return 0;
}