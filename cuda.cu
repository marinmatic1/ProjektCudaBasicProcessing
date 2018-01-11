#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "loading.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include<cmath>

const int xBlockSize = 32;
const int yBlockSize = 32;

typedef struct {
	int x, y;
	float *grey;

} d_image;
void mat_u_niz(PGMImage izvorna, float *pomIzv)
{
	for (int i = 0; i<izvorna.y; i++)
		for (int j = 0; j<izvorna.x; j++)
			pomIzv[i*izvorna.x + j] = izvorna.grey[i][j];
}

__global__ void NekaFunkcija(d_image d_izvorna, float *nizS)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int num = d_izvorna.x;
	float Kernel[3][3] = {
		{ 0.07513,0.12384,0.07513 },
		{ 0.12384,0.20418,0.12384 },
		{ 0.07513,0.12384,0.07513 }
	};

	float sum;
	if (row<(d_izvorna.y - 1) && (col<d_izvorna.x - 1))
	{
		sum = 0.0;
		for (int k = -1; k <= 1; k++)
		{
			for (int j = -1; j <= 1; j++)
			{
				sum = sum + Kernel[j + 1][k + 1] * (d_izvorna.grey[(row - j)*num + (col - k)]);
			}
		}
		nizS[row*num + col] = sum;
	}
}


int main()
{
	GpuTimer timer;
	PGMImage izvorna, nova;

	d_image d_izvorna;
	float *d_nova;
	float *izvorniNiz, *pom;
	cudaError_t err;
	char file1[1024] = "fpmoz01.pgm";
	char file2[1024] = "fpmoz10.pgm";

	ucitajPGM(file1, &izvorna);

	alloc_matrix(&nova.grey, izvorna.y, izvorna.x);
	nova.x = izvorna.x;
	nova.y = izvorna.y;

	//alociramo memoriju za pomocni niz i izvorni niz
	pom = (float *)malloc(nova.x*nova.y*sizeof(float));
	izvorniNiz = (float *)malloc(nova.x*nova.y*sizeof(float));

	//prebacujemo izvornu matricu(sliku) u niz 
	mat_u_niz(izvorna, izvorniNiz);

	d_izvorna.x = izvorna.x;
	d_izvorna.y = izvorna.y;

	// veličinu memorije u bajtovima za niz
	int size = d_izvorna.x * d_izvorna.y * sizeof(float);
	//alociranje memorije naGPU
	cudaMalloc((void **)& d_izvorna.grey, size);
	cudaMalloc((void **)& d_nova, size);

	//kopiranje na GPU
	cudaMemcpy(d_izvorna.grey, izvorniNiz, size, cudaMemcpyHostToDevice);
	//u blkSize funkciji imamo broj dretvi po bloku
	dim3 blkSize(xBlockSize, yBlockSize);
	//numBlock nam pokazuje koliko blokova imamo
	dim3 numBlock(ceil((float)izvorna.x / xBlockSize), ceil((float)izvorna.y / yBlockSize));

	//pocetak mjerenja
	timer.Start();
	//jezgrena funkcija
	NekaFunkcija << < numBlock, blkSize >> > (d_izvorna, d_nova);
	err = cudaDeviceSynchronize();
	//zavrsetak mjerenja
	timer.Stop();

	//ceka zavrsetak svih radnji za nastavak

	printf("Izvedba kernela: %s\n", cudaGetErrorString(err));

	printf("Vrijeme izvrsenja u cudi je = %g ms\n", timer.Elapsed());

	//kopiranje sa GPU na host
	err = cudaMemcpy(pom, d_nova, size, cudaMemcpyDeviceToHost);
	printf("Kopiranje na host: %s\n", cudaGetErrorString(err));

	//prebacivanje niza u matricu
	for (int i = 0; i < nova.y; i++)
		for (int j = 0; j < nova.x; j++)
			nova.grey[i][j] = pom[i*nova.x + j];

	zapisiPGM(file2, &nova);

	//oslobađanje memorije
	cudaFree(d_izvorna.grey);
	cudaFree(d_nova);
	free(pom);
	free(izvorniNiz);

	disalloc_matrix(izvorna.grey, izvorna.y, izvorna.x);
	disalloc_matrix(nova.grey, nova.y, nova.x);

	printf("Press any key...");
	getchar();
}
