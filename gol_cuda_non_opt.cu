#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <algorithm>
#include <omp.h>

#define BLOCK_SIZE 32
typedef unsigned char ubyte;
typedef unsigned int uint;

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

void init_world(uint8_t* world, int width, int height){
	for(int i = 0; i< width*height; i++)
		world[i] = rand()%2;
}

void print_world(uint8_t* world, int width, int height){
	for(int i = 0; i< width; i++){
		for(int j =0; j < height; j++)
			printf("%d ", world[i*width+j]);
		printf("\n");
	}
}

__global__ void game_kernel(ubyte* current, ubyte* next, uint width, uint height, uint board_size){
	uint x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	uint y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	uint index = y*width+x;
	uint x_left = y*width + ((x+width-1)&(width-1));
	uint x_right = y*width + ((x+1)&(width-1));
	uint y_up = ((y+width-1)&(width-1))*width+x;
	uint y_down = ((y+1)&(width-1))*width+x;
	uint i_ne = ((y+width-1)&(width-1))*width+((x+1)&(width-1));
	uint i_nw = ((y+width-1)&(width-1))*width+((x+width-1)&(width-1));
	uint i_se = ((y+1)&(width-1))*width+((x+1)&(width-1));
	uint i_sw = ((y+1)&(width-1))*width+((x+width-1)&(width-1));

	int alive_cell = current[x_left]+current[x_right]+current[y_up]+current[y_down]+current[i_ne]+current[i_nw]+current[i_se]+current[i_sw];
	//printf("%d, %d", index, alive_cell);
	//int alive_cell = current[x-1*width+y]+current[x+1*width+y]+current[x*width+y-1]+current[x*width+y+1]+current[x-1*width+y-1]+current[x-1*width+y+1]+current[x+1*width+y-1]+current[x+1*width+y+1];
	next[index] = alive_cell == 3 || (alive_cell ==2 && current[index]) ? 1:0;

}	

void run_game(ubyte* world, ubyte* current, ubyte* next, uint width, uint height, int iteration){
	dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_dim(width/BLOCK_SIZE, height/BLOCK_SIZE);
	printf("%d, %d\n", width/BLOCK_SIZE, height/BLOCK_SIZE);

	for(int i=0; i< iteration; i++){
		//printf("%d\n", i);
		game_kernel<<<grid_dim, block_dim>>>(current, next, width, height, width*height);
/*		ubyte *temp = current;
		current = next;
		next = temp;*/

		std::swap(current, next);
		Check_CUDA_Error("test");
		//printf("itr: %d\n", i);
		//cudaMemcpyAsync(world, current, width * height * sizeof(ubyte), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		//print_world(world, width, height);
	}
}


int main(int argc, char const *argv[]){
	/* code */
	uint width = 1024;
	uint height = 1024;
	int iteration = 100000;
	srand(time(NULL));

	ubyte *world;
	cudaMallocHost((void**)&world, width * height * sizeof(ubyte));
	init_world(world, width, height);
	//print_world(world, width, height);

	ubyte *g_world_current, *g_world_next;
	cudaMalloc(&g_world_current, width * height * sizeof(ubyte));
	cudaMalloc(&g_world_next, width * height * sizeof(ubyte));
	cudaMemcpyAsync(g_world_current, world, width * height * sizeof(ubyte), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(g_world_next, world, width * height * sizeof(ubyte), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	double tt = omp_get_wtime();
	run_game(world, g_world_current, g_world_next, width, height, iteration);
	printf("time: %lf\n", (omp_get_wtime()-tt));
	cudaFree(g_world_current);
	cudaFree(g_world_next);
	cudaFree(world);
	return 0;
}