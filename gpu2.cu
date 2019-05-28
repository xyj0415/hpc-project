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

	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint lowmask = width - 1;
	uint highmask = 0xffffffff - lowmask;
	uint left = (index & highmask) | ((index - 1) & lowmask);
	uint right = (index & highmask) | ((index + 1) & lowmask);
	uint top = (index + width) & (board_size - 1);
	uint bottom = (index - width) & (board_size - 1);
	uint topleft = (left + width) & (board_size - 1);
	uint topright = (right + width) & (board_size - 1);
	uint bottomleft = (left - width) & (board_size - 1);
	uint bottomright = (right - width) & (board_size - 1);

	ubyte alive_cell = current[left] + current[right] + current[top] + current[bottom] + current[topleft] + current[topright] + current[bottomleft] + current[bottomright];

	//printf("%d, %d", index, alive_cell);
	//int alive_cell = current[x-1*width+y]+current[x+1*width+y]+current[x*width+y-1]+current[x*width+y+1]+current[x-1*width+y-1]+current[x-1*width+y+1]+current[x+1*width+y-1]+current[x+1*width+y+1];
	// next[index] = (alive_cell == 3 || (alive_cell == 2 && current[index] == 1)) ? 0 : 1;
	next[index] = !((alive_cell | current[index]) ^ 3);
}	

void run_game(ubyte* world, ubyte* current, ubyte* next, uint width, uint height, int iteration, int block_size){
	// dim3 block_dim(32, 32);
	// dim3 grid_dim(width/BLOCK_SIZE, height/BLOCK_SIZE);
	uint board_size = width * height;
	//uint thread_num = 512;
	uint block_num = board_size / block_size;
	for(int i=0; i< iteration; i++){
		//printf("%d\n", i);
		// game_kernel<<<grid_dim, block_dim>>>(current, next, width, height);
		game_kernel<<<block_num, block_size>>>(current, next, width, height, board_size);
		std::swap(current, next);

		// printf("itr: %d\n", i);
		
	}
	// cudaMemcpyAsync(world, current, width * height * sizeof(ubyte), cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();
	// print_world(world, width, height);
}


int main(int argc, char const *argv[]){
	/* code */
	uint width;
	uint height;
	int iteration;
	int block_size;
	if(argc !=4){
        fprintf(stderr, "usage: ./gpu2 [dimension] [iterations] [block size]\n", );
        exit(EXIT_FAILURE);
    }
	sscanf(argv[1], "%d", &width);
	sscanf(argv[1], "%d", &height);
    sscanf(argv[2], "%d", &iteration);
    sscanf(argv[3], "%d", &block_size);
	srand(time(NULL));

	ubyte *world;
	cudaMallocHost((void**)&world, width * height * sizeof(ubyte));
	init_world(world, width, height);
	// print_world(world, width, height);

	ubyte *g_world_current, *g_world_next;
	cudaMalloc(&g_world_current, width * height * sizeof(ubyte));
	cudaMalloc(&g_world_next, width * height * sizeof(ubyte));
	cudaMemcpyAsync(g_world_current, world, width * height * sizeof(ubyte), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(g_world_next, world, width * height * sizeof(ubyte), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	double tt = omp_get_wtime();
	run_game(world, g_world_current, g_world_next, width, height, iteration, block_size);
	printf("time: %lf\n", (omp_get_wtime()-tt));
	cudaFree(g_world_current);
	cudaFree(g_world_next);
	cudaFree(world);
	return 0;
}
