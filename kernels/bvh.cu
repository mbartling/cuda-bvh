#include <stdio.h>
#include "bvh.h"

__global__ void hello()
{
   printf("Hello world! I'm a thread in block %d\n", blockIdx.x);

}


void bvh(void)
{
   //launch the kernel
   hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

   //force the printf()s to flush
   cudaDeviceSynchronize();

   printf("That's all!\n");

}
