// MP 1
#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<len) out[idx] = in1[idx] + in2[idx];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength, " elements");


    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    int byteSize =sizeof(float) * inputLength;

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here

    cudaMalloc((void **) &deviceInput1, byteSize);
    cudaMalloc((void **) &deviceInput2, byteSize);
	cudaMalloc((void **) &deviceOutput, byteSize);


    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    cudaMemcpy(deviceInput1, hostInput1, byteSize,cudaMemcpyHostToDevice);

    cudaMemcpy(deviceInput2, hostInput1, byteSize,cudaMemcpyHostToDevice);


    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
     int block_size = 16;
     int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1);

    dim3 dimGrid(n_blocks,1, 1) ;
    dim3 dimBlock(block_size, 1,1);

#ifndef CUDA_EMU
    vecAdd<<< n_blocks, block_size>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
#else
    setupCudaSim (dimGrid , dimBlock , boost::bind(vecAdd,deviceInput1,deviceInput2,deviceOutput, inputLength));
#endif


    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, byteSize,cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here


    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
