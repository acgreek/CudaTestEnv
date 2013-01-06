// MP 5 
#include	<wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void vecCumSum(float * in1, float * out, int len) {

}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostOutput;
    float * deviceInput1;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength, " elements");

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    int byteSize =inputLength * sizeof(float);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here

    wbCheck(cudaMalloc((void **) &deviceInput1, byteSize));
    wbCheck(cudaMalloc((void **) &deviceOutput, byteSize));

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    wbCheck(cudaMemcpy(deviceInput1, hostInput1, byteSize,cudaMemcpyHostToDevice));


    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
     int block_size = 16;
     int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1);


#ifndef CUDA_EMU
    vecCumAdd<<< n_blocks, block_size>>>(deviceInput1, deviceOutput, inputLength);
#else
    setupCudaSim (n_blocks, block_size, boost::bind(vecCumSum,deviceInput1,deviceOutput, inputLength));
#endif


    wbCheck(cudaThreadSynchronize());
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, byteSize,cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here

    wbCheck(cudaFree(deviceInput1));
    wbCheck(cudaFree(deviceOutput));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostOutput);

    return 0;
}
