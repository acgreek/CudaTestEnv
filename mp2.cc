// MP 3: Due Sunday, Dec 30, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define tile_width 8 
#define tile_height 8
#define TILE_WIDTH 8 
      
// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    wbLog(DEBUG, "ARow ", numARows);
    wbLog(DEBUG, "AColumn ", numAColumns);
  
    //@@ Allocate the hostC matrix
    hostC =(float *)  malloc (numCRows * numCColumns * sizeof(float) );
    if (NULL == hostC) {
       wbLog(ERROR, "Failed to allocate hostC memory");
       return -1;
    } 
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc((void **) &deviceA,numARows * numAColumns * sizeof(float) ));
    wbCheck(cudaMalloc((void **) &deviceB,numBRows * numBColumns * sizeof(float) ));
    wbCheck(cudaMalloc((void **) &deviceC,numCRows * numCColumns * sizeof(float) ));
    
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA,numARows * numAColumns * sizeof(float) , cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB,numBRows * numBColumns * sizeof(float) , cudaMemcpyHostToDevice));
  
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here


           
    dim3 dimGrid(ceil(numCColumns/(float) tile_width), ceil(numCRows/(float) tile_height), 1) ;
    dim3 dimBlock( tile_width, tile_height,1);
  
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
#ifndef CUDA_EMU
    matrixMultiplyShared <<< dimGrid , dimBlock >>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
#else 
    
    setupCudaSim (dimGrid , dimBlock , boost::bind(matrixMultiplyShared,deviceA,deviceB,deviceC,numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns));
#endif
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostC, deviceC,numCRows * numCColumns * sizeof(float) , cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));
  
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
