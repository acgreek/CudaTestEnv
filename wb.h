#ifndef WB_H
#define WB_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/thread/tss.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "thread_processor.hpp"

#include <fstream>

using namespace std;

#define __global__
#define __shared__ volatile static


#define blockIdx getBlockIdx()
#define threadIdx getThreadIdx()
#define blockDim getBlockDim()

typedef struct _Block_t  {
	int x;
	int y;
	int z;
} Block_t;

struct CudaThreadLocal {
	Block_t block;
	Block_t thread;
	int phase1;
	int phase2;
	CudaThreadLocal() : phase1(0),phase2(0)  {}
};

static void doNothing(CudaThreadLocal * ptr) {
}

boost::thread_specific_ptr<CudaThreadLocal> tls (doNothing);

Block_t getBlockIdx() {
	CudaThreadLocal  *p = tls.get();
	Block_t  d;
	d.x=p->block.x;
	d.y=p->block.y;
	return d;
}
Block_t getThreadIdx() {
	CudaThreadLocal  *p = tls.get();
	Block_t  d;
	d.x=p->thread.x;
	d.y=p->thread.y;
	return d;
}
static Block_t g_blockDim;
Block_t getBlockDim() {
	return g_blockDim;
}

static int g_num_threads;

boost::shared_ptr<boost::barrier> g_barrierp;
boost::shared_ptr<boost::barrier> g_barrier_2p;
boost::mutex g_b_mutex1;
boost::mutex g_b_mutex2;
bool has_t1_set_mutex=true;
int b_phase1=0;
int b_phase2=0;


void __syncthreads() {
	Block_t bl = getThreadIdx() ;
	CudaThreadLocal  *p = tls.get();

//	printf("thread %d %d is waiting on barrier 1\n", bl.x, bl.y);
	g_barrierp->wait();
	{
		boost::mutex::scoped_lock lock( g_b_mutex1);
		if (b_phase1 == p->phase1) {
			g_barrier_2p.reset ( new boost::barrier ( g_num_threads )) ;
			printf("thread %d %d reset barrier 1\n", bl.x, bl.y);
			b_phase1++;
		}
		p->phase1 = b_phase1 ;

	}
//	printf("thread %d %d is done waiting on barrier 1\n", bl.x, bl.y);
	g_barrier_2p->wait();
	{
		boost::mutex::scoped_lock lock( g_b_mutex2);
		if (b_phase2 == p->phase2) {
			g_barrierp.reset ( new boost::barrier ( g_num_threads )) ;
			printf("thread %d %d reset barrier 2\n", bl.x, bl.y);
			b_phase2++;
		}
		p->phase2 = b_phase2 ;

	}
}


typedef int wbArg_t;

char * filen[2];

int wbArg_read(int argc __attribute__((unused)), char * argv[] __attribute__((unused))) {
	if (argc< 2) {
		fprintf(stderr, "wrong number of args, requires 2\n");
		exit(-1);
	}
	filen[0] = argv[1];
	filen[1] = argv[2];

	return 0;
}


char *wbArg_getInputFile(int v __attribute__((unused)) , int index) {
	if (index > 1) {
		fprintf(stderr, "wrong index used, should be 1 or 2\n");
		exit(-1);
	}
	return filen[index];
}


// Stupid error treatment, to be substituted with something more elegant.
void die_if(bool cond, const char *msg1, const char *msg2)
{
	if (cond) {
		cerr << msg1 << msg2 << endl;
		exit(-1);
	}
}

//* read the file into 2 dimensional array
void *wbImport(const char *filename, int *rowp, int *columnp)
{
	int rows, cols;

	fstream f(filename);
	die_if(!f, "Error opening file: ", filename);

	f >> rows >> cols;

	float *array = (float *) malloc(rows * cols * sizeof(float));
	die_if(!array, "Error allocating memory for array.", "");

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			f >> array[i * cols + j];	// equivalent to array[i][j]
		}
	}
	f.close();

	*rowp = rows;
	*columnp = cols;
	return array;
}

//* read the file int to one dimensional array
void *wbImport(const char *filename, int *array_sizep)
{
	int items;

	fstream f(filename);
	die_if(!f, "Error opening file: ", filename);

	f >> items;

	float *array = (float *) malloc(sizeof(float) * items);
	die_if(!array, "Error allocating memory for array.", "");

	for (int i = 0; i < items; i++) {
		f >> array[i];
	}
	f.close();

	*array_sizep = items;
	return array;
}
#define Generic 1
#define GPU 1
#define Compute 1
#define Copy 1
void wbTime_start(...) {
}
void wbTime_stop(...) {
}


#define ERROR 0
#define TRACE 1
#define DEBUG 2


class wbLogger {
	public:
		template<typename T>
			wbLogger &operator,(const T &t) { std::cout << t; return *this; }
};
#define wbLogN(LINE,type,args...) do { wbLogger wbLogger##LINE; wbLogger##LINE, ##args; } while(0)
#define wbLog(type,args...) wbLogN(__LINE__,type,##args,"\n")

typedef int cudaError_t;


#define cudaSuccess  0
cudaError_t cudaMalloc(void ** ptr, int size) {
	*ptr = malloc(size);
	return cudaSuccess;
}
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
cudaError_t cudaMemcpy(void * dest, void * src, int size, int type __attribute__((unused))){
	memcpy(dest,src,size);
	return cudaSuccess;
}
cudaError_t cudaFree(void * ptr) {
	free(ptr);
	return cudaSuccess;
}

void cudaThreadSynchronize() {
}

typedef struct _dim3 {
	int x_;
	int y_;
	int z_;
	_dim3(int  x, int y, int z) {
		x_ = x;
		y_= y;
		z_= z;
	}
} dim3;

float *  computeCorrectResults(wbArg_t args, int *correct_columnp, int *correct_rowsp) {
	float * hostA; // The A matrix
	float * hostB; // The B matrix
	float * hostC; // The output C matrix
	int numARows; // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows; // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows; // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)
	hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

	numCRows = numARows;
	numCColumns = numBColumns;

    	hostC =(float *)  malloc (numCRows * numCColumns * sizeof(float) );

	for (int i=0; i < numCRows; i++ ) {
		for (int j=0; j< numCColumns; j++ ) {
			int sum =0;
			for (int k=0; k<numAColumns; k++) {
				sum += hostA[ numAColumns* i + k]  * hostB[numCColumns*k +j];
			}
			hostC[ numCColumns * i + j ] = sum;
		}
	}


	free(hostA);
	free(hostB);
	(*correct_columnp) = numCColumns;
	(*correct_rowsp) = numCRows;
	return hostC;
}

// two dimensional array solution check
void wbSolution(wbArg_t args, float *hostC, int numCRows, int numCColumns) {
	int correct_column;
	int correct_row;
	float * correct_results;
	correct_results = computeCorrectResults(args, &correct_column, &correct_row);

	if (numCColumns!= correct_column) {
		printf("ERROR Wrong number of Columns, expect %d, actual %d\n", correct_column, numCColumns);
		goto end;
	}
	if (numCRows!= correct_row) {

		printf("ERROR Wrong number of Rows, expect %d, actual %d\n", correct_row, numCRows);
		goto end;
	}
	for (int i=0; i< correct_row; i++) {
		for (int j=0; j< correct_column; j++) {
			int index = correct_column * i +j ;
			if (correct_results[index] != hostC[index] ) {

				printf("ERROR wrong value at row %d column %d: expect %g, actual %g\n",i, j, correct_results[index],hostC[index] );
				goto end;
			}
		}
	}
	printf("GOOD, Solution appears to be correct\n");
end:
	free(correct_results);

}

float *  computeCorrectResults(wbArg_t args, int *correct_columnp) {
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);

    hostOutput = (float *) malloc(inputLength * sizeof(float));
    for (int i=0; i < inputLength; i++ ) {
	    hostOutput[i] = hostInput1[i] +  hostInput1[i];
    }
    free(hostInput1);
    free(hostInput2);
    (*correct_columnp) = inputLength;

    return hostOutput;
}

// one dimensional array solution check
void wbSolution(wbArg_t args, float *hostC, int length ) {

	float * correct_results;
	int correct_column;
	correct_results = computeCorrectResults(args, &correct_column);


	if (length != correct_column) {
		printf("ERROR Wrong number of Columns, expect %d, actual %d\n", correct_column, length);
		goto end;
	}
	for (int j=0; j< correct_column; j++) {
		int index = j ;
		if (correct_results[index] != hostC[index] ) {

			printf("ERROR wrong value at column %d: expect %g, actual %g\n", j, correct_results[index],hostC[index] );
			goto end;
		}
	}
	printf("GOOD, Solution appears to be correct\n");
end:
	free(correct_results);

}


void setLocalAndRun(CudaThreadLocal l_tls, boost::function <void ()> func) {
	tls.reset(&l_tls);
	func();
}


void setupCudaSim (dim3 blocks, dim3 blocksize, boost::function <void ()  > func) {
	int numThreads = blocksize.x_ * blocksize.y_;
	ThreadProcessor processor( numThreads *2, numThreads);
	g_num_threads = numThreads ;

	g_blockDim.x = blocksize.x_;
	g_blockDim.y = blocksize.y_;
	g_blockDim.z = blocksize.z_;
	for (int b_x=0; b_x< blocks.x_; b_x++) {

		for (int b_y=0; b_y< blocks.y_; b_y++) {

			BatchTracker currentJob(&processor);
			g_barrierp. reset ( new boost::barrier ( g_num_threads )) ;
			for (int t_x=0; t_x< blocksize.x_; t_x++) {
				for (int t_y=0; t_y< blocksize.y_; t_y++) {
					CudaThreadLocal tl;
					tl.block.x =b_x;
					tl.block.y =b_y;
					tl.thread.x =t_x;
					tl.thread.y =t_y;
					currentJob.post(boost::bind(setLocalAndRun,tl, func));
				}
			}
			currentJob.wait_until_done();
		}
	}

	return ;
}

#endif
