#ifndef CUDA_EMU_HPP
#define CUDA_EMU_HPP

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/thread/tss.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "thread_processor.hpp"

#define UNUSED __attribute__((unused))

typedef int cudaError_t;


#define cudaSuccess  0
cudaError_t cudaMalloc(void ** ptr, int size) {
	*ptr = malloc(size);
	return cudaSuccess;
}
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
cudaError_t cudaMemcpy(void * dest, void * src, int size, int type UNUSED){
	memcpy(dest,src,size);
	return cudaSuccess;
}
cudaError_t cudaMemcpy2D(void * dest, int dpitch, void * src, int spitch, int size, int rows, int type __attribute__((unused))){
       for (int n=0;n<rows;++n) {
               memcpy(dest,src,size);
               dest=(char*)dest+dpitch;
               src=(char*)src+spitch;
       }
       return cudaSuccess;
}
cudaError_t cudaFree(void * ptr) {
	free(ptr);
	return cudaSuccess;
}

void cudaThreadSynchronize() {
}


#define __global__
#define __shared__ volatile static


#define blockIdx getBlockIdx()
#define threadIdx getThreadIdx()
#define blockDim getBlockDim()

typedef struct _Block_t  {
	int x;
	int y;
	int z;
	_Block_t () : x(0), y(0), z(0) {
	}
	_Block_t (int  lx, int ly, int lz) :  x(lx),y(ly),z(lz) {
	}

} Block_t;

typedef Block_t dim3;

struct CudaThreadLocal {
	Block_t block;
	Block_t thread;
	int phase1;
	int phase2;
	CudaThreadLocal() : block(), thread(), phase1(0),phase2(0)  {}
};

static void doNothing(CudaThreadLocal * ptr UNUSED) {
}

// we don't want the thread local to call delete on the object stored in this because it is allocated off of the stack
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

static boost::shared_ptr<boost::barrier> g_barrierp;
static boost::shared_ptr<boost::barrier> g_barrier_2p; 
static boost::mutex g_b_mutex1;
static boost::mutex g_b_mutex2;
static volatile int b_phase1=0;
static volatile int b_phase2=0;


/**
 * I'm sure there are better ways to do this. 
 * the problem here is that the boost::barrier can not be destroyed while there are there are threads that have not yet exited out of the boost:barrier wait function (it core dumps) . 
 * It would be great if the barrier object had a reset function that when called would reset the wait barrier wait count, all current threads waiting on it would be allowed to exit the function safely, and new calls to wait would decrement from wait on the new counter.
 *
 *
 * anyway, this seems to work 
 */
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


void setLocalAndRun(CudaThreadLocal l_tls, boost::function <void ()> func) {
	tls.reset(&l_tls);
	func();	
}


void setupCudaSim (dim3 blocks, dim3 blocksize, boost::function <void ()  > func) {
	int numThreads = blocksize.x * blocksize.y;
	ThreadProcessor processor( numThreads *2, numThreads);
	g_num_threads = numThreads ;

	g_blockDim.x = blocksize.x;
	g_blockDim.y = blocksize.y;
	g_blockDim.z = blocksize.z;
	for (int b_x=0; b_x< blocks.x; b_x++) {

		for (int b_y=0; b_y< blocks.y; b_y++) {

			BatchTracker currentJob(&processor);
			g_barrierp. reset ( new boost::barrier ( g_num_threads )) ; 
			for (int t_x=0; t_x< blocksize.x; t_x++) {
				for (int t_y=0; t_y< blocksize.y; t_y++) {
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
