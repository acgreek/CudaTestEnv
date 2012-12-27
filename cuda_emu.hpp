#ifndef CUDA_EMU_HPP
#define CUDA_EMU_HPP

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/thread/tss.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "thread_processor.hpp"

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
