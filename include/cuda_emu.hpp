#ifndef CUDA_EMU_HPP
#define CUDA_EMU_HPP

#include <malloc.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/thread/tss.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "thread_processor.hpp"

#define UNUSED __attribute__((unused))

typedef int cudaError_t;


const static int cudaSuccess = 0;

cudaError_t cudaMalloc(void **ptr, int size)
{
	*ptr = memalign(256, size);
	return cudaSuccess;
}

cudaError_t cudaMallocPitch(void **ptr, size_t *pitch, size_t width, size_t height)
{
	*pitch = (width + 255) & ~255;
	*ptr = memalign(256, (*pitch) * height);
	return cudaSuccess;
}

enum direction_t {
	cudaMemcpyHostToDevice,
	cudaMemcpyDeviceToHost
};

cudaError_t cudaMemcpy(void *dest, void *src, int size, direction_t type UNUSED)
{
	memcpy(dest, src, size);
	return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t counter)
{
	memset(devPtr, value, counter);
	return cudaSuccess;

}

cudaError_t cudaMemcpy2D(void *dest, int dpitch, void *src, int spitch, int size, int rows, direction_t type UNUSED)
{
	for (int n = 0; n < rows; ++n) {
		memcpy(dest, src, size);
		dest = (char *) dest + dpitch;
		src = (char *) src + spitch;
	}
	return cudaSuccess;
}

cudaError_t cudaFree(void *ptr)
{
	free(ptr);
	return cudaSuccess;
}

cudaError_t cudaThreadSynchronize()
{
	return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize()
{
	return cudaSuccess;
}


#define __global__

#define __shared__ volatile static


#define blockIdx getBlockIdx()
#define threadIdx getThreadIdx()
#define blockDim getBlockDim()

struct Block_t {
	int x;
	int y;
	int z;
	Block_t () : x(0), y(0), z(0) {
	}
	Block_t (int  lx, int ly, int lz) :  x(lx), y(ly), z(lz) {
	}
};

typedef Block_t dim3;

struct CudaThreadLocal {
	Block_t block;
	Block_t thread;
	int phase1;
	int phase2;
	CudaThreadLocal() : block(), thread(), phase1(0), phase2(0)  {}
	CudaThreadLocal(Block_t  lblock, Block_t lthread) : block(lblock), thread(lthread), phase1(0), phase2(0)  {}

};

static void doNothing(CudaThreadLocal *ptr UNUSED)
{
}

// we don't want the thread local to call delete on the object stored in this because it is allocated off of the stack
boost::thread_specific_ptr<CudaThreadLocal> tls(doNothing);

const Block_t getBlockIdx()
{
	CudaThreadLocal *p = tls.get();
	return p->block;
}

const Block_t getThreadIdx()
{
	CudaThreadLocal *p = tls.get();
	return p->thread;
}

static Block_t g_blockDim;

Block_t getBlockDim()
{
	return g_blockDim;
}

static int g_num_threads;

static boost::shared_ptr<boost::barrier> g_barrierp;
static boost::shared_ptr<boost::barrier> g_barrier_2p;
static boost::mutex g_b_mutex1;
static boost::mutex g_b_mutex2;
static volatile int b_phase1 = 0;
static volatile int b_phase2 = 0;


/**
 * I'm sure there are better ways to do this.
 * the problem here is that the boost::barrier can not be destroyed while there are there are threads that have not yet exited out of the boost:barrier wait function (it core dumps) .
 * It would be great if the barrier object had a reset function that when called would reset the wait barrier wait count, all current threads waiting on it would be allowed to exit the function safely, and new calls to wait would decrement from wait on the new counter.
 *
 *
 * anyway, this seems to work
 */
void __syncthreads()
{
//	Block_t bl = getThreadIdx() ;
	CudaThreadLocal *p = tls.get();

//	printf("thread %d %d is waiting on barrier 1\n", bl.x, bl.y);
	g_barrierp->wait();
	{
		boost::mutex::scoped_lock lock(g_b_mutex1);
		if (b_phase1 == p->phase1) {
			g_barrier_2p.reset(new boost::barrier(g_num_threads));
//			printf("thread %d %d reset barrier 1\n", bl.x, bl.y);
			b_phase1++;
		}
		p->phase1 = b_phase1;

	}
//	printf("thread %d %d is done waiting on barrier 1\n", bl.x, bl.y);
	g_barrier_2p->wait();
	{
		boost::mutex::scoped_lock lock(g_b_mutex2);
		if (b_phase2 == p->phase2) {
			g_barrierp.reset(new boost::barrier(g_num_threads));
//			printf("thread %d %d reset barrier 2\n", bl.x, bl.y);
			b_phase2++;
		}
		p->phase2 = b_phase2;

	}
}


void setLocalAndRun(CudaThreadLocal l_tls, boost::function <void ()> func)
{
	tls.reset(&l_tls);
	func();
}


void setupCudaSim(dim3 blocks, dim3 blocksize, boost::function <void ()  > func)
{
	int numThreads = blocksize.x * blocksize.y;
	ThreadProcessor processor(numThreads * 2, numThreads);
	g_num_threads = numThreads;

	g_blockDim.x = blocksize.x;
	g_blockDim.y = blocksize.y;
	g_blockDim.z = blocksize.z;
	for (int b_x = 0; b_x < blocks.x; b_x++) {

		for (int b_y = 0; b_y < blocks.y; b_y++) {

			for (int b_z = 0; b_z < blocks.z; b_z++) {
				BatchTracker currentJob(&processor);
				g_barrierp.reset(new boost::barrier(g_num_threads));
				for (int t_x = 0; t_x < blocksize.x; t_x++) {
					for (int t_y = 0; t_y < blocksize.y; t_y++) {
						for (int t_z = 0; t_z < blocksize.z; t_z++) {
							CudaThreadLocal tl(Block_t(b_x, b_y, b_z), Block_t(t_x, t_y, t_z));
							currentJob.post(boost::bind(setLocalAndRun, tl, func));
						}
					}
				}
				currentJob.wait_until_done();
			}
		}
	}

	return;
}

void setupCudaSim(const unsigned blocks_x, const unsigned int blocksize_x, boost::function <void ()  > func)
{
	dim3 dimGrid(blocks_x, 1, 1);
	dim3 dimBlock(blocksize_x, 1, 1);
	setupCudaSim(dimGrid, dimBlock, func);
}

#endif
