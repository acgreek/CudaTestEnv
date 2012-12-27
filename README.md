CudaTestEnv
===========

Basic, incomplete, host-based CUDA API implementation use for testing my
programming assignments for the Coursera "Heterogeneous Parallel
Programming" class. It does not use an nVidia GPU but instead simulates the
threading and thread local variables, shared memory and barriers.  It only
supports 1 and 2 dimensional array because that is all that is needed so far
to complete my assignments.

Note, since this is using only using the standard `g++` compiler, there is
no error checking for using non-CUDA functions and constructs with in the
device functions. In fact, the `__global__` is just a no-op.

You will also need to call the device function using standard C-compliant
syntax and also pass the function to `setupCudaSim`. Here is an example:

    #ifndef CUDA_EMU
        vecAdd<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    #else
        setupCudaSim (dimGrid, dimBlock, boost::bind(vecAdd, deviceInput1,
                      deviceInput2, deviceOutput, inputLength));
    #endif

I developed this on Cygwin in Windows 7. You will need `g++` (I used version
4.5) and Boost's `thread` and `system`.

To build and run my example, simply execute:

    make run

then the `mp1` binary is created and use the file `foo` as the first and
second argument:

    ./mp1 foo foo


To use your own input, create text files for input1 and input2. Each integer
must be space separated and each row is terminated by a newline.
