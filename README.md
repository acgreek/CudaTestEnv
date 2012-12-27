CudaTestEnv
===========

Basic incomplete host based cuda api implementation use for testing my programming assignments for the Coursera "Heterogeneous Parallel Programming" class

test harness for developing solutions for the coursera 'Heterogeneous Parallel Programming'. It does not use a nvidia Gpu but instead simulates the threading and thread local variables, shared memory and barriers. It only support 1 and 2 dimensional array because that is all needed so far to complete my assignements.

Note, since this is using only using the standard g++ compiler, there is no error checking for using non cuda functions and contrustructs with in the device functions. In fact, the __global__ is just a noop.

you will also need to call the device function using standard c compliant syntax and also pass the function to setupCudaSim. Here is an example 

<pre><code>
  #ifndef CUDA_EMU
    vecAdd<<< dimGrid, dimBlock >>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  #else 
    setupCudaSim (dimGrid , dimBlock , boost::bind(vecAdd,deviceInput1,deviceInput2,deviceOutput, inputLength));
  #endif
 </code></pre>

I developed this on cygwin in Windows 7. You will need g++  (I used version 4.5) and boost thread and system.


to build and run my example, simply execute 

make run1

the is create the mp1 GenDataMP1 binary, generates a vector test set of 90 and then run mp1 binary with the generated data



to use your own input create text files for input1 and input2. Each integer must be space separated and each row is terminated by a newline

Contributors 
============

Myself, as well. I also used some code from the coursera git repo, and their contributors can be found at the following: 
[View list of contributors](https://github.com/ashwin/coursera-heterogeneous/contributors)
We welcome improvements to this code. Fork it, make your change and give me a pull request. Please follow the coding conventions already in use in the source files.


License
=======

All the files in this project are shared under the [MIT License](http://opensource.org/licenses/mit-license.php).

Copyright (C) 2012 [Contributors](https://github.com/ashwin/coursera-heterogeneous/contributors)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
