// wb.h: Header file for Heterogeneous Parallel Programming course (Coursera)

#pragma once

////
// Headers
////

// C++
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <list>
#include <sstream>
#include <string>
#include <vector>

// CUDA
#ifndef CUDA_EMU 
#include <cuda.h>
#include <cuda_runtime.h>
#else 
#include "cuda_emu.hpp"

#endif

////
// Logging
////

enum wbLogLevel
{
    OFF,
    FATAL,
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
    wbLogLevelNum, // Keep this at the end
};

const char* const _wbLogLevelStr[] =
{
    "Off",
    "Fatal",
    "Error",
    "Warn",
    "Info",
    "Debug",
    "Trace",
    "***InvalidLogLevel***", // Keep this at the end
};

const char* _wbLogLevelToStr(wbLogLevel level)
{
    assert(level >= OFF && level <= TRACE);
    if (level >= OFF && level <= TRACE)
        return _wbLogLevelStr[level];
    return _wbLogLevelStr[wbLogLevelNum];
}

class wbLogger {
    public:
        template<typename T>
        wbLogger &operator,(const T &t) { std::cout << t; return *this; }
};

#define wbLog(level, ...)                                     \
    do                                                        \
    {                                                         \
        std::cout << _wbLogLevelToStr(level) << " ";          \
        std::cout << __FUNCTION__ << "::" << __LINE__ << " "; \
        wbLogger logger;                                      \
        logger, __VA_ARGS__;                                  \
        std::cout << std::endl;                               \
    } while (0)

////
// Input arguments
////

struct wbArg_t
{
    int    argc;
    char** argv;
};

wbArg_t wbArg_read(int argc, char** argv)
{
    wbArg_t argInfo = { argc, argv };
    return argInfo;
}

const char* wbArg_getInputFile(const wbArg_t &argInfo, int argNum)
{
    if (argNum >= 0 && argNum < (argInfo.argc - 1))
        return argInfo.argv[argNum + 1];
    return NULL;
}

// For assignment MP1
float* wbImport(const char* fname, int* itemNum)
{
    // Open file

    if (!fname)
    {
        std::cout << "No input file given\n";
        exit(1);
    }

    std::ifstream inFile(fname);

    if (!inFile)
    {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit(1);
    }

    // Read from file

    inFile >> *itemNum;

    float* fBuf = (float*) malloc( *itemNum * sizeof(float) );

    std::string sval;
    int idx = 0;

    while (inFile >> sval)
    {
        std::istringstream iss(sval);
        iss >> fBuf[ idx++ ];
    }

    return fBuf;
}

// For assignment MP2
float* wbImport(const char* fname, int* numRows, int* numCols)
{
    // Open file

    if (!fname)
    {
        std::cout << "No input file given\n";
        exit(1);
    }

    std::ifstream inFile(fname);

    if (!inFile)
    {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit(1);
    }

    // Read file to vector

    std::string sval;
    float fval;
    std::vector<float> fVec;
    int itemNum = 0;

    // Read in matrix dimensions
    inFile >> *numRows;
    inFile >> *numCols;

    while (inFile >> sval)
    {
        std::istringstream iss(sval);
        iss >> fval;
        fVec.push_back(fval );
    }

    // Vector to malloc memory

    if (fVec.size() != (*numRows * *numCols))
    {
        std::cout << "Error reading matrix content for a " << *numRows << " * " << *numCols << "matrix!\n";
        exit(1);
    }

    itemNum = *numRows * *numCols;

    float* fBuf = (float*) malloc(itemNum * sizeof(float));

    for (int i = 0; i < itemNum; ++i)
    {
        fBuf[i] = fVec[i];
    }

    return fBuf;
}

////
// Timer
////

// Namespace because windows.h causes errors
namespace CudaTimerNS
{
#if defined (_WIN32)
    #include <Windows.h>

    // CudaTimer class from: https://bitbucket.org/ashwin/cudatimer
    class CudaTimer
    {
    private:
        double        _freq;
        LARGE_INTEGER _time1;
        LARGE_INTEGER _time2;

    public:
        CudaTimer::CudaTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            _freq = 1.0 / freq.QuadPart;
            return;
        }

        void start()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&_time1);
            return;
        }

        void stop()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&_time2);
            return;
        }

        double value() const
        {
            return (_time2.QuadPart - _time1.QuadPart) * _freq * 1000;
        }
    };
#elif defined (__APPLE__)
    #include <mach/mach_time.h>

    class CudaTimer
    {
    private:
        uint64_t _start;
        uint64_t _end;

    public:
        void start()
        {
            cudaDeviceSynchronize();
            _start = mach_absolute_time();
        }

        void stop()
        {
            cudaDeviceSynchronize();
            _end = mach_absolute_time();
        }

        double value() const
        {
            static mach_timebase_info_data_t tb;

            if (0 == tb.denom)
                (void) mach_timebase_info(&tb); // Calculate ratio of mach_absolute_time ticks to nanoseconds

            return ((double) _end - (double) _start) * (tb.numer / tb.denom) / 1000000000ULL;
        }
    };
#else
    #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
        #include<time.h>
    #else
        #include<sys/time.h>
    #endif

    class CudaTimer
    {
    private:
        long long _start;
        long long _end;

        long long getTime() const
        {
            long long time = 0LL;
        #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0

            struct timespec ts;

            if ( 0 == clock_gettime(CLOCK_REALTIME, &ts) )
            {
                time  = 1000000000LL; // seconds->nanonseconds
                time *= ts.tv_sec;
                time += ts.tv_nsec;
            }
        #else
            struct timeval tv;

            if ( 0 == gettimeofday(&tv, NULL) )
            {
                time  = 1000000000LL; // seconds->nanonseconds
                time *= tv.tv_sec;
                time += tv.tv_usec * 1000; // ms->ns
            }
        #endif

            return time;
        }

    public:
        void start()
        {
            _start = getTime();
        }

        void stop()
        {
            _end = getTime();
        }

        double value() const
        {
            return ((double) _end - (double) _start) / 1000000000LL;
        }
    };
#endif
}

enum wbTimeType
{
    Generic,
    GPU,
    Compute,
    Copy,
    wbTimeTypeNum, // Keep this at the end
};

const char* const wbTimeTypeStr[] =
{
    "Generic",
    "GPU    ",
    "Compute",
    "Copy   ",
    "***InvalidTimeType***", // Keep this at the end
};

const char* wbTimeTypeToStr(wbTimeType t)
{
    assert(t >= Generic && t < wbTimeTypeNum);
    if (t >= Generic && t < wbTimeTypeNum)
        return wbTimeTypeStr[t];
    return wbTimeTypeStr[wbTimeTypeNum];
}

struct wbTimerInfo
{
    wbTimeType             type;
    std::string            name;
    CudaTimerNS::CudaTimer timer;

    wbTimerInfo(wbTimeType type, const std::string &name,
        const CudaTimerNS::CudaTimer &timer = CudaTimerNS::CudaTimer()):
        type(type), name(name), timer(timer)
    {
    }
    bool operator == (const wbTimerInfo& t2) const
    {
        return (type == t2.type && (0 == name.compare(t2.name)));
    }
};

typedef std::list< wbTimerInfo> wbTimerInfoList;
wbTimerInfoList gTimerInfoList;

void wbTime_start(wbTimeType timeType, const std::string &timeStar)
{
    CudaTimerNS::CudaTimer timer;
    timer.start();

    wbTimerInfo tInfo(timeType, timeStar, timer);

    gTimerInfoList.push_front(tInfo);

    return;
}

void wbTime_stop(wbTimeType timeType, const std::string &timeStar)
{
    // Find timer

    const wbTimerInfo searchInfo(timeType, timeStar);
    const wbTimerInfoList::iterator iter = std::find( gTimerInfoList.begin(), gTimerInfoList.end(), searchInfo );

    // Stop timer and print time

    wbTimerInfo& timerInfo = *iter;

    timerInfo.timer.stop();

    std::cout << "[" << wbTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << std::fixed << std::setprecision(10) << timerInfo.timer.value() << " ";
    std::cout << timerInfo.name << std::endl;

    // Delete timer from list
    gTimerInfoList.erase(iter);

    return;
}

////
// Solution
////

// For assignment MP1
template < typename T, typename S >
void wbSolution(wbArg_t args, const T& t, const S& s)
{
    int solnItems;
    float *soln = (float *) wbImport(wbArg_getInputFile(args, 2), &solnItems);

    if (solnItems != s)
    {
        std::cout << "Number of items in solution does not match. ";
        std::cout << "Expecting " << s << " but got " << solnItems << ".\n";
        free(soln);
        return;
    }
    
    // Check answer

    int item;
    int errCnt = 0;

    for (item = 0; item < solnItems; item++)
    {
        if (abs(soln[item] - t[item]) > .005f)
        {
            std::cout << "Solution does not match at item " << item << ". ";
            std::cout << "Expecting " << soln[item] << " but got " << t[item] << ".\n";
            errCnt++;
        }
    }

    free(soln);

    if (!errCnt)
        std::cout << "All tests passed!\n";
    else
        std::cout << errCnt << " tests failed.\n";
        
    return;
}

// For assignment MP2
template < typename T, typename S, typename U >
void wbSolution(wbArg_t args, const T& t, const S& s, const U& u)
{
    int solnRows, solnColumns;
    float *soln = (float *) wbImport(wbArg_getInputFile(args, 2), &solnRows, &solnColumns);

    if (solnRows != s || solnColumns != u)
    {
        std::cout << "Size of solution does not match. ";
        std::cout << "Expecting " << solnRows << " x " << solnColumns << " but got " << s << " x " << u << ".\n";
        free(soln);
        return;
    }
    
    // Check solution

    int errCnt = 0;
    int row, col;

    for (row = 0; row < solnRows; row++)
    {
        for (col = 0; col < solnColumns; col++)
        {
            float expected = *(soln + row * solnColumns + col);
            float got = *(t + row * solnColumns + col);

            if (abs(expected - got) > 0.005f)
            {
                std::cout << "Solution does not match at (" << row << ", " << col << "). ";
                std::cout << "Expecting " << expected << " but got " << got << ".\n";
                errCnt++;
            }
        }
    }

    free(soln);

    if (!errCnt)
        std::cout << "All tests passed!\n";
    else
        std::cout << errCnt << " tests failed.\n";
        
    return;
}
