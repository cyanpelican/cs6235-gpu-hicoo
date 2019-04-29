
#ifndef COMMON_HPP
#define COMMON_HPP
#include <chrono>
#include <ctime>
#include <iostream>

// This is just to get CLion happy
// I'll try not to stage it, so if you see this, feel free to throw all the
// blame at me... -GW
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#endif // __JETBRAINS_IDE__


#define cudaErrorCheck(err) \
 { auto __err = err;\
  if(__err != cudaSuccess) {\
   printf("error in cuda call at line %d, file %s. Error name = %s\n", __LINE__, __FILE__, cudaGetErrorName(__err));\
  }\
 }


// set to true to slow down the entire world and make sure accesses are good
#define ENABLE_ACCESS_ASSERTS false

// set to true to enable debug prints
#define DEBUG_PRINT_ENABLE false

#if DEBUG_PRINT_ENABLE && !defined NDEBUG
  inline void __print_debug_header(std::string file, unsigned int line) {
    // time-as-string from https://stackoverflow.com/questions/16357999/current-date-and-time-as-string
    //  and https://stackoverflow.com/questions/14370279/prevent-endline-after-printing-system-time
    auto t = std::time(nullptr);
    auto tm = std::localtime(&t);
    char* timeStr = asctime(tm);

    timeStr[strlen(timeStr)-1] = 0;

    std::cout << "\tDEBUG " << timeStr <<
        " : \t" << file << " : " << line << ":::  \t ";
  }
  #define DEBUG_PRINT(...) \
    __print_debug_header(__FILE__, __LINE__);\
    printf( __VA_ARGS__ );\
    fflush(stdout)
#else
  #define DEBUG_PRINT(...) /* */
#endif

#endif //COMMON_HPP
