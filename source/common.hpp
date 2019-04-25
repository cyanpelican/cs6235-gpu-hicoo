
#ifndef COMMON_HPP
#define COMMON_HPP

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
 if(err != cudaSuccess) {\
  printf("error in cuda call at line %d, file %s\n", __LINE__, __FILE__);\
 }

// set to true to enable debug prints
#define DEBUG_PRINT_ENABLE false

#if DEBUG_PRINT_ENABLE
 #define DEBUG_PRINT(...) printf("\tDEBUG %s : %d:::   ", __FILE__, __LINE__);\
  printf( __VA_ARGS__ );\
  fflush(stdout)
#else
#define DEBUG_PRINT(...) /* */
#endif

#endif
