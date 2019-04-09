
#ifndef COMMON_HPP
#define COMMON_HPP

#define cudaErrorCheck(err) \
 if(err != cudaSuccess) {\
  printf("error in cuda call at line %d\n", __LINE__);\
 }

#endif
