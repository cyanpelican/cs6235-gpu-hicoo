
#ifndef COMMON_HPP
#define COMMON_HPP

#define cudaErrorCheck(err) \
 if(err != cudaSuccess) {\
  printf("error in cuda call at line %d, file %s\n", __LINE__, __FILE__);\
 }

#endif
