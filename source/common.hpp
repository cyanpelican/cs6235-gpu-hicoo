#define cudaErrorCheck(err) \
 if(err != cudaSuccess) {\
  printf("error in cuda call at line %d\n", __LINE__);\
 }

