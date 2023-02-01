double dum_array[N_NEURONS] ; 
__device__ double d_dum_array[N_NEURONS] ; 

__host__ void init_dum() { 
  for(int i=0; i<N_NEURONS; i++) 
    dum_array[i] = (double) i ; 
  
}

__global__ void d_dum_func(curandState *state) { 

  unsigned long id = threadIdx.x + blockIdx.x * blockDim.x ;
  
  if(id<N_NEURONS) 
    for(int i=0; i<N_NEURONS; i++) 
      d_dum_array[i] = unif_dist(state, id) ; 
  
} 

__global__ void print_d_dum() {
 
  unsigned long id = threadIdx.x + blockIdx.x * blockDim.x ; 
  
  if(id<2) { 
    cuPrintf("d_dum_array: ") ; 
    for(int i=0; i<3;i++) 
      cuPrintf("%.2f ", d_dum_array[i] ) ; 
    cuPrintf("\n") ; 
  }

}
