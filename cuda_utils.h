#ifndef __CUDA_UTILS__
#define __CUDA_UTILS__

__device__ uint wang_hash(uint seed) {
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}

__global__ void setup_kernel() { 
  unsigned long id = threadIdx.x + blockIdx.x * blockDim.x ; 
  /* unsigned long i_neuron = (unsigned long) ( id + dev_i_chunck * N_NEURONS_PER_CHUNCK ) ; */
  /* Each thread gets different seed, a different sequence number, no offset */
  
  if(id < N_NEURONS_PER_CHUNCK)
    if(IF_CON_DIR)
      curand_init( wang_hash( (unsigned int) SEED_CON), id, 0, &dev_states[id]) ; 
    else 
      curand_init( wang_hash( (unsigned int) clock64()), id, 0, &dev_states[id]) ; 

  curand_init( wang_hash( (unsigned int) clock64()), id, 0, &dev_states_randi[id]) ;
  
  /* if(id<2) { */
  /*   cuPrintf("currand init: ") ;  */
  /*   cuPrintf("%d", dev_states[id]) ;  */
  /* } */
}

__device__ float unif_dist(unsigned long id) { 
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/ 
  float randNumber= 0.0 ; 
  
  if(id < N_NEURONS_PER_CHUNCK) { 
    /* save state in local memory for efficiency */ 
    curandState localState = dev_states[id] ; 
    randNumber = curand_uniform(&localState) ; 
    dev_states[id] = localState ; 
  } 
  
  return randNumber ; 
}


__device__ float unif_dist_randi(unsigned long id) {
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/
  float randNumber= 0.0 ;
  
  if(id < N_NEURONS_PER_CHUNCK) {
    /* save state in local memory for efficiency */
    curandState localState = dev_states_randi[id] ;
    randNumber = curand_uniform(&localState) ;
    dev_states_randi[id] = localState ;
  }
  
  return randNumber ;
}

__device__ unsigned long rand_int(unsigned long id, unsigned long MIN, unsigned long MAX) {
  float rand ;
  unsigned long randi ; 
  
  rand = unif_dist_randi(id) * (float) (MAX - MIN) + (float) MIN ; 
  randi = (unsigned long) truncf(rand) ; 
  
  return randi ;
}

__device__ void init_dev_theta() { 
  
  for(int i=0; i<n_pop; i++) 
    for(unsigned long j = dev_cum_n_per_pop[i]; j<dev_cum_n_per_pop[i+1]; j++) 
      dev_theta[j] = 2.0 * (float) M_PI * (float) (j-dev_cum_n_per_pop[i]) / (float) dev_n_per_pop[i] ; 
  
  /* unsigned long id = (unsigned long) threadIdx.x + blockIdx.x * blockDim.x ;  */
  
  /* if(id<1) {  */
  /*   cuPrintf("theta:") ; */
  /*   for(int i=0; i<5; i++)  */
  /*     cuPrintf("%.2f", dev_theta[i]) ;  */
  /*   cuPrintf("\n") ;  */
  /* } */
  
} 

__global__ void kernel_gen_con_prob() { 
  
  unsigned long id = (unsigned long) threadIdx.x + blockIdx.x * blockDim.x ; 
  unsigned long i_neuron = (unsigned long) ( id + dev_i_chunck * N_NEURONS_PER_CHUNCK ) ; 

  float kappa = 2.0 * KAPPA ; 
  float kappa_K_N = KAPPA * dev_K_over_Na[0] / sqrt(E_frac*K) ; 
  
  DEV_IF_STRUCTURE = IF_SPEC || IF_RING ; 
  
  if(IF_RING || IF_SPEC)
    init_dev_theta() ; 
  
  if(IF_SPEC) 
    kappa /= sqrt_K ; 
  
  if(id < N_NEURONS_PER_CHUNCK & i_neuron < N_NEURONS) {     
    dev_pre_pop = dev_which_pop[i_neuron] ; 
    
    /* if(dev_pre_pop==1) */
    /*   kappa = 2.0 * KAPPA_I ;  */
    
    for(unsigned long i=0; i<N_NEURONS; i++) { // id (pre) -> i (post)
      
      dev_post_pop = dev_which_pop[i] ; 
      /* dev_con_prob_chunck[id + i * N_NEURONS_PER_CHUNCK ] = dev_K_over_Na[dev_pre_pop] ; */ 
      dev_con_prob_chunck[i + id * N_NEURONS] = dev_K_over_Na[dev_pre_pop] ; 
      
      if(IF_RING || IF_SPEC) 
		if(DEV_IS_STRUCT_SYN[dev_pre_pop + dev_post_pop * n_pop]!=0)
		  dev_con_prob_chunck[i + id * N_NEURONS] *= ( 1.0 + kappa
													   * DEV_IS_STRUCT_SYN[dev_pre_pop + dev_post_pop * n_pop]
													   * cos( dev_theta[i_neuron] - dev_theta[i] )
													   ) ;
      
      if(IF_LOW_RANK)
		if(dev_pre_pop==0 && dev_post_pop==0) {
	  
		  dev_con_prob_chunck[i + id * N_NEURONS] += kappa_K_N * dev_ksi[i] * dev_ksi[i_neuron] ;
		  if(RANK==2)
			dev_con_prob_chunck[i + id * N_NEURONS] += kappa_K_N * KAPPA_FRAC * dev_ksi_1[i] * dev_ksi_1[i_neuron] ;
		  // KAPPA_1 is KAPPA_FRAC * KAPPA
		  dev_con_prob_chunck[i + id * N_NEURONS] = cut_LR(dev_con_prob_chunck[i + id * N_NEURONS]) ;
		}
      
    } 
  } 
} 


__global__ void kernel_gen_con_vec() { 

  unsigned long id = (unsigned long) threadIdx.x + blockIdx.x * blockDim.x ; 
  unsigned long i_neuron = (unsigned long) ( id + dev_i_chunck * N_NEURONS_PER_CHUNCK ) ; 

  int count=0;
  unsigned long randi ;
  
  /* if(id<1) {  */
  /*   cuPrintf("DEV_IF_STRUCTURE %d \n", DEV_IF_STRUCTURE) ;     */ 
  /*   cuPrintf("dev_K_over_Na:") ;  */
  /*   for(int i=0; i<n_pop; i++)  */
  /*     cuPrintf("%.2f", dev_K_over_Na[i]) ; */
  /*   cuPrintf("\n") ; */
  /* } */

  /* if(id<1) { */
  /*   cuPrintf("check random seed: ") ; */
  /* /\*   for(int i=0; i<5; i++) *\/ */
  /* /\*     cuPrintf("%.2f", unif_dist(i) ) ;  *\/ */
  /*   cuPrintf("1 %.2f", unif_dist(id) ) ; */
  /*   cuPrintf("2 %.2f", unif_dist(id) ) ; */
  /*   cuPrintf("\n") ; */
  /* } */
  
  if(id < N_NEURONS_PER_CHUNCK && i_neuron < N_NEURONS) { // presynaptic neuron (columns) 
    for(unsigned long i=0; i<N_NEURONS; i++) { // post
      /* if( dev_con_prob_chunck[id + i * N_NEURONS_PER_CHUNCK] >= unif_dist(id) ) { // WARNING must be id inside unif otherwise problems */
      /* 	dev_con_vec_chunck[id + i * N_NEURONS_PER_CHUNCK] = 1 ;  */
      /* 	dev_total_n_post++ ;  */
      /* 	dev_n_post[i_neuron]++ ;  */
      /* } */

      if( dev_con_prob_chunck[i + id * N_NEURONS] >= unif_dist(id) ) { 
		dev_con_vec_chunck[i + id * N_NEURONS] = 1 ;
		dev_total_n_post++ ;
		dev_n_post[i_neuron]++ ;
      }             
      /* else  */ 
      /* 	dev_con_vec_chunck[id + i * N_NEURONS_PER_CHUNCK] = 0 ;  */
    }  
    
    if(IF_REMAP_SQRT_K) {
      
      while (count < (int) K) {
	
    	randi = rand_int(id, 0, N_NEURONS) ;
	
    	if(id<1 && count==0) {
    	  cuPrintf("rand int: ") ;
    	  cuPrintf("%d", randi ) ;
    	  cuPrintf("\n") ;
    	}
	
    	if( dev_con_prob_chunck[randi + id * N_NEURONS] >= unif_dist_randi(id) )
    	  dev_con_vec_chunck[randi + id * N_NEURONS] = 1 ;
    	else
    	  dev_con_vec_chunck[randi + id * N_NEURONS] = 0 ;
    	count++ ;
      } 
    } 
  } 
  
  /* if(id<1) {  */
  /*   cuPrintf("dev_n_post: ") ; */
  /*   for(int i=0; i<5; i++)  */
  /*     cuPrintf("%d ", dev_n_post[i]) ; */
  /*   cuPrintf("\n") ;  */
  /* }  */
  
} 

#endif 
