#ifndef _LR_UTILS_ 
#define _LR_UTILS_

/* #include "rmvnorm.h" */

__device__ float cut_LR(float x) {
  if(x>1.0)
    x = 1.0 ;
  if(x<0.0)
    x = 0 ;  
  return x ; 
}

__host__ void create_LR_con_dir() {
  
  ksi_path += "../../cpp/model/connectivity/" + to_string(n_pop) +"pop" ;
  ksi_path += "/NE_" + to_string(n_per_pop[0]/1000) +  "_NI_" + to_string(n_per_pop[1]/1000) ; 
  ksi_path += "/low_rank/rank_" + to_string(RANK) ; 
  
  if(FIX_KSI_SEED)
    ksi_path += "/seed_ksi_" + to_string(SEED_KSI) ;
  
  make_dir(ksi_path) ; 
} 

__host__ void gen_ksi() {

  /* cout << "Generate ksi " << endl ;  */
  
  /* float *array ;  */
  /* random_normal_multivariate(COV_MAT, array, 4, n_per_pop[0]) ;  */
  
  /* for(i=0; i<10; i++) {  */
  /*   for(int j=0; j<4; j++) */
  /*     printf("%f ", array[j+i*4] ) ;  */
  /*   printf("\n") ;  */
  /* } */
  
  /* for(i=0; i<n_per_pop[0]; i++) {  */
  /*   sample_A[i] = array[0+i*4] ;  */
  /*   sample_B[i] = array[1+i*4] ;  */
  /*   ksi[i] = array[2+i*4] ; */
  /*   if(RANK==2)  */
  /*     ksi_1[i] = array[3+i*4] ;  */
  /* } */
  
  /* cout << "sample A: " ; */
  /* cout << "mean " << mean_array(sample_A, n_per_pop[0]) << " var " << var_array(sample_A, n_per_pop[0]) << endl ;  */
  
  /* cout << "sample B: " ; */
  /* cout << "mean " << mean_array(sample_B, n_per_pop[0]) << " var " << var_array(sample_B, n_per_pop[0]) << endl ;  */

  /* cout << "covar sample A/B " << covar_array(sample_A, sample_B, n_per_pop[0]) << endl ;  */
  
  /* cout << "ksi: " ; */
  /* cout << "mean " << mean_array(ksi, n_per_pop[0]) << " var " << var_array(ksi, n_per_pop[0]) << endl ;  */
  /* cout << "covar ksi/samples " << covar_array(ksi, sample_A, n_per_pop[0]) << " covar dist " << covar_array(ksi, sample_B, n_per_pop[0]) ;  */

  /* if(RANK==2) { */
  /*   cout << "ksi_1: " ; */
  /*   cout << "mean " << mean_array(ksi_1, n_per_pop[0]) << " var " << var_array(ksi_1, n_per_pop[0]) << endl ; */
  /*   cout << "covar ksi/samples " << covar_array(ksi_1, sample_A, n_per_pop[0]) << " covar dist " << covar_array(ksi_1, sample_B, n_per_pop[0]) ;  */
  /*   cout << "covar ksi/ksi_1 " << covar_array(ksi, ksi_1, n_per_pop[0]) << endl ;  */
  /* } */
  
  /* cout << "###############################################" << endl ; */
  
  /* write_to_file(ksi_path, "ksi", ksi , n_per_pop[0]) ; */
  
  /* if(RANK==2) */
  /*   write_to_file(ksi_path, "ksi_1", ksi_1 , n_per_pop[0]) ;  */
  
  /* write_to_file(ksi_path, "sample_A", sample_A , n_per_pop[0]) ;  */
  /* write_to_file(ksi_path, "sample_B", sample_B , n_per_pop[0]) ;      */
  
}


__host__ void get_ksi(){
  
  read_from_file(ksi_path, "ksi", ksi, n_per_pop[0]) ; 
  
  cout << "ksi "<< n_per_pop[0] << " " << endl ; 
  for(i=0;i<10;i++)
    cout << ksi[i] << " " ;
  cout << endl ;
  
  if(RANK==2) {
    read_from_file(ksi_path, "ksi_1", ksi_1, n_per_pop[0]) ;
    
    cout << "ksi_1 " << endl ;
    for(i=0;i<10;i++)
      cout << ksi_1[i] << " " ;
    cout << endl ;
  }  
    
}


__host__ void copy_ksi_to_dev() {

  create_LR_con_dir() ;
  
  if(IF_GEN_KSI)
    gen_ksi() ;
  else
    get_ksi() ;

  cout << "copy ksi to dev" << endl ;
  
  cudaCheck(cudaMemcpyToSymbol( dev_ksi, &ksi, n_per_pop[0] * sizeof(float) ) ) ; 
  if(RANK==2) 
    cudaCheck(cudaMemcpyToSymbol( dev_ksi_1, &ksi_1, n_per_pop[0] * sizeof(float) ) ) ;   
}

#endif
