#ifndef __UTILS__ 
#define __UTILS__ 

int n_post[N_NEURONS], avg_n_post[2*n_pop] ; 
int n_pres[N_NEURONS], avg_n_pres[2*n_pop] ; 
unsigned long *id_post = NULL, *idx_post = NULL ; 
unsigned long long total_n_post=0 ; 

__host__ void init_globals() { 
  
  /* n_per_pop = (unsigned long *) malloc( (unsigned long) n_pop * sizeof(unsigned long) ) ; */
  
  printf("n_per_pop: ") ; 
  for(i=0;i<n_pop;i++) { 
    n_per_pop[i] = (unsigned long) ( n_frac[i] * N_NEURONS ) ; 
    printf("%lu ", n_per_pop[i] ) ;
  }
  printf("\n") ;
  
  /* cum_n_per_pop = (unsigned long *) malloc( (unsigned long) (n_pop+1) * sizeof(unsigned long) ) ;  */
  
  printf("cum_n_per_pop: ") ; 
  for(i=0;i<n_pop+1;i++) {
    cum_n_per_pop[i] = 0 ;
    
    for(j=0;j<i;j++) 
      cum_n_per_pop[i] += n_per_pop[j] ; 
    
    printf("%lu ", cum_n_per_pop[i] ) ; 
  } 
  printf("\n") ; 
  
  /* which_pop = (int *) malloc( (unsigned long) N_NEURONS * sizeof(int) ) ; */
  
  for(i=0;i<n_pop;i++) 
    for(j=0; j<N_NEURONS; j++) 
      if(j>=cum_n_per_pop[i] && j<cum_n_per_pop[i+1]) 
	which_pop[j] = i ; 
  
  /* K_over_Na = (float *) malloc( (unsigned long) n_pop * sizeof(float) ) ;  */
  
  printf("K_over_Na: ") ; 
  for(i=0;i<n_pop;i++) {
    K_over_Na[i] = (float) K  * n_frac[i] / (float) n_per_pop[i] ; 
    printf("%.2f ", K_over_Na[i] ) ;     
  }
  printf("\n") ; 
  
}

__host__ void gen_con_sparse_rep() { 

  unsigned long counter = 0 ; 
  
  printf("generate sparse representation: \n") ; 
  
  /* n_post = (int *) malloc( (unsigned long) N_NEURONS * sizeof(int) ) ;  */ 
  idx_post = (unsigned long *) malloc( N_NEURONS * sizeof(unsigned long) ) ; 
  id_post = (unsigned long *) malloc( (unsigned long long) N_NEURONS * ( 2ULL + (unsigned long long) K + N_NEURONS ) * sizeof( unsigned long ) ) ; 
  /* id_post = (unsigned long *) malloc( (unsigned long long) ((unsigned long long) N_NEURONS * (unsigned long long) N_NEURONS * (unsigned long long) ( K + 2.0 * sqrt_K ) ) * sizeof(unsigned long) ) ;  */ 
  /* id_post = (unsigned long *) malloc( total_n_post * sizeof(unsigned long) ) ;  */ 
  
  /* avg_n_post = (int *) malloc( n_pop * n_pop * sizeof(int) ) ; */ 
        
  for(j=0; j<N_NEURONS; j++) {
    pre_pop = which_pop[j] ; 
    for(i=0; i<N_NEURONS; i++) { 
      post_pop = which_pop[i] ; 
      if(con_vec[i + j * N_NEURONS]) { // Jij j (pre) to i (post) 
	id_post[counter] = i ; 
	
	n_post[j]++ ; 
      	avg_n_post[pre_pop + post_pop*n_pop]++ ; 
	
	n_pres[i]++ ; 
      	avg_n_pres[pre_pop + post_pop*n_pop]++ ; 
	
      	total_n_post++ ; 
	counter++ ; 
      }
    }
  }
  
  free(con_vec) ; 
  
  printf(" total_n_con: %llu \n", total_n_post ) ; 

  printf(" average number of presynaptic connections per pop: ") ; 
  for(i=0; i<2*n_pop; i++) 
    if(i==0 || i==2) 
      printf("%lu ", avg_n_pres[i] / n_per_pop[0]) ; 
    else 
      printf("%lu ", avg_n_pres[i] / n_per_pop[1]) ; 
  printf("\n") ; 

  for(i=0; i<n_pop; i++)
    for(j=0; j<n_pop; j++)
      printf("%lu ", avg_n_pres[j+i*n_pop] / n_per_pop[i]); 
  printf("\n") ; 
  
  printf(" average number of postsynaptic connections per pop: ") ; 
  for(i=0; i<2*n_pop; i++) 
    if(i==0 || i==2) 
      printf("%lu ", avg_n_post[i] / n_per_pop[0]) ; 
    else 
      printf("%lu ", avg_n_post[i] / n_per_pop[1]) ; 
  printf("\n") ; 

  for(i=0; i<n_pop; i++)
    for(j=0; j<n_pop; j++)
      printf("%lu ", avg_n_post[j+i*n_pop] / n_per_pop[i]); 
  printf("\n") ; 
  
  printf(" average number of postsynaptic connections per neuron: ") ; 
  for(i=0; i<10; i++) 
    printf("%d ", n_post[i]) ; 
  printf("\n") ; 
  
  idx_post[0] = 0 ; 
  for(i=1; i<N_NEURONS; i++) 
    idx_post[i] = idx_post[i-1] + n_post[i-1] ; 
  
  /* free(avg_n_post) ;  */
  
}

__host__  void create_con_dir() { 
  
  string mkdirp = "mkdir -p " ;

  path += to_string(n_pop)+"pop" ;

  ostringstream str_seed ; 
  str_seed << SEED_CON ; 
  
  if(IF_CON_DIR)
    path += "/seed_" + str_seed.str() ;
  
  if(n_pop==1) 
    path += "/N" + to_string(n_per_pop[0]/1000) ; 
  else 
    path += "/NE_" + to_string(n_per_pop[0]/1000) +  "_NI_" + to_string(n_per_pop[1]/1000) ; 

  path += "/K" + to_string((int)K) ; 

  ostringstream str_kappa ;
  str_kappa << fixed << setprecision(2) << KAPPA ; 
  
  if(IF_STRUCTURE) { 
    if(IF_SPEC) 
      path += "/spec/kappa_" + str_kappa.str() ; 
    if(IF_RING) 
      path += "/ring/kappa_" + str_kappa.str() ; 
  }

  if(IF_LOW_RANK) {
    
    path += "/low_rank/rank_" + to_string(RANK) ; 
    
    if(FIX_KSI_SEED)
      path += "/seed_ksi_" + to_string(SEED_KSI) ;
  }
  
  mkdirp += path ; 

  const char * cmd = mkdirp.c_str();
  const int dir_err = system(cmd);

  if(-1 == dir_err) 
    cout << "error creating directories" << endl ;
  
  cout << "Created directory : " ;
  cout << path << endl ;

}

__host__ void save_to_con_file() {
  
  cout << "save to file" << endl ;

  FILE *file_con_mat ; 
  string file_path ; 

  if(IF_SAVE_CON_VEC) {
    cout << "saving connectivity vector to: "; 
    file_path = path + "/con_vec.dat" ; 
    cout << file_path << endl ; 
    
    file_con_mat = fopen(file_path.c_str() , "wb") ; 
    fwrite( con_vec, sizeof(*con_vec) , (unsigned long) (N_NEURONS*N_NEURONS), file_con_mat) ; 
    fclose(file_con_mat) ; 
  }
  
  FILE *file_id_post, *file_n_post, *file_idx_post ; 
  
  if(IF_SAVE_SPARSE_REP) { 
    
    file_path = path + "/id_post.dat" ; 
    cout << file_path << endl ; 

    cout << "id_post: " ;
    for(i=0;i<5;i++)
      cout << id_post[i] << " " ;
    cout << endl ;
    
    file_id_post = fopen(file_path.c_str() , "wb") ; 
    fwrite(id_post, sizeof(*id_post) , total_n_post, file_id_post) ; 
    fclose(file_id_post) ; 
    
    file_path = path + "/idx_post.dat" ; 
    cout << file_path << endl ; 

    cout << "idx_post: " ;
    for(i=0;i<5;i++)
      cout << idx_post[i] << " " ;
    cout << endl ;
    
    file_idx_post = fopen(file_path.c_str(), "wb") ; 
    fwrite(idx_post, sizeof(*idx_post) , N_NEURONS, file_idx_post) ; 
    fclose(file_idx_post) ; 
    
    file_path = path + "/n_post.dat" ; 
    cout << file_path << endl ; 

    cout << "n_post: " ;
    for(i=0;i<5;i++) 
      cout << n_post[i] << " " ; 
    cout << endl ;
    
    file_n_post = fopen(file_path.c_str(), "wb") ; 
    fwrite(n_post, sizeof(*n_post) , N_NEURONS, file_n_post) ; 
    fclose(file_n_post) ; 
  }
}


__host__ void delete_globals() { 

  free(id_post) ;
  free(idx_post) ;
  
}

__host__ void read_from_file(string path, string file_name, float *array, size_t array_size) {

  int dum=0 ; 
  
  string file_path ; 
  file_path = path + "/" + file_name + ".dat" ; 
  cout << "reading from: " << file_path << endl ; 
  
  struct stat buffer ; 
  FILE *file ; 
  
  if (stat (file_path.c_str(), &buffer) == 0) { 
    file = fopen(file_path.c_str(), "rb") ; 
    dum = fread(&array[0], sizeof array[0], array_size, file) ; 
    fclose(file) ; 
  } 
  else { 
    cout << "ERROR: " << file_name << ".dat not found" << endl ; 
    exit(-1) ; 
  }

  dum = 1 ;
  cout << dum << endl ;  
}

__host__ void make_dir(string path) {
  string mkdirp = "mkdir -p " ; 
  mkdirp += path ; 
  
  const char * cmd = mkdirp.c_str() ; 
  const int dir_err = system(cmd) ; 
  
  cout << path << endl ; 
}

#endif 
