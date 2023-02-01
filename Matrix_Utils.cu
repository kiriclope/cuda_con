#ifndef __MATRIXUTILS__
#define __MATRIXUTILS__

///////////////////////////////////////////////////////////////////    

__global__ void setup_kernel(curandState *state, unsigned long long seed ) {
  unsigned long int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets different seed, a different sequence number, no offset */
  if(id < N_NEURONS) 
    curand_init(seed * (id + 7), id, 0, &state[id]);
}

///////////////////////////////////////////////////////////////////    

__device__ float randkernel(curandState *state, unsigned long int kNeuron) {
  /*RETURNS ONE SAMPLE FROM UNIFORM DISTRIBUTION*/
  float randNumber= 0.0 ; 
  if(kNeuron < N_NEURONS) {
    curandState localState = state[kNeuron]; /* state in global memory */
    randNumber = curand_uniform(&localState);
    state[kNeuron] = localState;
  }
  return randNumber;
}

__device__ float normal_dist(curandState *state, unsigned long int kNeuron, float mean, float std) {
  /*RETURNS ONE SAMPLE FROM NORMAL DISTRIBUTION*/
  float randNumber= 0.0 ; 
  if(kNeuron < N_NEURONS) {
    curandState localState = state[kNeuron]; /* state in global memory */
    randNumber = curand_normal(&localState) ;
    state[kNeuron] = localState;
  }
  return randNumber;
}

///////////////////////////////////////////////////////////////////    

__host__ void nbNeurons(unsigned long * &nbN) {
  
  cudaCheck(cudaMallocHost((void **)&nbN, nbpop * sizeof( unsigned long )));
  printf("Number of neurons : ") ;
  unsigned long i = 0; 
  while(i<nbpop) {
    if(i==0) 
      nbN[i] = N_NEURONS*popSize ;
    else
      nbN[i] = (unsigned long) ( N_NEURONS - nbN[0] ) / max( (nbpop-1), 1 ) ;
       
    printf("%lu ", nbN[i]) ;
    ++i ;
  }
  printf("\n") ;
}

///////////////////////////////////////////////////////////////////    
 
__host__ void CptNeurons(unsigned long* nbN, unsigned long* &Cpt) {
  cudaCheck(cudaMallocHost((void **)&Cpt, nbpop * sizeof( unsigned long int)));
  printf("Counter : ") ;

  unsigned long i,j;
  for(i=0;i<nbpop+1;i++) {
    Cpt[i] = 0 ;
    for(j=0;j<i;j++) {
      Cpt[i] = Cpt[i] + nbN[j] ; 
    } 
    printf("%lu ", Cpt[i]) ;
  } 
  printf("\n") ; 
}

///////////////////////////////////////////////////////////////////    

__global__ void initConVec(float *dev_conVec, int lChunck, unsigned long int maxNeurons) {
  unsigned long id = threadIdx.x + blockIdx.x * blockDim.x; 
  unsigned long kNeuron = id + lChunck * maxNeurons;
  unsigned long i;
  if(id < maxNeurons && kNeuron < N_NEURONS) 
    for(i = 0; i < N_NEURONS; i++) 
      dev_conVec[i + id * N_NEURONS] = 0 ; 
      // dev_conVec[id + i * N_NEURONS] = 0 ; 
}

///////////////////////////////////////////////////////////////////    

__host__ __device__ int whichPop(unsigned long int neuronIdx) {
  
  // const double PROPORTIONS[4] = {.75,.25,0,0} ;

  int popIdx = 0 ;
  unsigned long propPop = N_NEURONS*popSize ;
  unsigned long propPop0 = N_NEURONS*popSize ;
  
  while( neuronIdx > propPop-1 ) {
    popIdx++ ;
    // propPop += propPop ;
    propPop += (unsigned long ) ( N_NEURONS - propPop0 ) / max( (nbpop-1), 1 ) ;
    // propPop += int( N_NEURONS * ( 1. - popSize ) ) / max( (nbpop-1), 1 ) ;
  }
  return popIdx ;
}

///////////////////////////////////////////////////////////////////    

__host__ int dirExists(const char *path) {
  struct stat info;

  if(stat( path, &info ) != 0)
    return 0;
  else if(info.st_mode & S_IFDIR)
    return 1;
  else
    return 0;
}

///////////////////////////////////////////////////////////////////    

__host__ void CreatePath(char *&path) {
  

  char* cdum ;
  cdum = (char *) malloc(500 * sizeof(char) ) ;

  char* strCrec ;
  strCrec = (char *) malloc(500 * sizeof(char)) ;

  char* strAuta ;
  strAuta = (char *) malloc(500 * sizeof(char)) ;

  switch(AUTA_Pop) {
  case 0 :
    if(nbpop>1)
      sprintf(strAuta,"AutaE%.2f", AUTA_p[0]) ;
    else
      sprintf(strAuta,"AutaI%.2f", AUTA_p[0]) ;
    break ;
  case 1 :
    sprintf(strAuta,"AutaI%.2f", AUTA_p[1]) ;
    break ;
  case 2 :
    sprintf(strAuta,"AutaE%.2fI%.2f", AUTA_p[0], AUTA_p[1] ) ; 
    break ;
  }
  
  if(nbpop==1) 
    sprintf(strCrec,"CrecI%.4f",Sigma[0]);
  if(nbpop==2) {
    if(IF_Dij)
      sprintf(strCrec,"CrecEE%.4fCrecEI%.4fCrecIE%.4fCrecII%.4f",Sigma[0]*Dij[0],Sigma[1]*Dij[1],Sigma[0]*Dij[2],Sigma[1]*Dij[3]);
    else 
      sprintf(strCrec,"CrecE%.4fCrecI%.4f",Sigma[0],Sigma[1]);    
  }
  if(nbpop==3) {
    if(IF_Dij)
      sprintf(strCrec,"CrecEE%.4fCrecEI%.4fCrecES%.4fCrecIE%.4fCrecII%.4fCrecIS%.4fCrecSE%.4fCrecSI%.4fCrecSS%.4f",Sigma[0]*Dij[0],Sigma[1]*Dij[1],Sigma[2]*Dij[2],Sigma[0]*Dij[3],Sigma[1]*Dij[4],Sigma[2]*Dij[5],Sigma[0]*Dij[6],Sigma[1]*Dij[7],Sigma[2]*Dij[8]);
    else 
      sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4f",Sigma[0],Sigma[1],Sigma[2]);
  }
  if(nbpop==4) 
    sprintf(strCrec,"CrecE%.4fCrecI%.4fCrecS%.4fCrecV%.4f",Sigma[0],Sigma[1],Sigma[2],Sigma[3]);

  if(!IF_RING && !IF_SPACE)
    sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f", nbpop, (int) (N_NEURONS/nbPref), K) ; 
  
  if(IF_RING) {
    if(IF_SPEC) 
      sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/Ring/Spec/%s", nbpop, (int) (N_NEURONS/nbPref), K, strCrec) ;    
    else
      sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/Ring/%s", nbpop, (int) (N_NEURONS/nbPref), K, strCrec) ;  
  }

  if(IF_SPACE) {
    if(IF_SPEC)
      sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/Gauss/Spec/%s", nbpop, (int) (N_NEURONS/nbPref), K, strCrec) ; 
    else {
      if(IF_GAUSS)
	if(DIMENSION==1)
	  sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/Gauss/%s", nbpop, (int) (N_NEURONS/nbPref), K, strCrec) ;  
	else
	  sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/Gauss2D/%s", nbpop, (int) (N_NEURONS/nbPref), K, strCrec) ;  
      if(IF_EXP)
	sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/Exp/%s", nbpop, (int) (N_NEURONS/nbPref), K, strCrec) ;  
    }
  }

  // if(IF_AUTA)
  //   sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/%s", nbpop, (int) (N_NEURONS/nbPref), K, strAuta) ; 
  // if(IF_SHARED)
  //   sprintf(cdum, "../../Connectivity/%dpop/N%d/K%.0f/Shared%.2f/Cluster%.2f", nbpop, (int) (N_NEURONS/nbPref), K, PROP_SHARED, CLUSTER_SIZE) ;
  
  
  path = (char *) malloc( (strlen(cdum) + 500) * sizeof(char) ) ;
  strcpy(path,cdum) ;

  char *mkdirp ;   
  mkdirp = (char *) malloc( (strlen(path) + 500) * sizeof(char) );
  
  strcpy(mkdirp,"mkdir -p ") ; 
  strcat(mkdirp,path) ; 
  printf("%s\n",mkdirp) ;

  const int dir_err = system(mkdirp) ; 
  printf("%d\n",dir_err) ;

  if(-1 == dir_err) {
    printf("error creating directory : ");
  }
  else { 
    printf("Created directory : ") ;
  }
  printf("%s\n",path) ;
}

///////////////////////////////////////////////////////////////////    

__host__ void CheckPres(char *path, unsigned long* Cpt, unsigned long *IdPost, int *nbPost, int **nbPreSab) {

  // for(unsigned long k=0;k<N_NEURONS;k++) 
  //   nbPreSab[ whichPop(k) ][ whichPop( IdPost[k] ) ] += nbPost[k] ;
  
  printf("\nAverage nbPreS : ");

  const char* str ;
  if(IF_Nk)
    str ="/nbPreSab_Nk.txt" ;
  else
    str ="/nbPreSab.txt" ;

  char *strPreSab ;
  strPreSab =  (char *) malloc( (strlen(path) + strlen(str) + 100 ) * sizeof(char) ) ;

  strcpy(strPreSab,path) ;
  strcat(strPreSab,str) ;

  FILE * pFile ;
  pFile = fopen (strPreSab,"w");

  for(int i=0;i<nbpop;i++) 
    for(int j=0;j<nbpop;j++) { 
      printf("%.3f ", (double) nbPreSab[i][j] / (double) ( Cpt[i+1]-Cpt[i] ) ) ;
      fprintf(pFile,"%.3f ", nbPreSab[i][j] / (double) (Cpt[i+1]-Cpt[i]) ) ;
    }
  printf("\n");

  fclose(pFile);
}

///////////////////////////////////////////////////////////////////    

__host__ void WritetoFile(char *path, unsigned long int *IdPost, int *nbPost, unsigned long int *idxPost) {

  printf(" Writing to Files :\n") ;

  char *nbpath  ;
  char *idxpath ;
  char  *Idpath ;

  const char *strIdPost = "/IdPost.dat";
  const char *stridxPost = "/idxPost.dat";
  const char *strnbPost = "/nbPost.dat"; 

  if(IF_Nk) {
    strIdPost = "/IdPost_Nk.dat";
    stridxPost = "/idxPost_Nk.dat";
    strnbPost = "/nbPost_Nk.dat"; 
  }

  if(IF_PRES) {
    strIdPost = "/IdPreS.dat" ; 
    stridxPost = "/idxPreS.dat"; 
    strnbPost = "/nbPreS.dat"; 
  }
  
  nbpath =  (char *) malloc( (strlen(path)+strlen(strnbPost) + 100) * sizeof(char) ) ;
  idxpath = (char *) malloc( (strlen(path)+strlen(stridxPost) + 100) * sizeof(char) ) ;
  Idpath = (char *)  malloc( (strlen(path)+strlen(strIdPost) + 100 ) * sizeof(char) ) ;

  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strnbPost) ;
  strcat(idxpath,stridxPost) ;
  strcat(Idpath,strIdPost) ;

  unsigned long nbCon = 0;
  for(unsigned long i = 0; i < N_NEURONS; i++) 
    nbCon += nbPost[i];
  
  FILE *fIdPost, *fnbPost, *fidxPost ;
  
  printf("sizeof IdPost %lu \n",  nbCon) ;

  for(int i=0;i<10;i++)
    printf("%lu ",IdPost[i]);
  printf("\n"); 

  fIdPost = fopen(Idpath, "wb");
  fwrite(IdPost, sizeof(*IdPost) , nbCon , fIdPost);
  fclose(fIdPost);

  printf("%s\n",Idpath) ;

  for(unsigned long i=1;i<N_NEURONS;i++)
    idxPost[i] = idxPost[i-1] + nbPost[i-1] ;

  for(int i=0;i<10;i++)
    printf("%lu ",idxPost[i]);
  printf("\n"); 

  fidxPost = fopen(idxpath, "wb") ;
  fwrite(idxPost, sizeof(*idxPost) , N_NEURONS , fidxPost); 
  fclose(fidxPost);

  printf("%s\n",idxpath) ;

  for(int i=0;i<10;i++)
    printf("%d ",nbPost[i]);
  printf("\n"); 

  fnbPost = fopen(nbpath, "wb") ;
  fwrite(nbPost, sizeof(*nbPost) , N_NEURONS , fnbPost) ;
  fclose(fnbPost);

  printf("%s\n",nbpath) ;
  printf("Done\n") ;

}

///////////////////////////////////////////////////////////////////    

__host__ void WritetoFileLarge(char *path, unsigned long int *IdPost, int *nbPost, unsigned long int *idxPost, char *AtoB) {

  printf(" Writing to Files :\n") ;

  char *nbpath  ;
  char *idxpath ;
  char  *Idpath ;

  char strIdPost[20] ;
  char stridxPost[20] ;
  char strnbPost[20] ; 

  if(IF_Nk) {
    sprintf(strIdPost, "/IdPost_%s_Nk.dat", AtoB);
    sprintf(stridxPost, "/idxPost_%s_Nk.dat", AtoB);
    sprintf(strnbPost, "/nbPost_%s_Nk.dat", AtoB);
  }
  else { 

    if(IF_PRES) {
      sprintf(strIdPost, "/IdPreS_%s.dat", AtoB);
      sprintf(stridxPost, "/idxPreS_%s.dat", AtoB);
      sprintf(strnbPost, "/nbPreS_%s.dat", AtoB);      
    }
    else {
      sprintf(strIdPost, "/IdPost_%s.dat", AtoB);
      sprintf(stridxPost, "/idxPost_%s.dat", AtoB);
      sprintf(strnbPost, "/nbPost_%s.dat", AtoB);
    }
  }

  nbpath =  (char *) malloc( ( strlen(path)+strlen(strnbPost) + 100 ) * sizeof(char) );
  idxpath = (char *) malloc( (strlen(path)+strlen(stridxPost) + 100 ) * sizeof(char) );
  Idpath = (char *)  malloc( (strlen(path)+strlen(strIdPost) + 100 ) * sizeof(char) );

  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strnbPost) ;
  strcat(idxpath,stridxPost) ;
  strcat(Idpath,strIdPost) ;

  unsigned long nbCon = 0;
  for(unsigned long i = 0; i < N_NEURONS; i++) 
    nbCon += nbPost[i];
  
  FILE *fIdPost, *fnbPost, *fidxPost ;
  
  printf("sizeof IdPost %lu \n",  nbCon) ;

  for(int i=0;i<10;i++)
    printf("%lu ",IdPost[i]);
  printf("\n"); 

  fIdPost = fopen(Idpath, "wb");
  fwrite(IdPost, sizeof(*IdPost) , nbCon , fIdPost);
  fclose(fIdPost);

  printf("%s\n",Idpath) ;

  for(unsigned long i=1;i<N_NEURONS;i++)
    idxPost[i] = idxPost[i-1] + nbPost[i-1] ;

  for(int i=0;i<10;i++)
    printf("%lu ",idxPost[i]);
  printf("\n"); 

  fidxPost = fopen(idxpath, "wb") ;
  fwrite(idxPost, sizeof(*idxPost) , N_NEURONS , fidxPost); 
  fclose(fidxPost);

  printf("%s\n",idxpath) ;

  for(int i=0;i<10;i++)
    printf("%d ",nbPost[i]);
  printf("\n"); 

  fnbPost = fopen(nbpath, "wb") ;
  fwrite(nbPost, sizeof(*nbPost) , N_NEURONS , fnbPost) ;
  fclose(fnbPost);

  printf("%s\n",nbpath) ;
  printf("Done\n") ;

}

///////////////////////////////////////////////////////////////////    

__host__ void WriteMatrix(char *path, float *conVec) {

  printf("Writing Cij Matrix to : \n") ;   
  const char* strMatrix = "/Cij_Matrix.dat"; 
  char *pathMatrix ; 
  pathMatrix = (char *) malloc( (strlen(path)+strlen(strMatrix)+100) * sizeof(char) ) ;

  strcpy(pathMatrix,path) ;
  strcat(pathMatrix,strMatrix) ;

  printf("%s\n",pathMatrix);
  
  FILE *Out;
  Out = fopen(pathMatrix,"wb");  
  fwrite(conVec, sizeof(float), (unsigned long) N_NEURONS * N_NEURONS , Out) ;
  fclose(Out) ;
}

__host__ void CheckSparseVec(char * path) {

  char *nbpath  ;
  char *idxpath ;
  char *Idpath ;

  const char *strIdPost = "/IdPost.dat";
  const char *stridxPost = "/idxPost.dat";
  const char *strnbPost = "/nbPost.dat"; 

  if(IF_Nk) {
    strIdPost = "/IdPost_Nk.dat" ; 
    stridxPost = "/idxPost_Nk.dat"; 
    strnbPost = "/nbPost_Nk.dat";    
  }

  if(IF_PRES) {
    strIdPost = "/IdPreS.dat" ; 
    stridxPost = "/idxPreS.dat"; 
    strnbPost = "/nbPreS.dat";    
  }

  nbpath =  (char *) malloc( (strlen(path)+strlen(strnbPost) + 100 ) * sizeof(char) ) ;
  idxpath = (char *) malloc( (strlen(path)+strlen(stridxPost) + 100 ) * sizeof(char) ) ;
  Idpath = (char *)  malloc( (strlen(path)+strlen(strIdPost) + 100 ) * sizeof(char) ) ;
  
  strcpy(nbpath,path) ;
  strcpy(idxpath,path) ;
  strcpy(Idpath,path) ;

  strcat(nbpath,strnbPost) ;
  strcat(idxpath,stridxPost) ;
  strcat(Idpath,strIdPost) ;

  int *nbPost ;
  unsigned long int *idxPost ;
  unsigned long int *IdPost ;

  nbPost = new int [N_NEURONS] ;
  idxPost = new unsigned long int [N_NEURONS] ;

  FILE *fnbPost, *fidxPost, *fIdPost ;
  
  int dum ;
  
  fnbPost = fopen(nbpath, "rb") ;
  dum = fread(&nbPost[0], sizeof nbPost[0], N_NEURONS , fnbPost);  
  fclose(fnbPost);
  
  fidxPost = fopen(idxpath, "rb") ;
  dum = fread(&idxPost[0], sizeof idxPost[0], N_NEURONS , fidxPost);
  fclose(fidxPost);
  
  unsigned long int nbposttot = 0 ;
  for(int j=0 ; j<N_NEURONS; j++)
    nbposttot += nbPost[j] ;
  
  IdPost = new unsigned long int [nbposttot] ;
  
  fIdPost = fopen(Idpath, "rb");
  dum = fread(&IdPost[0], sizeof IdPost[0], nbposttot , fIdPost); 
  fclose(fIdPost);
  
  printf("Writing Cij Matrix to : \n") ;
  const char* strMatrix = "/Cij_Matrix.dat";
  char *pathMatrix ;
  pathMatrix = (char *) malloc( (strlen(path)+strlen(strMatrix)+100) * sizeof(char) ) ;

  strcpy(pathMatrix,path) ;
  strcat(pathMatrix,strMatrix) ;

  printf("%s\n",pathMatrix);
  
  FILE *Out;
  Out = fopen(pathMatrix,"wb");
  
  float **M ;
  M = new float*[N_NEURONS] ;
  for(int i=0;i<N_NEURONS;i++) 
    M[i] = new float[N_NEURONS]() ;

  for(int i=1;i<N_NEURONS;i++)
    idxPost[i] = idxPost[i-1] + nbPost[i-1] ;
  
  for(int i=0;i<N_NEURONS;i++) 
    for(int l=idxPost[i]; l<idxPost[i]+nbPost[i]; l++) 
      M[IdPost[l]][i] = 1 ;
  
  for (int i=0; i<N_NEURONS; i++) 
    fwrite(M[i], sizeof(M[i][0]), N_NEURONS, Out) ; 
  
  fclose(Out) ; 
  delete [] M ;

}


///////////////////////////////////////////////////////////////////    

__global__ void kernelGenSparseRep(float *dev_conVec, unsigned long *dev_IdPost, int *dev_nbPost, int lChunk, unsigned long maxNeurons) {
  
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread 
  unsigned long kNeuron = id + lChunk * maxNeurons; 
  unsigned long i ; 
  int nbPost ; 

  if(id < maxNeurons & kNeuron < N_NEURONS) {

    dev_nbPost[kNeuron] = 0 ; 
    nbPost = 0 ; 
    
    for(i=0;i<N_NEURONS;i++) 
      if(dev_conVec[i + id * N_NEURONS]) {// id-->i column to row
	dev_IdPost[i + id * N_NEURONS] = i ; 
	nbPost += 1 ; 
      } 
    dev_nbPost[kNeuron] = nbPost ; 
  }  
}

///////////////////////////////////////////////////////////////////    

__global__ void kernelGenConMat(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int* nbN) { 

  /* indexing of matrix row + clm x N_NEURONS*/ 
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x ; 
  unsigned long kNeuron = id + lChunck * maxNeurons ; 
  unsigned long i ;
  
  if(id < maxNeurons && kNeuron < N_NEURONS) 
    for(i=0; i<N_NEURONS; i++) { // i is row and id is clmn 
      // cuPrintf("id %d i %d \n",id,i) ;     
      if( (float) ( K ) / (float) ( nbN[whichPop(i)] ) >= randkernel(state, kNeuron) ) // neuron[id] receives input from i 
    	dev_conVec[i + id * N_NEURONS] = 1 ; 
      else 
    	dev_conVec[i + id * N_NEURONS] = 0 ; 
    } 
} 

///////////////////////////////////////////////////////////////////    

__global__ void kernelGenConMatShared(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int* nbN) { 

  /* indexing of matrix row + clm x N_NEURONS*/ 
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x ; 
  unsigned long int kNeuron = id + lChunck * maxNeurons ; 
  unsigned long int i;
  int j=0 ;

  if(id < maxNeurons && kNeuron < N_NEURONS ) {

    if(whichPop(kNeuron)==0) {

      for(i=0; i<N_NEURONS/nbpop; i++) { // i is row and id is clmn 
	if( (float) ( K ) / (float) ( nbN[whichPop(kNeuron)] ) >= randkernel(state, kNeuron) ) // neuron[id] receives input from i 
	  dev_conVec[i + id * N_NEURONS] = 1 ; 
	else 
	  dev_conVec[i + id * N_NEURONS] = 0 ; 
      }
      
      if(kNeuron< CLUSTER_SIZE * N_NEURONS / nbpop ) { // CLUSTER projecting mostly to SOM	
	for(j=0;j<nbpop-1;j++)
	  for(i=(j+1)*N_NEURONS/nbpop; i<(j+2)*N_NEURONS/nbpop; i++) { // i is row and id is clmn 
	    if( (float) ( j - (2.0*j-1.0) * PROP_SHARED )* K / (float) ( CLUSTER_SIZE * nbN[whichPop(kNeuron)] ) >= randkernel(state, kNeuron) ) // neuron[id] receives input from i 
	      dev_conVec[id + i * maxNeurons] = 1 ; 
	    else 
	      dev_conVec[id + i * maxNeurons] = 0 ; 
	  }
      }

      if(kNeuron>= CLUSTER_SIZE * N_NEURONS / nbpop && nbpop>2) {
	for(j=0;j<nbpop-1;j++)
	  for(i=(j+1)*N_NEURONS/nbpop; i<(j+2)*N_NEURONS/nbpop; i++) { // i is row and id is clmn 
	    if( (float)  ( ( j - (2.0*j-1.0) * (1.0-PROP_SHARED) )* K ) / (float) ( (1.0 - CLUSTER_SIZE) * nbN[whichPop(kNeuron)] ) >= randkernel(state, kNeuron) ) // neuron[id] receives input from i 
	      dev_conVec[id + i * maxNeurons] = 1 ; 
	    else 
	      dev_conVec[id + i * maxNeurons] = 0 ; 
	  }
      }
      
    }
    else
      for(i=0; i<N_NEURONS; i++) { // i is row and id is clmn 
	if( (float) ( K ) / (float) ( nbN[whichPop(kNeuron)] ) >= randkernel(state, kNeuron) ) // neuron[id] receives input from i 
	  dev_conVec[id + i * maxNeurons] = 1 ; 
	else 
	  dev_conVec[id + i * maxNeurons] = 0 ; 
      }    
  }
  
}
///////////////////////////////////////////////////////////////////    

__global__ void kernelGenConAuta(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int* nbN, const double *AUTA_p) {

  /* indexing of matrix row + clm x N_NEURONS*/
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons ;
  unsigned long int i ;
  
  if(id < maxNeurons && kNeuron < N_NEURONS) 
    for(i=0; i<N_NEURONS; i++) { // i is row and id is clmn 
      // cuPrintf("id %d i %d \n",id,i) ; 
      if( i==kNeuron && ( whichPop(kNeuron) == AUTA_Pop || AUTA_Pop==2 ) ) 
	// if( (float) ( K ) / (float) ( nbN[whichPop(kNeuron)] ) * ( 1.0 + (float) AUTA_p[whichPop(kNeuron)] / sqrt(K) )  >= randkernel(state, kNeuron) ) 
	if(  (float) AUTA_p[whichPop(kNeuron)] >= randkernel(state, kNeuron) ) 
	  dev_conVec[id + i * maxNeurons] = 1 ; 
	else 
	  dev_conVec[id + i * maxNeurons] = 0 ; 
      else 
	if( (float) ( K ) / (float) ( nbN[whichPop(kNeuron)] ) >= randkernel(state, kNeuron) ) // neuron[id] receives input from i 
	  dev_conVec[id + i * maxNeurons] = 1 ; 
	else 
	  dev_conVec[id + i * maxNeurons] = 0 ; 

    }
}

///////////////////////////////////////////////////////////////////    

// __host__ void GenSparseVec(float *conVec, unsigned long int *IdPost, int *nbPost, int *nbPreS, int lChunk, unsigned long int maxNeurons) {
    
//     unsigned long int counter = 0 ;
    
//     for(int i=0;i<nbpop;i++) 
//       for(unsigned long int k=Cpt[i];k<Cpt[i+1];k++) { //Presynaptic neurons
// 	for(int j=0;j<nbpop;j++) 
// 	  for(unsigned long int l=Cpt[j];l<Cpt[j+1];l++) //Postsynaptic neurons
// 	    if(fullConVec[k + N_NEURONS * l]) { // k-->l column to row
// 	      IdPost[counter] = l ;
// 	      nbPost[k]++ ;
// 	      nbPreSab[j][i]++ ;
// 	      counter+=1 ;
// 	    }   
// 	// printf("PresId %d, nPost %d \r",k,nbPost[k]);
//       }

// }

#endif
