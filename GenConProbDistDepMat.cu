#ifndef _CONNECTION_PROB_
#define _CONNECTION_PROB_
#include <stdio.h>
#include <cuda.h>
#include "devHostConstants.h"

///////////////////////////////////////////////////////////////////    

__global__ void initPreFactor(float *dev_preFactor) {
  unsigned long i;
  for(i = 0; i < nbpop * N_NEURONS ; i++) 
    dev_preFactor[i] = 0 ; 
}

/* GENERATE CONNECTION MATRIX */

__device__ double XCordinate(unsigned long int neuronIdx, unsigned long int *nbN, unsigned long int *Cpt) { 
  double X = 0 ;
  int i = whichPop(neuronIdx) ;

  if(DIMENSION==1)
    // X = fmod( (double) (neuronIdx-Cpt[i]), ( (double) nbN[i]-1.0 ) ) * L / ( (double) nbN[i]-1.0 ) ; 
    X = fmod( (double) (neuronIdx-Cpt[i]), ( (double) nbN[i] ) ) * L / ( (double) nbN[i] ) ; 
  else
    X = fmod( (double) (neuronIdx-Cpt[i]),  sqrt( (double) nbN[i]-1.0) ) * L / sqrt( (double) nbN[i]-1.0 ) ; 
  
  return X ;
}

__device__ double YCordinate(unsigned long int neuronIdx, unsigned long int *nbN, unsigned long int *Cpt) { 
  double Y = 0 ;
  int i = whichPop(neuronIdx) ; 
  Y = floor( (double) (neuronIdx-Cpt[i]) / sqrt( (double) nbN[i]-1.0 ) ) * L / sqrt( (double) nbN[i]-1.0 )  ;

  return Y ;
}

///////////////////////////////////////////////////////////////////    

__device__ double Gaussian1D(double mu, double sigma) {
  if(sigma!=0.)
    return exp(-mu*mu/2./sigma/sigma)/sqrt(2.*M_PI)/sigma ;
  else
    return 1. ; 
}

__device__ double Exp(double mu, double sigma) {
  if(sigma!=0.)
    return  exp(-abs(mu)/sigma) /sigma ;
  else
    return 1. ;
}

///////////////////////////////////////////////////////////////////    

 __device__ double ShortestDistOnCirc(double X, double Y) { // NOT WORKING !!!!!!!!!!!
  double dist = 0.0;
  
  if(X==Y)
    dist=0 ; 
  else {
    dist = fmod(abs(X-Y),L) ; 
  
    if(dist > 0.5*L)
      dist = dist-L ;
    else
      dist = .5*dist ; 
  }
  return dist;
}

///////////////////////////////////////////////////////////////////    

__device__ double ConProb(double xa, double xb, double varianceOfGaussian) {
  double distX = 0.0, outX=0.0 ; 
  int k=0 ;

  if(IF_PERIODIC) {
    // distX = ShortestDistOnCirc(xa, xb) ; 
    for(k=-4;k<=4;k++) { 
      distX = xa - xb -L*(double)k  ; 
      if(IF_GAUSS)
	outX += Gaussian1D(distX, varianceOfGaussian) ; 
      if(IF_EXP)
	outX += Exp(distX, varianceOfGaussian) ; 
    }
  }
  else {
    distX = abs(xa - xb) ; 
    if(IF_GAUSS)
      outX = Gaussian1D(distX, varianceOfGaussian) ;
    if(IF_EXP)
      outX = Exp(distX, varianceOfGaussian) ; 
  }

  return outX ;
}

///////////////////////////////////////////////////////////////////    

__device__ double ConProb2D(double xa, double xb, double ya, double yb, double varianceOfGaussian) {
  double distX = 0.0, distY = 0.0 ;
  double outX = 0.0, outY = 0.0;
  int k=0 ;

  if(IF_PERIODIC) {
    for(k=-2;k<=2;k++) { 
      distX = xa - xb -L*(double)k  ; 
      distY = ya - yb -L*(double)k  ; 

      outX += Gaussian1D(distX, varianceOfGaussian) ;
      outY += Gaussian1D(distY, varianceOfGaussian) ;
    }
  }
  else {
    distX = abs(xa - xb) ; 
    distY = abs(ya - yb) ; 

    outX = Gaussian1D(distX, varianceOfGaussian) ;
    outY = Gaussian1D(distY, varianceOfGaussian) ;
  }
  // return Gaussian1D(distX, varianceOfGaussian) * Gaussian1D(distY, varianceOfGaussian) ; 

  return outX*outY ; 
}

/////////////////////////////////////////////////////////////////// 

__global__ void KernelGenConProbMat(float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int *nbN, unsigned long int *Cpt, const double *Sigma, const double *Dij) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long kNeuron = id + lChunck * maxNeurons ;
  unsigned long i = 0.0 ;
  double xPreS = 0.0, xPost = 0.0 ;
  int PreS = 0, Post = 0 ;
  
  if(id < maxNeurons & kNeuron < N_NEURONS) { // M[clm + row*N]
    Post = whichPop(kNeuron) ; 
    xPost = XCordinate(kNeuron,nbN,Cpt) ; // Mij clm to row 
    for(i=0; i < N_NEURONS; i++) { // i preS/clm to id post/row , P[row][clm] = G(X[row],X[clm],Sigma[clm]) 
      PreS = whichPop(i) ; 
      xPreS = XCordinate(i,nbN,Cpt) ; 
      dev_conVec[i + id * N_NEURONS ] = (float) ConProb(xPreS, xPost, Dij[ PreS + Post * nbpop ] * Sigma[PreS] ) ; // Pb[id<-i]
    }
  }
}
///////////////////////////////////////////////////////////////////    

__global__ void KernelGenConProbMat2D(float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int *nbN, unsigned long int *Cpt, const double *Sigma) {
  
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long int kNeuron = id + lChunck * maxNeurons ;
  unsigned long int i ; 
  double xa, ya ;

  if(id < maxNeurons & kNeuron < N_NEURONS) {    
    xa = XCordinate(kNeuron,nbN,Cpt) ; // Mij column to row 
    ya = YCordinate(kNeuron,nbN,Cpt) ; 
    for(i=0; i < N_NEURONS; i++)  // i post/row to id post/clm, P[row][clm] = G(X[row],X[clm],Sigma[clm]) 
      dev_conVec[i + id * N_NEURONS ] = (float) ConProb2D(xa, XCordinate(i,nbN,Cpt), ya, YCordinate(i,nbN,Cpt), Sigma[whichPop(i)] ) ; 
  }
}

///////////////////////////////////////////////////////////////////    

__global__ void KernelConProbPreFactor(float *dev_conVec,float *dev_preFactor, int lChunck, unsigned long int maxNeurons) { 

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread 
  unsigned long kNeuron = id + lChunck * maxNeurons ; 
  unsigned long i ; 
  
  if(id < maxNeurons & kNeuron < N_NEURONS) {// i pres/clm to id post/row 
    for(i=0;i<N_NEURONS;i++)  // sum over i preS/clm Zb[row] = K / sum_clm C[row][clm] 
      dev_preFactor[ kNeuron + whichPop(i) * N_NEURONS ] += (double) dev_conVec[i + id * N_NEURONS] ; // sum_i Pb[id<-i]
    // dev_preFactor is [N,nbpop] so that neuron (i0, j) <- sum_over preS pop j 
  }
}

/////////////////////////////////////////////////////////////////// 

__global__ void KernelConProbNorm(float *dev_conVec, float *dev_preFactor, int lChunck, unsigned long int maxNeurons) {
  
  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread
  unsigned long kNeuron = id + lChunck * maxNeurons ; 
  unsigned long i ; 
  float preFactor = 0 ; 
  int PreS ;
  
  if(id < maxNeurons & kNeuron< N_NEURONS) { 
    for(i=0;i<N_NEURONS;i++) { // i pres/clm to id post/row, P[row][clm] = Zb[row]*C[row][clm] 
      PreS = whichPop(i) ; 
      if(dev_preFactor[kNeuron + PreS * N_NEURONS] !=0) {
	preFactor = K / dev_preFactor[kNeuron + PreS * N_NEURONS] ; 
	
	// if(PreS==1 && whichPop(kNeuron)==0)
	//   preFactor = (K-5.0*sqrt(K)) / dev_preFactor[kNeuron + PreS * N_NEURONS] ;
	
	dev_conVec[i + id * N_NEURONS] *= preFactor ; 
      }
      // if(PreS==1 && whichPop(kNeuron)==0) 
      // 	dev_conVec[i + id * N_NEURONS] += 5.*sqrt(K)/10000. ; 
    }
  }
}

/////////////////////////////////////////////////////////////////// 

__global__ void KernelSumConVec(float *dev_conVec, float *dev_conVecLR, int lChunck, unsigned long int maxNeurons, const double *DijLR) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x; // each clm is a thread
  unsigned long kNeuron = id + lChunck * maxNeurons ; 
  unsigned long i ; 
  int PreS = 0, Post = 0 ;

  if(id < maxNeurons & kNeuron< N_NEURONS) { 
    Post = whichPop(kNeuron) ; 
    for(i=0;i<N_NEURONS;i++) { // i pres/clm to id post/row, P[row][clm] = Zb[row]*C[row][clm] 
      PreS = whichPop(i) ; 
      if( DijLR[ PreS + Post * nbpop ] !=0 )
	dev_conVec[i + id * N_NEURONS] =  (K - AmpLR * sqrt(K)) / K * dev_conVec[i + id * N_NEURONS] + AmpLR * sqrt(K) / K * dev_conVecLR[i + id * N_NEURONS] ; 
    }
  }  
}

/////////////////////////////////////////////////////////////////// 

__global__ void KernelGenConRing(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons, unsigned long int *nbN, unsigned long int *Cpt, const double *Sigma, const double *Dij) { 

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long kNeuron = id + lChunck * maxNeurons ;
  unsigned long i ;
  double xPreS, xPost ;
  int PreS, Post;
  
  if(id < maxNeurons && kNeuron < N_NEURONS) { 
    Post = whichPop(kNeuron) ; 
    xPost = XCordinate(kNeuron,nbN,Cpt) ; // Mij column to row 
    for(i=0; i < N_NEURONS; i++) { // id-->i column to row, P[row][clm] = G(X[row],X[clm],Sigma[clm]) 
      PreS = whichPop(i) ; 
      xPreS = XCordinate(i,nbN,Cpt) ; 
      dev_conVec[i + id * N_NEURONS] = (float) ( K / (float) nbN[PreS] ) * ( 2.0 * Sigma[PreS] * Dij[ PreS + Post * nbpop ] * cos( 2.0 * (xPreS-xPost) ) ) ; 
      // dev_conVec[i + id * N_NEURONS] = (float) ( K / (float) nbN[PreS] ) * ( 1.0 + 2.0 * Sigma[PreS] * Dij[ PreS + Post * nbpop ] * cos( 2.0 * (xPreS-xPost) ) ) ; 
      // cuPrintf("id %d pop %d | i %d pop %d | idx %d Dij %.0f\n", kNeuron, whichPop(kNeuron), i, whichPop(i), whichPop(kNeuron) + whichPop(i) * nbpop, Dij[ whichPop(kNeuron) + whichPop(i) * nbpop ]) ;     
    }
  }
}

///////////////////////////////////////////////////////////////////  

__global__ void KernelGenDistDepConMat(curandState *state, float *dev_conVec, int lChunck, unsigned long int maxNeurons) {

  unsigned long id =  (unsigned long int)threadIdx.x + blockIdx.x * blockDim.x;
  unsigned long kNeuron = id + lChunck * maxNeurons, i;
  
  if(id < maxNeurons && kNeuron < N_NEURONS)
    for(i=0; i<N_NEURONS; i++) { 
      
      if(dev_conVec[i + id * N_NEURONS] >= randkernel(state, kNeuron) ) 
      	dev_conVec[i + id * N_NEURONS] = 1. ;
      else
      	dev_conVec[i + id * N_NEURONS] = 0. ; 
    } 
} 

#endif