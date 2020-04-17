#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define NUM_ITER 1000

#define pHorizon 1
#define nState 29
#define nInput 7
#define nOutput 7
#define nDis 1

#define erc 1e-6
#define eac 1e-6
#define eaj 1e-6
#define erj 1e-6

__global__ void printMat(float *mat, int N, int M)
{
	printf("printing mat\n");
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<M;j++)
		{
			printf("%f ",mat[i*M+j]);
		}
		printf("\n");
	}
	printf("\n");
}



__global__ void initMatCuda(float *mat, float val, int N)								// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<N)	
	{
		mat[id] = val;
	}
}

void initMat(float *mat, float val, int N)								// parallel
{
	dim3 block = 1024;
	dim3 grid = (N+1024-1)/1024;

	initMatCuda<<<grid, block>>>(mat, val, N);
}



float *newMatrixCUDA(int n, int m)			
{
	float *tmp = NULL;
	
	cudaError_t err = cudaMalloc((void **)&tmp, n*m*sizeof(float));

	if ( err != cudaSuccess )
	{
		printf (" Failed to allocate device matrix! %s\n", cudaGetErrorString(err));
		exit ( EXIT_FAILURE ) ;
	}

	initMat(tmp, 0, n*m);
	return tmp;
}

float *newMatrix(int n, int m)			
{
	float *tmp = (float *)malloc(n*m*sizeof(float));
	for(int i=0;i<n*m;i++)
	{
		tmp[i] = 0;
	}
	return tmp;
}



void copyToDevice(float *dM, float *hM, int n, int m)
{
	int size = n*m;
	cudaMemcpy (dM ,hM, size * sizeof ( float ) , cudaMemcpyHostToDevice );
}

void copyToHost(float *hM, float *dM, int n, int m)
{
	int size = n*m;
	cudaMemcpy (hM ,dM, size * sizeof ( float ) , cudaMemcpyDeviceToHost );
}



__global__ void copyMatrixCuda(float *output, float *mat, int a, int b)		// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<a*b)	
	{
		output[id] = mat[id];
	}
}

void copyMatrix(float *output, float *mat, int a, int b)		// parallel
{
	dim3 block = 1024;
	dim3 grid = (a*b+1024-1)/1024;

	copyMatrixCuda<<<grid,block>>>(output, mat, a, b);
}



__global__ void transposeCuda(float *odata, float *idata, int n, int m)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(x<n && y<m)
		odata[y*n+x] = idata[x*m+y];
}

void transpose(float *odata, float *idata, int n, int m)
{
	dim3 block(32,32,1);
	dim3 grid((n+31)/32, (m+31)/32);
	
	transposeCuda<<<grid,block>>>(odata,idata,n,m);
}



__global__ void matrixMultiplyCuda(float *output, float *matrix1, float *matrix2, int a, int b, int c) 		//mat1-a*b	mat2-b*c
{		
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
		
	if(x<a && y<c)
	{
		float val = 0;
		for(int k=0;k<b;k++)
		{
			val += matrix1[x*b+k]*matrix2[k*c+y];
		}
		output[x*c+y] = val;
	}
}

void matrixMultiply(float *output, float *mat1, int transpose1, float *mat2, int transpose2, int a, int b, int c) 		//mat1-a*b	mat2-b*c 	// parallel
{
	float *tmp = newMatrixCUDA(a,c);
	
	float *matrix1;
	float *matrix2;
	
	if(transpose1)
	{
		matrix1 = newMatrixCUDA(a,b);
		transpose(matrix1, mat1, b,a);
	}
	else
	{
		matrix1 = mat1;
	}
	
	if(transpose2)
	{
		matrix2 = newMatrixCUDA(b,c);
		transpose(matrix2, mat1, c,b);
	}
	else
	{
		matrix2 = mat2;
	}
	
	dim3 block(32,32,1);
	dim3 grid((a+31)/32, (c+31)/32);
	matrixMultiplyCuda<<<grid, block>>>(output, matrix1, matrix2, a,b,c);
	
	if(transpose1)
	{
		cudaFree(matrix1);
	}
	if(transpose2)
	{
		cudaFree(matrix2);
	}

	cudaFree(tmp);
}



__global__ void matrixAddCuda(float *A, float *B, float sign, int a, int b) 			// adds b to a 	// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<a*b)	
	{
		A[id] += sign * B[id];
	}
}

void matrixAdd(float *A, float *B, float sign, int a, int b) 			// adds b to a 	// parallel
{
	dim3 block = 1024;
	dim3 grid = (a*b+1024-1)/1024;

	matrixAddCuda<<<grid,block>>>(A,B,sign,a,b);
}



__global__ void negateMatrixCuda(float *mat, int n, int m)			// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<n*m)	
	{
		mat[id] = -mat[id];
	}
}

void negateMatrix(float *mat, int n, int m)			// parallel
{
	dim3 block = 1024;
	dim3 grid = (n*m+1024-1)/1024;

	negateMatrixCuda<<<grid,block>>>(mat,n,m);
}



__global__ void matrixPosCuda(float *mat1, float *mat2, int n, int m)			// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<n*m)	
	{
		mat1[id] = fmaxf(0.0, mat2[id]);

	}
}

void matrixPos(float *mat1, float *mat2, int n, int m)			// parallel
{
	dim3 block = 1024;
	dim3 grid = (n*m+1024-1)/1024;

	matrixPosCuda<<<grid,block>>>(mat1,mat2,n,m);
}



__global__ void matrixNegCuda(float *mat1, float *mat2, int n, int m)			// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<n*m)	
	{
		mat1[id] = fmaxf(0.0, -mat2[id]);
	}
}

void matrixNeg(float *mat1, float *mat2, int n, int m)			// parallel
{
	dim3 block = 1024;
	dim3 grid = (n*m+1024-1)/1024;

	matrixNegCuda<<<grid,block>>>(mat1,mat2,n,m);
}



__global__ void diagonalAddCuda(float *theta, float *tmp, int N)			// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<N)	
	{
		// printf("tmp %f\n",tmp[i]);
		theta[id*N+id] = fmaxf(tmp[id],5.0);
	}
}

void diagonalAdd(float *theta, float *tmp, int N)			// parallel
{
	dim3 block = 1024;
	dim3 grid = (N+1024-1)/1024;

	diagonalAddCuda<<<grid,block>>>(theta,tmp,N);
}



__global__ void compareCuda(float *GpU, float *Kp, int *re, int N)				// parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	if(id<N)	
	{
		if(GpU[id] > Kp[id]+fmaxf(erc*Kp[id], eac))
		{
			*re = 0;
		}
	}
}

void compare(float *GpU, float *Kp, int *re, int N)				// parallel
{
	dim3 block = 1024;
	dim3 grid = (N+1024-1)/1024;

	compareCuda<<<grid,block>>>(GpU, Kp, re, N);
}




__global__ void updYCuda(float *Y_next, float *numerator, float *denominator, float *Y, int N)   // parallel
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int id = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
	
	if(id<N)
	{
		Y_next[id] = numerator[id]/denominator[id]*Y[id];
	}
}

void updY(float *Y_next, float *numerator, float *denominator, float *Y, int N)   // parallel
{
	dim3 block = 1024;
	dim3 grid = (N+1023)/1024;
	
	updYCuda<<<grid, block>>>(Y_next, numerator, denominator, Y, N);
}




void Gauss_Jordan(float *A,float *res, int N)
{
    /*
    size=Size of input matrix
    A=input matrix
    res= inverted matrix
    */
    float temp;
    float *matrix = newMatrix(N, 2*N);

    for (int i = 0; i < N; i++) 
    { 
        for (int j = 0; j < 2 * N; j++) 
        { 
            matrix[i*2*N+j]=0;
            if (j == (i + N)) 
                matrix[i*2*N+j] = 1; 
        } 
    }

    for (int i = 0; i < N; i++) 
    { 
        for (int j = 0; j < N; j++) 
        { 
            matrix[i*2*N+j]=A[i*N+j];

        } 
    }

    for (int i = N - 1; i > 0; i--) 
    { 
        if (matrix[(i - 1)*2*N+0] < matrix[i*2*N+0]) 
            for (int j = 0; j < 2 * N; j++) 
            { 
                temp = matrix[i*2*N+j]; 
                matrix[i*2*N+j] = matrix[(i - 1)*2*N+j]; 
                matrix[(i - 1)*2*N+j] = temp; 
            } 
    }

    for (int i = 0; i < N; i++)
    { 

        for (int j = 0; j < N; j++) 
        { 
            if (j != i) 
            { 
                temp = matrix[j*2*N+i] / matrix[i*2*N+i]; 
                for (int k = 0; k < 2 * N; k++) 
                { 
                    matrix[j*2*N+k] -= matrix[i*2*N+k] * temp; 
                } 
            } 
        } 
    } 

    for (int i = 0; i < N; i++)
    { 
        temp = matrix[i*2*N+i]; 
        for (int j = 0; j < 2 * N; j++)
        {
            matrix[i*2*N+j] = matrix[i*2*N+j] / temp; 
        } 
    }

    for (int i = 0; i < N; i++) 
    { 
        for (int j = N; j <2*N; j++) 
        { 
            res[i*N+j-N]=matrix[i*2*N+j];

        } 
    }

    free(matrix);
}



void computeUfromY(float *U, float *Y, float *Fp, float *Gp, float *Qp_inv, int N, int M)
{
	float *tmp = newMatrixCUDA(M,1);
	matrixMultiply(tmp, Gp, 1, Y, 0, M, N, 1);
	matrixAdd(tmp, Fp, 1, M, 1);
	matrixMultiply(U, Qp_inv, 0, tmp, 0, M, M, 1);
	negateMatrix(U, M, 1);
	cudaFree(tmp);
}

void computeFp(float *Fp, float *Fp1, float *Fp2, float *Fp3, float *D, float *x)
{
	matrixMultiply(Fp, Fp1, 0, D, 0, nInput*pHorizon, nDis*pHorizon, 1);
	float *Fp2x = newMatrixCUDA(nInput*pHorizon,1);
	matrixMultiply(Fp2x, Fp2, 0, x, 0, nInput*pHorizon, nState, 1);
	matrixAdd(Fp, Fp2x, 1, nInput*pHorizon, 1);
	matrixAdd(Fp, Fp3, -1, nInput*pHorizon, 1);
	
	cudaFree(Fp2x);	
	// for(int i=0;i<nInput*pHorizon;i++)
	// {
	// 	printf("%f\n", Fp[i]);
	// }
	// printf("\n");
	// printf("%d\n", Fp);
}

void computeMp(float *Mp, float *Mp1, float *Mp2, float *Mp3, float *Mp4, float *Mp5, float *Mp6, float *D, float *x)
{
	initMat(Mp, 0, 1);

	float *tmp = newMatrixCUDA(1,nState);
	matrixMultiply(tmp, x, 1, Mp1, 0, 1, nState, nState);
	matrixMultiply(tmp, tmp, 0, x, 0, 1, nState, 1);

	matrixAdd(Mp, tmp, 0.5, 1,1);
//	printMat<<<1,1>>>(Mp, 1, 1);

	matrixMultiply(tmp, D, 1, Mp2, 0, 1, nDis*pHorizon, nState);
	matrixMultiply(tmp, tmp, 0, x, 0, 1, nState, 1);

	matrixAdd(Mp, tmp, 0.5, 1,1);
	
	matrixMultiply(tmp, Mp4, 1, x, 0, 1, nState, 1);

	matrixAdd(Mp, tmp, 0.5, 1,1);

	cudaFree(tmp);
	tmp = newMatrixCUDA(1, nDis*pHorizon);
	matrixMultiply(tmp, D, 1, Mp3, 0, 1, nDis*pHorizon, nDis*pHorizon);
	matrixMultiply(tmp, tmp, 0, D, 0, 1, nDis*pHorizon, 1);

	matrixAdd(Mp, tmp, 0.5, 1,1);

	matrixMultiply(tmp, Mp5, 1, D, 0, 1, nDis*pHorizon, 1);

	matrixAdd(Mp, tmp, 0.5, 1,1);

	matrixAdd(Mp, Mp6, 0.5, 1,1);
	cudaFree(tmp);
}

void computeQd(float *Qd, float *Gp_Qp_inv, float *Gp, int N, int M)
{
	matrixMultiply(Qd, Gp_Qp_inv, 0, Gp, 1, N, M, N);	
}

void computeFd(float *Fd, float *Gp_Qp_inv, float *Fp, float *Kp, int N, int M)
{
	matrixMultiply(Fd, Gp_Qp_inv, 0, Fp, 0, N, M, 1);
	matrixAdd(Fd, Kp, 1, N, 1);
}

void computeMd(float *Md, float *Fp, float* Qp_inv, float* Mp, int N, int M)
{
	float *tmp = newMatrixCUDA(1,M);
	matrixMultiply(tmp, Fp, 1, Qp_inv, 0, 1, M, M);
	matrixMultiply(Md, tmp, 0, Fp, 0, 1, M, 1);
	matrixAdd(Md, Mp, -1, 1, 1);
	cudaFree(tmp);
}

void convertToDual(float *Qd, float *Fd, float *Md, float *Qp_inv, float *Gp, float *Kp, float *Fp, float *Mp, int N, int M)
{	
	float *Gp_Qp_inv = newMatrixCUDA(N,M);
	matrixMultiply(Gp_Qp_inv, Gp, 0, Qp_inv, 0, N, M, M);
	computeQd(Qd, Gp_Qp_inv, Gp, N, M);
	computeFd(Fd, Gp_Qp_inv, Fp, Kp, N, M);
	computeMd(Md, Fp, Qp_inv, Mp, N, M);

	cudaFree(Gp_Qp_inv);
}

void computeTheta(float *theta, float *Qd, int N)
{
	float *Qdn = newMatrixCUDA(N,N);
	matrixNeg(Qdn, Qd, N, N);

	float *one = newMatrixCUDA(N,1);
	initMat(one, 1, N);

	float *tmp = newMatrixCUDA(N,1);
	matrixMultiply(tmp, Qdn, 0, one, 0, N,N,1);

	diagonalAdd(theta, tmp, N);

	cudaFree(Qdn);
	cudaFree(one);
	cudaFree(tmp);
}

void computeQdp_theta(float *Qdp_theta, float *Qd, float *theta, int N)
{
	matrixPos(Qdp_theta, Qd, N, N);
	matrixAdd(Qdp_theta, theta, 1, N, N);
}

void computeQdn_theta(float *Qdn_theta, float *Qd, float *theta, int N)
{
	matrixNeg(Qdn_theta, Qd, N, N);
	matrixAdd(Qdn_theta, theta, 1, N, N);
}

void computealphaY(float *alphaY, float *ph, float *Qd, float *Y, float *Fd, int N)
{
	float *temp = newMatrixCUDA(1,N);

	matrixMultiply(temp, ph, 1, Qd, 0, 1, N, N);
	matrixMultiply(temp, temp, 0, ph, 0, 1, N, 1);

	if(temp[0] > 0)
	{
		float *temp2 = newMatrixCUDA(1,N);

		matrixMultiply(temp2, Y, 1, Qd, 0, 1, N, N);
		
		matrixAdd(temp2, Fd, 1, 1, N);
		
		matrixMultiply(temp2, temp2, 0, ph, 0, 1, N, 1);

		*alphaY = -temp2[0]/temp[0];

		cudaFree(temp2);
	}
	else
	{
		alphaY = 0;
	}

	cudaFree(temp);
}

void updateY1(float *Y_next, float *Y, float alphaY, float *ph, int N)
{
	copyMatrix(Y_next, Y, N, 1);
	matrixAdd(Y_next, ph, alphaY, N, 1);
}

void updateY2(float *Y_next, float *Y, float *Qdp_theta, float *Qdn_theta, float *Fd, float *Fdp, float *Fdn, int N)
{
	float *numerator = newMatrixCUDA(N,1);
	float *denominator = newMatrixCUDA(N,1);

	matrixMultiply(numerator, Qdn_theta, 0, Y, 0, N, N, 1);
	matrixMultiply(denominator, Qdp_theta, 0, Y, 0, N, N, 1);

	matrixAdd(numerator, Fdn, 1, N, 1);
	matrixAdd(denominator, Fdp, 1, N, 1);

	updY(Y_next, numerator, denominator, Y, N);

	cudaFree(numerator);
	cudaFree(denominator);
}

void computeph(float *ph, float *Qd, float *Y, float *Fd, int N)
{
	matrixMultiply(ph, Qd, 0, Y, 0, N, N, 1);
	matrixAdd(ph, ph, 1, N, 1);
	matrixNeg(ph, ph, N, 1);
}

int checkFeas(float *U, float *Gp, float *Kp, int N, int M)
{
	float *tmp = newMatrixCUDA(N,1);
	matrixMultiply(tmp, Gp, 0, U, 0, N, M, 1);
	int re = 1;
	compare(tmp, Kp, &re, N);

	cudaFree(tmp);
	return re;
}	

float computeCost(float *Z, float *Q, float *F, float *M, int N)
{
	float *J=newMatrixCUDA(1,1);

	float *tmp = newMatrixCUDA(1,N);
	matrixMultiply(tmp, Z, 1, Q, 0, 1, N, N);
	matrixMultiply(tmp, tmp, 0, Z, 0, 1, N, 1);

	matrixAdd(J, tmp, 0.5, 1,1);

	matrixMultiply(tmp, F, 1, Z, 0, 1, N, 1);

	matrixAdd(J, tmp, 1, 1,1);
//	printMat<<<1,1>>>(J,1,1);
//	printMat<<<1,1>>>(M,1,1);
	matrixAdd(J, M, 0.5, 1,1);
	

	float *hJ = newMatrix(1,1);
	copyToHost(hJ,J,1,1);

	float cost = hJ[0];
	free(hJ);
	cudaFree(J);
	cudaFree(tmp);

	return cost;
}

int terminate(float *Y, float *Qd, float *Fd, float *Md, float *U, float *Qp, float *Qp_inv, float *Fp, float *Mp, float *Gp, float *Kp, int N, int M)
{
	computeUfromY(U, Y, Fp, Gp, Qp_inv, N, M);

	if(!checkFeas(U, Gp, Kp, N, M))	return 0;

	float Jd = computeCost(Y, Qd, Fd, Md, N);
	float Jp = computeCost(U, Qp, Fp, Mp, M);

	if(Jp>-Jd)	return 0;
	if(Jp+Jd>eaj)	return 0;
	if((Jp+Jd)/fabs(Jd)>erj) return 0;

	return 1;
}

void solveQuadraticDual(float *Y, float *Qd, float *Fd, float *Md, float *U, float *Qp, float *Qp_inv, float *Fp, float *Mp, float *Gp, float *Kp, int N, int M)
{
	float *theta = newMatrixCUDA(N,N);
	float *Qdp_theta = newMatrixCUDA(N,N);
	float *Qdn_theta = newMatrixCUDA(N,N);
	float *Y_next = newMatrixCUDA(N,1);
	
	float *Fdn = newMatrixCUDA(N,1);
	float *Fdp = newMatrixCUDA(N,1);

	matrixPos(Fdp, Fd, N, 1);
	matrixNeg(Fdn, Fd, N, 1);
	
	computeTheta(theta, Qd, N);
	computeQdp_theta(Qdp_theta, Qd, theta, N);
	computeQdn_theta(Qdn_theta, Qd, theta, N);

	initMat(Y, 1000.0, N);
	// for(int i=0;i<N;i++) Y[i] = i+1;

	float *ph = newMatrixCUDA(N,1);
	long int h=1;
	float alphaY=0;

//	 while(h<NUM_ITER)
	while(!terminate(Y, Qd, Fd, Md, U, Qp, Qp_inv, Fp, Mp, Gp, Kp, N, M))
	{	
		// if(h>100000) break;
//		 printf("h %ld\n",h);
		if(1)
		{
			//update
			// printf("here\n");
			updateY2(Y_next, Y, Qdp_theta, Qdn_theta, Fd, Fdp, Fdn, N);			
			// printf("there\n");
		}
		else
		{
			// printf("accelerating\n");
			// accelerate
			computeph(ph, Qd, Y, Fd, N);
			computealphaY(&alphaY, ph, Qd, Y, Fd, N);
			// printf("alpha %f\n", alphaY);
			
			updateY1(Y_next, Y, alphaY/10, ph, N);

		}

		copyMatrix(Y, Y_next, N, 1);
		// for(int i=0;i<N;i++)
		// {
		// 	printf("%f ",Y[i]);
		// }
		// printf("\n\n");

		h++;
	}
	printf("Printing number of iterations = %ld\n",h);

	cudaFree(theta);
	cudaFree(Qdp_theta);
	cudaFree(Qdn_theta);
	cudaFree(Y_next);
	cudaFree(ph);
	cudaFree(Fdp);
	cudaFree(Fdn);
}

void input(float* qp_inv, float* Fp1, float* Fp2, float * Fp3, float * Mp1, float * Mp2, float * Mp3, float* Mp4, float* Mp5, float* Mp6, float* Gp, float* Kp, float* x, float* D, float* theta, float* Z)
{
	FILE *fptr;
	int i,j;
	float num;

	//Fill Qp_inverse	
	fptr = fopen("./example/Qp_inv.txt","r");
	for(i=0;i<pHorizon*nInput;i++)
	{
		for(j=0;j<pHorizon*nInput;j++)
		{
			fscanf(fptr,"%f", &num);
			qp_inv[j*pHorizon*nInput+i] = num;
		}
	}
	fclose(fptr);

	//Fill Fp1
	fptr = fopen("./example/Fp1.txt","r");
	for(i=0;i<nDis*pHorizon;i++)
	{
		for(j=0;j<nInput*pHorizon;j++)
		{
			fscanf(fptr,"%f", &num);
			Fp1[j*nDis*pHorizon+i] = num;
		}
	}
	fclose(fptr);

	//Fill Fp2
	fptr = fopen("./example/Fp2.txt","r");
	for(i=0;i<nState;i++)
	{
		for(j=0;j<nInput*pHorizon;j++)
		{
			fscanf(fptr,"%f", &num);
			Fp2[j*nState+i] = num;
		}
	}
	fclose(fptr);

	//Fill Fp3
	fptr = fopen("./example/Fp3.txt","r");
	for(j=0;j<nInput*pHorizon;j++)
	{
		fscanf(fptr,"%f", &num);
		Fp3[j] = num;
	}
	fclose(fptr);

	//Fill Mp1
	fptr = fopen("./example/Mp1.txt","r");
	for(i=0;i<nState;i++)
	{
		for(j=0;j<nState;j++)
		{
			fscanf(fptr,"%f", &num);
			Mp1[j*nState+i] = num;
		}
	}
	fclose(fptr);

	//Fill Mp2
	fptr = fopen("./example/Mp2.txt","r");
	for(i=0;i<nState;i++)
	{
		for(j=0;j<nDis*pHorizon;j++)
		{
			fscanf(fptr,"%f", &num);
			Mp2[j*nState+i] = num;
		}
	}
	fclose(fptr);

	//Fill Mp3
	fptr = fopen("./example/Mp3.txt","r");
	for(i=0;i<nDis*pHorizon;i++)
	{
		for(j=0;j<nDis*pHorizon;j++)
		{
			fscanf(fptr,"%f", &num);
			Mp3[j*nDis*pHorizon+i] = num;
		}
	}
	fclose(fptr);

	//Fill Mp4
	fptr = fopen("./example/Mp4.txt","r");
	for(i=0;i<nState;i++)
	{
		fscanf(fptr,"%f", &num);
		Mp4[i] = num;
	}
	fclose(fptr);

	//Fill Mp5
	fptr = fopen("./example/Mp5.txt","r");
	for(i=0;i<nDis*pHorizon;i++)
	{
		fscanf(fptr,"%f", &num);
		Mp5[i] = num;
	}
	fclose(fptr);

	//Fill Mp6
	fptr = fopen("./example/Mp6.txt","r");
	fscanf(fptr,"%f", &num);
	Mp6[0] = num;
	fclose(fptr);

	//Fill Gp
	fptr = fopen("./example/Gp.txt","r");
	for(i=0;i<pHorizon*nInput;i++)
	{
		for(j=0;j<4*pHorizon*nInput;j++)
		{
			fscanf(fptr,"%f", &num);
			Gp[j*pHorizon*nInput+i] = num;
		}
	}
	fclose(fptr);

	//Fill Kp
	fptr = fopen("./example/Kp.txt","r");
	for(i=0;i<4*pHorizon*nInput;i++)
	{
		fscanf(fptr,"%f", &num);
		Kp[i] = num;
	}
	fclose(fptr);

	//Fill Z
	fptr = fopen("./example/Z.txt","r");
	for(i=0;i<nState;i++)
	{
		for(j=0;j<nOutput*pHorizon;j++)
		{
			fscanf(fptr,"%f", &num);
			Z[j*nState+i] = num;
		}
	}
	fclose(fptr);

	//Fill Theta
	fptr = fopen("./example/Theta.txt","r");
	for(i=0;i<nDis*pHorizon;i++)
	{
		for(j=0;j<nOutput*pHorizon;j++)
		{
			fscanf(fptr,"%f", &num);
			theta[j*nDis*pHorizon+i] = num;
		}
	}
	fclose(fptr);

	//Fill D
	fptr = fopen("./example/D.txt","r");
	for(i=0;i<nDis*pHorizon;i++)
	{
		fscanf(fptr,"%f", &num);
		D[i] = num;
	}
	fclose(fptr);

	//Fill x
	fptr = fopen("./example/x.txt","r");
	for(i=0;i<nState;i++)
	{
		fscanf(fptr,"%f", &num);
		x[i] = num;
	}
	fclose(fptr);
}

int main()
{
	// QP is of parametric from 
	// J(U) = min U 1/2*U'QpU + Fp'U + 1/2*Mp
	// st GpU <= Kp
	
	cudaDeviceReset();
	 
	int N, M;

	M = pHorizon*nInput;
	N = 4*pHorizon*nInput;

	// host matrix
	float *hQp_inv = newMatrix(M,M);
	float *hQp = newMatrix(M,M);

	float *hFp1;
	float *hFp2;
	float *hFp3;

	float *hMp1;
	float *hMp2;
	float *hMp3;
	float *hMp4;
	float *hMp5;
	float *hMp6;

	float *hFp = newMatrix(nInput*pHorizon,1);
	float *hMp = newMatrix(1,1);
	float *hGp;
	float *hKp;
	float *hx;
	float *hD; 
	float *htheta; 
	float *hZ; 

	hFp1 = newMatrix(nInput*pHorizon, nDis*pHorizon);
	hFp2 = newMatrix(nInput*pHorizon, nState);
	hFp3 = newMatrix(1, nInput*pHorizon);
	hMp1 = newMatrix(nState, nState);
	hMp2 = newMatrix(nDis*pHorizon, nState);
	hMp3 = newMatrix(nDis*pHorizon, nDis*pHorizon);
	hMp4 = newMatrix(1, nState);
	hMp5 = newMatrix(1, nDis*pHorizon);
	hMp6 = newMatrix(1,1);
	hGp = newMatrix(4*pHorizon*nInput, nInput*pHorizon);
	hKp = newMatrix(1,4*pHorizon*nInput);
	hZ = newMatrix(nOutput*pHorizon, nState);
	htheta = newMatrix(nOutput*pHorizon, nDis*pHorizon);
	hD = newMatrix(nDis*pHorizon,1);
	hx = newMatrix(nState, 1);

	// device matrix
	float *Qp_inv = newMatrixCUDA(M,M);
	float *Qp = newMatrixCUDA(M,M);

	float *Fp1;
	float *Fp2;
	float *Fp3;

	float *Mp1;
	float *Mp2;
	float *Mp3;
	float *Mp4;
	float *Mp5;
	float *Mp6;

	float *Fp = newMatrixCUDA(nInput*pHorizon,1);
	float *Mp = newMatrixCUDA(1,1);
	float *Gp;
	float *Kp;
	float *x;
	float *D; 
	float *theta; 
	float *Z; 

	Fp1 = newMatrixCUDA(nInput*pHorizon, nDis*pHorizon);
	Fp2 = newMatrixCUDA(nInput*pHorizon, nState);
	Fp3 = newMatrixCUDA(1, nInput*pHorizon);
	Mp1 = newMatrixCUDA(nState, nState);
	Mp2 = newMatrixCUDA(nDis*pHorizon, nState);
	Mp3 = newMatrixCUDA(nDis*pHorizon, nDis*pHorizon);
	Mp4 = newMatrixCUDA(1, nState);
	Mp5 = newMatrixCUDA(1, nDis*pHorizon);
	Mp6 = newMatrixCUDA(1,1);
	Gp = newMatrixCUDA(4*pHorizon*nInput, nInput*pHorizon);
	Kp = newMatrixCUDA(1,4*pHorizon*nInput);
	Z = newMatrixCUDA(nOutput*pHorizon, nState);
	theta = newMatrixCUDA(nOutput*pHorizon, nDis*pHorizon);
	D = newMatrixCUDA(nDis*pHorizon,1);
	x = newMatrixCUDA(nState, 1);	

	input(hQp_inv, hFp1, hFp2, hFp3, hMp1, hMp2, hMp3, hMp4, hMp5, hMp6, hGp, hKp, hx, hD, htheta, hZ);
	Gauss_Jordan(hQp_inv, hQp, M);
	copyToDevice(Qp_inv, hQp_inv, M, M);
	copyToDevice(Qp, hQp, M, M);
	copyToDevice(Fp1, hFp1, nInput*pHorizon, nDis*pHorizon);
	copyToDevice(Fp2, hFp2, nInput*pHorizon, nState);
	copyToDevice(Fp3, hFp3, 1, nInput*pHorizon);
	copyToDevice(Mp1, hMp1, nState, nState);
	copyToDevice(Mp2, hMp2, nDis*pHorizon, nState);
	copyToDevice(Mp3, hMp3, nDis*pHorizon, nDis*pHorizon);
	copyToDevice(Mp4, hMp4, 1, nState);
	copyToDevice(Mp5, hMp5, 1, nDis*pHorizon);
	copyToDevice(Mp6, hMp6, 1,1);
	copyToDevice(Gp, hGp, 4*pHorizon*nInput, nInput*pHorizon);
	copyToDevice(Kp, hKp, 1,4*pHorizon*nInput);
	copyToDevice(Z, hZ, nOutput*pHorizon, nState);
	copyToDevice(D, hD, nDis*pHorizon,1);
	copyToDevice(theta, htheta, nOutput*pHorizon, nDis*pHorizon);
	copyToDevice(x, hx, nState, 1);

	computeFp(Fp, Fp1, Fp2, Fp3, D, x);
	computeMp(Mp, Mp1, Mp2, Mp3, Mp4, Mp5, Mp6, D, x);
	// printf("Mp %f\n", Mp[0]);
	// printf("er\n");
//	printMat<<<1,1>>>(Mp,1,1);
	// matrices and vectors required for dual form of QP
	float *Qd = newMatrixCUDA(N,N);
	float *Fd = newMatrixCUDA(N,1);
	float *Md = newMatrixCUDA(1,1);	
	float *Y  = newMatrixCUDA(N,1);
	float *U  = newMatrixCUDA(M,1);
	// printf("er\n");
	convertToDual(Qd, Fd, Md, Qp_inv, Gp, Kp, Fp, Mp, N, M);
	// printf("Qd\n");
	// for(int i=0;i<N;i++)
	// {
	// 	for(int j=0;j<N;j++)
	// 	{
	// 		printf("%f ", Qd[i*N+j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("Fd\n");
	// printf("%f\n", Md[0]);
	// for(int i=0;i<N;i++)
	// {
	// 	printf("%f ", Fp[i]);
	// }
	// printf("\n");
	solveQuadraticDual(Y, Qd, Fd, Md, U, Qp, Qp_inv, Fp, Mp, Gp, Kp, N, M);
	// printf("erer\n");

	computeUfromY(U, Y, Fp, Gp, Qp_inv, N, M);

	// U[0] = -6.399018;
	// U[1] = -10.648726;
	// U[2] = -4.792378;
	// U[3] = -7.033428;
	// U[4] = -4.792378;
	// U[5] = -10.648726;
	// U[6] = -6.399018;

	// U[0] = -6.398985;
	// U[1] = -10.646729;
	// U[2] = -4.792132;
	// U[3] = -7.027614;
	// U[4] = -4.792255;
	// U[5] = -10.643004;
	// U[6] = -6.398996;

	float Jp = computeCost(U, Qp, Fp, Mp, M);
	float Jd = computeCost(Y, Qd, Fd, Md, N);

	printf("Jp = %f\n", Jp);
	printf("Jd = %f\n", Jd);
	
	float *hU = newMatrix(M,1);
	float *hY = newMatrix(N,1);

	copyToHost(hU,U,M,1);
	copyToHost(hY,Y,N,1);

	// printf("Printing Y*\n");
	// for(int i=0;i<N;i++)
	// {
	// 	printf("%f\n", hY[i]);
	// }
	printf("Printing U*\n");
	for(int i=0;i<M;i++)
	{
		printf("\t%f\n", hU[i]);
	}

	free(hQp_inv);
	free(hQp);
	free(hFp1);
	free(hFp2);
	free(hFp3);
	free(hMp1);
	free(hMp2);
	free(hMp3);
	free(hMp4);
	free(hMp5);
	free(hMp6); 
	free(hFp);
	free(hMp);
	free(hGp);
	free(hKp);
	free(hx);
	free(hD);
	free(htheta);
	free(hZ);

	cudaFree(Qp_inv);
	cudaFree(Qp);
	cudaFree(Fp1);
	cudaFree(Fp2);
	cudaFree(Fp3);
	cudaFree(Mp1);
	cudaFree(Mp2);
	cudaFree(Mp3);
	cudaFree(Mp4);
	cudaFree(Mp5);
	cudaFree(Mp6); 
	cudaFree(Fp);
	cudaFree(Mp);
	cudaFree(Gp);
	cudaFree(Kp);
	cudaFree(x);
	cudaFree(D);
	cudaFree(theta);
	cudaFree(Z);
	
	cudaFree(Qd);
	cudaFree(Fd);
	cudaFree(Md);
	cudaFree(Y);
	cudaFree(U);
}
