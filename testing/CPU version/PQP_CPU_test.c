/**************************************************************************
* This file contains implementation of pqp (parallel quadratic programming)
* CPU version for MPC Term Project of HP3 Course.
* Group 7 CSE Dept. IIT KGP
*	Objective function: 1/2 U'QpU + Fp'U + 1/2 Mp
*	Constraints: GpU <= Kp
**************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define pHorizon 1
#define nState 29
#define nInput 7
#define nOutput 7
#define nDis 1

#define erc 7
#define eac 100000
#define eaj 100000
#define erj 7

#define NUM_ITER 100


/**************************************************************************
* This is utility function used to find maximum
*   1. Parameter is float type
*   2. Return float type max value
**************************************************************************/
float max(float a, float b)
{
	if(a>b)	return a;
	else return b;
}

/**************************************************************************
* This is utility function initialize the matrix
*   1. Parameter is float type matrix pointer (*mat), float val, 
*		size of matrix 
*   2. Return type void
**************************************************************************/
void initMat(float *mat, float val, int N)								
{
	for(int i=0;i<N;i++)	
	{
		mat[i] = val;
	}
}

/**************************************************************************
* This is utility function for create new  matrix
*   1. Parameter is (int n, int m) dimension of (n X m matrix) , 
*	2. Return pointer of new matrix
*   3. This function create dynamic size matrix using malloc
**************************************************************************/
float *newMatrix(int n, int m)			
{
	float *tmp = (float *)malloc(n*m*sizeof(float));
	initMat(tmp, 0, n*m);
	return tmp;
}

/**************************************************************************
* This is utility function for making copy of a matrix
*   1. Parameter is (pointer of output, pointer of input int n, int m)
*		dimension of (n X m matrix) , 
*	2. Return pointer of new matrix   
**************************************************************************/
void copyMatrix(float *output, float *mat, int a, int b)		
{
	for(int i=0;i<a*b;i++)
	{
		output[i] = mat[i];
	}
}

/**************************************************************************
* This is utility function generate transpose of matrix
*   1. Parameter is (pointer of mat, pointer of input int n, int m)
*		dimension of (n X m matrix) 
**************************************************************************/
void matrixMultiply(float *output, float *mat1, int transpose1, float *mat2, int transpose2, int a, int b, int c) 		//mat1-a*b	mat2-b*c 	
{
	float *tmp = newMatrix(a,c);

	if(!transpose1 && !transpose2)
	{
		for(int i=0;i<a;i++)
		{
			for(int j=0;j<c;j++)
			{
				for(int k=0;k<b;k++)
				{
					tmp[i*c+j] += mat1[i*b+k] * mat2[k*c+j];
				}
			}
		}
	}

	if(transpose1 && !transpose2)
	{
		for(int i=0;i<a;i++)
		{
			for(int j=0;j<c;j++)
			{
				for(int k=0;k<b;k++)
				{
					tmp[i*c+j] += mat1[k*a+i] * mat2[k*c+j];
				}
			}
		}
	}

	if(!transpose1 && transpose2)
	{
		for(int i=0;i<a;i++)
		{
			for(int j=0;j<c;j++)
			{
				for(int k=0;k<b;k++)
				{
					tmp[i*c+j] += mat1[i*b+k] * mat2[j*b+k];
				}
			}
		}
	}

	if(transpose1 && transpose2)
	{
		for(int i=0;i<a;i++)
		{
			for(int j=0;j<c;j++)
			{
				for(int k=0;k<b;k++)
				{
					tmp[i*c+j] += mat1[k*a+i] * mat2[j*b+k];
				}
			}
		}
	}

	copyMatrix(output, tmp, a,c);

	free(tmp);
}

/**************************************************************************
* This is utility function for generating addition or substraction 
*	of two matrix
*   1. Parameter is (pointer of matrix1, pointer of matrix2, float sign,int n int m)
*		dimension of (n X m matrix) 
*	2. sign parameters for decide addition or substraction
*	3. Result write back in matrix1
**************************************************************************/
void matrixAdd(float *A, float *B, float sign, int a, int b) 			// adds b to a 	
{
	for(int i=0;i<a*b;i++)
	{
		A[i] += sign * B[i];
	}
}
/**************************************************************************
* This is utility function for generating negation of matrix elementwise 
*	of matrix
*   1. Parameter is (pointer of matrix1,int n int m)
*		dimension of (n X m matrix)
*	2. Result write back in matrix1
**************************************************************************/
void negateMatrix(float *mat, int n, int m)			
{
	for(int i=0;i<n*m;i++)
	{
		mat[i] = -mat[i];
	}
}

/**************************************************************************
* This is utility function for generating positive of matrix elementwise 
*	of matrix
*   1. Parameter is (pointer of matrix1,pointer of matrix2,int n int m)
*		dimension of (n X m matrix)
*	2. Result in matrix1
*	3. This function utilised during Qd_plus, Fd_plus generation
*	4. This function uses max function defined above
**************************************************************************/
void matrixPos(float *mat1, float *mat2, int n, int m)			
{
	for(int i=0;i<n*m;i++)
	{
		mat1[i] = max(0.0, mat2[i]);

	}
}
/**************************************************************************
* This is utility function for generating negative of matrix elementwise 
*	of matrix
*   1. Parameter is (pointer of matrix1,pointer of matrix2,int n int m)
*		dimension of (n X m matrix)
*	2. Result in matrix1
*	3. This function utilised during Qd_minus, Fd_minus generation
*	4. This function uses max function defined above
**************************************************************************/
void matrixNeg(float *mat1, float *mat2, int n, int m)			
{
	for(int j=0;j<n*m;j++)
	{
		mat1[j] = max(0.0, -mat2[j]);
	}
}

void isSymmetric(float *mat, int N, int b)			
{
	for(int i=0;i<N;i++)
	{
		for(int j=i+1;j<N;j++)
		{
			if(mat[i*N + j] != mat[j*N + i])
			{
				b = 0;
			}
		}
	}
}
/**************************************************************************
* This is utility function for diagonal addition matrix elementwise 
*	of matrix
*   1. Parameter is (pointer of matrix1,pointer of matrix2,int N)
*		dimension of (n X m matrix)
*	2. Result in matrix1
*	3. This function utilised during Qd_minus+theta, Qd_plus+theta generation
*	4. This function uses max function defined above
**************************************************************************/
void diagonalAdd(float *theta, float *tmp, int N)			
{
	for(int i=0;i<N;i++)
	{
		// printf("tmp %f\n",tmp[i]);
		theta[i*N+i] = max(tmp[i],100);
	}
}
/**************************************************************************
* This is utility function for finding inversion matrix 
*   1. Parameter is (pointer of matrix1,pointer of matrix2,int N)
*		dimension of (n X m matrix)
*	2. Result in res matrix
*	3. This function utilised during Qd_minus+theta, Qd_plus+theta generation
*	4. This function uses max function defined above
**************************************************************************/
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

/**************************************************************************
* This is utility function for two matrix 
*   1. Parameter is (pointer of matrix1,pointer of matrix2,int N)
*		dimension of (n X m matrix)
*	2. Result in res matrix
**************************************************************************/
void compare(float *GpU, float *Kp, int *re, int N)				
{
	for(int i=0;i<N;i++)
	{
		if(GpU[i]>Kp[i]+max(erc*Kp[i], eac))
		{
			*re = 0;
		}
	}
}
/**************************************************************************
* This is PQP utility function for compute U from Y
*   1. Parameter is (pointer of U vector,pointer of Y vector,
		pointer of Fp, pointer of Gp, pointer of Qp_inv,int N, int M)
*		dimension of (n X m matrix)
*	2. Result in Vector U
**************************************************************************/

void computeUfromY(float *U, float *Y, float *Fp, float *Gp, float *Qp_inv, int N, int M)
{
	float *tmp = newMatrix(M,1);
	matrixMultiply(tmp, Gp, 1, Y, 0, M, N, 1);
	matrixAdd(tmp, Fp, 1, M, 1);
	matrixMultiply(U, Qp_inv, 0, tmp, 0, M, M, 1);
	negateMatrix(U, M, 1);
	free(tmp);
}

/**************************************************************************
* This is PQP utility function for compute Fp from Fp1, Fp2, Fp3
*   1. Parameter is (pointer of Fp,pointer of Fp1,
		pointer of Fp2, pointer of Fp3, pointer of D, ponter of x)
*		dimension of (n X m matrix)
*	2. Result in Fp
*   Formula for Fp generation
*		Fp = Fp1*D+Fp2*x-Fp3
*  This function uses utility function matrixMultiply(), matrixAdd() defined above
**************************************************************************/

void computeFp(float *Fp, float *Fp1, float *Fp2, float *Fp3, float *D, float *x)
{
	matrixMultiply(Fp, Fp1, 0, D, 0, nInput*pHorizon, nDis*pHorizon, 1);
	float *Fp2x = newMatrix(nInput*pHorizon,1);
	matrixMultiply(Fp2x, Fp2, 0, x, 0, nInput*pHorizon, nState, 1);
	matrixAdd(Fp, Fp2x, 1, nInput*pHorizon, 1);
	matrixAdd(Fp, Fp3, -1, nInput*pHorizon, 1);
	
	free(Fp2x);		
}
/**************************************************************************
* This is PQP utility function for compute Mp from Mp1, Mp2, Mp3, Mp4, Mp5, Mp6, D, X
*   1. Parameter is (pointer of Mp,pointer of Mp1,
		pointer of Mp2, pointer of Mp3,pointer of Mp4, pointer of Mp5,
		pointer of Mp6,pointer of D, ponter of x)

*	2. Result in Mp
*   Formula for Mp generation
*	Mp= 0.5 .* x'*Mp1*x + D'*Mp2*x+ 0.5 .*D'*Mp3*D - 0.5 .*Mp4*x - 0.5.*Mp5*D+0.5*Mp6
	[ Where, '=> transpose operation and .* => element wise multiplication]
*  This function uses utility function matrixMultiply(), matrixAdd() defined above
**************************************************************************/
void computeMp(float *Mp, float *Mp1, float *Mp2, float *Mp3, float *Mp4, float *Mp5, float *Mp6, float *D, float *x)
{
	Mp[0] = 0;

	float *tmp = newMatrix(1,nState);
	matrixMultiply(tmp, x, 1, Mp1, 0, 1, nState, nState);
	matrixMultiply(tmp, tmp, 0, x, 0, 1, nState, 1);

	Mp[0] += tmp[0]/2;

	matrixMultiply(tmp, D, 1, Mp2, 0, 1, nDis*pHorizon, nState);
	matrixMultiply(tmp, tmp, 0, x, 0, 1, nState, 1);

	Mp[0] += tmp[0]/2;

	matrixMultiply(tmp, Mp4, 1, x, 0, 1, nState, 1);

	Mp[0] += tmp[0]/2;

	free(tmp);
	tmp = newMatrix(1, nDis*pHorizon);
	matrixMultiply(tmp, D, 1, Mp3, 0, 1, nDis*pHorizon, nDis*pHorizon);
	matrixMultiply(tmp, tmp, 0, D, 0, 1, nDis*pHorizon, 1);

	Mp[0] += tmp[0]/2;

	matrixMultiply(tmp, Mp5, 1, D, 0, 1, nDis*pHorizon, 1);

	Mp[0] += tmp[0]/2;

	Mp[0] += Mp6[0]/2;

	free(tmp);
}
/**************************************************************************
* This is PQP utility function for compute Qd
*   1. Parameter is (pointer of Qd,pointer of Gp_Qp_inv,
		pointer of Gp, int N, int M)

*	2. Result in Qd
*   Formula for Qd generation
*	Qd= 
	[ Where, '=> transpose operation and .* => element wise multiplication]
*  This function uses utility function matrixMultiply(), matrixAdd() defined above
**************************************************************************/
void computeQd(float *Qd, float *Gp_Qp_inv, float *Gp, int N, int M)
{
	matrixMultiply(Qd, Gp_Qp_inv, 0, Gp, 1, N, M, N);	
}

/**************************************************************************
* This is PQP utility function for compute Fd
*   1. Parameter is (pointer of Fd,pointer of Gp_Qp_inv,
		pointer of Fp, pointer of Kp,int N, int M)

*	2. Result in Fd
*   Formula for Fd generation
*	Fd= 
	[ Where, '=> transpose operation and .* => element wise multiplication]
*  This function uses utility function matrixMultiply(), matrixAdd() defined above
**************************************************************************/
void computeFd(float *Fd, float *Gp_Qp_inv, float *Fp, float *Kp, int N, int M)
{
	matrixMultiply(Fd, Gp_Qp_inv, 0, Fp, 0, N, M, 1);
	matrixAdd(Fd, Kp, 1, N, 1);
}
/**************************************************************************
* This is PQP utility function for compute Md
*   1. Parameter is (pointer of Fd,pointer of Fp,pointer of Qp_inv,
		 pointer of Mp,int N, int M)

*	2. Result in Md
*   Formula for Fd generation
*	Md= 
	[ Where, '=> transpose operation and .* => element wise multiplication]
*  This function uses utility function matrixMultiply(), matrixAdd() defined above
**************************************************************************/
void computeMd(float *Md, float *Fp, float* Qp_inv, float* Mp, int N, int M)
{
	float *tmp = newMatrix(1,M);
	matrixMultiply(tmp, Fp, 1, Qp_inv, 0, 1, M, M);
	matrixMultiply(Md, tmp, 0, Fp, 0, 1, M, 1);
	free(tmp);
	Md[0] -= Mp[0];
}
/**************************************************************************
* This is PQP utility function for convert primal to dual form of PQP
*   1. Parameter is (pointer of Qd,pointer of Fd,pointer of Md,pointer of Qp_inv,
		 pointer of Gp,pointer of Kp,pointer of Fp,pointer of Mp,int N, int M)

*  This function uses utility function matrixMultiply(), matrixAdd()
	computeQd(), computeFd(), computeMd()defined above
*  temp variables to keep intermediate result
**************************************************************************/
void convertToDual(float *Qd, float *Fd, float *Md, float *Qp_inv, float *Gp, float *Kp, float *Fp, float *Mp, int N, int M)
{	
	float *Gp_Qp_inv = newMatrix(N,M);
	matrixMultiply(Gp_Qp_inv, Gp, 0, Qp_inv, 0, N, M, M);
	computeQd(Qd, Gp_Qp_inv, Gp, N, M);
	computeFd(Fd, Gp_Qp_inv, Fp, Kp, N, M);
	computeMd(Md, Fp, Qp_inv, Mp, N, M);

	free(Gp_Qp_inv);
}
/**************************************************************************
* This is PQP utility function for compute theta 
*   1. Parameter is (pointer of theta,pointer of Qd,int N)
**************************************************************************/
void computeTheta(float *theta, float *Qd, int N)
{
	float *Qdn = newMatrix(N,N);
	matrixNeg(Qdn, Qd, N, N);

	float *one = newMatrix(N,1);
	initMat(one, 1, N);

	float *tmp = newMatrix(N,1);
	matrixMultiply(tmp, Qdn, 0, one, 0, N,N,1);

	diagonalAdd(theta, tmp, N);

	free(Qdn);
	free(one);
	free(tmp);
}
/**************************************************************************
* This is PQP utility function for compute Qd_plus + theta 
*   1. Parameter is (pointer of theta,pointer of Qd,int N)
**************************************************************************/
void computeQdp_theta(float *Qdp_theta, float *Qd, float *theta, int N)
{
	matrixPos(Qdp_theta, Qd, N, N);
	matrixAdd(Qdp_theta, theta, 1, N, N);
}
/**************************************************************************
* This is PQP utility function for compute Qd_minus + theta 
*   1. Parameter is (pointer of theta,pointer of Qd,int N)
**************************************************************************/
void computeQdn_theta(float *Qdn_theta, float *Qd, float *theta, int N)
{
	matrixNeg(Qdn_theta, Qd, N, N);
	matrixAdd(Qdn_theta, theta, 1, N, N);
}

/**************************************************************************
* This is PQP utility function for compute alpha Y 
*   1. Parameter is (pointer,int N)
*	2. use for acceleration

**************************************************************************/
void computealphaY(float *alphaY, float *ph, float *Qd, float *Y, float *Fd, int N)
{
	float *temp = newMatrix(1,N);

	matrixMultiply(temp, ph, 1, Qd, 0, 1, N, N);
	matrixMultiply(temp, temp, 0, ph, 0, 1, N, 1);

	if(temp[0] > 0)
	{
		float *temp2 = newMatrix(1,N);

		matrixMultiply(temp2, Y, 1, Qd, 0, 1, N, N);
		
		matrixAdd(temp2, Fd, 1, 1, N);
		
		matrixMultiply(temp2, temp2, 0, ph, 0, 1, N, 1);

		*alphaY = -temp2[0]/temp[0];

		free(temp2);
	}
	else
	{
		alphaY = 0;
	}

	free(temp);
}
/**************************************************************************
* This is PQP utility function for update Y1 
*   1. Parameter is (pointer,int N)
*	2. use for update intermediate Y during iteration

**************************************************************************/
void updateY1(float *Y_next, float *Y, float alphaY, float *ph, int N)
{
	copyMatrix(Y_next, Y, N, 1);
	matrixAdd(Y_next, ph, alphaY, N, 1);
}
/**************************************************************************
* This is PQP utility function for update Y1 
*   1. Parameter is (pointer,int N)
*	2. use for update intermediate Y during iteration

**************************************************************************/
void updY(float *Y_next, float *numerator, float *denominator, float *Y, int N)   
{
	for(int i=0;i<N;i++)
	{
		Y_next[i] = numerator[i]/denominator[i]*Y[i];
	}
}
/**************************************************************************
* This is PQP utility function for update Y1 
*   1. Parameter is (pointer,int N)
*	2. use for update intermediate Y during iteration

**************************************************************************/
void updateY2(float *Y_next, float *Y, float *Qdp_theta, float *Qdn_theta, float *Fd, float *Fdp, float *Fdn, int N)
{
	float *numerator = newMatrix(N,1);
	float *denominator = newMatrix(N,1);

	matrixMultiply(numerator, Qdn_theta, 0, Y, 0, N, N, 1);
	matrixMultiply(denominator, Qdp_theta, 0, Y, 0, N, N, 1);

	matrixAdd(numerator, Fdn, 1, N, 1);
	matrixAdd(denominator, Fdp, 1, N, 1);

	updY(Y_next, numerator, denominator, Y, N);

	free(numerator);
	free(denominator);
}
/**************************************************************************
* This is PQP utility function for computeph for update acceleration
*   1. Parameter is (pointer,int N)
*	2. use for update intermediate Y during iteration

**************************************************************************/
void computeph(float *ph, float *Qd, float *Y, float *Fd, int N)
{
	matrixMultiply(ph, Qd, 0, Y, 0, N, N, 1);
	matrixAdd(ph, ph, 1, N, 1);
	matrixNeg(ph, ph, N, 1);
}

int checkFeas(float *U, float *Gp, float *Kp, int N, int M)
{
	float *tmp = newMatrix(N,1);
	matrixMultiply(tmp, Gp, 0, U, 0, N, M, 1);
	int re = 1;
	compare(tmp, Kp, &re, N);
	// if(!re) printf("0\n");
	free(tmp);
	return re;
}	
/**************************************************************************
* This is PQP utility function for cost
*   1. Parameter is (pointer,int N)
*	2. use for update intermediate Y during iteration

**************************************************************************/
float computeCost(float *Z, float *Q, float *F, float *M, int N)
{
	float J=0;
	float *tmp = newMatrix(1,N);
	matrixMultiply(tmp, Z, 1, Q, 0, 1, N, N);
	matrixMultiply(tmp, tmp, 0, Z, 0, 1, N, 1);

	J+=0.5*tmp[0];

	matrixMultiply(tmp, F, 1, Z, 0, 1, N, 1);

	J+=tmp[0];

	J+= M[0]/2;

	free(tmp);

	return J;
}
/**************************************************************************
* This is PQP utility function for termination condition of iteration steps
*   1. Parameter is (pointer,int N)
*	2. 

**************************************************************************/
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
/**************************************************************************
* This is PQP utility function main function for solving quadratic dual
*   1. Parameter is (pointer,int N)
*	2. use for update intermediate Y during iteration

**************************************************************************/
void solveQuadraticDual(float *Y, float *Qd, float *Fd, float *Md, float *U, float *Qp, float *Qp_inv, float *Fp, float *Mp, float *Gp, float *Kp, int N, int M)
{
	float *theta = newMatrix(N,N);
	float *Qdp_theta = newMatrix(N,N);
	float *Qdn_theta = newMatrix(N,N);
	float *Y_next = newMatrix(N,1);
	float *Fdn = newMatrix(N,1);
	float *Fdp = newMatrix(N,1);

	matrixPos(Fdp, Fd, N, 1);
	matrixNeg(Fdn, Fd, N, 1);

	computeTheta(theta, Qd, N);
	computeQdp_theta(Qdp_theta, Qd, theta, N);
	computeQdn_theta(Qdn_theta, Qd, theta, N);

	initMat(Y, 1000.0, N);
	// for(int i=0;i<N;i++) Y[i] = i+1;

	float *ph = newMatrix(N,1);
	long int h=1;
	float alphaY=0;

	while(h<NUM_ITER)
	// while(!terminate(Y, Qd, Fd, Md, U, Qp, Qp_inv, Fp, Mp, Gp, Kp, N, M))
	{	
		
		if(1)
		{
			// float J1 = computeCost(Y, Qd, Fd, Md, N);
			
			updateY2(Y_next, Y, Qdp_theta, Qdn_theta, Fd, Fdp, Fdn, N);			
			// float J2 = computeCost(Y_next, Qd, Fd, Md, N);
			
			// printf("J1 %f\n",J1);
			// printf("J2 %f\n",J2);
		}
		else
		{
			
			computeph(ph, Qd, Y, Fd, N);
			computealphaY(&alphaY, ph, Qd, Y, Fd, N);	
			
			updateY1(Y_next, Y, alphaY/10, ph, N);

		}
		// printf("h = %d\n", h);
		copyMatrix(Y, Y_next, N, 1);
		
		h++;
	}
	printf("Printing number of iterations = %ld\n",h);

	free(theta);
	free(Qdp_theta);
	free(Qdn_theta);
	free(Y_next);
	free(ph);
	free(Fdp);
	free(Fdn);
}

/**************************************************************************
* This utitlity function for read input data from text file
*  1. Pointer of variables use for store value
*  2. input file kept in folder (examples) in inside the same folder where codes kept
**************************************************************************/
// void input(float* qp_inv, float* Fp1, float* Fp2, float * Fp3, float * Mp1, float * Mp2, float * Mp3, float* Mp4, float* Mp5, float* Mp6, float* Gp, float* Kp, float* x, float* D, float* theta, float* Z)
// {
// 	FILE *fptr;
// 	int i,j;
// 	float num;

// 	//Fill Qp_inverse	
// 	fptr = fopen("./example/Qp_inv.txt","r");
// 	for(i=0;i<pHorizon*nInput;i++)
// 	{
// 		for(j=0;j<pHorizon*nInput;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			qp_inv[j*pHorizon*nInput+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Fp1
// 	fptr = fopen("./example/Fp1.txt","r");
// 	for(i=0;i<nDis*pHorizon;i++)
// 	{
// 		for(j=0;j<nInput*pHorizon;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			Fp1[j*nDis*pHorizon+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Fp2
// 	fptr = fopen("./example/Fp2.txt","r");
// 	for(i=0;i<nState;i++)
// 	{
// 		for(j=0;j<nInput*pHorizon;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			Fp2[j*nState+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Fp3
// 	fptr = fopen("./example/Fp3.txt","r");
// 	for(j=0;j<nInput*pHorizon;j++)
// 	{
// 		fscanf(fptr,"%f", &num);
// 		Fp3[j] = num;
// 	}
// 	fclose(fptr);

// 	//Fill Mp1
// 	fptr = fopen("./example/Mp1.txt","r");
// 	for(i=0;i<nState;i++)
// 	{
// 		for(j=0;j<nState;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			Mp1[j*nState+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Mp2
// 	fptr = fopen("./example/Mp2.txt","r");
// 	for(i=0;i<nState;i++)
// 	{
// 		for(j=0;j<nDis*pHorizon;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			Mp2[j*nState+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Mp3
// 	fptr = fopen("./example/Mp3.txt","r");
// 	for(i=0;i<nDis*pHorizon;i++)
// 	{
// 		for(j=0;j<nDis*pHorizon;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			Mp3[j*nDis*pHorizon+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Mp4
// 	fptr = fopen("./example/Mp4.txt","r");
// 	for(i=0;i<nState;i++)
// 	{
// 		fscanf(fptr,"%f", &num);
// 		Mp4[i] = num;
// 	}
// 	fclose(fptr);

// 	//Fill Mp5
// 	fptr = fopen("./example/Mp5.txt","r");
// 	for(i=0;i<nDis*pHorizon;i++)
// 	{
// 		fscanf(fptr,"%f", &num);
// 		Mp5[i] = num;
// 	}
// 	fclose(fptr);

// 	//Fill Mp6
// 	fptr = fopen("./example/Mp6.txt","r");
// 	fscanf(fptr,"%f", &num);
// 	Mp6[0] = num;
// 	fclose(fptr);

// 	//Fill Gp
// 	fptr = fopen("./example/Gp.txt","r");
// 	for(i=0;i<pHorizon*nInput;i++)
// 	{
// 		for(j=0;j<4*pHorizon*nInput;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			Gp[j*pHorizon*nInput+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Kp
// 	fptr = fopen("./example/Kp.txt","r");
// 	for(i=0;i<4*pHorizon*nInput;i++)
// 	{
// 		fscanf(fptr,"%f", &num);
// 		Kp[i] = num;
// 	}
// 	fclose(fptr);

// 	//Fill Z
// 	fptr = fopen("./example/Z.txt","r");
// 	for(i=0;i<nState;i++)
// 	{
// 		for(j=0;j<nOutput*pHorizon;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			Z[j*nState+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill Theta
// 	fptr = fopen("./example/Theta.txt","r");
// 	for(i=0;i<nDis*pHorizon;i++)
// 	{
// 		for(j=0;j<nOutput*pHorizon;j++)
// 		{
// 			fscanf(fptr,"%f", &num);
// 			theta[j*nDis*pHorizon+i] = num;
// 		}
// 	}
// 	fclose(fptr);

// 	//Fill D
// 	fptr = fopen("./example/D.txt","r");
// 	for(i=0;i<nDis*pHorizon;i++)
// 	{
// 		fscanf(fptr,"%f", &num);
// 		D[i] = num;
// 	}
// 	fclose(fptr);

// 	//Fill x
// 	fptr = fopen("./example/x.txt","r");
// 	for(i=0;i<nState;i++)
// 	{
// 		fscanf(fptr,"%f", &num);
// 		x[i] = num;
// 	}
// 	fclose(fptr);
// }

void input(float *Qp_inv, float *Fp, float *Mp, float *Gp, float *Kp, float *x, float *D, float *theta, float *Z, int N, int M, FILE *fp)
{
	for(int i=0;i<M;i++)
	{
		fscanf(fp, "%f", &Qp_inv[i*M+i]);
	}

	for(int i=0;i<M;i++)
	{
		fscanf(fp, "%f", &Fp[i]);
	}

 	fscanf(fp, "%f", Mp);

	for(int i=0;i<N;i++)
	{
		fscanf(fp, "%f", &Kp[i]);
	}

	for(int i=0;i<N;i++)
	{
		Kp[i] = fabs(10.0*rand()/RAND_MAX);
		for(int j=0;j<M;j++)
		{
			int tmp;
			fscanf(fp, "%d",&tmp);
			if(tmp%3 == 0)
			{
				Gp[i*M+j] = 0;
			}
			else if(tmp%3==2)
			{
				Gp[i*M+j] = -1;
			}
			else
			{
				Gp[i*M+j] = 1;
			}
		}
	}
}

/**************************************************************************
* This driver function
**************************************************************************/
int main(int argc, char *argv[])
{
	
	int N=1000, M=500;
	// FILE *fp;
	// fp = fopen(argv[1], "r");
	// fscanf(fp, "%d%d", &M, &N);

	float *Qp_inv = newMatrix(M,M);
	float *Qp = newMatrix(M,M);

	float *Fp = newMatrix(M,1);
	float *Mp = newMatrix(1,1);
	float *Gp;
	float *Kp;
	float *x;
	float *D; 
	float *theta; 
	float *Z; 

	float *Fp1;
	float *Fp2;
	float *Fp3;

	float *Mp1;
	float *Mp2;
	float *Mp3;
	float *Mp4;
	float *Mp5;
	float *Mp6;

	Gp = newMatrix(N,M);
	Kp = newMatrix(N,1);
	Z = newMatrix(nOutput*pHorizon, nState);
	theta = newMatrix(nOutput*pHorizon, nDis*pHorizon);
	D = newMatrix(nDis*pHorizon,1);
	x = newMatrix(nState, 1);

	float *Qd = newMatrix(N,N);
	float *Fd = newMatrix(N,1);
	float *Md = newMatrix(1,1);	
	float *Y  = newMatrix(N,1);
	float *U  = newMatrix(M,1);

	// input(Qp_inv, Fp, Mp, Gp, Kp, x, D, theta, Z, N, M, fp);
	// fclose(fp);
	Gauss_Jordan(Qp_inv, Qp, M);

	// computeFp(Fp, Fp1, Fp2, Fp3, D, x);
	// computeMp(Mp, Mp1, Mp2, Mp3, Mp4, Mp5, Mp6, D, x);
	
	convertToDual(Qd, Fd, Md, Qp_inv, Gp, Kp, Fp, Mp, N, M);
	
	solveQuadraticDual(Y, Qd, Fd, Md, U, Qp, Qp_inv, Fp, Mp, Gp, Kp, N, M);
	

	computeUfromY(U, Y, Fp, Gp, Qp_inv, N, M);


	float Jp = computeCost(U, Qp, Fp, Mp, M);
	float Jd = computeCost(Y, Qd, Fd, Md, N);

	printf("Jp = %f\n", Jp);
	printf("Jd = %f\n", Jd);

	
	printf("Printing U*\n");
	for(int i=0;i<M;i++)
	{
		printf("\t%f\n", U[i]);
	}
	// code below for free all dynamic memory used before exist from program
	free(Qp_inv);
	free(Qp);
	
	free(Fp);
	free(Mp);
	free(Gp);
	free(Kp);
	

	free(Qd);
	free(Fd);
	free(Md);
	free(Y);
	free(U);
}
