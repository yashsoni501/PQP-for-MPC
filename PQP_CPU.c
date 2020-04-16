#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define pHorizon 1
#define nState 29
#define nInput 7
#define nOutput 7
#define nDis 1

#define erc 1e-6
#define eac 1e-6
#define eaj 1e-6
#define erj 1e-6

#define NUM_ITER 1000

float max(float a, float b)
{
	if(a>b)	return a;
	else return b;
}

void initMat(float *mat, float val, int N)								// parallel
{
	for(int i=0;i<N;i++)	
	{
		mat[i] = val;
	}
}

float *newMatrix(int n, int m)			
{
	float *tmp = (float *)malloc(n*m*sizeof(float));
	initMat(tmp, 0, n*m);
	return tmp;
}

void copyMatrix(float *output, float *mat, int a, int b)		// parallel
{
	for(int i=0;i<a*b;i++)
	{
		output[i] = mat[i];
	}
}

float transpose(float *mat, int tr, int i, int j, int n, int m)
{
	if(tr)
		return mat[j*n+i];
	else 
		return mat[i*m+j];
}

// void transposeMatrix(float *mat, int m, int n)
// {
// 	float *tps = newMatrix(n,m);
// 	for(int i=0;i<n;i++)
// 	{
// 		for(int j=0;j<m;j++)
// 		{
// 			tps[i*m+j] = mat[j*n+i];
// 		}
// 	}
// 	float *tmp = mat;
// 	mat = tps;

// 	// free(tmp);
// }

void matrixMultiply(float *output, float *mat1, int transpose1, float *mat2, int transpose2, int a, int b, int c) 		//mat1-a*b	mat2-b*c 	// parallel
{
	float *tmp = newMatrix(a,c);

	for(int i=0;i<a;i++)
	{
		for(int j=0;j<c;j++)
		{
			for(int k=0;k<b;k++)
			{
				tmp[i*c+j] += transpose(mat1, transpose1, i, k, a, b) * transpose(mat2, transpose2, k, j, b, c);
			}
		}
	}

	copyMatrix(output, tmp, a,c);

	free(tmp);
}

void matrixAdd(float *A, float *B, int sign, int a, int b) 			// adds b to a 	// parallel
{
	for(int i=0;i<a*b;i++)
	{
		A[i] += sign * B[i];
	}
}

void negateMatrix(float *mat, int n, int m)			// parallel
{
	for(int i=0;i<n*m;i++)
	{
		mat[i] = -mat[i];
	}
}

void matrixPos(float *mat1, float *mat2, int n, int m)			// parallel
{
	for(int i=0;i<n*m;i++)
	{
		mat1[i] = max(0.0, mat2[i]);

	}
}

void matrixNeg(float *mat1, float *mat2, int n, int m)			// parallel
{
	for(int j=0;j<n*m;j++)
	{
		mat1[j] = max(0.0, -mat2[j]);
	}
}

void isSymmetric(float *mat, int N, int b)			// parallel
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

void diagonalAdd(float *theta, float *tmp, int N)			// parallel
{
	for(int i=0;i<N;i++)
	{
		// printf("tmp %f\n",tmp[i]);
		theta[i*N+i] = max(tmp[i],5);
	}
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

int compare(float *GpU, float *Kp, int *re, int N)				// parallel
{
	for(int i=0;i<N;i++)
	{
		if(GpU[i]>Kp[i]+max(erc*Kp[i], eac))
		{
			*re = 0;
		}
	}
}

void computeUfromY(float *U, float *Y, float *Fp, float *Gp, float *Qp_inv, int N, int M)
{
	float *tmp = newMatrix(M,1);
	matrixMultiply(tmp, Gp, 1, Y, 0, M, N, 1);
	matrixAdd(tmp, Fp, 1, M, 1);
	matrixMultiply(U, Qp_inv, 0, tmp, 0, M, M, 1);
	negateMatrix(U, M, 1);
	free(tmp);
}

void computeFp(float *Fp, float *Fp1, float *Fp2, float *Fp3, float *D, float *x)
{
	matrixMultiply(Fp, Fp1, 0, D, 0, nInput*pHorizon, nDis*pHorizon, 1);
	float *Fp2x = newMatrix(nInput*pHorizon,1);
	matrixMultiply(Fp2x, Fp2, 0, x, 0, nInput*pHorizon, nState, 1);
	matrixAdd(Fp, Fp2x, 1, nInput*pHorizon, 1);
	matrixAdd(Fp, Fp3, -1, nInput*pHorizon, 1);
	
	free(Fp2x);	
	// for(int i=0;i<nInput*pHorizon;i++)
	// {
	// 	printf("%f\n", Fp[i]);
	// }
	// printf("\n");
	// printf("%d\n", Fp);
}

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
	matrixMultiply(Md, Fp, 1, Qp_inv, 0, 1, M, M);
	matrixMultiply(Md, Md, 0, Fp, 0, 1, M, 1);
	Md[0] -= Mp[0];
}

void convertToDual(float *Qd, float *Fd, float *Md, float *Qp_inv, float *Gp, float *Kp, float *Fp, float *Mp, int N, int M)
{	
	float *Gp_Qp_inv = newMatrix(N,M);
	matrixMultiply(Gp_Qp_inv, Gp, 0, Qp_inv, 0, N, M, M);
	computeQd(Qd, Gp_Qp_inv, Gp, N, M);
	computeFd(Fd, Gp_Qp_inv, Fp, Kp, N, M);
	computeMd(Md, Fp, Qp_inv, Mp, N, M);

	free(Gp_Qp_inv);
}

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

void updateY1(float *Y_next, float *Y, float alphaY, float *ph, int N)
{
	copyMatrix(Y_next, Y, N, 1);
	matrixAdd(Y_next, ph, alphaY, N, 1);
}

void updY(float *Y_next, float *numerator, float *denominator, float *Y, int N)   // parallel
{
	for(int i=0;i<N;i++)
	{
		// if(denominator[i]<1e-6)	
		// {
		// printf("%f %f %f\n",Y[i], numerator[i], denominator[i]);
		// }
		Y_next[i] = numerator[i]/denominator[i]*Y[i];
	}
}

void updateY2(float *Y_next, float *Y, float *Qdp_theta, float *Qdn_theta, float *Fd, int N)
{
	float *numerator = newMatrix(N,1);
	float *denominator = newMatrix(N,1);

	matrixMultiply(numerator, Qdn_theta, 0, Y, 0, N, N, 1);
	matrixMultiply(denominator, Qdp_theta, 0, Y, 0, N, N, 1);

	float *Fdn = newMatrix(N,1);
	float *Fdp = newMatrix(N,1);

	matrixPos(Fdp, Fd, N, 1);
	matrixNeg(Fdn, Fd, N, 1);

	matrixAdd(numerator, Fdn, 1, N, 1);
	matrixAdd(denominator, Fdp, 1, N, 1);

	updY(Y_next, numerator, denominator, Y, N);

	free(numerator);
	free(denominator);
}

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
	if(!re) printf("0\n");
	free(tmp);
	return re;
}	

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

int terminate(float *Y, float *Qd, float *Fd, float *Md, float *U, float *Qp, float *Qp_inv, float *Fp, float *Mp, float *Gp, float *Kp, int N, int M)
{
	computeUfromY(U, Y, Fp, Gp, Qp_inv, N, M);

	if(!checkFeas(U, Gp, Kp, N, M))	return 0;

	float Jd = computeCost(Y, Qd, Fd, Md, N);
	float Jp = computeCost(U, Qp, Fp, Mp, M);

	if(Jp+Jd>eaj)	return 0;
	if((Jp+Jd)/fabs(Jd)>erj) return 0;

	return 1;
}

void solveQuadraticDual(float *Y, float *Qd, float *Fd, float *Md, float *U, float *Qp, float *Qp_inv, float *Fp, float *Mp, float *Gp, float *Kp, int N, int M)
{
	float *theta = newMatrix(N,N);
	float *Qdp_theta = newMatrix(N,N);
	float *Qdn_theta = newMatrix(N,N);
	float *Y_next = newMatrix(N,1);

	computeTheta(theta, Qd, N);
	computeQdp_theta(Qdp_theta, Qd, theta, N);
	computeQdn_theta(Qdn_theta, Qd, theta, N);

	initMat(Y, 10000.0, N);
	// for(int i=0;i<N;i++) Y[i] = i+1;
	int term1=0, term2 = 0;

	float *ph = newMatrix(N,1);
	long int h=1;
	float alphaY=0;

	// while(h<NUM_ITER)
	while(!terminate(Y, Qd, Fd, Md, U, Qp, Qp_inv, Fp, Mp, Gp, Kp, N, M))
	{	
		// if(h>100000) break;
		if(1)
		{
			//update
			// printf("here\n");
			updateY2(Y_next, Y, Qdp_theta, Qdn_theta, Fd, N);			
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

	free(theta);
	free(Qdp_theta);
	free(Qdn_theta);
	free(Y_next);
	free(ph);
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


	// float *tmp1 = newMatrix(10,10);
	// float *tmp2 = newMatrix(10,10);
	// float *tmp = NULL;
	// for(int i=0;i<10;i++)
	// {
	// 	tmp1[11*i] = i;
	// 	tmp2[11*i] = i;
	// }
	// matrixMultiply(tmp, tmp1, 0, tmp2, 0, 10, 10, 10);
	// for(int i=0;i<10;i++)
	// {
	// 	for(int j=0;j<10;j++)
	// 	{
	// 		printf("%f ",tmp[10*i+j]);
	// 	}
	// 	printf("\n");
	// }
	// return 0;
	int N, M;

	M = pHorizon*nInput;
	N = 4*pHorizon*nInput;

	float *Qp_inv = newMatrix(M,M);
	float *Qp = newMatrix(M,M);

	float *Fp1;
	float *Fp2;
	float *Fp3;

	float *Mp1;
	float *Mp2;
	float *Mp3;
	float *Mp4;
	float *Mp5;
	float *Mp6;

	float *Fp = newMatrix(nInput*pHorizon,1);
	float *Mp = newMatrix(1,1);
	float *Gp;
	float *Kp;
	float *x;
	float *D; 
	float *theta; 
	float *Z; 

	Fp1 = newMatrix(nInput*pHorizon, nDis*pHorizon);
	Fp2 = newMatrix(nInput*pHorizon, nState);
	Fp3 = newMatrix(1, nInput*pHorizon);
	Mp1 = newMatrix(nState, nState);
	Mp2 = newMatrix(nDis*pHorizon, nState);
	Mp3 = newMatrix(nDis*pHorizon, nDis*pHorizon);
	Mp4 = newMatrix(1, nState);
	Mp5 = newMatrix(1, nDis*pHorizon);
	Mp6 = newMatrix(1,1);
	Gp = newMatrix(4*pHorizon*nInput, nInput*pHorizon);
	Kp = newMatrix(1,4*pHorizon*nInput);
	Z = newMatrix(nOutput*pHorizon, nState);
	theta = newMatrix(nOutput*pHorizon, nDis*pHorizon);
	D = newMatrix(nDis*pHorizon,1);
	x = newMatrix(nState, 1);

	input(Qp_inv, Fp1, Fp2, Fp3, Mp1, Mp2, Mp3, Mp4, Mp5, Mp6, Gp, Kp, x, D, theta, Z);
	Gauss_Jordan(Qp_inv, Qp, M);

	computeFp(Fp, Fp1, Fp2, Fp3, D, x);
	computeMp(Mp, Mp1, Mp2, Mp3, Mp4, Mp5, Mp6, D, x);
	// printf("Mp %f\n", Mp[0]);
	// printf("er\n");
	// matrices and vectors required for dual form of QP
	float *Qd = newMatrix(N,N);
	float *Fd = newMatrix(N,1);
	float *Md = newMatrix(1,1);	
	float *Y  = newMatrix(N,1);
	float *U  = newMatrix(M,1);
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

	// printf("Printing Y*\n");
	// for(int i=0;i<N;i++)
	// {
	// 	printf("%f\n", Y[i]);
	// }
	printf("Printing U*\n");
	for(int i=0;i<M;i++)
	{
		printf("\t%f\n", U[i]);
	}

	free(Qp_inv);
	free(Qp);
	free(Fp1);
	free(Fp2);
	free(Fp3);
	free(Mp1);
	free(Mp2);
	free(Mp3);
	free(Mp4);
	free(Mp5);
	free(Mp6); 
	free(Fp);
	free(Mp);
	free(Gp);
	free(Kp);
	free(x);
	free(D);
	free(theta);
	free(Z);
}