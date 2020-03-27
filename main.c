#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define erc 0.1
#define eac 0.1

double max(double a, double b)
{
	if(a>b)	return a;
	else return b;
}

double *newVector(int n)
{
	double *tmp = (double *)malloc(n*sizeof(double));
	for(int i=0;i<n;i++)
	{
		tmp[i] = 0.00;
	}
	return tmp;
}

double **newMatrix(int n, int m)
{
	double **tmp = (double **)malloc(n*sizeof(double *));
	for(int i=0;i<n;i++)
	{
		tmp[i] = newVector(m);
	}
	return tmp;
}

void deleteVector(double *vec)
{
	free(vec);
}

void deleteMatrix(double **Mat, int m)
{
	for(int i=0;i<m;i++)
	{
		free(Mat[i]);
	}
	free(Mat);
}

void computeInv(double **invMat, double **Mat, int N)
{

}

void copyMatrix(double **output, double **mat, int a, int b)
{
	for(int i=0;i<a;i++)
	{
		for(int j=0;j<b;j++)
		{
			output[i][j] = mat[i][j];
		}
	}
}

double transpose(double **mat, int tr, int i, int j)
{
	if(tr)
		return mat[j][i];
	else 
		return mat[i][j];
}

void matrixMultiply(double **output, double **mat1, int transpose1, double **mat2, int transpose2, int a, int b, int c) 		//mat1-a*b	mat2-b*c
{
	double **tmp = newMatrix(a,c);

	for(int i=0;i<a;i++)
	{
		for(int j=0;j<b;j++)
		{
			for(int k=0;k<c;k++)
			{
				tmp[i][j] += transpose(mat1, transpose1, i, j) * transpose(mat2, transpose2, j, k);
			}
		}
	}

	copyMatrix(output, mat1, a, c);
	deleteMatrix(tmp, c);
}

void matrixAdd(double **A, double **B, int sign, int a, int b) 			// adds b to a
{
	for(int i=0;i<a;i++)
	{
		for(int j=0;j<b;j++)
		{
			A[i][j] += sign * B[i][j];
		}
	}
}

void negateMatrix(double **mat, int n, int m)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			mat[i][j] = -mat[i][j];
		}
	}
}

void matrixPos(double **mat1, double **mat2, int n, int m)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			mat1[i][j] = max(0.0, mat2[i][j]);
		}
	}
}

void matrixNeg(double **mat1, double **mat2, int n, int m)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			mat1[i][j] = max(0.0, -mat2[i][j]);
		}
	}
}

void getCofactor(double **mat, double **temp, int p, int q, int n) 
{ 
    int i = 0, j = 0; 
  
    for (int row = 0; row < n; row++) 
    { 
        for (int col = 0; col < n; col++) 
        {  
            if (row != p && col != q) 
            { 
                temp[i][j++] = mat[row][col]; 
   
                if (j == n - 1) 
                { 
                    j = 0; 
                    i++; 
                } 
            } 
        } 
    } 
} 

double determinantOfMatrix(double **mat, int n) 
{ 
    double D = 0;
  
    if (n == 1) 
        return mat[0][0]; 
  
    double **temp = newMatrix(n-1,n-1); 
    int sign = 1;   

    for (int f = 0; f < n; f++) 
    { 
        getCofactor(mat, temp, 0, f, n); 
        D += sign * mat[0][f] * determinantOfMatrix(temp, n - 1); 
  
        sign = -sign; 
    } 

    deleteMatrix(temp, n-1);
    return D; 
} 

void isSymmetric(double **mat, int N, int &b)
{
	for(int i=0;i<N;i++)
	{
		for(int j=i+1;j<N;j++)
		{
			if(mat[i][j] != mat[j][i])
			{
				b = 0;
			}
		}
	}
}

void computeQd(double **Qd, double **Gp, double **Qp_inv, int N)
{
	matrixMultiply(Qd, Gp, 0, Qp_inv, 0, N, N, N);
	matrixMultiply(Qd, Qd, 0, Gp, 1, N, N, N);	
}

void computeSd(double **Sd, double **Gp, double **Qp_inv, double ** Cp, double **Sp, int N)
{
	matrixMultiply(Sd, Gp, 0, Qp_inv, 0, N, N, N);
	matrixMultiply(Sd, Sd, 0, Cp, 0, N, N, N);
	matrixAdd(Sd, Sp, 1, N, N);
}

void computeWd(double **Wd, double **Wp, int N)
{
	copyMatrix(Wd, Wp, N, 1);
}

void computeOd(double **Od, double **Cp, double **Qp_inv, double **Op, int N)
{
	matrixMultiply(Od, Cp, 1, Qp_inv, 0, N, N, N);
	matrixMultiply(Od, Od, 0, Cp, 0, N, N, N);
	matrixAdd(Od, Op, -1, N, N);
}

void computeEd(double **Ed, double **Qp_inv, double **Gp, int N)
{
	matrixMultiply(Ed, Qp_inv, 0, Gp, 0, N, N, N);
	negateMatrix(Ed, N, N);
}

void convertFromParametricToDual(double **Qd, double **Sd, double **Wd, double **Od, double **Qp, double **Cp, double **Op, double **Gp, double **Sp, double **W, double **Td, double **Ed, int N)
{	
	double **Qp_inv = newMatrix(N,N);
	computeInv(Qp_inv, Qp, N);
	computeQd(Qd, Gp, Qp_inv, N);
	computeSd(Sd, Gp, Qp_inv, Cp, Sp, N);
	computeWd(Wd, W, N);
	computeOd(Od, Cp, Qp_inv, Op, N);
	computeEd(Ed, Qp_inv, Gp, N);
	computeEd(Td, Qp_inv, Cp, N);		// computeTd and computeEd are of same form
	deleteMatrix(Qp_inv, N);
}

void computeFd(double **Fd, double **Sd, double **x, double **Wd, int N)
{
	matrixMultiply(Fd, Sd, 0, x, 0, N, N, 1);
	matrixAdd(Fd, Wd, 1, N, 1);
}

void computeTheta(double **theta, double **Qd, int N)
{
	for(int i=0;i<N;i++)
	{
		theta[i][i] = 2*max(0, -Qd[i][i]) + 1.0;
	}
}

void computeQdp_theta(double **Qdp_theta, double **Qd, double **theta, int N)
{
	matrixPos(Qdp_theta, Qd, N, N);
	matrixAdd(Qdp_theta, theta, 1, N, N);
}

void computeQdn_theta(double **Qdn_theta, double **Qd, double **theta, int N)
{
	matrixNeg(Qdn_theta, Qd, N, N);
	matrixAdd(Qdn_theta, theta, 1, N, N);
}

void initializeY(double **Y, int N)
{
	for(int i=0;i<N;i++)
	{
		Y[i][0] = i+1;
	}
}

int isEqual(double **a, double **b, int n, int m)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
			if(fabs(a[i][j]-b[i][j]) > 1e-7)
			{
				return 0;
			}
	}

	return 1;
}

void computealphaY(double &alphaY, double **ph, double **Qd, double **Y, double **Fd, int N)
{
	double **temp = newMatrix(1,N);

	matrixMultiply(temp, ph, 1, Qd, 0, 1, N, N);
	matrixMultiply(temp, temp, 0, ph, 0, 1, N, 1);

	if(temp[0][0] > 0)
	{
		double **temp2 = newMatrix(1,N);

		matrixMultiply(temp2, Y, 1, Qd, 0, 1, N, N);
		
		for(int i=0;i<N;i++)
		{
			temp2[0][i] += Fd[i][0];
		}
		
		matrixMultiply(temp2, temp2, 0, ph, 0, 1, N, 1);

		alphaY = -temp2[0][0]/temp[0][0];

		deleteMatrix(temp2, N);
	}
	else
	{
		alphaY = 0;
	}

	deleteMatrix(temp, N);
}

void updateY1(double **Y_next, double **Y, double alphaY, double **ph, int N)
{
	for(int i=0;i<N;i++)
	{
		Y_next[i][0] = Y[i][0] + alphaY * ph[i][0];
	}
}

void updateY2(double **Y_next, double **Y, double **Qdp_theta, double **Qdn_theta, double **Fd, int N)
{
	double **numerator = newMatrix(N,1);
	double **denominator = newMatrix(N,1);

	matrixMultiply(numerator, Qdn_theta, 0, Y, 0, N, N, 1);
	matrixMultiply(denominator, Qdp_theta, 0, Y, 0, N, N, 1);

	double **Fdn = newMatrix(N,1);
	double **Fdp = newMatrix(N,1);

	matrixPos(Fdp, Fd, N, 1);
	matrixNeg(Fdn, Fd, N, 1);

	matrixAdd(numerator, Fdn, 1, N, N);
	matrixAdd(denominator, Fdp, 1, N, N);

	for(int i=0;i<N;i++)
	{
		Y_next[i][0] = numerator[i][0]/denominator[i][0]*Y[i][0];
	}
}

void computeph(double **ph, double **Qd, double **Y, double **Fd, int N)
{
	matrixMultiply(ph, Qd, 0, Y, 0, N, N, 1);
	matrixAdd(ph, ph, 1, N, 1);
	matrixNeg(ph, ph, N, 1);
}

void updateTerminationCondition1(int &term1, double **Sd, double **Wd, double **Qd, double **Y, double **Sp, double **x, double **Wp, int N)
{	
	double **lhs = newMatrix(N,1);

	matrixMultiply(lhs, Sd, 0, x, 0, N, N, 1);
	matrixAdd(lhs, Wd, 1, N, 1);

	double **tmp = newMatrix(N,1);
	matrixMultiply(tmp, Qd, 0, Y, 0, N, N, 1);
	matrixAdd(lhs, tmp, 1, N, 1);
	negateMatrix(lhs, N, 1);

	double **rhs = newMatrix(N,1);
	matrixMultiply(rhs, Sp, 0, x, 0, N, N, 1);
	matrixAdd(rhs, Wp, 1, N, 1);

	for(int i=0;i<N;i++)
	{
		rhs[i][0] = max(erc*fabs(rhs[i][0]), eac);;
	}

	term1 = 1;
	for(int i=0;i<N;i++)
	{
		if(lhs[i][0]>rhs[i][0])
		{
			term1 = 0;
		}
	}

	deleteMatrix(lhs,N);
	deleteMatrix(rhs,N);
}

void updateTerminationCondition2(int &term2, double **Y, double **Qd, double **Sd, double **x, double **Wd, double **Od, int N)
{
	
}


void solveQuadraticDual(double **Y, double **x, double **Qd, double **Sd, double **Wd, double **Od, double **Sp, double **Wp, int N)
{
	double **Fd = newMatrix(N,1);
	double **pd = newMatrix(N,N);
	double **theta = newMatrix(N,N);
	double **Qdp_theta = newMatrix(N,N);
	double **Qdn_theta = newMatrix(N,N);
	double **Y_next = newMatrix(N,1);

	computeFd(Fd, Sd, x, Wd, N);
	computeTheta(theta, Qd, N);
	computeQdp_theta(Qdp_theta, Qd, theta, N);
	computeQdn_theta(Qdn_theta, Qd, theta, N);

	initializeY(Y, N);

	int term1=0, term2 = 0;

	while(!term1 || !term2)
	{
		double **ph = newMatrix(N, 1);

		double alphaY=0;

		if(!isEqual(Y, Y_next, N, 1))
		{
			computeph(ph, Qd, Y, Fd, N);
			computealphaY(alphaY, ph, Qd, Y, Fd, N);
			
			updateY1(Y_next, Y, alphaY, ph, N);			
		}
		else
		{
			updateY2(Y_next, Y, Qdp_theta, Qdn_theta, Fd, N);
		}

		copyMatrix(Y, Y_next, N, 1);

		updateTerminationCondition1(term1, Sd, Wd, Qd, Y, Sp, x, Wp, N);
		updateTerminationCondition2(term2, Y, Qd, Sd, x, Wd, Od, N);
	}
}

int main()
{
	// QP is of parametric from 
	// J(U) = min U 1/2*U'QpU + x'Cp'U + 1/2*x'Opx
	// st GpU <= Spx + W

	int N;

	// dimensions
	double **Y = newMatrix(N,1);				// Y = N vector
	double **U = newMatrix(N,1);				// U = N vector
	double **Qp = newMatrix(N,N);			// Qp = N*N matrix
	double **Cp = newMatrix(N,N);			// Cp = N*N matrix
	double **Op = newMatrix(N,N);			// Op = N*N matrix 		Omega p
	double **Gp = newMatrix(N,N);			// Gp = N*N matrix
	double **Sp = newMatrix(N,N);			// Sp = N*N matrix
	double **W = newMatrix(N,1);				// W = N vector

	double **x = newMatrix(N,1);				// x = N vector 	state vector

	// input(U, Qp, Cp, Op, Gp, Sp, N);

	// matrices and vectors required for dual form of QP
	double **Qd = newMatrix(N,N);
	double **Sd = newMatrix(N,N);
	double **Wd = newMatrix(N,1);
	double **Od = newMatrix(N,N);
	

	double **Ed = newMatrix(N,1);			// Ed = N*N matrix 		Xi d
	double **Td = newMatrix(N,1);			// Td = N*N matrix 		Tou d

	convertFromParametricToDual(Qd, Sd, Wd, Od, Qp, Cp, Op, Gp, Sp, W, Td, Ed, N);

	solveQuadraticDual(Y, x, Qd, Sd, Wd, Od, Sp, W, N);

}