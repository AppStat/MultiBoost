#include "blaswrap.h"
#include "f2c.h"

#define DIM 1024
#define WSIZE DIM * DIM

//for MDL
static doublereal       work[WSIZE];
static integer          ipiv[WSIZE];


//////////////////////////////////////////////////////////////////////////////////////////////
// extern for lapack routines
extern "C" /* Subroutine */ int dgetri_(integer *n, doublereal *a, integer *lda, integer 
	*ipiv, doublereal *work, integer *lwork, integer *info);
extern "C" /* Subroutine */ int sgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, real *alpha, real *a, integer *lda, real *b, integer *
	ldb, real *beta, real *c__, integer *ldc);
extern "C" /* Subroutine */ int dgetrf_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);
extern "C" void solveEquationSystem( double* X, double* b, int* N );

int matrixInverse( integer *n, doublereal *a );
void solveEquationSystem( double* X, double* b, int* N );




void solveEquationSystem( double* X, double* b, int* N ) {
	matrixInverse( (integer*) N, (doublereal*) X );
	double* sum = new double[ *N ];
	for( int i = 0; i < *N; i++ ) {
		sum[i] = 0.0;
		for( int j = 0; j < *N; j++ ) {
			sum[i] += (X[(i*(*N))+j] * b[j]);
		}
	}
	for( int i = 0; i < *N; i++ ) {
		b[i] = sum[i];
	}
	
	delete [] sum;
}


int matrixInverse( integer *n, doublereal *a ) {
    integer info;
    static integer lwork = WSIZE;
    
	dgetrf_( n, n, a, n, ipiv, &info);
    dgetri_( n, a, n, ipiv, work, &lwork, &info);    
    return info;
}

