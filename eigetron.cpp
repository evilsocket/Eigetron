#include <math.h>
#include <stdio.h>
#include <vector>
#include <CImg.h>

using namespace std;
using namespace cimg_library;

typedef unsigned int  uint;
typedef unsigned long ulong;
			  
typedef struct {
	uint  faces;
	ulong width;
	ulong height;
}
eigen_db_t;

void     process_image  ( CImg<double>& image, CImg<double>& mean, uint trainingset_size );
void     normalize_image( CImg<double>& image, uint index, CImg<double>& mean, double **& normalized_m );
void     eigen_covariance( double **& normalized_m, double **& covariance_m, uint vector_size, uint trainingset_size );
int      eigen_decomposition( double **& matrix, uint m_size, double *& eigenvalues, double **& eigenvectors );
double **eigen_project( double **& m_normalized, double **& eigenvectors, double *& eigenvalues, uint vector_size, uint trainingset_size );
double  *eigen_weights( CImg<double>& face, CImg<double>& mean, double **& projections, uint trainingset_size );

void     eigen_build_db();

void eigen_load_db( eigen_db_t *db ){
	FILE *fp = fopen( "faces/eigen.db.dat", "rb" );
	fread( db, sizeof(eigen_db_t), 1, fp );	
	fclose(fp);
}

double **eigen_load_vectors( eigen_db_t& db ){
	double **projections;
	uint i, j;
	
	projections = new double * [db.faces];
	for( i = 0; i < db.faces; i++ ){
		projections[i] = new double[db.width * db.height];	
	}
		
	FILE *fp = fopen( "faces/eigen.vectors.dat", "rb" ); 
	for( i = 0; i < db.faces; i++ ){
		for( j = 0; j < db.width * db.height; j++ ){
			fread( &projections[i][j], sizeof(double), 1, fp );
		}	
	}
	fclose(fp);
	
	return projections;
} 

double **eigen_load_weigths( eigen_db_t& db ){
	double **weights;
	uint i, j;
	
	weights = new double * [db.faces];
	for( i = 0; i < db.faces; i++ ){
		weights[i] = new double[db.faces];	
	}
	FILE *fp = fopen( "faces/eigen.weights.dat", "rb" ); 
	for( i = 0; i < db.faces; i++ ){
		for( j = 0; j < db.faces; j++ ){
			fread( &weights[i][j], sizeof(double), 1, fp );
		}	
	}
	fclose(fp);
	
	return weights;
} 

int main( int argc, char **argv ){
	
	if( argc <= 1 ){
		eigen_build_db();
	}
	else if( strcmp( argv[1], "match" ) == 0 ){
		CImg<double> mean( "faces/eigen.mean.bmp" ), 
					 face;
		eigen_db_t db;
		char filename[0xFF] = {0x00};
		uint idx = atoi(argv[2]), i, j;
		double **projections,
		       **weights,
			   	*iweights;
		
		sprintf( filename, "faces/face%.2d.jpg", idx );
		face.load(filename);
		
		eigen_load_db(&db);
		projections = eigen_load_vectors(db);		
		weights     = eigen_load_weigths(db);
				
		iweights = eigen_weights( face, mean, projections, db.faces );
		
		double min, mag;
		int closest = 0;

		mag = 0.0;
		min = -1000;
		for( i = 0; i < db.faces; i++ ){
			for( j = 0; j < db.faces; j++ ){
				mag += pow( iweights[j] - weights[i][j], 2.0 );
			}
			//printf( "sqrt(mag)[%d] = %g\n", i, sqrt(mag) );
			mag = sqrt(mag);
			if( i == 0 || mag < min ){
				min = mag;
				closest = i;
			} 
			mag = 0;
		}
		
		printf( "Closest is faces/face%.2d.jpg with %g\n", closest, min );
	}
	
	return 0;	
}

void eigen_build_db(){      
	eigen_db_t db;
	
	vector< CImg<double> * > trainingset;
	         
	CImg<double> mean( 57, 61, 1, 1 ); // mean face from training set
	
	double      **normalized_m,     // normalized faces matrix A
				**covariance_m,     // covariance matrix (C = tA*A --> C = A*tA)
				**eigenvectors,     // eigenvectors of jacobi decomposition
				*eigenvalues,       // eigenvalues of jacobi decomposition
				**eigenprojections, // eigenvectors projected to face space
				**eigenweights;     // weights
				
	int         i, j;
	char        path[] = { "faces/face%.2d.jpg" },
				filename[0xFF] = { 0x00 };
  	
	printf( "--- Building Eigen faces database ---\n" );
	
  	printf( "\t@ Loading training set ...\n" );
	for( i = 0; i < 20; i++ ){
		sprintf( filename, path, i );		
		trainingset.push_back( new CImg<double>(filename) );
	}
				
	printf( "\t@ Processing mean face ...\n" );
	for( i = 0; i < 20; i++ ){
		process_image( *trainingset[i], mean, trainingset.size() );
	}
	mean.save("faces/eigen.mean.bmp");
	
	printf( "\t@ Normalizing faces ...\n" );
	normalized_m = new double * [ trainingset.size() ];
	for( i = 0; i < trainingset.size(); i++ ){
		normalized_m[i] = new double[ mean.width * mean.height ];
	}
	for( i = 0; i < trainingset.size(); i++ ){
		normalize_image( *trainingset[i], i, mean, normalized_m );
	}
	
	printf( "\t@ Computing covariance ...\n" );
	covariance_m = new double * [ trainingset.size() ];
	for( i = 0; i < trainingset.size(); i++ ){
		covariance_m[i] = new double[ trainingset.size() ];
	}
	eigen_covariance( normalized_m, covariance_m, mean.width * mean.height, trainingset.size() );
	
	printf( "\t@ Computing Jacobi decomposition ... " );
	eigenvalues  = new double   [trainingset.size()];
	eigenvectors = new double * [trainingset.size()];
	for( i = 0; i < trainingset.size(); i++ ){
		eigenvectors[i] = new double [trainingset.size()];
	}
	int ret = eigen_decomposition( covariance_m, trainingset.size(), eigenvalues, eigenvectors );
	printf( "[%d]\n", ret );
	
	
	printf( "\t@ Projecting eigenvectors ...\n" );
	eigenprojections = eigen_project( normalized_m, eigenvectors, eigenvalues, mean.width * mean.height, trainingset.size() );
	
	printf( "\t@ Saving eigen values ...\n" );
	FILE *fp = fopen( "faces/eigen.values.dat", "w+b" );
	for( i = 0; i < trainingset.size(); i++ ){
		fwrite( &eigenvalues[i], sizeof(double), 1, fp );	
	}
	fclose(fp);
	
	printf( "\t@ Computing eigen weights ...\n" );
	eigenweights = new double * [trainingset.size()];
	for( i = 0; i < trainingset.size(); i++ ){
		eigenweights[i] = eigen_weights( *trainingset[i], mean, eigenprojections, trainingset.size() );
	}
	fp = fopen( "faces/eigen.weights.dat", "w+b" );
	for( i = 0; i < trainingset.size(); i++ ){
		for( j = 0; j < trainingset.size(); j++ ){
			fwrite( &eigenweights[i][j], sizeof(double), 1, fp );	
		}
	}
	fclose(fp);
	
	printf( "\t@ Saving db info ...\n" );
	db.faces  = trainingset.size();
	db.width  = mean.width;
	db.height = mean.height;
	fp = fopen( "faces/eigen.db.dat", "w+b" );
	fwrite( &db, sizeof(eigen_db_t), 1, fp );	
	fclose(fp);
}

void process_image( CImg<double>& image, CImg<double>& mean, uint trainingset_size ){
	int i, j;
		
	for( i = 0; i < image.width; i++ ){
		for( j = 0; j < image.height; j++ ){
  			*mean.ptr( i, j ) += *image.ptr( i, j ) / (trainingset_size - 1);
		}
	}
}

void normalize_image( CImg<double>& image, uint index, CImg<double>& mean, double **&normalized_m ){
	uint i, j, k;
	
	for( i = 0, k = 0; i < image.width; i++ ){
		for( j = 0; j < image.height; j++, k++ ){
			normalized_m[index][k] = *image.ptr( i, j ) - *mean.ptr( i, j );
		}	
	}
}

void eigen_covariance( double **& normalized_m, double **& covariance_m, uint vector_size, uint trainingset_size ){
	double **transpose;
	uint i, j, m;
	
	transpose = new double * [vector_size];
	for( i = 0; i < vector_size; i++ ){
		transpose[i] = new double [trainingset_size];
	}
	
	for( i = 0; i < vector_size; i++ ){
		for( j = 0; j < trainingset_size; j++ ){
			transpose[i][j] = normalized_m[j][i];	
		}	
	}
	
	for( m = 1; m < trainingset_size; m++ ){
		for( i = 1; i < trainingset_size; i++ ){
			covariance_m[i][m] = 0;
			for( j = 1; j < vector_size; j++ ){
				covariance_m[i][m] += normalized_m[m][j] * transpose[j][i];
			} 
		}
	}
	
	for( i = 0; i < vector_size; i++ ){
		delete[] transpose[i];
	}
	delete[] transpose;
}

int eigen_decomposition( double **& matrix, uint m_size, double *& eigenvalues, double **& eigenvectors ){
	uint i, j, k, iq, ip;
	double threshold, theta, tau, t, sm, s, h, g, c,
		   p, *b, *z;
	int jiterations;
	// max iterations in Jacobi decomposition algorithm
	ulong jacobi_max_iterations = 500;
	

	b = new double[m_size * sizeof(double)];
	b--;
	z = new double[m_size * sizeof(double)];
	z--;
	
	/* initialize eigenvectors and eigen values */
	for( ip = 1; ip < m_size; ip++ ){
		for (iq = 1; iq < m_size; iq++){
			eigenvectors[ip][iq] = 0.0;
		}
		eigenvectors[ip][ip] = 1.0;
	}
	for( ip = 1; ip < m_size; ip++ ){
		b[ip] = eigenvalues[ip] = matrix[ip][ip];
		z[ip] = 0.0;
	}
	
	jiterations = 0;
	for( i = 0; i <= jacobi_max_iterations; i++ ){
		sm = 0.0;
		for( ip = 1; ip < m_size; ip++ ){
			for( iq = ip + 1; iq < m_size; iq++ ){
				sm += fabs(matrix[ip][iq]);
			}
		}
		
		if( sm == 0.0 ){
			/* eigenvalues & eigenvectors sorting */
			for( i = 1; i < m_size; i++ ){
				p = eigenvalues[k = i];
				for( j = i + 1; j < m_size; j++ ){
					if( eigenvalues[j] >= p ){
						p = eigenvalues[k = j];
					}
				}
				if( k != i ){
					eigenvalues[k] = eigenvalues[i];
					eigenvalues[i] = p;
					for( j = 1; j < m_size; j++ ){
						p                  = eigenvectors[j][i];
						eigenvectors[j][i] = eigenvectors[j][k];
						eigenvectors[j][k] = p;
					}
				}
			}

			/* restore symmetric matrix's matrix */
			for( i = 2; i < m_size; i++ ){
				for( j = 1; j < i; j++ ){
					matrix[j][i] = matrix[i][j];
				}
			}
			
			z++;
			delete z;
			b++;
			delete b;
			return jiterations;
		}

		threshold = ( i < 4 ? 0.2 * sm / (m_size * m_size) : 0.0 );		
		for( ip = 1; ip < m_size; ip++ ){
			for( iq = ip + 1; iq < m_size; iq++ ){
				g = 100.0 * fabs(matrix[ip][iq]);
					
				if( i > 4 && 
					fabs(eigenvalues[ip]) + g == fabs(eigenvalues[ip]) && 
					fabs(eigenvalues[iq]) + g == fabs(eigenvalues[iq]) ){
					matrix[ip][iq] = 0.0;
				}
				else if( fabs(matrix[ip][iq]) > threshold ){
					h = eigenvalues[iq] - eigenvalues[ip];
					if( fabs(h) + g == fabs(h) ){
						t = matrix[ip][iq] / h;
					}
					else {
						theta = 0.5 * h / matrix[ip][iq];
						t     = 1.0 / ( fabs(theta) + sqrt( 1.0 + theta * theta ) );
						if( theta < 0.0 ){
							t = -t;
						}
					}
					
					c                = 1.0 / sqrt(1 + t * t);
					s                = t * c;
					tau              = s / (1.0 + c);
					h                = t * matrix[ip][iq];
					z[ip]           -= h;
					z[iq]           += h;
					eigenvalues[ip] -= h;
					eigenvalues[iq] += h;
					matrix[ip][iq]   = 0.0;
					
					#define M_ROTATE(M,i,j,k,l) g = M[i][j]; \
											    h = M[k][l]; \
											    M[i][j] = g - s * (h + g * tau); \
			  								    M[k][l] = h + s * (g - h * tau)
			  
					for( j = 1; j < ip; j++ ){
						M_ROTATE( matrix, j, ip, j, iq );
					}
					for( j = ip + 1; j < iq; j++ ){
						M_ROTATE( matrix, ip, j, j, iq );
					}
					for( j = iq + 1; j < m_size; j++ ){
						M_ROTATE( matrix, ip, j, iq, j );
					}
					for( j = 1; j < m_size; j++ ){
						M_ROTATE( eigenvectors, j, ip, j, iq );
					}
					++jiterations;
				}
			}
		}
		
		for( ip = 1; ip < m_size; ip++ ){
			b[ip]          += z[ip];
			eigenvalues[ip] = b[ip];
			z[ip]           = 0.0;
		}
	}
	
	z++;
	delete z;
	b++;
	delete b;
	
	return -1;
}

double ** eigen_project( double **& m_normalized, double **& eigenvectors, double *& eigenvalues, uint vector_size, uint trainingset_size ){
	double **transpose;
	int i,j,l,p;
	long k,m = 0;
	double value = 0,max,mag,
		   **projections;
	
	projections = new double * [trainingset_size];
	for( i = 0; i < trainingset_size; i++ ){
		projections[i] = new double[vector_size];	
	}
		
	transpose = new double * [vector_size];
	for( i = 0; i < vector_size; i++ ){
		transpose[i] = new double[trainingset_size];
	}
	for( i = 0; i < vector_size; i++ ){ 
		for( j = 0; j < trainingset_size; j++ ){
			transpose[i][j] = m_normalized[j][i];
		}
	}
	
	FILE *fp = fopen( "faces/eigen.vectors.dat", "w+b" );
	for( k = 0; k < trainingset_size; k++ ){
		for( i = 1, m = 0; i < vector_size; i++, m++ ){
			for( j = 1; j < trainingset_size; j++ ){
				value += transpose[i][j] * eigenvectors[j][k];
			}
			projections[k][m] = value; 
			value   = 0;
		}

		/* eigenfaces normalization */
		mag = 0;
		for( l = 0; l < vector_size; l++ ){
			mag += pow( projections[k][l], 2.0 );
		}
		mag = sqrt(mag);
		for( l = 0; l < vector_size; l++ ){
			projections[k][l] /= mag;
		}
		
		max = 0;
		for( p = 0; p < vector_size; p++ ){ 
			if( projections[k][p] > max ){
				max = projections[k][p];
			}
		}

		/* save the projected eigenvector */		
		for( p = 0; p < vector_size; p++ ){
			fwrite( &projections[k][p], sizeof(double), 1, fp );
		}
	}
	fclose(fp);
	
	return projections;
}

double *eigen_weights( CImg<double>& face, CImg<double>& mean, double **& projections, uint trainingset_size ){
	CImg<double> normalized = face - mean;
	uint m, n, i, index;	
	double *weights, w;

	weights = new double[trainingset_size];
	
	for( index = 0; index < trainingset_size; index++ ){
		w = 0.0;
		for( m = 0, i = 0; m < normalized.width; m++ ){
			for( n = 0; n < normalized.height; n++, i++ ){
				w += projections[index][i] * *normalized.ptr(m,n);	
			}	
		}
		weights[index] = w;
	}
	
	return weights;
}
