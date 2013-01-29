// GenDataMP6.cpp: Generate data for assignment MP1

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

typedef std::vector< float > FloatVec;

float genRandomFloat()
{
    return ( (float) rand() / RAND_MAX );
}

void genVector( FloatVec& vec, const int xLen,const int yLen, const int cLen)
{
    for ( int i = 0; i < (xLen*yLen*cLen); ++i )
        vec.push_back( genRandomFloat() );
}
void genMatrix( FloatVec&mat, int rows, int cols )
{
    for ( int r = 0; r < rows; ++r )
        for ( int c = 0; c < cols; ++c )
		mat.push_back(genRandomFloat());
        //    mat[ cols * r + c ] = genRandomFloat();
}

#define max(A,B) (A> B? A:B)
#define min(A,B) (A< B? A:B)

float sumAt( FloatVec& in, int xC, int yC, int cC, int xMax, int yMax, int cMax) {
    	float sum=0;
	for (int x = max(xC -2, 0); x < min(xMax,xC+2); x++) {
		for (int y =max(yC -2,0); y < min(yMax,yC+2); y++) {
			int index = (y*xMax +x)*cMax + cC;
			sum+= in[index];
		}
	}

	return sum;
}
void sumVector( FloatVec& in,FloatVec &out, const int xLen,const int yLen, const int cLen)
{
    out.clear();
    for ( int x = 0; x < (int) xLen; ++x ) {
	    for ( int y = 0; y < (int) yLen; ++y) {
	    	for ( int c = 0; c < (int) cLen; ++c) {
        		out.push_back(sumAt(in, x,y,c, xLen, yLen, cLen));
		}
	    }
    }
}

void writeVector( const FloatVec& vec, const int xLen,const int yLen, const int cLen, const char* fname )
{
    std::ofstream outFile( fname );

    if ( !outFile )
    {
        std::cout << "Error! Opening file: " << fname << " for writing vector.\n";
        exit(1);
    }

    std::cout << "Writing image to file: " << fname << std::endl;


    outFile << xLen << std::endl;
    outFile << yLen << std::endl;
    outFile << cLen << std::endl;

    for ( int i = 0; i < (xLen*yLen*cLen); ++i ) {
        outFile << vec[i] << std::endl;
    }
}
void writeMatrix(const FloatVec& mat, int rows, int cols, const char* fname )
{
    std::ofstream outFile( fname );

    if ( !outFile )
    {
        std::cout << "Error! Opening file: " << fname << " for writing matrix\n";
        exit(1);
    }

    std::cout << "Writing matrix to file: " << fname << std::endl;

    outFile << rows << std::endl;
    outFile << cols << std::endl;

    int idx = 0;

    for ( int r = 0; r < rows; ++r )
    {
        for ( int c = 0; c < cols; ++c )
        {
            outFile << mat[ idx++ ] << " ";
        }

        outFile << std::endl;
    }
}

int main( int argc, const char** argv )
{
    // Info for user

    std::cout <<argv[0] << ": Generates data files to use as input for assignment MP.\n";
    std::cout << "Invoke as: " << argv[0] << " [X] [Y] [Channels] [MaskX] [MaskY]\n\n";

    // Read input

    if ( 6 != argc )
    {
        std::cout << "Error! Wrong number of arguments to program.\n";
        return 0;
    }

    // Create vectors

    const int xLen = atoi( argv[1] );
    const int yLen = atoi( argv[2] );
    const int cLen = atoi( argv[3] );
    const int mXLen = atoi( argv[4] );
    const int mYLen = atoi( argv[5] );

    FloatVec inputImage;
    FloatVec maskMatrix;
    FloatVec outputImage;

    genVector(inputImage, xLen,yLen,cLen);
    genMatrix(maskMatrix, mXLen, mYLen);
    sumVector(inputImage, outputImage,  xLen,yLen,cLen);

    // Write to files

    writeVector(inputImage, xLen,yLen,cLen, "imageInput.txt" );
    writeMatrix(maskMatrix, mXLen,mYLen, "convolutionMatrix.txt" );
    writeVector(outputImage, xLen,yLen,cLen, "imageOutput.txt" );
    return 0;
}
