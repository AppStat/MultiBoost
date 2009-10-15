#include "Exp3G2.h"
#include <limits>
#include <set>
#include <math.h>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


Exp3G2::Exp3G2( void ) : Exp3G()
{
	_gamma = 0.05;
	_eta = 0.5;
}





//----------------------------------------------------------------
//----------------------------------------------------------------

void Exp3G2::initialize( vector< double >& vals )
{
	//for serialization
	if ( _serializationFlag ) writeOutInitialArray( vals );

	_p.resize( _numOfArms );
	_w.resize( _numOfArms );
	_tmpW.resize( _numOfArms );

	fill( _p.begin(), _p.end(), 1.0 / _numOfArms );
	fill( _w.begin(), _w.end(), 1.0 );


	copy( vals.begin(), vals.end(), _X.begin() );
	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] = _eta * _X[i];
	}

	setInitializedFlagToTrue();
}



//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3G2::receiveReward( int armNum, double reward )
{

	_T[ armNum ]++;
	// calculate the feedback value

	incIter();
	 
	//update 
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] += ( _eta * reward );
	}

	//_w[armNum] += ( _eta * reward );

	updateithValue( armNum );		
}

//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3G2::receiveReward( vector<double> reward )
{
	incIter();
	 
	//update 
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] += ( _eta * reward[i] );
	}

	//_w[armNum] += ( _eta * reward );

	updateithValue( 0 );		
}


} // end namespace MultiBoost