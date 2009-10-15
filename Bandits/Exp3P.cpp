#include "Exp3P.h"
#include <limits>
#include <set>
#include <math.h>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


Exp3P::Exp3P( void ) : Exp3G()
{
	_gamma = 0.05;
	_eta = 0.6;
	_horizon = 100.0;
}



//----------------------------------------------------------------
//----------------------------------------------------------------

void Exp3P::initialize( vector< double >& vals )
{
	//for serialization
	if ( _serializationFlag ) writeOutInitialArray( vals );

	_p.resize( _numOfArms );
	_w.resize( _numOfArms );
	_tmpW.resize( _numOfArms );

	fill( _p.begin(), _p.end(), 1.0 / _numOfArms );
	fill( _w.begin(), _w.end(), 1.0 );

	//min( 3/5, \sqrt{3/5 * (MlogM) / T } ) the horizon is considered 100000
	//_gamma = 2.0 * pow( (3.0 / 5.0) * ( ( (double)_numOfArms * log( (double)_numOfArms ) ) / 100000 ), 0.5 );
	//if ( _gamma > (3.0/5.0) ) _gamma=(3.0/5.0);
	

	// \delta = 0.1 T=100000
	//_eta = 2.0* pow( (log( (double)_numOfArms) * 100000.0 ) , 0.5);
	//if ( _gamma > 1.0 ) _gamma = 1.0;



	copy( vals.begin(), vals.end(), _X.begin() );
	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] = ((_gamma * _eta ) / 3.0 ) * pow( ( _horizon / ((double)_numOfArms) ) , 0.5 ) ;
		_w[i]+=	( (_gamma / (3 * (double)_numOfArms )) * // _gamma / 3K
				(vals[i]*((double)_numOfArms ))  // x hat
				+ ( ( ( _eta * (double)_numOfArms ) / (  pow( (double)_numOfArms * _horizon,0.5 )))));
	}

	setInitializedFlagToTrue();
}
//----------------------------------------------------------------
//----------------------------------------------------------------


void Exp3P::receiveReward( int armNum, double reward )
{
	_T[ armNum ]++;
	// calculate the feedback value

	incIter();
	 
	//update 
	double xHat = reward / _p[armNum]; 
	_w[armNum] +=  (_gamma / (3 * (double)_numOfArms )) * ( xHat + ( _eta / ( _p[armNum] * pow( (double)_numOfArms * _horizon,0.5 )) ) );

	/*
	double wsum = 0.0;
	for( int i=0; i<_numOfArms; i++ ) 
	{
		wsum += _w[i];
	}
	for( int i=0; i<_numOfArms; i++ ) 
	{
		_w[i] /= wsum;
	}
	*/

	updateithValue( armNum );		
}


} // end namespace MultiBoost