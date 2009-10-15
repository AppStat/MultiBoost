#include "Exp3.h"
#include <limits>
#include <set>
#include <math.h>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


Exp3::Exp3( void ) : GenericBanditAlgorithm()
{
	_gamma = 0.01;
	//_eta = 0.05;
	//_alpha = 0.05;
}


//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3::initLearningOptions(const nor_utils::Args& args) 
{
	if ( args.hasArgument( "gamma" ) ){
		_gamma = args.getValue<double>("gamma", 0 );
	} 

}


//----------------------------------------------------------------
//----------------------------------------------------------------
int Exp3::getNextAction()
{
	vector< double > cumsum( getArmNumber()+1 );
	int i;

	cumsum[0] = 0.0;
	for( int i=0; i < getArmNumber(); i++ )
	{
		cumsum[i+1] = cumsum[i] + _pHat[i];
	}

	for( i=0; i < getArmNumber(); i++ )
	{
		cumsum[i+1] /= cumsum[  getArmNumber() ];
	}

	double r = rand() / (double) RAND_MAX;

	for( i=0; i < getArmNumber(); i++ )
	{
		if ( (cumsum[i] <= r) && ( r<=cumsum[i+1] )  ) break;
	}

	return i;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void Exp3::initialize( vector< double >& vals )
{
	//for serialization
	if ( _serializationFlag ) writeOutInitialArray( vals );

	_p.resize( _numOfArms );
	_pHat.resize( _numOfArms );

	fill( _p.begin(), _p.end(), 1.0 / _numOfArms );
	fill( _pHat.begin(), _pHat.end(), 1.0 / _numOfArms );

	//_alpha = pow( 4*_numOfArms*log((double)_numOfArms) * ( 1.0 / 100000 ), 1.0/3.0);
	//_gamma = pow( (_numOfArms*log((double)_numOfArms))/( 2.0 * 100000 ) , 0.5);
	//if ( _gamma > 1.0 ) _gamma = 1.0;


	//copy the initial values to X
	copy( vals.begin(), vals.end(), _X.begin() );

	for( int i=0; i < _X.size(); i++ ) _X[i] *= ( _gamma /(double) _numOfArms* _numOfArms);

	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );


	setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void Exp3::updateithValue( int i )
{
	double sumexp = 0.0;
	double sum = 0.0;


	double max = -numeric_limits<double>::max();
	for( int i=0; i<_numOfArms; i++ ) 
	{
		if ( max < _X[i] ) max = _X[i];
	}

	for( int i=0; i<_numOfArms; i++ ) 
	{
		_p[i] = exp( _X[i]-max);// * log( 1.0 + _alpha ); 
		sumexp += _p[i];
	}
	
	for( int i=0; i<_numOfArms; i++ ) 
	{
		_p[i] /= sumexp;
		//_p[i] = exp( _p[i] );
	
		//sum += _p[i];
	}


	for( int i=0; i<_numOfArms; i++ ) 
	{
		_pHat[i] = (1-_gamma)*_p[i] + _gamma / _numOfArms;
	}

}

//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3::receiveReward( int armNum, double reward )
{
	if ( _serializationFlag )	
		writeOutActionAndReward( armNum, reward );

	_T[ armNum ]++;
	double xHat = reward / _pHat[armNum];
	_X[ armNum ] += ( ( _gamma  * xHat )/ _numOfArms );
	incIter();
	updateithValue( armNum );		
}


} // end namespace MultiBoost