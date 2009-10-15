#include "Exp3G.h"
#include <limits>
#include <set>
#include <math.h>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


Exp3G::Exp3G( void ) : GenericBanditAlgorithm(), _actions(0)
{
	_gamma = 0.05;
	_eta = 0.5;
}


//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3G::initLearningOptions(const nor_utils::Args& args) 
{
	if ( args.hasArgument( "gamma" ) ){
		_gamma = args.getValue<double>("gamma", 0 );
	} 
	if ( args.hasArgument( "eta" ) ){
		_eta = args.getValue<double>("eta", 0 );
	} 
}


//----------------------------------------------------------------
//----------------------------------------------------------------


int Exp3G::getNextAction()
{
	vector< double > cumsum( getArmNumber()+1 );
	int i;

	cumsum[0] = 0.0;
	for( int i=0; i < getArmNumber(); i++ )
	{
		cumsum[i+1] = cumsum[i] + _p[i];
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

void Exp3G::initialize( vector< double >& vals )
{
	//for serialization
	if ( _serializationFlag ) writeOutInitialArray( vals );

	_p.resize( _numOfArms );
	_w.resize( _numOfArms );
	_tmpW.resize( _numOfArms );

	fill( _p.begin(), _p.end(), 1.0 / _numOfArms );
	fill( _w.begin(), _w.end(), 1.0 );

	//_eta = pow( log((double)_numOfArms)/ (4.0 * 2.0) , 0.5);
	//_gamma = 0.1;
	//if ( _gamma > 1.0 ) _gamma = 1.0;


	//copy the initial values to X
	copy( vals.begin(), vals.end(), _X.begin() );
	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	_sideInformation.resize( _numOfArms );
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_sideInformation[i].resize( _numOfArms );
		fill( _sideInformation[i].begin(), _sideInformation[i].end(), 0 );
	}
	
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] = _eta * _X[i];
	}

	setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------
/*
void Exp3G::updateithValue( int i )
{
	double sum = 0.0;

	for( int i=0; i<_numOfArms; i++ ) 
	{
		sum += _w[i];
	}
	for( int i=0; i<_numOfArms; i++ ) 
	{
		//_p[i] = ( 1 - _gamma ) * exp( _w[i] / sum ) + ( _gamma / (double)getIterNum() );
		_p[i] = ( 1 - _gamma ) * ( _w[i] / sum ) + ( _gamma / (double)getIterNum() );
	}
}
*/

void Exp3G::updateithValue( int arm )
{
	//double sum = 0.0;
	double max = -numeric_limits<double>::max();
	for( int i=0; i<_numOfArms; i++ ) 
	{
		//sum += _w[i];
		if ( max < _w[i] ) max = _w[i];
	}
	//double mean = sum / ( double ) _numOfArms;
	double expSum = 0.0;
	
	for( int i=0; i<_numOfArms; i++ ) 
	{
		_tmpW[i] = _w[i] - max;
		expSum += exp( _tmpW[i] );
	}
	


	for( int i=0; i<_numOfArms; i++ ) 
	{
		//_p[i] = ( 1 - _gamma ) * exp( _w[i] / sum ) + ( _gamma / (double)getIterNum() );
		_p[i] = ( 1 - _gamma ) * ( exp( _tmpW[i] ) / expSum ) + ( _gamma / (double)getIterNum() );
	}
}

//----------------------------------------------------------------
//----------------------------------------------------------------
void Exp3G::receiveReward( int armNum, double reward )
{
	if ( _serializationFlag )	
		writeOutActionAndReward( armNum, reward );

	_T[ armNum ]++;
	// calculate the feedback value

	int prevArm;
	if ( _actions.size() >= 1 )
	{
		prevArm = _actions.back();
		_actions.push_back( armNum );
		_sideInformation[armNum][prevArm]++;
	} else {
		_sideInformation[armNum][armNum]++;
		_actions.push_back( armNum );
		prevArm = armNum;
	}
	incIter();
	 
	//update 
	for( int i=0; i < _numOfArms; i++ ) 
	{
		_w[i] += ( ( ( (double)_sideInformation[i][prevArm] ) / ( (double)_sideInformation[armNum][prevArm] ) ) * ( _eta * reward ) );
	}

	//_w[armNum] += ( _eta * reward );

	updateithValue( armNum );		
}


} // end namespace MultiBoost