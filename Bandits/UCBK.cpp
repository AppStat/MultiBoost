#include "UCBK.h"
#include <limits>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


UCBK::UCBK( void ) : GenericBanditAlgorithm(), _valuesList(0)
{
}


//----------------------------------------------------------------
//----------------------------------------------------------------


void UCBK::getKBestAction( const int k, vector<int>& bestArms )
{
	bestArms.resize(k);
	int i=0; 
	for( list< pair< double,int>* >::iterator it = _valuesList.begin(); i<k; it++, i++ )
	{
		bestArms[i] = (**it).second;
	}
}
//----------------------------------------------------------------
//----------------------------------------------------------------

int UCBK::getNextAction()
{
	return _valuesList.front()->second;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBK::initialize( vector< double >& vals )
{
	//for serialization
	if ( _serializationFlag ) writeOutInitialArray( vals );

	_valueRecord.resize( _numOfArms );
		
	for( int i=0; i < _numOfArms; i++ ) {
		pair< double, int >* tmpPair = new pair< double, int >(0.0,i);

		_valuesList.push_back( tmpPair );
		_valueRecord[i] = tmpPair;
	}


	//copy the initial values to X
	copy( vals.begin(), vals.end(), _X.begin() );
	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	//update the values
	for( int i = 0; i < _numOfArms; i++ )
	{
		_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	}
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );

	setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBK::updateithValue( int i )
{
	//update the value
	_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );
}

//----------------------------------------------------------------
//----------------------------------------------------------------


} // end namespace MultiBoost