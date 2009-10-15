#include "UCBKV.h"
#include <limits>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


UCBKV::UCBKV( void ) : GenericBanditAlgorithm(), _valuesList(0), _table( 0 ), _kszi( 1.0 ), _c(1.0/3.0), _b(1)
{
}


//----------------------------------------------------------------
//----------------------------------------------------------------


void UCBKV::getKBestAction( const int k, vector<int>& bestArms )
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

int UCBKV::getNextAction()
{
	return _valuesList.front()->second;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBKV::initialize( vector< double >& vals )
{
	//for serialization
	if ( _serializationFlag ) writeOutInitialArray( vals );

	int i;

	_valueRecord.resize( _numOfArms );
		
	for( i=0; i < _numOfArms; i++ ) {
		pair< double, int >* tmpPair = new pair< double, int >(0.0,i);

		_valuesList.push_back( tmpPair );
		_valueRecord[i] = tmpPair;
	}


	//copy the initial values to X
	copy( vals.begin(), vals.end(), _X.begin() );
	
	//copy the initial values into the table
	_table.resize( _numOfArms );
	for( i = 0; i < _numOfArms; i++ ) 
	{
		_table[i].resize( 1 );
		_table[i][0] = vals[i];
	}

	//one pull for all arm
	fill( _T.begin(), _T.end(), 1 );
	
	//update the values
	for( i = 0; i < _numOfArms; i++ )
	{
		//_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
		_valueRecord[i]->first = _X[i] / (double) _T[i] + _c * ( ( 3 * _b * _kszi *  log( (double)getIterNum() ) )/ _T[i]); //second term is zero because of the variance is eqaul to zero
	}
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );

	setInitializedFlagToTrue();
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBKV::updateithValue( int i )
{
	//update the value
	double mean = _X[i] / (double) _T[i];
	double variance = 0.0;

	for( int j = 0; j < _T[i]; j++ )
	{
		variance += (( _table[i][j] - mean )*( _table[i][j] - mean ));
	}
	variance /= _T[i];

	_valueRecord[i]->first = mean + sqrt( ( 2.0 * _kszi * variance * log( (double)getIterNum() ) ) / _T[i] ) + 
									_c * ( ( 3 * _b * _kszi *  log( (double)getIterNum() ) )/ _T[i]) ;
	
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void UCBKV::receiveReward( int armNum, double reward )
{
	if ( _serializationFlag )	
		writeOutActionAndReward( armNum, reward );

	_T[ armNum ]++;
	_X[ armNum ] += reward;
	_table[armNum].push_back( reward );
	incIter();
	updateithValue( armNum );		
}

//----------------------------------------------------------------
//----------------------------------------------------------------


} // end namespace MultiBoost