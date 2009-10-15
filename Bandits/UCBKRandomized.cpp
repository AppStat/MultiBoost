#include "UCBKRandomized.h"
#include <limits>
#include <set>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


UCBKRandomized::UCBKRandomized( void ) : UCBK()
{
}


//----------------------------------------------------------------
//----------------------------------------------------------------


void UCBKRandomized::getKBestAction( const int k, vector<int>& bestArms )
{
	set<int> s;
	for( int i=0; i<k; i++ )
	{
		s.insert( getNextAction() );
	}
	bestArms.clear();
	for( set<int>::iterator it = s.begin(); it != s.end(); it++ )
	{
		bestArms.push_back( *it );
	}
}
//----------------------------------------------------------------
//----------------------------------------------------------------

int UCBKRandomized::getNextAction()
{
	vector< double > cumsum( getArmNumber()+1 );
	int i;

	cumsum[0] = 0.0;
	for( int i=0; i < getArmNumber(); i++ )
	{
		cumsum[i+1] = cumsum[i] + _valueRecord[i]->first;
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

void UCBKRandomized::updateithValue( int i )
{
	//update the value
	//_valueRecord[i]->first = _X[i] / (double) _T[i] + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	_valueRecord[i]->first = _X[i] / (double) _T[i];// + sqrt( ( 2 * log( (double)getIterNum() ) ) / _T[i] );
	//_valueRecord[i]->first = exp( _valueRecord[i]->first );
	//sort them according to the values the arms
	_valuesList.sort( nor_utils::comparePairP< 1, double, int, greater<double> >() );
}
//----------------------------------------------------------------
//----------------------------------------------------------------


} // end namespace MultiBoost