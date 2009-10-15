#include "Random.h"
#include <limits>
#include <set>

namespace MultiBoost {
//----------------------------------------------------------------
//----------------------------------------------------------------


Random::Random( void ) : GenericBanditAlgorithm()
{
}


//----------------------------------------------------------------
//----------------------------------------------------------------


void Random::getKBestAction( const int k, vector<int>& bestArms )
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

int Random::getNextAction()
{
	int arm = (int) (rand() % getArmNumber() );

	return arm;
}

//----------------------------------------------------------------
//----------------------------------------------------------------

void Random::updateithValue( int i )
{
}

//----------------------------------------------------------------
//----------------------------------------------------------------


} // end namespace MultiBoost