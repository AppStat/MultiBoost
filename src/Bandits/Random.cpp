/*
*
*    MultiBoost - Multi-purpose boosting package
*
*    Copyright (C)        AppStat group
*                         Laboratoire de l'Accelerateur Lineaire
*                         Universite Paris-Sud, 11, CNRS
*
*    This file is part of the MultiBoost library
*
*    This library is free software; you can redistribute it
*    and/or modify it under the terms of the GNU General Public
*    License as published by the Free Software Foundation
*    version 2.1 of the License.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    General Public License for more details.
*
*    You should have received a copy of the GNU General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
*
*    Contact: : multiboost@googlegroups.com
*
*    For more information and up-to-date version, please visit
*
*                       http://www.multiboost.org/
*
*/


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

