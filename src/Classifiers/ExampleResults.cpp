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


#include "Classifiers/ExampleResults.h"
#include "Utils/Utils.h" // for nor_utils::comparePairOnSecond
#include <cassert>
#include <functional> // for greater
#include <algorithm> // for sort

namespace MultiBoost {

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

    pair<int, AlphaReal> ExampleResults::getWinner(int rank)
    {
        assert(rank >= 0);

        vector< pair<int, AlphaReal> > rankedList;
        getRankedList(rankedList); // get the sorted rankings
        return rankedList[rank];
    }

// -------------------------------------------------------------------------

    bool ExampleResults::isWinner(const Example& example, int atLeastRank) const
    {
        assert(atLeastRank >= 0);

        vector< pair<int, AlphaReal> > rankedList;
        getRankedList(rankedList); // get the sorted rankings

        for (int i = 0; i <= atLeastRank; ++i)
        {
            if ( example.hasPositiveLabel(rankedList[i].first) )
                return true;
        }

        return false;
    }

// -------------------------------------------------------------------------

    void ExampleResults::getRankedList( vector< pair<int, AlphaReal> >& rankedList ) const
    {
        rankedList.resize( _votesVector.size() );

        vector<AlphaReal>::const_iterator vIt;
        const vector<AlphaReal>::const_iterator votesVectorEnd = _votesVector.end();
        int i;
        for (vIt = _votesVector.begin(), i = 0; vIt != votesVectorEnd; ++vIt, ++i )
            rankedList[i] = make_pair(i, *vIt);

        sort( rankedList.begin(), rankedList.end(), 
              nor_utils::comparePair<2, int, AlphaReal, greater<AlphaReal> >() );
    }

// -------------------------------------------------------------------------

} // end of namespace MultiBoost
