/*
* This file is part of MultiBoost, a multi-class 
* AdaBoost learner/classifier
*
* Copyright (C) 2005-2006 Norman Casagrande
* For informations write to nova77@gmail.com
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*
*/

// Indexes: i = loop on examples
//          j = loop on columns
//          l = loop on classes

#include <iostream> // for cerr
#include <algorithm> // for sort
#include <functional> // for less
#include <fstream>

#include "IO/TxtParser.h"
#include "IO/ArffParser.h"

#include "Utils/Utils.h" // for white_tabs
#include "IO/InputData.h"

//#include <cassert>
//#include <cmath>  //for fabs

//
namespace MultiBoost {

	// ------------------------------------------------------------------------
	int		InputData::loadIndexSet( set< int > ind ) {
		int i = 0;
		//upload the indirection
		this->_usedIndices.clear();
		this->_rawIndices.clear();

		map<int, int> tmpPointsPerClass;
		
		for( set< int >::iterator it = ind.begin(); it != ind.end(); it++ ) {
			this->_indirectIndices[i] = *it;
			this->_usedIndices.insert( *it );

			this->_rawIndices[*it] = i;

			i++;

			const vector<Label>& labels = _pData->getLabels( *it );
			vector<Label>::const_iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				switch ( this->_pData->getLabelRep() ) {
					case LR_DENSE:
						if ( lIt->y > 0 )
							tmpPointsPerClass[lIt->idx]++;
						break;
					case LR_SPARSE:
						if ( lIt->y > 0 )
							tmpPointsPerClass[lIt->idx]++;
						break;
				}
			}
		}
		
		_nExamplesPerClass.clear();
		for (int l = 0; l < this->_pData->getNumClasses(); ++l)
			_nExamplesPerClass.push_back( tmpPointsPerClass[l] );


		this->_numExamples = ind.size();
		return 0;
	}


	// ------------------------------------------------------------------------
	void		InputData::clearIndexSet( void ) {
		this->_usedIndices.clear();
		this->_rawIndices.clear();

		for( int i = 0; i < this->_pData->getNumExample(); i++ ) {
			this->_indirectIndices[ i ] = i;
			this->_usedIndices.insert( i );
			this->_rawIndices[ i ] = i;
		}
		this->_numExamples = this->_pData->getNumExample();
		
		_nExamplesPerClass.clear();
		_nExamplesPerClass = this->_pData->getExamplesPerClass();
	}

} // end of namespace MultiBoost
