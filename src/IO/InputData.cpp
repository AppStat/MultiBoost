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
 *    Contact: multiboost@googlegroups.com 
 * 
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
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
    
    
    void InputData::addExample( Example example ) { 
        
        int exampleIndex = _pData->getNumExample();
        
        _pData->addExample(example);
        const vector<Label> & labels = example.getLabels();
        vector<Label>::const_iterator lIt;
        
        for (lIt = labels.begin(); lIt != labels.end(); ++lIt) {
            if ( lIt->y > 0) {
                _nExamplesPerClass[lIt->idx]++;
            }
        }
        
        this->_indirectIndices.push_back(exampleIndex);
        this->_rawIndices[exampleIndex] = this->_numExamples;
        
        this->_numExamples++;
        
        //why shouldn't we do the same direct update of _subset in loadIndexSet() ?
        _subset.push_back(example);
        
        //otherwise : _subsetAlreadyComputed = false;
    }
        
    // ------------------------------------------------------------------------
        
    int     InputData::loadIndexSet( set< int > ind ) {
        int i = 0;
        //upload the indirection
        for (int j = 0; j < this->_rawIndices.size(); j++) this->_rawIndices[j] = -1;

        map<int, int> tmpPointsPerClass;
                
        for( set< int >::iterator it = ind.begin(); it != ind.end(); it++ ) {
            this->_indirectIndices[i] = *it;
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
                
                
        this->_numExamples = (int)ind.size();
        
        _subsetAlreadyComputed = false;
        
        return 0;
    }
        
        
    // ------------------------------------------------------------------------
        
    void InputData::clearIndexSet( void ) {
                
        for( int i = 0; i < this->_pData->getNumExample(); i++ ) {
            this->_indirectIndices[ i ] = i;
            this->_rawIndices[ i ] = i;
        }
        this->_numExamples = this->_pData->getNumExample();
                
        _nExamplesPerClass.clear();
        _nExamplesPerClass = this->_pData->getExamplesPerClass();
        
        _subsetAlreadyComputed = false;
    }
        
} // end of namespace MultiBoost
