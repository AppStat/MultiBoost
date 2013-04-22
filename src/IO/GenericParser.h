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


/**
 * \file GenericParser.h Defines an abstract class for parsing files
 */

#ifndef __GENERIC_PARSER_H
#define __GENERIC_PARSER_H

#include <string>
#include <vector>
#include <algorithm>

#include "Others/Example.h"
#include "NameMap.h"
#include "InputData.h"

using namespace std;

namespace MultiBoost
{
        
    ////////////////////////////////////////////////////////////////
        
    class GenericParser 
    { 
    public:
        /**
         * The constructor. It initializes the data file and the header file.
         * \remark The data and label representation are set to dense by default.
         * \data 30/07/2011 
         */
    GenericParser(const string& fileName, const string& headerFileName)
        : _fileName(fileName), _headerFileName(headerFileName), 
            _dataRep(DR_DENSE), _labelRep(LR_DENSE), _hasWeigthInit(false) {}
                
        /**
         * Abstract function for reading data.
         * \param examples The vector of examples to be filled up.
         * \param classMap The map which stores the class names.
         * \param enumMaps It stores the mapping of nominal values into natural numbers.
         * \param attributeNameMap The name of attributes given by the user.
         * \param attributeTypes The type of attributes, for example numerical or nominal. \see RawData::eAttributeType
         * \data 30/07/2011 
         */
        virtual void            readData(vector<Example>& examples, NameMap& classMap, 
                                         vector<NameMap>& enumMaps, NameMap& attributeNameMap,
                                         vector<RawData::eAttributeType>& attributeTypes) = 0;
                
        /**
         * It retrurns the number of features. 
         * \remark It might be implemented here, and not as an abstract function.
         * \return The number of features.
         * \data 30/07/2011 
         */
        virtual int       getNumAttributes() const = 0;
                
        /**
         * It gets the data representation, i.e. sparse or dense.
         * \return The type of data representation 
         * \data 30/07/2011              
         */
        const eDataRep    getDataRep()  const { return _dataRep; }
                
        /**
         * It gets the label representation, i.e. sparse or dense.
         * \return The type of label representation 
         * \data 30/07/2011              
         */             
        const eLabelRep   getLabelRep() const { return _labelRep; }
                                
        /**
         * It gets whether there is an intial weighting or not.
         * \remark Initial weights can be set only with sparse labels.
         * \return If the user gave initial weights for the labels, it returns with true, othervise with false.
         * \data 30/07/2011 
         */
        bool  hasWeightInitialized()    const { return _hasWeigthInit; }

        /** 
         * Verbose Level for parsing stages
         *
         */
        int _verboseLevel;

    protected:
        /**
         * The data file name.
         * \data 30/07/2011 
         */                             
        const string& _fileName;
                
        /**
         * If there is given a separate header file then this variable stores its name.
         * \data 30/07/2011 
         */
        const string& _headerFileName;
                
        /**
         * The data representation, i.e. sparse or dense.
         * \data 30/07/2011 
         */             
        eDataRep      _dataRep;
                
        /**
         * The label representation.
         * \data 30/07/2011 
         */
        eLabelRep     _labelRep;
                
        /**
         * In the case of sparse label representation the user is allowed to provide initial weigths for each label. 
         * If the user do this this flag is set to true.
         * \data 30/07/2011 
         */
        bool          _hasWeigthInit;
        

    };
        
    ////////////////////////////////////////////////////////////////
        
};

#endif // __GENERIC_PARSER_H
