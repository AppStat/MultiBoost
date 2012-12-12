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
 * \file ArffParser.h A parser for ARFF file format
 */
#pragma warning( disable : 4786 )


#ifndef __ARFF_PARSER_H
#define __ARFF_PARSER_H

#include <fstream>
#include <sstream>
#include "GenericParser.h"
#include "NameMap.h"
#include "InputData.h"

using namespace std;

namespace MultiBoost {
        
    class ArffParser : public GenericParser
    {
    public:
                
        /**
         * The constructor. It initializes the file names and the separators.
         * \date 30/07/2010
         */
        ArffParser(const string& fileName, const string& headerFileName);
                
                
        /**
         * Read the data.
         * \see GenericParser::readData
         * \date 30/07/2010
         */             
        virtual void readData(vector<Example>& examples, NameMap& classMap, 
                              vector<NameMap>& enumMaps, NameMap& attributeNameMap,
                              vector<RawData::eAttributeType>& attributeTypes);
                
        /**
         * It retrurns the number of features. 
         * \remark It might be implemented here, and not as an abstract function.
         * \return The number of features.
         * \data 30/07/2011 
         */             
        virtual int  getNumAttributes() const
        { return _numAttributes; }
                
    protected:
        /**
         * Read the header. It reads the class labels, attribute names, attribute types and
         * the mappings of nominal features.
         * \param in The file stream.
         * \param \see GenericParser::readData           
         * \data 30/07/2011 
         */
        void readHeader(ifstream& in, NameMap& classMap, 
                        vector<NameMap>& enumMaps, NameMap& attributeNameMap, 
                        vector<RawData::eAttributeType>& attributeTypes);

        /**
         * Read the data. The arff can be sparse and dense as well.
         * \param in The file stream.
         * \param \see GenericParser::readData           
         * \data 30/07/2011              
         */
        void readData(ifstream& in, vector<Example>& examples, NameMap& classMap, 
                      vector<NameMap>& enumMaps, 
                      const vector<RawData::eAttributeType>& attributeTypes);
                
        string readName(ifstream& in);
                
        void readDenseValues(ifstream& in, vector<FeatureReal>& values,
                             vector<NameMap>& enumMaps, 
                             const vector<RawData::eAttributeType>& attributeTypes);
                
        void readSparseValues(istringstream& ss, vector<FeatureReal>& values, vector<int>& idxs, map<int, int>& idxmap, 
                              vector<NameMap>& enumMaps, 
                              const vector<RawData::eAttributeType>& attributeTypes);
                
        /**
         * Read labels declared in the standard arff format:
         * each class (label) is set to +1 if it's there, otherwise is -1
         * i.e.:
         * \verbatim
         @ATTRIBUTE sepallength  NUMERIC
         @ATTRIBUTE sepalwidth   NUMERIC
         @ATTRIBUTE petallength  NUMERIC
         @ATTRIBUTE petalwidth   NUMERIC
         @ATTRIBUTE class        {Iris-setosa, Iris-versicolor, Iris-virginica}
         @DATA
         4,  2,  4,  2, Iris-setosa
         25, 23,  1,  0, Iris-versicolor, Iris-virginica
         0,  1, 10, 12, Iris-virginica
         \endverbatim
         * In this case the labels the the example will respectively be:
         \verbatim
         +1, -1, -1
         -1, +1, +1
         -1, -1, +1
         \endverbatim
        */
        void readSimpleLabels(istringstream& ss, vector<Label>& labels, NameMap& classMap);

        /**
         * Read labels declared in the standard arff format, but specified as 
         * multiple prefixed "class" Attributes, and with a value that 
         * can be positive, or negative.
         * i.e.:
         * \verbatim
         @ATTRIBUTE sepallength  NUMERIC
         @ATTRIBUTE sepalwidth   NUMERIC
         @ATTRIBUTE petallength  NUMERIC
         @ATTRIBUTE petalwidth   NUMERIC
         @ATTRIBUTE classIris-setosa NUMERIC
         @ATTRIBUTE classIris-versicolor NUMERIC
         @ATTRIBUTE classIris-virginica NUMERIC
         @DATA
         4,  2,  4,  2, +1, -2, -1
         25, 23,  1,  0, -1, 0, +1
         0,  1, 10, 12, -1, -3, +1
         \endverbatim
         * In this case the labels the the example will respectively be:
         \verbatim
         +1, -1, -1
         -1,  0, +1
         -1, -1, +1
         \endverbatim
         * and weights :
         \verbatim
         1, 2, 1,
         1, 0, 1
         1, 3, 1
         \endverbatim
        */
        void readDenseMultiLabels(istringstream& ss, vector<Label>& labels, NameMap& classMap);

        /**
         * Read labels in the same manner than \verb+readDenseMultiLabels+, but with
         * sparse data (hence sparse labels, then)
         */
        void readSparseMultiLabels(istringstream& ss, vector<Label>& labels, NameMap& classMap);
                
        /**
         * Read sparse labels declared in a non-standard arff variant:
         * each class (label) is declared with a value that can be positive, or negative
         * left out labels are automatically considered zero (abstention!).
         * i.e.:
         * \verbatim
         @ATTRIBUTE sepallength  NUMERIC
         @ATTRIBUTE sepalwidth   NUMERIC
         @ATTRIBUTE petallength  NUMERIC
         @ATTRIBUTE petalwidth   NUMERIC
         @ATTRIBUTE class        {Iris-setosa, Iris-versicolor, Iris-virginica}
         @DATA
         4,  2,  4,  2, {Iris-setosa -2} 
         25, 23,  1,  0, {Iris-versicolor 1, Iris-virginica -1}
         0,  1, 10, 12, {Iris-setosa +2, Iris-versicolor -1, Iris-virginica -3}
         \endverbatim
         * The sign is used to set the value of y[l], and the magnitude to initialize the weights.
         * In this case the labels the the example will respectively be:
         \verbatim
         -1, 0, 0
         0, +1, -1
         +1, -1, -1
         \endverbatim
         * \remark Internally this type of label is stored as sparse. This will have
         * a small hit in terms of memory, but nothing in terms of performance.
         */
        void readExtendedLabels(istringstream& ss, vector<Label>& labels, NameMap& classMap);
                
        enum eTokenType
        {
            TT_EOF,
            TT_COMMENT,
            TT_RELATION,
            TT_ATTRIBUTE,
            TT_DATA,
            TT_UNKNOWN
        };
                
        eTokenType getNextTokenType(ifstream& in);
                
        int            _numAttributes;
        int                        _lastIdx;
        string         _headerFileName;
                
        locale         _denseLocale;
        locale         _sparseLocale;
        bool           _hasName;
        bool           _hasAttributeClassForm;        

    };
        
    // -----------------------------------------------------------------------------
        
    inline string ArffParser::readName(ifstream& in)
    {
        const locale& originalLocale = in.imbue(_denseLocale); 
        string name;
        in >> name;
        in.imbue(originalLocale);
        return name;
    }
        
    // -----------------------------------------------------------------------------
        
        
} // end of namespace MultiBoost

#endif // __ARFF_PARSER_H
