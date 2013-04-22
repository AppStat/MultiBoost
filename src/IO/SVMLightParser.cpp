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


#include <iostream>
#include <cmath> // for abs

#include "IO/SVMLightParser.h"
#include "Utils/Utils.h"

namespace MultiBoost {
        
    // ------------------------------------------------------------------------
        
    SVMLightParser::SVMLightParser(const string& fileName,const string& headerFileName)
        : GenericParser(fileName,headerFileName), _hasName(false), _numRows(0)
    {
        _sparseLocale = locale(locale(), new nor_utils::white_spaces(": "));
        _labelLocale = locale(locale(), new nor_utils::white_spaces(", "));
    }
        
    // ------------------------------------------------------------------------
        
    void SVMLightParser::readData( vector<Example>& examples, NameMap& classMap, 
                                   vector<NameMap>& enumMaps, NameMap& attributeNameMap,
                                   vector<RawData::eAttributeType>& attributeTypes )
    {
        // open file
        ifstream inFile(_fileName.c_str());
        if ( !inFile.is_open() )
        {
            cerr << "\nERROR: Cannot open file <" << _fileName << ">!!" << endl;
            exit(1);
        }
                
        _dataRep = DR_SPARSE;
        _labelRep = LR_DENSE;
                
        if ( ! _headerFileName.empty() ) // there is no file name
            readHeader( classMap, enumMaps, attributeNameMap, attributeTypes);
        else
            getHeaderInfoFromData(inFile, classMap, enumMaps, attributeNameMap, attributeTypes );
        readData(inFile, examples, classMap, enumMaps, attributeNameMap, attributeTypes);
    }
        
    // ------------------------------------------------------------------------
    void SVMLightParser::getHeaderInfoFromData(ifstream& in, NameMap& classMap, 
                                               vector<NameMap>& enumMaps, NameMap& attributeNameMap,
                                               vector<RawData::eAttributeType>& attributeTypes)
    {
        string tmpLine;
        int maxLabel = -1;
        int maxFeatureIndex = -1;
        // get read pointer             
        int numOfRaws = 0;
                
        while (getline(in, tmpLine) )
        {
            if (tmpLine[0] == '#' ) continue;
                        
            numOfRaws++;
                        
            istringstream ss;
            ss.imbue(_sparseLocale);
            ss.clear();
            ss.str(tmpLine);
                        
            string lab;
            ss >> lab;                      
                        
            stringstream labelStream(lab);
            labelStream.imbue( _labelLocale );
                        
            // read the label
            while (!labelStream.eof())
            {
                int tmpLab;
                labelStream >> tmpLab;
                                
                if (tmpLab>maxLabel) maxLabel = tmpLab;
            }
                        
                        
                        
            //read the name of features
            while (!ss.eof() && !ss.fail())
            {
                //read the index of the feature                         
                string tmpFeatName;
                ss >> tmpFeatName;
                if (tmpFeatName.compare("qid")!=0) { 
                    int featureIndex;
                    istringstream iss(tmpFeatName);
                    iss >> featureIndex;
                    if (maxFeatureIndex<featureIndex) maxFeatureIndex = featureIndex;
                }
                                
                FeatureReal tmpVal;
                ss >> tmpVal;
            }
                        
        }
                
        // Put read pointer back to where it was
        in.clear();
        in.seekg(0, ios::beg);
        //in.clear();
        if (!in.good())
            cout << "File is not good" << endl;
                
        _numRows = numOfRaws;
                
        // fill out the type attributes
        for(int i = 1; i <= maxFeatureIndex; ++i ) 
        {
            stringstream ss;
            ss << i;
            attributeNameMap.addName( ss.str() );
        }
                
        attributeTypes.resize( attributeNameMap.getNumNames() );
        fill( attributeTypes.begin(), attributeTypes.end(), RawData::ATTRIBUTE_NUMERIC );

                
        for(int i = 0; i <= maxLabel; ++i ) 
        {
            stringstream ss;
            ss << i;
            classMap.addName( ss.str() );
        }
                
    }
        
        
    // ------------------------------------------------------------------------
    void SVMLightParser::readHeader( NameMap& classMap, 
                                     vector<NameMap>& enumMaps, NameMap& attributeNameMap,
                                     vector<RawData::eAttributeType>& attributeTypes )
    {
        if (_verboseLevel > 0) cout << "Reading header file (" << _headerFileName << ")...";
        ifstream inHeaderFile(_headerFileName.c_str());
        if ( !inHeaderFile.is_open() )
        {
            cerr << "\nERROR: Cannot open file <" << _headerFileName << ">!!" << endl;
            exit(1);
        }
                
        string tmpLine;
                
        getline(inHeaderFile, tmpLine);
        istringstream ss;
        ss.imbue(_sparseLocale);
        ss.clear();
        ss.str(tmpLine);
                
        //read the class labels
        while (!ss.eof())
        {                       
            //read the name of the label
            string tmpLab;
            ss >> tmpLab;
                        
            if ( tmpLab.empty() ) continue;
            nor_utils::trim( tmpLab );
            if ( tmpLab.empty() ) continue;
                        
            //add the class name to the namemap if it doesn't exist
            classMap.addName( tmpLab );
        }
                
        // read the feature names
        getline(inHeaderFile, tmpLine);
        ss.clear();
        ss.str(tmpLine);
                
        while (!ss.eof())
        {
            //read the name of the label
            string tmpFeatName;
            ss >> tmpFeatName;                                              
                        
            if ( tmpFeatName.empty() ) continue;
            nor_utils::trim( tmpFeatName );
            if ( tmpFeatName.empty() ) continue;
                        
            //add the feat name to the namemap if it doesn't exist
            attributeNameMap.addName( tmpFeatName );
        }
                
        attributeTypes.resize( attributeNameMap.getNumNames() );
        fill( attributeTypes.begin(), attributeTypes.end(), RawData::ATTRIBUTE_NUMERIC );
                
        // read the label weighting if it is given
        getline(inHeaderFile, tmpLine);
        if ( ! tmpLine.empty() )
        {
            if (_verboseLevel > 0) cout << "Read weighting...";
            _weightOfClasses.clear();
                        
            ss.clear();
            ss.str(tmpLine);
            for ( int i=0; i < classMap.getNumNames(); i++ )
            {
                //read the name of the label
                AlphaReal tmpWeight;
                ss >> tmpWeight;
                                
                _weightOfClasses.insert( pair<int, AlphaReal>( i, tmpWeight ) );                                   
            }
                        
        }
                
        inHeaderFile.close();

        if (_verboseLevel > 0) cout << "Done." << endl;
    }
    // ------------------------------------------------------------------------
        
    void SVMLightParser::readData( ifstream& in, vector<Example>& examples,
                                   NameMap& classMap, vector<NameMap>& enumMaps, NameMap& attributeNameMap,
                                   vector<RawData::eAttributeType>& attributeTypes )
    {
        char firstChar = 0;
        string tmpLine;
        int maxColumnIdx = 0;;
                
        istringstream ss;
        ss.imbue(_sparseLocale);
                
        if ( _numRows == 0 ) // if header file was provided, the data had not been parsed
        {
            if (_verboseLevel > 0) cout << "Counting rows.." << flush;
            _numRows = nor_utils::count_rows(in);
        }
                
        if (_verboseLevel > 0) cout << "Allocating.." << flush;
        try {
            examples.resize(_numRows);
        } 
        catch(...) {
            cerr << "ERROR: Cannot allocate memory for storage!" << endl;
            exit(1);
        }
        if (_verboseLevel > 0) cout << "Done!" << endl;
                
        if (_verboseLevel > 0) cout << "Now reading file.." << flush;
        int i;
        vector<vector<int> > tmpLabelIdxs( _numRows );
                
        size_t currentSize = 0;
        for (i = 0; i < _numRows; ++i)
        {
            while ( isspace(firstChar = in.get()) && !in.eof() );
                        
            if (in.eof())
                break;                                          
                        
            //////////////////////////////////////////////////////////////////////////
            // first read the data
            if ( firstChar == '#' ) // comment!
            {
                getline(in, tmpLine); // read the comment line and continue
                continue;
            }
                        
                        
            in.putback(firstChar);
                                                
            // read the next line
            getline(in, tmpLine);
            tmpLine = nor_utils::trim( tmpLine );
            ss.clear();
            ss.str(tmpLine);
                        
            // now read the labels                  
            string currLab;
            ss >> currLab;
                                                
            stringstream labelStream(currLab);
            labelStream.imbue( _labelLocale );
            tmpLabelIdxs[currentSize].clear();
            // read the label
            while (!labelStream.eof())
            {
                string tmp;
                labelStream >> tmp;
                                
                int tmpLabIDX = classMap.addName( tmp );
                tmpLabelIdxs[currentSize].push_back( tmpLabIDX );
            }                                               
                        
            Example& currExample = examples[currentSize];
            //now read values
            readSparseValues(ss, currExample.getValues(), currExample.getValuesIndexes(), currExample.getValuesIndexesMap(),
                             enumMaps, attributeTypes, attributeNameMap );
                        
            currentSize++;
        }
                
        if ( attributeTypes.empty() )
        {
            attributeTypes.resize( attributeNameMap.getNumNames() );
            fill( attributeTypes.begin(), attributeTypes.end(), RawData::ATTRIBUTE_NUMERIC );
        }
        if ( currentSize != _numRows )
        {
            // for example last row was empty!
            examples.resize( currentSize );
        }
                
        for( int i=0; i<examples.size(); i++ )
        {
            Example& currExample = examples[i];
            allocateSimpleLabels( tmpLabelIdxs[i], currExample.getLabels(), classMap );
        }
                
        if (_verboseLevel > 0) cout << "Done!" << endl;
                
        // set the number of attributes, it can be problem, that the test dataset contains such attributes which isn't presented in the training data
        _numAttributes = attributeNameMap.getNumNames();
                
        // sparse representation always set the weight!
        if ( _labelRep == LR_SPARSE )
            _hasWeigthInit = true;
    }
        
        
        
    // ------------------------------------------------------------------------
        
    void SVMLightParser::readDenseValues(ifstream& in, vector<FeatureReal>& values,
                                         vector<NameMap>& enumMaps, 
                                         const vector<RawData::eAttributeType>& attributeTypes )
    {
        const locale& originalLocale = in.imbue(_denseLocale); 
                
        values.reserve(_numAttributes);
        string tmpVal;
                
        for ( int j = 0; j < _numAttributes; ++j )
        {
            in >> tmpVal;
            if ( attributeTypes[j] == RawData::ATTRIBUTE_NUMERIC ) 
                if ( ( ! tmpVal.compare( "NaN" ) ) || ( ! tmpVal.compare( "?" ) ) )
                    values.push_back( numeric_limits<FeatureReal>::infinity() );
                else
                    values.push_back(atof(tmpVal.c_str()));
            else //if ( attributeTypes[i] == RawData::ATTRIBUTE_ENUM ) 
                values.push_back( enumMaps[j].getIdxFromName(tmpVal) );
        }
                
        in.imbue(originalLocale);
    }
        
    // -----------------------------------------------------------------------------
        
    void SVMLightParser::readSparseValues(istringstream& ss, vector<FeatureReal>& values, 
                                          vector<int>& idxs, map<int, int>& idxmap, vector<NameMap>& enumMaps, 
                                          const vector<RawData::eAttributeType>& attributeTypes, NameMap& attributeNameMap)
    {
        FeatureReal tmpFeatVal;
        string tmpFeatName;
        int i = 0;
        while (!ss.eof() && !ss.fail())
        {
            //read the name of the next feature name and its value
            ss >> tmpFeatName;
            ss >> tmpFeatVal;
                        
            // discard the query ID
            if (tmpFeatName.compare("qid")==0) continue;
                        
            //add the feature to the namemap if it doesn't exist
            int tmpIdx = attributeNameMap.addName( tmpFeatName );
                        
            // add the feature value and its index
            idxs.push_back(tmpIdx);
            idxmap[ tmpIdx ] = i++;
            values.push_back( tmpFeatVal );
        }
    }
        
    // ------------------------------------------------------------------------
        
    void SVMLightParser::allocateSimpleLabels( vector<int>& labelIdx, vector<Label>& labels,
                                               NameMap& classMap )
    {
                
        const int numClasses = classMap.getNumNames();
        labels.resize(numClasses);
                
        for ( int i = 0; i < numClasses; ++i )
        {
            labels[i].idx = i;
            labels[i].y = -1;
        }
                
        // now set the declared labels
        for ( int i = 0; i < labelIdx.size(); ++i )
            labels[ labelIdx[i] ].y = +1;
                
        if ( ! _weightOfClasses.empty() ) // weighting
        {
            for ( int i = 0; i < labelIdx.size(); ++i )
                labels[ labelIdx[i] ].weight = _weightOfClasses[ labelIdx[i] ];         
        }
    }
        
} // end of namespace MultiBoost
