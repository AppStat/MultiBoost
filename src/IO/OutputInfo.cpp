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


#include <limits>

#include <math.h>

#include "OutputInfo.h"
#include "NameMap.h"
#include "WeakLearners/BaseLearner.h"
#include "Others/Example.h"
#include "Utils/Utils.h"

namespace MultiBoost {

    // -------------------------------------------------------------------------

    OutputInfo::OutputInfo(const nor_utils::Args& args, bool customUpdate, const string & clArg)
    {
        _customTablesUpdate = customUpdate;
     
        _historyStartingIteration = 0;
        _outputList.clear();
        
        string outputInfoFile;
        args.getValue(clArg, 0, outputInfoFile);
        
        if ( args.getNumValues(clArg) > 1)
        {
            string outList;
            args.getValue(clArg, 1, outList);
            getOutputListFromString(outList, &args);
        }
        else
        {
            _outputList[defaultOutput] = BaseOutputInfoType::createOutput(defaultOutput);
        }

        // open the stream
        
        ios_base::openmode outputfileflag = ios_base::out;
        if (args.hasArgument("resume") &&
            !args.hasArgument("slowresumeprocess")) {
            outputfileflag = ios_base::in | ios_base::out | ios_base::ate;
        }
        _outStream.open(outputInfoFile.c_str(), outputfileflag);
        
        // is it really open?
        if ( !_outStream.is_open() )
        {
            outputfileflag = ios_base::out;
            _outStream.open(outputInfoFile.c_str(), outputfileflag);
            if ( !_outStream.is_open() ) {
                cerr << "ERROR: cannot open the output steam (<"
                     << outputInfoFile << ">) for the step-by-step info!" << endl;
                exit(1);
            }
        }
        
        // the header file
        string headerFileName = outputInfoFile + ".header";
        ios_base::openmode headerflag = ios_base::out;
        if (outputfileflag != ios_base::out) {
            headerflag = ios_base::in;
        }
        
        _headerOutStream.open(headerFileName.c_str(), headerflag);
        if ( !_headerOutStream.is_open() )
        {
            cerr << "WARNING: cannot find the header output steam (<"
                 << headerFileName << ">). A new header file will be created." << endl;
            
            // try to create the file if it doesn't exist
            headerflag = ios_base::out;
            _headerOutStream.open(headerFileName.c_str(), headerflag);
            if ( !_headerOutStream.is_open()) {
                cerr << "ERROR: cannot open the header output steam (<"
                     << headerFileName << ">) for the step-by-step info!" << endl;
                exit(1);
            }
        }
        
        _timeBias = 0;
        
        if (headerflag == ios_base::in) {
            
            // look for the position of the time
            // column (if it exists) in the header
            int position = -1, i = 0;
                        
            while (! _headerOutStream.eof()) {
                string column;
                _headerOutStream >> column;
                if (column.compare("time") == 0) {
                    position = i;
                    break;
                }
                ++i;
            }
            
            // if the time actually belongs to the header
            if (position >= 0) {
                // if "t" belongs to the header and the outputfile is not empty
                
                string readLine;
                _outStream.seekg(0, ios_base::beg);
                
                while (! _outStream.eof()) {
                    string tmpLine;
                    getline(_outStream, tmpLine);
                    if (! tmpLine.empty()) {
                        readLine = tmpLine;
                    }

                }
                
                if (! readLine.empty()) {
                    istringstream tmpLineSs(readLine);
                    string timeToken;
                    while (! tmpLineSs.eof()) {
                        string tmpToken("0");
                        getline(tmpLineSs, tmpToken, '\t');
                        if (! tmpToken.empty()) {
                            timeToken = tmpToken;
                        }
                    }
                    istringstream tmpOss(timeToken);
                    tmpOss  >> _timeBias;

                }
                
                _outStream.clear();
                _outStream.seekp(0, ios_base::end);

            }
        }
    }

    // -------------------------------------------------------------------------
        
    void OutputInfo::setOutputList(const string& list, const nor_utils::Args* args)
    {
        getOutputListFromString(list, args);
    }
    // -------------------------------------------------------------------------
    
    void OutputInfo::getOutputListFromString(const string& outList,  const nor_utils::Args* args)
    {
        for (int i = 0; i < outList.size(); i+=3) 
        {
            string outputName = outList.substr(i, 3);
            if (_outputList.find(outputName) == _outputList.end())
            {
                BaseOutputInfoType* t = BaseOutputInfoType::createOutput(outputName, args);
                if ( t )  _outputList[outputName] = t;
            }
        }
        
        if ( _outputList.size() == 0 )
            _outputList[defaultOutput] = BaseOutputInfoType::createOutput(defaultOutput);


    }
    
    // -------------------------------------------------------------------------
    
    
    void OutputInfo::outputHeader(const NameMap& namemap, bool outputIterations, bool outputTime, bool endline)
    { 
        if (outputIterations) {
            _headerOutStream << "t" << OUTPUT_SEPARATOR;
        }
        
        OutInfIt outputIt;
        long numDatasets = _gTableMap.size()  ;

        // the number of datasets used
        for (int i = 0; i < numDatasets; ++i) {
            for (outputIt = _outputList.begin(); outputIt != _outputList.end(); ++outputIt) {
                outputIt->second->outputHeader(_headerOutStream, namemap);
                _headerOutStream << OUTPUT_SEPARATOR;
            }
        }
        
        if (outputTime) {
            _headerOutStream << "time" << OUTPUT_SEPARATOR;
        }
        
        if (endline) {
            _headerOutStream << endl << endl << "Column description:\n-------------------\n";
            
            // description of the different columns
            for (outputIt = _outputList.begin(); outputIt != _outputList.end(); ++outputIt) {
                outputIt->second->outputDescription(_headerOutStream);
                _headerOutStream << endl;
            }
        }
    }
        
    // -------------------------------------------------------------------------
        
    void OutputInfo::outputCustom(InputData* pData, BaseLearner* pWeakHypothesis)
    {
        if (! _customTablesUpdate) {
            updateTables(pData, pWeakHypothesis);
        }
        
        size_t numOutput = _outputList.size();
        
        _outStream << setiosflags(ios::fixed) << setprecision(6);
        
        OutInfIt outputIt;
        int i = 0;
        for (outputIt = _outputList.begin(); outputIt != _outputList.end(); ++outputIt, ++i) {
            outputIt->second->computeAndOutput(_outStream, pData, _gTableMap, _margins, _alphaSums, pWeakHypothesis);
            if ((i+1) != numOutput) _outStream << OUTPUT_SEPARATOR;
        } 
        
    }
    
    // -------------------------------------------------------------------------
    void OutputInfo::updateTables(InputData* pData, BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
                
        table& g = _gTableMap[pData];
        table& margins = _margins[pData];
        
        vector<Label>::const_iterator lIt;
                
        // Building the strong learner (discriminant function)
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
                        
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                AlphaReal hx = pWeakHypothesis->getAlpha() * // alpha
                    pWeakHypothesis->classify(pData, i, lIt->idx);
                
                // update the posteriors table
                g[i][lIt->idx] += hx ;
                
                // update the margins table
                // FIXME: redundancy 
                margins[i][lIt->idx] += hx * lIt->y ;
            }
        }
        
        // update the sum of alphas
        _alphaSums[pData] += pWeakHypothesis->getAlpha();
        
    }

    // -------------------------------------------------------------------------

//    void OutputInfo::updateSpecificInfo(const string& type, AlphaReal value) 
//    {
//        OutInfIt outputIt;
//        for (outputIt = _outputList.begin(); outputIt != _outputList.end(); ++outputIt) {
//            (*outputIt)->updateSpecificInfo(type, value);
//        } 
//    }
    
    // -------------------------------------------------------------------------
    
    BaseOutputInfoType* OutputInfo::getOutputInfoObject(const string& type)
    {
        return _outputList[type];
    }
    
    // -----------------------------------------------------------------------------------
    
    AlphaReal OutputInfo::getOutputHistory(InputData *pData, const string& outputName, int iteration)
    { return _outputList[outputName]->getOutputHistory(pData, iteration - _historyStartingIteration); }
    
    // -----------------------------------------------------------------------------------
    
    bool OutputInfo::outputIsActivated(const string& outputName)
    {
        OutInfIt outputIt = _outputList.find(outputName);
        return !(outputIt == _outputList.end());
    }
    
    
    // -------------------------------------------------------------------------
    
    void OutputInfo::outputIteration(int t)
    { 
        _outStream << (t+1) << OUTPUT_SEPARATOR; // just output t
    }
    // -------------------------------------------------------------------------
        
    void OutputInfo::outputCurrentTime( void )
    { 
        time_t seconds;
        seconds = time (NULL);
                
        _outStream << OUTPUT_SEPARATOR << (seconds - _beginingTime + _timeBias); // just output current time in seconds
    }
        
    // -------------------------------------------------------------------------
        
    void OutputInfo::initialize(InputData* pData)
    { 
        _beginingTime = time( NULL );
        int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();
        
        table& g = _gTableMap[pData];
        
        g.resize(numExamples);
        
        for ( int i = 0; i < numExamples; ++i )
        {
            //if (pData->isSparseLabel())
            //   numClasses = pData->getNumNonzeroLabels(i);
            g[i].resize(numClasses);
            for (int l = 0; l < numClasses; ++l)
                g[i][l] = 0;
        }
        
        table& margins = _margins[pData];
        margins.resize(numExamples);
        
        for ( int i = 0; i < numExamples; ++i )
        {
            //if (pData->isSparseLabel())
            //   numClasses = pData->getNumNonzeroLabels(i);
            margins[i].resize(numClasses);
            for (int l = 0; l < numClasses; ++l)
                margins[i][l] = 0;
        }
        
        _alphaSums[pData] = 0;

    }
        
#pragma mark -
#pragma mark factory method 
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------
    
    BaseOutputInfoType* BaseOutputInfoType::createOutput(string type, const nor_utils::Args* args)
    {
        if ( type.compare("e01") == 0 ) return new ZeroOneErrorOutput();                 // 0-1 error
        if ( type.compare("w01") == 0 ) return new WeightedZeroOneErrorOutput(); // weighted 0-1 error
        if ( type.compare("ham") == 0 ) return new HammingErrorOutput();                 // Hamming loss
        if ( type.compare("wha") == 0 ) return new WeightedHammingErrorOutput();                 // Weighted Hamming loss               
        if ( type.compare("r01") == 0 ) return new RestrictedZeroOneError();     // restricted 0-1 error
        if ( type.compare("wer") == 0 ) return new WeightedErrorOutput();
        if ( type.compare("ber") == 0 ) return new BalancedErrorOutput();
        if ( type.compare("mae") == 0 ) return new MAEOuput();
        if ( type.compare("mar") == 0 ) return new MarginsOutput();
        if ( type.compare("edg") == 0 ) return new EdgeOutput();
        if ( type.compare("auc") == 0 ) return new AUCOutput();
        if ( type.compare("roc") == 0 ) return new TPRFPROutput();
        if ( type.compare("sca") == 0 ) return new SoftCascadeOutput(*args); 
        if ( type.compare("pos") == 0 ) return new PosteriorsOutput();
        
        cout << "Warning ! Unknown output type provide : " << type << endl;
        
        return NULL;
    }

#pragma mark Subclasses 
    
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void RestrictedZeroOneError::computeAndOutput(ostream& outStream, InputData* pData, 
                                                  map<InputData*, table>& gTableMap, 
                                                  map<InputData*, table>& marginsTableMap, 
                                                  map<InputData*, AlphaReal>& alphaSums,
                                                  BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        
        table& g = gTableMap[pData];
        
        vector<Label>::const_iterator lIt;
        
        int numErrors = 0;   
        
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
            
            // the vote of the winning negative class
            AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
            // the vote of the winning positive class
            AlphaReal minPosClass = numeric_limits<AlphaReal>::max();
            
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the negative winner class
                if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
                    maxNegClass = g[i][lIt->idx];
                
                // get the positive winner class
                if ( lIt->y > 0 && g[i][lIt->idx] < minPosClass )
                    minPosClass = g[i][lIt->idx];
            }
            
            // if the vote for the worst positive label is lower than the
            // vote for the highest negative label -> error
            if (minPosClass <= maxNegClass)
                ++numErrors;
        }
        
        
        // The error is normalized by the number of points
        AlphaReal output = (AlphaReal)(numErrors)/(AlphaReal)(numExamples);
        outStream  << output ;
        _outputHistory[pData].push_back(output);
        
    }

    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------    
        
    void ZeroOneErrorOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                              map<InputData*, table>& gTableMap, 
                                              map<InputData*, table>& marginsTableMap, 
                                              map<InputData*, AlphaReal>& alphaSums,
                                              BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        
        table& g = gTableMap[pData];
        
        vector<Label>::const_iterator lIt;
        
        int numErrors = 0;   
        
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
            
            // the vote of the winning negative class
            AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
            // the vote of the winning positive class
            AlphaReal maxPosClass = -numeric_limits<AlphaReal>::max();
            
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the negative winner class
                if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
                    maxNegClass = g[i][lIt->idx];
                
                // get the positive winner class
                if ( lIt->y > 0 && g[i][lIt->idx] > maxPosClass )
                    maxPosClass = g[i][lIt->idx];
            }
            
            // if the vote for the worst positive label is lower than the
            // vote for the highest negative label -> error
            if (maxPosClass <= maxNegClass)
                ++numErrors;
        }
        
        
        // The error is normalized by the number of points
        AlphaReal output = (AlphaReal)(numErrors)/(AlphaReal)(numExamples);
        outStream  << output ;
        _outputHistory[pData].push_back(output);
        
    }
                
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------    
        
    void WeightedZeroOneErrorOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                                      map<InputData*, table>& gTableMap, 
                                                      map<InputData*, table>& marginsTableMap, 
                                                      map<InputData*, AlphaReal>& alphaSums,
                                                      BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        
        table& g = gTableMap[pData];
        
        vector<Label>::const_iterator lIt;
        
        AlphaReal posWeights = 0.0;
        AlphaReal sumWeight = 0.0;
        int numErrors = 0;   
                
                
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
            
            // the vote of the winning negative class
            AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
            // the vote of the winning positive class
            AlphaReal maxPosClass = -numeric_limits<AlphaReal>::max();
            
            AlphaReal sumPerInstanceWeight = 0.0;
                        
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the negative winner class
                if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
                    maxNegClass = g[i][lIt->idx];
                
                // get the positive winner class
                if ( lIt->y > 0 && g[i][lIt->idx] > maxPosClass )
                    maxPosClass = g[i][lIt->idx];
                                
                if ( lIt->y > 0 ) sumPerInstanceWeight += (lIt->initialWeight>0.0) ? lIt->initialWeight : -lIt->initialWeight;
            }
            
            // if the vote for the worst positive label is lower than the
            // vote for the highest negative label -> error
            if (maxPosClass < maxNegClass) {
                ++numErrors;
            } else {
                posWeights += sumPerInstanceWeight;
            }
            sumWeight += sumPerInstanceWeight;
        }
        
        // The error is normalized by the sum of positive weights
        AlphaReal output = 1-(posWeights/sumWeight);
        outStream  << output ;
        _outputHistory[pData].push_back(output);
    }
    
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------    
        
    void HammingErrorOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                              map<InputData*, table>& gTableMap, 
                                              map<InputData*, table>& marginsTableMap, 
                                              map<InputData*, AlphaReal>& alphaSums,
                                              BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();
        
        table& g = gTableMap[pData];
        
        vector<Label>::const_iterator lIt;
        
        int numErrors = 0;   
        
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
            
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                if ( g[i][lIt->idx] * lIt->y < 0 ) numErrors++;
            }            
        }        
        
        // The error is normalized by the number of points
        AlphaReal output = (AlphaReal)(numErrors)/(AlphaReal)(numExamples*numClasses);
        outStream  << output ;
        _outputHistory[pData].push_back(output);
    }
    
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------    
        
    void WeightedHammingErrorOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                                      map<InputData*, table>& gTableMap, 
                                                      map<InputData*, table>& marginsTableMap, 
                                                      map<InputData*, AlphaReal>& alphaSums,
                                                      BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        
        table& g = gTableMap[pData];
        
        vector<Label>::const_iterator lIt;
        
        AlphaReal negWeights = 0.0;   
        AlphaReal sumWeights = 0.0;   
        
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
            
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                if ( g[i][lIt->idx] * lIt->y < 0 ) {
                    negWeights += lIt->initialWeight;
                }
                sumWeights += lIt->initialWeight;
            }            
        }
        
        
        // The error is normalized by the number of points
        AlphaReal output = (negWeights/sumWeights);
        outStream  << output ;
        _outputHistory[pData].push_back(output);
    }
        
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void WeightedErrorOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                               map<InputData*, table>& gTableMap, 
                                               map<InputData*, table>& marginsTableMap, 
                                               map<InputData*, AlphaReal>& alphaSums,
                                               BaseLearner* pWeakHypothesis)
    {
        int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();
                
        table& g = gTableMap[pData];
        vector<Label>::const_iterator lIt;
                
        AlphaReal posWeights = 0.0;
        AlphaReal sumWeight = 0;
        int numErrors = 0;   
                
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
                        
            // the vote of the winning negative class
            AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
            // the vote of the winning positive class
            AlphaReal minPosClass = numeric_limits<AlphaReal>::max();
                        
            AlphaReal sumPerInstanceWeight = 0.0;
                        
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the negative winner class
                if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
                    maxNegClass = g[i][lIt->idx];
                                
                // get the positive winner class
                if ( lIt->y > 0 && g[i][lIt->idx] < minPosClass )
                    minPosClass = g[i][lIt->idx];

                sumPerInstanceWeight += (lIt->initialWeight>0.0) ? lIt->initialWeight : -lIt->initialWeight;
            }
                        
            // if the vote for the worst positive label is lower than the
            // vote for the highest negative label -> error
            if (minPosClass <= maxNegClass) {
                ++numErrors; // just indicating that there is missclassification in this case                           
            } else {
                posWeights += sumPerInstanceWeight;
            }
            sumWeight += sumPerInstanceWeight;
        }
        
        // The error is normalized by the sum of positive weights
        AlphaReal output = 1-(posWeights/sumWeight);
        outStream  << output ;
        _outputHistory[pData].push_back(output);
    }

    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void BalancedErrorOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                               map<InputData*, table>& gTableMap, 
                                               map<InputData*, table>& marginsTableMap, 
                                               map<InputData*, AlphaReal>& alphaSums,
                                               BaseLearner* pWeakHypothesis)
    {
        
        
        int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();
        
        vector< int > tp( numClasses );   
        fill( tp.begin(), tp.end(), 0 );
        
        vector< int > tn( numClasses );   
        fill( tn.begin(), tn.end(), 0 );
        
        vector< AlphaReal > bacPerClass( numClasses );   
        fill( bacPerClass.begin(), bacPerClass.end(), 0.0 );
        
        table& g = gTableMap[pData];
        vector<Label>::const_iterator lIt;
        
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
            
            // the vote of the winning negative class
            AlphaReal maxNegClass = -numeric_limits<AlphaReal>::max();
            // the vote of the winning positive class
            AlphaReal minPosClass = numeric_limits<AlphaReal>::max();
            
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the negative winner class
                if ( lIt->y < 0 && g[i][lIt->idx] > maxNegClass )
                    maxNegClass = g[i][lIt->idx];
                    
                // get the positive winner class
                if ( lIt->y > 0 && g[i][lIt->idx] < minPosClass )
                    minPosClass = g[i][lIt->idx];
            }
            
            // if the vote for the worst positive label is higher than the
            // vote for the highest negative label -> good label
            if (minPosClass > maxNegClass){
                for ( lIt = labels.begin(); lIt != labels.end(); ++lIt ) {
                    if ( lIt->y > 0  ) {
                        tp[ lIt->idx]++;
                    } else { 
                        tn[ lIt->idx]++;
                    }
                }
            }
            
        }
        
        AlphaReal bACC = 0.0;
        
        for( int i = 0; i < numClasses; i++ ) {
            AlphaReal specificity = (((AlphaReal)tp[i]) / ((AlphaReal) pData->getNumExamplesPerClass( i ) ));
            AlphaReal sensitivity = ( ((AlphaReal)tn[i]) / ((AlphaReal) ( numExamples - pData->getNumExamplesPerClass( i ))) );
            bacPerClass[ i ] = 0.5 * ( specificity + sensitivity );
            bACC += bacPerClass[i];
        }
        
        bACC /= (AlphaReal) numClasses;
        
        outStream << bACC;
        _outputHistory[pData].push_back(bACC);
        
        for( int i = 0; i < numClasses; i++ ) {
            outStream << OUTPUT_SEPARATOR << bacPerClass[i];
        }
    }
    
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void MAEOuput::computeAndOutput(ostream& outStream, InputData* pData, 
                                    map<InputData*, table>& gTableMap, 
                                    map<InputData*, table>& marginsTableMap,
                                    map<InputData*, AlphaReal>& alphaSums,
                                    BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
                
        table& g = gTableMap[pData];
        vector<Label>::const_iterator lIt,maxlIt,truelIt;
        AlphaReal maxDiscriminant,mae = 0.0,mse = 0.0,tmpVal;
        char maxLabel;
                
        // Get label values: they must be convertible to AlphaReal
        vector<AlphaReal> labelValues;
        NameMap classMap = pData->getClassMap();
        for (int l = 0;l < classMap.getNumNames(); ++l)
            labelValues.push_back(atof(classMap.getNameFromIdx(l).c_str()));
                
        // Building the strong learner (discriminant function)
        for (int i = 0; i < numExamples; ++i){
            const vector<Label>& labels = pData->getLabels(i);
            maxDiscriminant = -numeric_limits<AlphaReal>::max();
            maxLabel = -100;
                        
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt ) {
                if ( g[i][lIt->idx] > maxDiscriminant ) {
                    maxDiscriminant = g[i][lIt->idx];
                    maxlIt = lIt;
                }
                if ( lIt->y > maxLabel ) {
                    maxLabel = lIt->y;
                    truelIt = lIt;
                }        
            }
            tmpVal = labelValues[truelIt->idx] - labelValues[maxlIt->idx];
            mae += fabs(tmpVal);                                  
            mse += tmpVal * tmpVal;                               
        }
        
        AlphaReal firstOutput = mae/(AlphaReal)(numExamples);
        outStream << firstOutput << OUTPUT_SEPARATOR << sqrt(mse/(AlphaReal)(numExamples));
        _outputHistory[pData].push_back(firstOutput);
        
    }
    
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void MarginsOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                         map<InputData*, table>& gTableMap, 
                                         map<InputData*, table>& marginsTableMap, 
                                         map<InputData*, AlphaReal>& alphaSums,
                                         BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
                
        //    to be continued: single/dense/sparse                
        table& margins = marginsTableMap[pData];
                
        AlphaReal minMargin = numeric_limits<AlphaReal>::max();
        
        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the minimum margin among classes and examples
                if (margins[i][lIt->idx] < minMargin)
                    minMargin = margins[i][lIt->idx];
            }
        }
                
        AlphaReal output = minMargin / alphaSums[pData] ;
        outStream << output;
        _outputHistory[pData].push_back(output);
    }
        
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void EdgeOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                      map<InputData*, table>& gTableMap, 
                                      map<InputData*, table>& marginsTableMap, 
                                      map<InputData*, AlphaReal>& alphaSums,
                                      BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
                
        AlphaReal gamma = 0; // the edge
        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                AlphaReal hy = pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
                    lIt->y;
                gamma += lIt->weight * hy;
            }
        }
                
        AlphaReal output = gamma; // edge
        outStream << output;
        _outputHistory[pData].push_back(output);
        
    }
        
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void AUCOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                     map<InputData*, table>& gTableMap, 
                                     map<InputData*, table>& marginsTableMap, 
                                     map<InputData*, AlphaReal>& alphaSums,
                                     BaseLearner* pWeakHypothesis)
    {
        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();
                
        vector< int > fp( numClasses );   
        fill( fp.begin(), fp.end(), 0 );
                
        table& g = gTableMap[pData];
        vector<Label>::const_iterator lIt;
                
        vector< pair< int, AlphaReal > > data( numExamples );
                
        vector< double > ROCscores( numClasses );
        fill( ROCscores.begin(), ROCscores.end(), 0.0 );
        double ROCsum = 0.0;
                
        for( int i=0; i < numClasses; i++ ) {
            if ( 0 < pData->getNumExamplesPerClass( i ) ) {
                                
                //fill( labels.begin(), labels.end(), 0 );
                AlphaReal mn = numeric_limits< AlphaReal >::max();
                AlphaReal mx = numeric_limits< AlphaReal >::min();
                                
                                
                                
                for( int j = 0; j < numExamples; j++ ) {
                    data[j].second = g[j][i];
                                        
                    if ( mn > data[j].second ) mn = data[j].second;
                    if ( mx < data[j].second ) mx = data[j].second;                                 
                                        
                    if ( pData->hasPositiveLabel( j, i ) ) data[j].first = 1;
                    else data[j].first = 0;
                }
                                
                mx -= mn;
                if ( mx > numeric_limits<AlphaReal>::epsilon() ) {
                    for( int j = 0; j < numExamples; j++ ) {
                        data[j].second -= mn;
                        data[j].second /= mx; 
                    }
                }
                                
                ROCscores[i] = nor_utils::getROC( data );
            } else {
                ROCscores[i] = 0.0;
            }
                        
            ROCsum += ROCscores[i];
        }
        ROCsum /= (double) numClasses;
        
        _outputHistory[pData].push_back(ROCsum);
        
        outStream << ROCsum; // mean of AUC
        for( int i=0; i < numClasses; i++ ) {
            outStream << OUTPUT_SEPARATOR << ROCscores[i];
        }
                
    }
        
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void TPRFPROutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                        map<InputData*, table>& gTableMap, 
                                        map<InputData*, table>& marginsTableMap, 
                                        map<InputData*, AlphaReal>& alphaSums,
                                        BaseLearner* pWeakHypothesis)

    {
        int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();
                
        table& g = gTableMap[pData];
                
        vector<int> origLabels(numExamples);
        vector<int> forecastedLabels(numExamples);
                
        for (int i = 0; i < numExamples; ++i)
        {
            const vector<Label>& labels = pData->getLabels(i);
                        
            vector<Label>::const_iterator lIt;
                        
            // the vote of the winning negative class
            AlphaReal maxClass = -numeric_limits<AlphaReal>::max();
                        
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                // get the negative winner class
                if ( g[i][lIt->idx] > maxClass )
                {
                    forecastedLabels[i]=lIt->idx;
                    maxClass = g[i][lIt->idx];
                }
                                
                // get the positive winner class
                if ( lIt->y > 0 )
                {
                    origLabels[i]=lIt->idx;
                }
            }                                               
        }
                
        vector<double> TPR(numClasses);
        vector<double> FPR(numClasses);
                
        vector<int> TP(numClasses);
        vector<int> FP(numClasses);
                
        vector<int> classDistr(numClasses);
                
        fill( classDistr.begin(), classDistr.end(), 0 );
        fill( TP.begin(), TP.end(), 0 );
        fill( FP.begin(), FP.end(), 0 );
                
        for (int i = 0; i < numExamples; ++i)
        {
            classDistr[origLabels[i]]++;
                        
            if (origLabels[i]==forecastedLabels[i]) // True positive
            {
                TP[origLabels[i]]++;
            } else {
                FP[forecastedLabels[i]]++;
            }                       
        }               
                
        double avgTPR = 0.0, avgFPR = 0.0;
        // print
        for( int l=0; l<numClasses; ++l ) {
            TPR[l] = TP[l]/((double)classDistr[l]);
            FPR[l] = FP[l]/((double)(numExamples-classDistr[l]));
                        
            avgTPR += TPR[l];
            avgFPR += FPR[l];
        }
                
        avgTPR /= numClasses;
        avgFPR /= numClasses;
                
        outStream << avgTPR; // mean of TPR
        for( int i=0; i < numClasses; i++ ) {
            outStream << OUTPUT_SEPARATOR << TPR[i];
        }
                
        outStream << OUTPUT_SEPARATOR << avgFPR; // mean of FPR
        for( int i=0; i < numClasses; i++ ) {
            outStream << OUTPUT_SEPARATOR << FPR[i];
        }                               
    }

    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        
    
    void SoftCascadeOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                             map<InputData*, table>& gTableMap, 
                                             map<InputData*, table>& marginsTableMap, 
                                             map<InputData*, AlphaReal>& alphaSums,
                                             BaseLearner* pWeakHypothesis) {
        set<int> indices ;
        pData->getIndexSet(indices);
        pData->clearIndexSet();
        
//        outStream << setprecision(4);
        
        table& g = gTableMap[pData];
        
        //resize in case of bootstrapping
        const int newDimension = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();
        const int oldDimension = (int)g.size();
        
        assert(newDimension >= oldDimension);
        g.resize(newDimension);
        
        _forecast.resize(newDimension);
        
        for (int i = oldDimension; i < newDimension; ++i) {
            g[i].resize(numClasses, 0.);
        }
        
        if (pWeakHypothesis != NULL) {
            _calibratedWeakHypotheses.push_back(pWeakHypothesis);
        }

        const int numExamples = pData->getNumExamples();
        
        const NameMap& namemap = pData->getClassMap();
        int positiveLabelIndex = namemap.getIdxFromName(_positiveLabelName);
        
        int P = pData->getNumExamplesPerClass(positiveLabelIndex);
        int N = pData->getNumExamplesPerClass(1 - positiveLabelIndex);
        int TP = 0, FP = 0;
        int err = 0;
        int numWhyp = 0;
        
        //for the ROC curve
        vector< pair< int, AlphaReal> > scores;
        scores.resize(0);
        scores.resize(numExamples);    
        
        AlphaReal alphaSum = 0. ;
        for (int wh = 0; wh < _calibratedWeakHypotheses.size(); ++wh) {
            alphaSum += _calibratedWeakHypotheses[wh]->getAlpha();
        }

        vector<AlphaReal>& rejectionThresholds = _rejectionThresholds;
        
        for (int i = 0; i < numExamples; ++i) {
            
            _forecast[i] = 1;
            
            AlphaReal posterior = 0. ;
            
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(positiveLabelIndex);

            int nbEvaluations = 0 ;
            
            for (int s = 0; s < _calibratedWeakHypotheses.size(); ++s) {
                
                nbEvaluations += 1;
                
                posterior += _calibratedWeakHypotheses[s]->getAlpha() * _calibratedWeakHypotheses[s]->classify(pData, i, positiveLabelIndex);
                if ( posterior < rejectionThresholds[s] ) {
                    _forecast[i] = -1;
                    break;
                }
            }
            
            // update the g table
//            vector<Label>::const_iterator lIt;                        
//                      for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
//                      {
//                              g[i][lIt->idx] = posterior;
//            }
//instead
            g[i][positiveLabelIndex] = posterior;
            
            
            scores[i].second = ( ( posterior/alphaSum ) + 1 ) / 2 ;                 
            
            if (labelY < 0) {
                numWhyp += nbEvaluations;
                scores[i].first = 0;
            }
            else 
                scores[i].first = 1;
            
//            outScores[i].resize(2);
//            outScores[i][0] = scores[i].second;
//            outScores[i][1] = nbEvaluations;
            
            if (_forecast[i] * labelY < 0) {
                err++;
            }
            
            if (_forecast[i] > 0)
            {
                if (labelY > 0) {
                    TP++;
                }
                else {
                    FP++;
                }                
            }
        }
        
        double tpRate = (double)TP/P;
        double fpRate = (double)FP/N;
        double eval = (double)numWhyp / N;
        double errRate = (double)err/numExamples;
        
        double rocScore = nor_utils::getROC( scores );  
        
        outStream << errRate;
        outStream << OUTPUT_SEPARATOR << rocScore;
        outStream << OUTPUT_SEPARATOR << fpRate;
        outStream << OUTPUT_SEPARATOR << tpRate;
        outStream << OUTPUT_SEPARATOR << eval;
        
        pData->loadIndexSet(indices);
        
    }
    
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

    void PosteriorsOutput::computeAndOutput(ostream& outStream, InputData* pData, 
                                            map<InputData*, table>& gTableMap, 
                                            map<InputData*, table>& marginsTableMap, 
                                            map<InputData*, AlphaReal>& alphaSums,
                                            BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        
        table& g = gTableMap[pData];
        vector<int>& idxV = _classIdx;
        

        for (int j = 0; j < idxV.size() ; ++j) {
            for (int i = 0; i < numExamples; ++i) {
                outStream << g[i][j] << " " ;
            }
            
            if (j != idxV.size()) {
                outStream << endl;
            }
            
        }
    }
    
    // -------------------------------------------------------------------------        
    // -------------------------------------------------------------------------        

} // end of namespace MultiBoost
