
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


#include "WeakLearners/BaseLearner.h"
#include "IO/InputData.h"
#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/OutputInfo.h"
#include "VJCascadeClassifier.h"
//#include "Classifiers/AdaBoostMHClassifier.h"
#include "Classifiers/ExampleResults.h"

#include "WeakLearners/SingleStumpLearner.h" // for saveSingleStumpFeatureData

#include <iomanip> // for setw
#include <cmath> // for setw
#include <functional>

namespace MultiBoost {
        
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
        
    VJCascadeClassifier::VJCascadeClassifier(const nor_utils::Args &args, int verbose)
        : _verbose(verbose), _args(args), _positiveLabelIndex(-1)
    {
        // The file with the step-by-step information
        if ( args.hasArgument("outputinfo") )
            args.getValue("outputinfo", 0, _outputInfoFile);
                
        if ( args.hasArgument("positivelabel") )
        {
            args.getValue("positivelabel", 0, _positiveLabelName);
        } else {
            cout << "The name of positive label has to be given!!!" << endl;
            exit(-1);
        }                               
    }
        
    // -------------------------------------------------------------------------
        
    void VJCascadeClassifier::run(const string& dataFileName, const string& shypFileName, 
                                  int numIterations, const string& outResFileName )
    {
        // loading data
        InputData* pData = loadInputData(dataFileName, shypFileName);
        const int numOfExamples = pData->getNumExamples();
                                
        //get the index of positive label               
        const NameMap& namemap = pData->getClassMap();
        _positiveLabelIndex = namemap.getIdxFromName( _positiveLabelName );                             
                
        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;
                
                
                
        // The class that loads the weak hypotheses
        UnSerialization us;
                
        // Where to put the weak hypotheses
        vector<vector<BaseLearner*> > weakHypotheses;
                
        // For stagewise thresholds 
        vector<AlphaReal> thresholds(0);
        
        // loads them
        //us.loadHypotheses(shypFileName, weakHypotheses, pData);
        us.loadCascadeHypotheses(shypFileName, weakHypotheses, thresholds, pData);
                

        // store result
        vector<CascadeOutputInformation> cascadeData(0);
        vector<CascadeOutputInformation>::iterator it;
                
        cascadeData.resize(numOfExamples);              
        for( it=cascadeData.begin(); it != cascadeData.end(); ++it )
        {
            it->active=true;
        }                                                                               
                
        if (!_outputInfoFile.empty())
        {
            outputHeader();
        }
                
        for(int stagei=0; stagei < weakHypotheses.size(); ++stagei )
        {
            // for posteriors
            vector<AlphaReal> posteriors(0);                
                        
            // calculate the posteriors after stage
            VJCascadeLearner::calculatePosteriors( pData, weakHypotheses[stagei], posteriors, _positiveLabelIndex );                        
                        
            // update the data (posteriors, active element index etc.)
            updateCascadeData(pData, weakHypotheses, stagei, posteriors, thresholds, _positiveLabelIndex, cascadeData);
                        
            if (!_outputInfoFile.empty())
            {
                _output << stagei + 1 << "\t";
                _output << weakHypotheses[stagei].size() << "\t";
                outputCascadeResult( pData, cascadeData );
            }
                        
            int numberOfActiveInstance = 0;
            for( int i = 0; i < numOfExamples; ++i )
                if (cascadeData[i].active) numberOfActiveInstance++;
                        
            if (_verbose > 0 )
                cout << "Number of active instances: " << numberOfActiveInstance << "(" << numOfExamples << ")" << endl;                                                                        
        }
                                
        vector<vector<int> > confMatrix(2);
        confMatrix[0].resize(2);
        fill( confMatrix[0].begin(), confMatrix[0].end(), 0 );
        confMatrix[1].resize(2);
        fill( confMatrix[1].begin(), confMatrix[1].end(), 0 );
                
        // print accuracy
        for(int i=0; i<numOfExamples; ++i )
        {               
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);
            
            if (labelY>0) // pos label                              
                if (cascadeData[i].forecast==1)
                    confMatrix[1][1]++;
                else
                    confMatrix[1][0]++;
            else // negative label
                if (cascadeData[i].forecast==0)
                    confMatrix[0][0]++;
                else
                    confMatrix[0][1]++;
        }                       
                
        double acc = 100.0 * (confMatrix[0][0] + confMatrix[1][1]) / ((double) numOfExamples);
        // output it
        cout << endl;
        cout << "Error Summary" << endl;
        cout << "=============" << endl;
                
        cout << "Accuracy: " << setprecision(4) << acc << endl;
        cout << setw(10) << "\t" << setw(10) << namemap.getNameFromIdx(1-_positiveLabelIndex) << setw(10) << namemap.getNameFromIdx(_positiveLabelIndex) << endl;
        cout << setw(10) << namemap.getNameFromIdx(1-_positiveLabelIndex) << setw(10) << confMatrix[0][0] << setw(10) << confMatrix[0][1] << endl;
        cout << setw(10) << namemap.getNameFromIdx(_positiveLabelIndex) << setw(10) << confMatrix[1][0] << setw(10) << confMatrix[1][1] << endl;                
                
        // output forecast 
        if (!outResFileName.empty() ) outputForecast(pData, outResFileName, cascadeData );
                                                
        // free memory allocation
        vector<vector<BaseLearner*> >::iterator bvIt;
        for( bvIt = weakHypotheses.begin(); bvIt != weakHypotheses.end(); ++bvIt )
        {
            vector<BaseLearner* >::iterator bIt;
            for( bIt = (*bvIt).begin(); bIt != (*bvIt).end(); ++bIt )
                delete *bIt;
        }
    }
        
    // -------------------------------------------------------------------------
    void VJCascadeClassifier::updateCascadeData(InputData* pData, vector<vector<BaseLearner*> >& weakHypotheses, 
                                                int stagei, const vector<AlphaReal>& posteriors, vector<AlphaReal>& thresholds, int positiveLabelIndex,
                                                vector<CascadeOutputInformation>& cascadeData)
    {
        const int numOfExamples = pData->getNumExamples();
        int sumOfWeakClassifier = 0;
        for(int i=0; i<=0; ++i) sumOfWeakClassifier += ((int)weakHypotheses[stagei].size());
                
        double sumalphas = 0.0;
        for(int i=0; i<weakHypotheses[stagei].size(); ++i)
        {
            sumalphas += weakHypotheses[stagei][i]->getAlpha();
        }
                
        for(int i=0; i<numOfExamples; ++i )
        {
            bool isPos;
            
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);
            
            if (labelY>0)                           
                isPos = true;
            else 
                isPos =false;                   
                        
                        
            //cout << posteriors[i] << " ";
            if (cascadeData[i].active) // active: it is not classified yet
            {
                cascadeData[i].score=((posteriors[i]/sumalphas)+1)/2;
                
                cascadeData[i].score += stagei - 1 ;
                
                if (posteriors[i]<thresholds[stagei])
                {
                    cascadeData[i].active = false; // classified
                    cascadeData[i].forecast=0;
                } else {
                    cascadeData[i].active = true; // continue
                    cascadeData[i].forecast=1;                                      
                }
                
                                
                cascadeData[i].classifiedInStage=stagei;
                cascadeData[i].numberOfUsedClassifier=sumOfWeakClassifier;                                                              
            }                                               
        }                               
                
                
    }
                                                           
    // -------------------------------------------------------------------------
        
    void VJCascadeClassifier::printConfusionMatrix(const string& dataFileName, const string& shypFileName)
    {
    }
        
    // -------------------------------------------------------------------------
        
    void VJCascadeClassifier::saveConfusionMatrix(const string& dataFileName, const string& shypFileName,
                                                  const string& outFileName)
    {
    }
        
    // -------------------------------------------------------------------------
        
    void VJCascadeClassifier::savePosteriors(const string& dataFileName, const string& shypFileName, 
                                             const string& outFileName, int numIterations)
    {
        // loading data
        InputData* pData = loadInputData(dataFileName, shypFileName);
        const int numOfExamples = pData->getNumExamples();
                
        //get the index of positive label               
        const NameMap& namemap = pData->getClassMap();
        _positiveLabelIndex = namemap.getIdxFromName( _positiveLabelName );
                
                
        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;
                
                
        // open outfile
        ofstream outRes(outFileName.c_str());
        if (!outRes.is_open())
        {
            cout << "Cannot open outfile!!! " << outFileName << endl;
        }
                                
                
        // The class that loads the weak hypotheses
        UnSerialization us;
                
        // Where to put the weak hypotheses
        vector<vector<BaseLearner*> > weakHypotheses;
                        
        // For stagewise thresholds 
        vector<AlphaReal> thresholds(0);
        // loads them
        //us.loadHypotheses(shypFileName, weakHypotheses, pData);
        us.loadCascadeHypotheses(shypFileName, weakHypotheses, thresholds, pData);
                
        // output the number of stages
        outRes << "StageNum " << weakHypotheses.size() << endl;
                
        // output original labels
        outRes << "Labels";
        for(int i=0; i<numOfExamples; ++i )
        {               
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);
            
            if (labelY>0) // pos label                              
                outRes << " 1";
            else
                outRes << " 0";
        }                               
        outRes << endl;
                
        // store result
        vector<CascadeOutputInformation> cascadeData(0);
        vector<CascadeOutputInformation>::iterator it;
                
        cascadeData.resize(numOfExamples);              
        for( it=cascadeData.begin(); it != cascadeData.end(); ++it )
        {
            it->active=true;
        }                                                                               
                
        for(int stagei=0; stagei < weakHypotheses.size(); ++stagei )
        {
            // for posteriors
            vector<AlphaReal> posteriors(0);                
                        
            // calculate the posteriors after stage
            VJCascadeLearner::calculatePosteriors( pData, weakHypotheses[stagei], posteriors, _positiveLabelIndex );                        
                        
            // update the data (posteriors, active element index etc.)
            //VJCascadeLearner::forecastOverAllCascade( pData, posteriors, activeInstances, thresholds[stagei] );
            updateCascadeData(pData, weakHypotheses, stagei, posteriors, thresholds, _positiveLabelIndex, cascadeData);
                        
                        
            int numberOfActiveInstance = 0;
            for( int i = 0; i < numOfExamples; ++i )
                if (cascadeData[i].active) numberOfActiveInstance++;
                        
            if (_verbose > 0 )
                cout << "Number of active instances: " << numberOfActiveInstance << "(" << numOfExamples << ")" << endl;                                                                        
                        
            // output stats
            outRes << "Stage " << stagei << " " << weakHypotheses[stagei].size() << endl; 

            outRes << "Forecast";
            for(int i=0; i<numOfExamples; ++i )
            {       
                outRes << " " << cascadeData[i].forecast;
            }                               
            outRes << endl;

            outRes << "Active";
            for(int i=0; i<numOfExamples; ++i )
            {       
                if( cascadeData[i].active)
                    outRes << " 1";
                else
                    outRes << " 0";
            }                               
            outRes << endl;

            outRes << "Posteriors";
            for(int i=0; i<numOfExamples; ++i )
            {       
                outRes << " " << cascadeData[i].score;
            }                               
            outRes << endl;
                        
        }                                               
                
        outRes.close();
                
        // free memory allocation
        vector<vector<BaseLearner*> >::iterator bvIt;
        for( bvIt = weakHypotheses.begin(); bvIt != weakHypotheses.end(); ++bvIt )
        {
            vector<BaseLearner* >::iterator bIt;
            for( bIt = (*bvIt).begin(); bIt != (*bvIt).end(); ++bIt )
                delete *bIt;
        }
    }
        
        
        
                
    // -------------------------------------------------------------------------
        
    InputData* VJCascadeClassifier::loadInputData(const string& dataFileName, const string& shypFileName)
    {
        // open file
        ifstream inFile(shypFileName.c_str());
        if (!inFile.is_open())
        {
            cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
            exit(1);
        }
                
        // Declares the stream tokenizer
        nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");
                
        // Move until it finds the multiboost tag
        if ( !UnSerialization::seekSimpleTag(st, "cascade") )
        {
            // no multiboost tag found: this is not the correct file!
            cerr << "ERROR: Not a valid Cascade Strong Hypothesis file!!" << endl;
            exit(1);
        }
                
        // Move until it finds the algo tag
        string basicLearnerName = UnSerialization::seekAndParseEnclosedValue<string>(st, "algo");
        
        // Check if the weak learner exists
        if ( !BaseLearner::RegisteredLearners().hasLearner(basicLearnerName) )
        {
            cerr << "ERROR: Weak learner <" << basicLearnerName << "> not registered!!" << endl;
            exit(1);
        }
                
        // get the training input data, and load it
        BaseLearner* baseLearner = BaseLearner::RegisteredLearners().getLearner(basicLearnerName);
        baseLearner->initLearningOptions(_args);
        InputData* pData = baseLearner->createInputData();
                
        // set the non-default arguments of the input data
        pData->initOptions(_args);
        // load the data
        pData->load(dataFileName, IT_TEST, _verbose);
                
        return pData;
    }
        
    // -------------------------------------------------------------------------
    void VJCascadeClassifier::outputForecast( InputData* pData, const string& outResFileName, vector<CascadeOutputInformation>& cascadeData )
    {
        ofstream out( outResFileName.c_str() );
        if ( ! out.is_open() )
        {
            cout << "Outfile (" << outResFileName << ") cannot be opened!" << endl;
            exit(-1);
        }
                
        // header
        out << "Labels\t| Forecast " << endl;
                
        const int positiveLabelIndex = pData->getClassMap().getIdxFromName( _positiveLabelName );               
        const int numOfExamples = pData->getNumExamples();
                
        string posLabelName = pData->getClassMap().getNameFromIdx(positiveLabelIndex);
        string negLabelName = pData->getClassMap().getNameFromIdx(1-positiveLabelIndex);
                
        for(int i=0; i<numOfExamples; ++i )
        {               
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);
            
            if (labelY>0) // pos label
            {
                out << posLabelName << "\t| ";
                                
                if (cascadeData[i].forecast==1)
                    out << posLabelName;
                else
                    out << negLabelName;
            }
            else // negative label
            {
                out << negLabelName << "\t| ";                          
                        
                if (cascadeData[i].forecast==0)
                    out << posLabelName;
                else
                    out << negLabelName;
            }
                        
            out << endl;
        }                       
                
                
        out.close();
    }
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    void VJCascadeClassifier::outputHeader()
    {
        // open outfile
        _output.open(_outputInfoFile.c_str());
        if ( ! _output.is_open() )
        {
            cout << "Cannot open output file" << endl;
            exit(-1);
        }       
        _output << "Stage\t";
        _output << "Number of weak hyp.\t";             
        //_output << "Stage\t"; 
                                
        _output << "Test FPR\t";
        _output << "Test TPR\t";                
        _output << "Test ROC\t";        
                                
        _output << endl << flush;
    }
        
        
    // -------------------------------------------------------------------------
    void VJCascadeClassifier::outputCascadeResult( InputData* pData, vector<CascadeOutputInformation>& cascadeData )
    {
        const int numOfExamples = pData->getNumExamples();
                
        int P=0,N=0;
        int TP=0,FP=0;
                
        for(int i=0; i<numOfExamples; ++i )
        {
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);
            
            if (labelY>0)
            {
                P++;
                if (cascadeData[i].forecast==1) TP++;
            } else { 
                N++;
                if (cascadeData[i].forecast==1) FP++;                           
            }
        }               
                
        _output << (FP/((double)N)) << "\t";
        _output << (TP/((double)P)) << "\t";
                
        //output ROC
        vector< pair< int, AlphaReal > > scores( numOfExamples );               
                
        for(int i=0; i<numOfExamples; ++i )
        {
            AlphaReal s = cascadeData[i].score;
            scores[i].second = s;
            //_output << "," << cascadeData[i].score;
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);
            
            if (labelY>0)
            {
                scores[i].first=1;                              
            } else { 
                scores[i].first=0;                              
            }                       
                        
        }               
                
        AlphaReal rocScore = nor_utils::getROC( scores );
        _output << rocScore << "\t";
        _output << endl;                
    }
        
        
} // end of namespace MultiBoost

