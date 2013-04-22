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

//
//  SoftCascadeClassifier.cpp
//  MultiBoost

#include "SoftCascadeClassifier.h"
#include "WeakLearners/BaseLearner.h"
#include "IO/InputData.h"
#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/OutputInfo.h"

namespace MultiBoost {
    
    // -------------------------------------------------------------------------
    
    SoftCascadeClassifier::SoftCascadeClassifier(const nor_utils::Args &args, int verbose)
        : _verbose(verbose), _args(args), _outputInfoFile("")
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
    
    InputData* SoftCascadeClassifier::loadInputData(const string& dataFileName, const string& shypFileName)
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
        if ( !UnSerialization::seekSimpleTag(st, "multiboost") )
        {
            // no multiboost tag found: this is not the correct file!
            cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file!!" << endl;
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
    void SoftCascadeClassifier::run(const string& dataFileName, const string& shypFileName, 
                                    int numIterations, const string& outFileName, 
                                    int numRanksEnclosed)

    {
        InputData* pData = loadInputData(dataFileName, shypFileName);
        const int numExamples = pData->getNumExamples();
        
        cout << "Number of examples : " << numExamples << endl;
        const NameMap& namemap = pData->getClassMap();
        
        int positiveLabelIndex = namemap.getIdxFromName(_positiveLabelName);
        
        
        OutputInfo* pOutInfo = NULL;
        
        if ( !_outputInfoFile.empty() ) 
        {
            pOutInfo = new OutputInfo(_args, true);
            pOutInfo->setOutputList("sca", &_args);
            
            pOutInfo->initialize(pData);
            
            pOutInfo->outputHeader(pData->getClassMap(), true, true, false);
            pOutInfo->outputUserHeader("thresh");
            pOutInfo->headerEndLine();
        }
        
                
        int P = pData->getNumExamplesPerClass(positiveLabelIndex);
        int N = pData->getNumExamplesPerClass(1 - positiveLabelIndex);
        
        cout << "\t positives : " << P << endl;
        cout << "\t negatives : " << N << endl;
        
        
        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;
        
        // The class that loads the weak hypotheses
        UnSerialization us;
        
        // Where to put the weak hypotheses
        vector<BaseLearner*> calibWeakHypotheses;
        vector<AlphaReal> rejectionThresholds;
                
        us.loadHypothesesWithThresholds(shypFileName, calibWeakHypotheses, rejectionThresholds, pData);
        
        
        for (int w = 0; w < calibWeakHypotheses.size(); ++w) {
            dynamic_cast<SoftCascadeOutput*>( pOutInfo->getOutputInfoObject("sca") )->appendRejectionThreshold(rejectionThresholds[w]);            
            printOutputInfo(pOutInfo, w, pData, calibWeakHypotheses[w], rejectionThresholds[w]);
        }
        
        
        vector<char> & forecast = dynamic_cast<SoftCascadeOutput*>( pOutInfo->getOutputInfoObject("sca") )->getForcastVector() ;

//        for (int j = 0; j < forecast.size(); j++) {
//            cout << (int) forecast[j] << " ";
//        }
//        cout << endl;
        
        vector<vector<int> > confMatrix(2);
        confMatrix[0].resize(2);
        fill( confMatrix[0].begin(), confMatrix[0].end(), 0 );
        confMatrix[1].resize(2);
        fill( confMatrix[1].begin(), confMatrix[1].end(), 0 );
                
        // print accuracy
        for(int i=0; i<numExamples; ++i )
        {               
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(positiveLabelIndex);

            if (labelY > 0) // pos label                            
                if (forecast[i]==1)
                    confMatrix[1][1]++;
                else
                    confMatrix[1][0]++;
            else // negative label
                if (forecast[i]==-1)
                    confMatrix[0][0]++;
                else
                    confMatrix[0][1]++;
        }                       
                
        double acc = 100.0 * (confMatrix[0][0] + confMatrix[1][1]) / ((double) numExamples);
        // output it
        cout << endl;
        cout << "Error Summary" << endl;
        cout << "=============" << endl;
                
        cout << "Accuracy: " << setprecision(4) << acc << endl;
        cout << setw(10) << "\t" << setw(10) << namemap.getNameFromIdx(1-positiveLabelIndex) << setw(10) << namemap.getNameFromIdx(positiveLabelIndex) << endl;
        cout << setw(10) << namemap.getNameFromIdx(1-positiveLabelIndex) << setw(10) << confMatrix[0][0] << setw(10) << confMatrix[0][1] << endl;
        cout << setw(10) << namemap.getNameFromIdx(positiveLabelIndex) << setw(10) << confMatrix[1][0] << setw(10) << confMatrix[1][1] << endl;         
        
        if (pData) {
            delete pData;
        }
    }

    // -------------------------------------------------------------------------
    
    void SoftCascadeClassifier::savePosteriors(const string& dataFileName, const string& shypFileName, 
                                               const string& outFileName, int numIterations )
    {
        InputData* pData = loadInputData(dataFileName, shypFileName);
        const int numExamples = pData->getNumExamples();
        
        cout << "Number of examples : " << numExamples << endl;
        const NameMap& namemap = pData->getClassMap();
        
        int positiveLabelIndex = namemap.getIdxFromName(_positiveLabelName);
        
        ofstream output;
        if (!_outputInfoFile.empty()) {
            output.open(_outputInfoFile.c_str());
            output << setiosflags(ios::fixed) << setprecision(6);
        }
        
        if ( ! output.is_open() ) {
            cout << "Cannot open output file" << endl;
            exit(-1);
        }
        
        ofstream outputPosteriors(outFileName.c_str());
        if (!outputPosteriors.is_open())
        {
            cout << "Cannot open file " << outFileName << endl;
        }
        
        int P = pData->getNumExamplesPerClass(positiveLabelIndex);
        int N = pData->getNumExamplesPerClass(1 - positiveLabelIndex);
        
        cout << "\t positives : " << P << endl;
        cout << "\t negatives : " << N << endl;
        
        
        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;
        
        // The class that loads the weak hypotheses
        UnSerialization us;
        
        // Where to put the weak hypotheses
        vector<BaseLearner*> calibWeakHypotheses;
        vector<AlphaReal> rejectionThresholds;
        
        us.loadHypothesesWithThresholds(shypFileName, calibWeakHypotheses, rejectionThresholds, pData);
        
        if (!_outputInfoFile.empty()) {
            //the header
            output  << "t" << OUTPUT_SEPARATOR
                    << "err" << OUTPUT_SEPARATOR
                    << "auc" << OUTPUT_SEPARATOR
                    << "fpr" << OUTPUT_SEPARATOR
                    << "tpr" << OUTPUT_SEPARATOR
                    << "nbeval" << endl;
        }            
                    
        
        
        vector<BaseLearner*> weakHypotheses;
        for (int w = 0; w < calibWeakHypotheses.size(); ++w) {
            weakHypotheses.push_back(calibWeakHypotheses[w]);
            
            AlphaReal alphaSum = 0. ;
            for (int wh = 0; wh < weakHypotheses.size(); ++wh) {
                alphaSum += weakHypotheses[wh]->getAlpha();
            }
            
            int TP = 0, FP = 0;
            int err = 0;
            int numWhyp = 0;
            vector< pair< int, AlphaReal> > scores;
            scores.resize(numExamples);
            
            vector<vector< AlphaReal> > outputs;
            outputs.resize(numExamples);
            
            for (int i = 0; i < numExamples; ++i) {
                int forecast = 1;
                AlphaReal posterior = 0. ;
                int nbEvaluations = 0 ;

                const Example& example = pData->getExample(i);
                int labelY = example.getLabelY(positiveLabelIndex);

                for (int s = 0; s < numIterations && s < weakHypotheses.size(); ++s) {
                    
                    nbEvaluations += 1;
                    
                    posterior += weakHypotheses[s]->getAlpha() * weakHypotheses[s]->classify(pData, i, positiveLabelIndex);
                    if (  posterior < rejectionThresholds[s] ) {
                        forecast = -1;
                        break;
                    }
                }
                
                outputs[i].resize(2);
                outputs[i][0] = posterior; // ( ( posterior/alphaSum ) + 1 ) / 2 ;;
                outputs[i][1] = nbEvaluations;
                
                scores[i].second = outputs[i][0];
                if (labelY < 0) {
                    scores[i].first = 0;
                    numWhyp += nbEvaluations;
                }
                else 
                    scores[i].first = 1;
                
                if (forecast * labelY < 0) {
                    err++;
                }
                
                if (forecast > 0)
                {
                    if (labelY > 0) {
                        TP++;
                    }
                    else {
                        FP++;
                    }                
                }
            }
            
            for (int i = 0; i < outputs.size(); ++i) {
                outputPosteriors << outputs[i][0] << " ";
            }
            outputPosteriors << endl;
            
            // number of used classifiers
            //            for (int i = 0; i < outputs.size(); ++i) {
            //                outputPosteriors << int(outputs[i][1]) << " ";
            //            }
            //            outputPosteriors << endl;
            
            if (! _outputInfoFile.empty()) {
                
                double tpRate = (double)TP/P;
                double fpRate = (double)FP/N;
                double eval = (double)numWhyp / N;
                double errRate = (double)err/numExamples;
                
                double rocScore = nor_utils::getROC( scores );  
                
                output << w + 1 << OUTPUT_SEPARATOR << errRate;
                output << OUTPUT_SEPARATOR << rocScore;
                output << OUTPUT_SEPARATOR << fpRate;
                output << OUTPUT_SEPARATOR << tpRate;
                output << OUTPUT_SEPARATOR << eval;
                output << endl;                
            }
        }
        
        if (pData) {
            delete pData;
        }
    }
    
    // -------------------------------------------------------------------------
    
    void SoftCascadeClassifier::printOutputInfo(OutputInfo* pOutInfo, int t, 
                                                InputData* pData, 
                                                BaseLearner* pWeakHypothesis,
                                                AlphaReal r)
    {
        pOutInfo->outputIteration(t);
        pOutInfo->outputCustom(pData, pWeakHypothesis);
        pOutInfo->outputCurrentTime();
        pOutInfo->separator();
        pOutInfo->outputUserData(r);
        pOutInfo->endLine();
        
    }
    
    
}



