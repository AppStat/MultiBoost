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


#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <limits>
#include <iomanip> // setprecision

#include "Utils/Utils.h" // for addAndCheckExtension
#include "Defaults.h" // for defaultLearner
#include "IO/OutputInfo.h"
#include "IO/InputData.h"
#include "IO/Serialization.h" // to save the found strong hypothesis

#include "WeakLearners/BaseLearner.h"
#include "StrongLearners/AdaBoostMHLearner.h"
#include "Classifiers/VJCascadeClassifier.h"
#include "VJCascadeLearner.h"

namespace MultiBoost {
        
    // -----------------------------------------------------------------------------------
        
    void VJCascadeLearner::getArgs(const nor_utils::Args& args)
    {
        if ( args.hasArgument("verbose") )
            args.getValue("verbose", 0, _verbose);
                
        // The file with the step-by-step information
        if ( args.hasArgument("outputinfo") )
            args.getValue("outputinfo", 0, _outputInfoFile);
                
        // The file with the stagewise posteriors
        if ( args.hasArgument("stagewiseposteriors") )
            args.getValue("stagewiseposteriors", 0, _outputPosteriorsFileName);
                
        ///////////////////////////////////////////////////
        // get the output strong hypothesis file name, if given
        if ( args.hasArgument("shypname") )
            args.getValue("shypname", 0, _shypFileName);
        else
            _shypFileName = string(SHYP_NAME);
                
        _shypFileName = nor_utils::addAndCheckExtension(_shypFileName, SHYP_EXTENSION);
                
        ///////////////////////////////////////////////////
        // get the output strong hypothesis file name, if given
        if ( args.hasArgument("shypcomp") )
            args.getValue("shypcomp", 0, _isShypCompressed );
        else
            _isShypCompressed = false;
                
        // get the name of the learner
        _baseLearnerName = "HaarSingleStumpLearner";
        if ( args.hasArgument("learnertype") )
            args.getValue("learnertype", 0, _baseLearnerName);
                
        if ( args.hasArgument("firstStage") )
            args.getValue("firstStage", 0, _stageStartNumber);
                
                
        // -train <dataFile> <nInterations>
        if ( args.hasArgument("train") )
        {
            cout << "Validation file is needed in VJ cascade!!!" << endl;
        }
        // -traintest <trainingDataFile> <testDataFile> <nInterations>
        else if ( args.hasArgument("traintest") ) 
        {
            args.getValue("traintest", 0, _trainFileName);
            args.getValue("traintest", 1, _testFileName);
            args.getValue("traintest", 2, _numIterations);
        }               
        // -traintest <trainingDataFile> <validDataFile> <testDataFile> <nInterations>
        else if ( args.hasArgument("trainvalidtest") ) 
        {
            args.getValue("trainvalidtest", 0, _trainFileName);
            args.getValue("trainvalidtest", 1, _validFileName);
            args.getValue("trainvalidtest", 2, _testFileName);
            args.getValue("trainvalidtest", 3, _numIterations);
        }
                
        // Set the value of Minimum Acceptable Detection Rate
        if ( args.hasArgument("minacctpr") )
            args.getValue("minacctpr", 0, _minAcceptableDetectionRate);  
                
        // Set the value of Maximum Acceptable False Positive Rate
        if ( args.hasArgument("maxaccfpr") )
            args.getValue("maxaccfpr", 0, _maxAcceptableFalsePositiveRate);  
                
                
        if ( args.hasArgument("positivelabel") )
        {
            args.getValue("positivelabel", 0, _positiveLabelName);
        } else {
            cout << "The name of positive label has to be given. \n Type --h vjcascade to know the mandatory options." << endl;
            exit(-1);
        }
                
        // --constant: check constant learner in each iteration
        if ( args.hasArgument("constant") )
            _withConstantLearner = true;                    
    }
        
    // -----------------------------------------------------------------------------------
        
    void VJCascadeLearner::run(const nor_utils::Args& args)
    {
        // load the arguments
        this->getArgs(args);
                
        outputHeader();         
                
        AlphaReal Fi=1.0;               
        AlphaReal Di=1.0;
        AlphaReal currentTPR = 0.0, currentFPR = 0.0;
        //int th[] = {2,10,10,25,25,60,60,100,100,100};
        _foundHypotheses.resize(0);
                
        // get the registered weak learner (type from name)
        BaseLearner* pWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
        // initialize learning options; normally it's done in the strong loop
        // also, here we do it for Product learners, so input data can be created
        pWeakHypothesisSource->initLearningOptions(args);
                
        BaseLearner* pConstantWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner("ConstantLearner");
                
        // get the training input data, and load it
                
        InputData* pTrainingData = pWeakHypothesisSource->createInputData();
        pTrainingData->initOptions(args);
        pTrainingData->load(_trainFileName, IT_TRAIN, _verbose);
                
        InputData* pValidationData = NULL;
        pValidationData = pWeakHypothesisSource->createInputData();
        pValidationData->initOptions(args);             
        if ( !_validFileName.empty() )
        {
            pValidationData->load(_validFileName, IT_TRAIN, _verbose);                              
        } else {
            // this solution is not memory efficient, should be improved later
            pValidationData->load(_trainFileName, IT_TRAIN, _verbose);
        }
                
        // get the testing input data, and load it
        InputData* pTestData = NULL;
        if ( !_testFileName.empty() )
        {
            pTestData = pWeakHypothesisSource->createInputData();
            pTestData->initOptions(args);
            pTestData->load(_testFileName, IT_TEST, _verbose);
        }                                               
                                                
        //get the index of positive label               
        const NameMap& namemap = pTrainingData->getClassMap();
        _positiveLabelIndex = namemap.getIdxFromName( _positiveLabelName );
                
        if (_verbose>3)
        {
            cout << "Positive label:\t" << _positiveLabelName << endl;
            cout << "Positive label index:\t" << _positiveLabelIndex << endl;
        }
                
        Serialization ss(_shypFileName, false );
        ss.writeCascadeHeader(_baseLearnerName); // this must go after resumeProcess has been called
                
                
        if (_verbose == 1)
            cout << "Learning in progress..." << endl;
                
        vector<CascadeOutputInformation> activeTrainInstances(pTrainingData->getNumExamples());
        vector<CascadeOutputInformation>::iterator it;
        for( it=activeTrainInstances.begin(); it != activeTrainInstances.end(); ++it )
        {
            it->active=true;
        }               
                
        vector<CascadeOutputInformation> activeValidationInstances(pValidationData->getNumExamples());
        for( it=activeValidationInstances.begin(); it != activeValidationInstances.end(); ++it )
        {
            it->active=true;
        }               
                
        vector<CascadeOutputInformation> activeTestInstances(0);
        if (pTestData)
        {
            activeTestInstances.resize(pTestData->getNumExamples());
            for( it=activeTestInstances.begin(); it != activeTestInstances.end(); ++it )
            {
                it->active=true;
            }                                       
        }
                
        set<int> trainingIndices;
        set<int> validationIndices;
                
        ///////////////////////////////////////////////////////////////////////
        // Starting the Cascad main loop
        ///////////////////////////////////////////////////////////////////////         
        for(int stagei=0; stagei < _numIterations; ++stagei )
        {
            resetWeights(pTrainingData);
            vector<AlphaReal> validPosteriors(0);
            vector<AlphaReal> trainPosteriors(0);
            vector<AlphaReal> testPosteriors(0);
                        
            //Fi=prevFi;
            Fi *= _maxAcceptableFalsePositiveRate;                                          
            Di *= _minAcceptableDetectionRate;
                        
            int t=0;
            _foundHypotheses.resize( _foundHypotheses.size()+1 );
                        
            // storing the posteriors for validation set
            trainPosteriors.resize(pTrainingData->getNumExamples());
            fill(trainPosteriors.begin(), trainPosteriors.end(), 0.0 );
            validPosteriors.resize(pValidationData->getNumExamples());
            fill(validPosteriors.begin(), validPosteriors.end(), 0.0 );
                        
            if (pTestData)
            {
                testPosteriors.resize(pTestData->getNumExamples());
                fill(testPosteriors.begin(), testPosteriors.end(), 0.0 );
            }
                        
            FeatureReal tunedThreshold=0.0;
            ///////////////////////////////////////////////////////////////////////
            // Starting the AdaBoost main loop
            ///////////////////////////////////////////////////////////////////////
            while (true)
            {
                if (_verbose > 1)
                    cout << "------- STAGE " << stagei << " WORKING ON ITERATION " << (t+1) << " -------" << endl;
                                
                BaseLearner* pWeakHypothesis = pWeakHypothesisSource->create();
                pWeakHypothesis->initLearningOptions(args);                             
                                
                pWeakHypothesis->setTrainingData(pTrainingData);
                                
                AlphaReal energy = pWeakHypothesis->run();
                                
                //float gamma = pWeakHypothesis->getEdge();
                //cout << gamma << endl;
                                
                if ( (_withConstantLearner) || ( energy != energy ) ) // check constant learner if user wants it (if energi is nan, then we chose constant learner
                {
                    BaseLearner* pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
                    pConstantWeakHypothesis->initLearningOptions(args);
                    pConstantWeakHypothesis->setTrainingData(pTrainingData);
                    AlphaReal constantEnergy = pConstantWeakHypothesis->run();
                                        
                    if ( (constantEnergy <= energy) || ( energy != energy ) || ( nor_utils::is_zero(constantEnergy - energy))) {
                        delete pWeakHypothesis;
                        pWeakHypothesis = pConstantWeakHypothesis;
                    }
                }
                                
                if (_verbose > 1)
                    cout << "Weak learner: " << pWeakHypothesis->getName()<< endl;
                // Output the step-by-step information
                //printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);
                                
                // Updates the weights and returns the edge
                AlphaReal gamma = updateWeights(pTrainingData, pWeakHypothesis);
                                
                //checkWeights(pTrainingData);
                                
                if (_verbose > 1)
                {
                    cout << setprecision(5)
                         << "--> Alpha = " << pWeakHypothesis->getAlpha() << endl
                         << "--> Edge  = " << gamma << endl
                         << "--> Energy  = " << energy << endl
                        //            << "--> ConstantEnergy  = " << constantEnergy << endl
                        //            << "--> difference  = " << (energy - constantEnergy) << endl
                        ;
                }
                                
                // If gamma <= theta the algorithm must stop.
                // If theta == 0 and gamma is 0, it means that the weak learner is no better than chance
                // and no further training is possible.
                if (gamma <= 0)
                {
                    if (_verbose > 0)
                    {
                        cout << "Can't train any further: edge = " << gamma << endl;
                    }
                    //cout << pWeakHypothesis->getEdge(false) << endl << flush; 
                    //          delete pWeakHypothesis;
                    //          break; 
                }
                                
                                
                // Add it to the internal list of weak hypotheses
                _foundHypotheses[stagei].push_back(pWeakHypothesis); 
                                
                // evaluate current detector on validation set
                updatePosteriors( pTrainingData, pWeakHypothesis, trainPosteriors, _positiveLabelIndex );
                updatePosteriors( pValidationData, pWeakHypothesis, validPosteriors, _positiveLabelIndex );
                if (pTestData) updatePosteriors( pTestData, pWeakHypothesis, testPosteriors, _positiveLabelIndex );
                                
                // caclualte the current detection rate and false positive rate                         
                //getTPRandFPR( pValidationData, validPosteriors, currentTPR, currentFPR );                             
                //if (_verbose>4)
                //      cout << "Current TPR: " << currentTPR << " Current FPR: " << currentFPR << endl << flush;
                                
                tunedThreshold = getThresholdBasedOnTPR( pValidationData, validPosteriors, Di, currentTPR, currentFPR );
                                
                if (_verbose>1)
                {
                    cout << "**** Threshold: " << tunedThreshold << endl; //"\tExpected detection rate: " << Di << endl;
                    cout << "**** Current TPR: " << currentTPR << "(Expected: " << Di << ")" << endl;
                    cout << "**** Current FPR: " << currentFPR << "(Expected: " << Fi << ")" << endl << flush;
                }
                                
                //if (((currentFPR<(prevFi*_maxAcceptableFalsePositiveRate)) || (t>1000))&&(!(t<th[stagei])))
                if (((currentFPR<(Fi)) || (t>10000))&&(!(t<1)))
                {
                    if (t>10000)
                    {
                        cout << "Warning maximal iteration number per stage has reached!!!!" << endl;
                    }
                                        
                    break;
                }
                                
                t++;
                                
            }  // loop on iterations
                        
            // store threshold
            _thresholds.push_back(tunedThreshold);
                        
            // calculate the overall cascade performance
            pValidationData->clearIndexSet();
            validPosteriors.resize(pValidationData->getNumExamples());
            calculatePosteriors( pValidationData, _foundHypotheses[stagei], validPosteriors, _positiveLabelIndex );
                        
            // this update the current forecast stored in activeValidationInstances
            forecastOverAllCascade( pValidationData, validPosteriors, activeValidationInstances, tunedThreshold );
            if (pTestData) forecastOverAllCascade( pTestData, testPosteriors, activeTestInstances, tunedThreshold );
                        
            _output << (stagei+1) << "\t";
            _output << _foundHypotheses[stagei].size() << "\t";
                        
                        
            //_output << "valid" << endl;
            outputCascadeResult( pValidationData, activeValidationInstances );
            if (pTestData) 
            {
                //_output << "test" << endl;
                outputCascadeResult( pTestData, activeTestInstances );                  
            }
                                                                        

            // filter training dataset data set and generate negative set for the next iteration
            trainingIndices.clear();
            pTrainingData->clearIndexSet();
            //cout << pTrainingData->getNumExamples() << endl << flush;
            trainPosteriors.resize(pTrainingData->getNumExamples());
            calculatePosteriors( pTrainingData, _foundHypotheses[stagei], trainPosteriors, _positiveLabelIndex );

            int trainPosNum=0;
            int trainNegNum=0;
            int validPosNum=0;
            int validNegNum=0;

                        
            for( int i=0; i < activeTrainInstances.size(); ++i )
            {       
                const Example& example = pTrainingData->getExample(i);
                int labelY = example.getLabelY(_positiveLabelIndex);

                if (labelY>0)
                {       
                    trainPosNum++;
                    activeTrainInstances[i].active=true; // all positive
                    trainingIndices.insert(i);
                } else {
                    if ( trainPosteriors[i] >= tunedThreshold )
                    {
                        trainNegNum++;
                        activeTrainInstances[i].active=true; // all false positive
                        trainingIndices.insert(i);
                    } else {
                        activeTrainInstances[i].active=false;
                    }
                }
            }               
            // filter training                      
            pTrainingData->loadIndexSet( trainingIndices );

            // output the actual training dataset size
            _output << (trainPosNum+trainNegNum) << "\t" << trainPosNum << "\t" << trainNegNum << "\t";
                        
            validationIndices.clear();
            pValidationData->clearIndexSet();
            validPosteriors.resize(pValidationData->getNumExamples());
            //calculatePosteriors( pValidationData, _foundHypotheses[stagei], validPosteriors );
            for( int i=0; i < activeValidationInstances.size(); ++i )
            {       
                const Example& example = pValidationData->getExample(i);
                int labelY = example.getLabelY(_positiveLabelIndex);
                                
                if (labelY>0)
                {       
                    validPosNum++;
                    activeValidationInstances[i].active=true; // all positive
                    validationIndices.insert(i);
                } else {
                    if ( validPosteriors[i] >= tunedThreshold )
                    {
                        validNegNum++;
                        activeValidationInstances[i].active=true; // all false positive
                        validationIndices.insert(i);
                    } else {
                        activeValidationInstances[i].active=false; // all false positive
                    }
                }
            }       
                                
            // filter validation dataset
            pValidationData->loadIndexSet(validationIndices);
                                
                                
            //cout << "The size of validation dataset: " << (validPosNum+validNegNum) << "(" << trainPosNum << "/" << validNegNum << ")" << endl;                           
                                                
            _output << (validPosNum+validNegNum) << "\t" << validPosNum << "\t" << validNegNum << "\t";
                        
            if (_verbose>0 )
            {
                cout << "****************************************************************" << endl;
                cout << "**** STOP ADABOOST****" << endl; 
                cout << "**** Stage:\t" << stagei+1 << endl; 
                cout << "**** It. num:\t" << _foundHypotheses[stagei].size() << endl;
                cout << "Validation set: " << (validPosNum+validNegNum) << "(" << trainPosNum << "/" << validNegNum << ")" << endl;
                cout << "Training set: \t" << (trainPosNum+trainNegNum) << "(" << trainPosNum << "/" << trainNegNum << ")" << endl;
                cout << "****************************************************************" << endl;
            }
                        
                        
            _output << endl;                        
                        
            if ( trainNegNum < 1 )
            {
                cout << "No negative remained in training set!!!!" << endl;
                exit(-1);
            }
                        
            //_maxAcceptableFalsePositiveRate
            //_minAcceptableDetectionRate
                        
            /////////////////////////////////////////////////////////
            // save the weak hypithesis of current stage
            ss.appendStageSeparatorHeader( stagei, _foundHypotheses[stagei].size(), _thresholds[stagei] );
            //ss.appendStageSeparatorFooter();
            // append the current weak learner to strong hypothesis file,
            // that is, serialize it.                                       
            for (int t=0 ; t < _foundHypotheses[stagei].size(); ++t )
                ss.appendHypothesis(t, _foundHypotheses[stagei][t]);
                        
                        
        }// end of cascade
                
                
        // write the footer of the strong hypothesis file
        ss.writeCascadeFooter();
                
        // write the weights of the instances if the name of weights file isn't empty
        //printOutWeights( pTrainingData );
                
                
        // Free the two input data objects
        if (pTrainingData) {                    
            delete pTrainingData;
            pTrainingData = NULL;
        }
        if (pValidationData)
            delete pValidationData;
        if (pTestData)
            delete pTestData;
                
        _output.close();
                
        if (_verbose > 0)
            cout << "Learning completed." << endl;
    }
        
    // -------------------------------------------------------------------------
    void VJCascadeLearner::updatePosteriors( InputData* pData, BaseLearner* weakHypotheses, vector<AlphaReal>& posteriors, int positiveLabelIndex )
    {
        const int numExamples = pData->getNumExamples();                
                
        AlphaReal alpha = weakHypotheses->getAlpha();
        // for every point
        for (int i = 0; i < numExamples; ++i)
        {
            // just for the negative class
            posteriors[i] += alpha * weakHypotheses->classify(pData, i, positiveLabelIndex);
        }                       
    }
        
        
    // -------------------------------------------------------------------------
    void VJCascadeLearner::calculatePosteriors( InputData* pData, vector<BaseLearner*>& weakHypotheses, vector<AlphaReal>& posteriors, int positiveLabelIndex )
    {
        const int numExamples = pData->getNumExamples();                        
                
        posteriors.resize(numExamples);
        fill( posteriors.begin(), posteriors.end(), 0.0 );
                
        vector<BaseLearner*>::iterator whyIt = weakHypotheses.begin();                          
        for (;whyIt != weakHypotheses.end(); ++whyIt )
        {
            BaseLearner* currWeakHyp = *whyIt;
            AlphaReal alpha = currWeakHyp->getAlpha();
                        
            // for every point
            for (int i = 0; i < numExamples; ++i)
            {
                // just for the negative class
                posteriors[i] += alpha * currWeakHyp->classify(pData, i, positiveLabelIndex);
            }                       
        }
        /*
          for (int i = 0; i < numExamples; ++i)
          {
          // just for the negative class
          posteriors[i] /= sumAlpha;
          }                      
        */
    }
        
        
    // -------------------------------------------------------------------------                                                     
    void VJCascadeLearner::classify(const nor_utils::Args& args)
    {
        VJCascadeClassifier classifier(args, _verbose);
                
        // -test <dataFile> <shypFile>
        string testFileName = args.getValue<string>("test", 0);
        string shypFileName = args.getValue<string>("test", 1);
        int numIterations = args.getValue<int>("test", 2);
                
        string outResFileName = "";
        if ( args.getNumValues("test") > 3 )
            args.getValue("test", 3, outResFileName);
                
        classifier.run(testFileName, shypFileName, numIterations, outResFileName);
    }
        
    // -------------------------------------------------------------------------
        
    void VJCascadeLearner::doConfusionMatrix(const nor_utils::Args& args)
    {
        VJCascadeClassifier classifier(args, _verbose);
                
        // -cmatrix <dataFile> <shypFile>
        if ( args.getNumValues("cmatrix") == 2 )
        {
            string testFileName = args.getValue<string>("cmatrix", 0);
            string shypFileName = args.getValue<string>("cmatrix", 1);
                        
            classifier.printConfusionMatrix(testFileName, shypFileName);
        }
        // -cmatrix <dataFile> <shypFile> <outFile>
        else if ( args.getNumValues("cmatrix") == 3)
        {
            string testFileName = args.getValue<string>("cmatrix", 0);
            string shypFileName = args.getValue<string>("cmatrix", 1);
            string outResFileName = args.getValue<string>("cmatrix", 2);
                        
            classifier.saveConfusionMatrix(testFileName, shypFileName, outResFileName);
        }
    }
        
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    void VJCascadeLearner::doPosteriors(const nor_utils::Args& args)
    {
        VJCascadeClassifier classifier(args, _verbose);
        //int numofargs = args.getNumValues( "posteriors" );
        // -posteriors <dataFile> <shypFile> <outFile> <numIters>
        string testFileName = args.getValue<string>("posteriors", 0);
        string shypFileName = args.getValue<string>("posteriors", 1);
        string outFileName = args.getValue<string>("posteriors", 2);
        int numStages = args.getValue<int>("posteriors", 3);
                
        classifier.savePosteriors(testFileName, shypFileName, outFileName, numStages);
    }
        
    // -------------------------------------------------------------------------    
    // -------------------------------------------------------------------------
        
    AlphaReal VJCascadeLearner::updateWeights(InputData* pData, BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();
                
        const AlphaReal alpha = pWeakHypothesis->getAlpha();
                
        float Z = 0; // The normalization factor
                
        _hy.resize(numExamples);
        for ( int i = 0; i < numExamples; ++i) {
            _hy[i].resize(numClasses);
            fill( _hy[i].begin(), _hy[i].end(), 0.0 );
        }
        // recompute weights
        // computing the normalization factor Z
                
        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                _hy[i][lIt->idx] = pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
                    lIt->y;
                Z += lIt->weight * // w
                    exp( 
                        -alpha * _hy[i][lIt->idx] // -alpha * h_l(x_i) * y_i
                        );
                // important!
                // _hy[i] must be a vector with different sizes, depending on the
                // example!
                // so it will become:
                // _hy[i][l] 
                // where l is NOT the index of the label (lIt->idx), but the index in the 
                // label vector of the example
            }
        }
                
        AlphaReal gamma = 0;
                
        // Now do the actual re-weight
        // (and compute the edge at the same time)
        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                AlphaReal w = lIt->weight;
                gamma += w * _hy[i][lIt->idx];
                //if ( gamma < -0.8 ) {
                //      cout << gamma << endl;
                //}
                // The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
                lIt->weight = w * exp( -alpha * _hy[i][lIt->idx] ) / Z;
            }
        }                               
        return gamma;
    }
        
    // -------------------------------------------------------------------------
        
    int VJCascadeLearner::resumeWeakLearners(InputData* pTrainingData)
    {
        if (_resumeShypFileName.empty())
            return 0;
                
        if (_verbose > 0)
            cout << "Reloading strong hypothesis file <" << _resumeShypFileName << ">.." << flush;
                
        // The class that loads the weak hypotheses
        UnSerialization us;
                
        // loads them stagewise
        for(int stagei=0; stagei < _numIterations; ++stagei )
        {               
            us.loadHypotheses(_resumeShypFileName, _foundHypotheses[stagei], pTrainingData, _verbose);
                        
        }
                
        if (_verbose > 0)
            cout << "Done!" << endl;
                
        // return the number of iterations found
        return static_cast<int>( _foundHypotheses.size() );
    }
        
    // -------------------------------------------------------------------------
    void VJCascadeLearner::checkWeights(InputData* pData)
    {
        const int numOfSamples = pData->getNumExamples();
        AlphaReal sumWeight = 0;
        for (int i = 0; i < numOfSamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            // first find the sum of the weights
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                sumWeight += lIt->weight;
        }
                
        cout << "Check weights: " << pData->getNumExamples() << endl << flush;
                
        if ( !nor_utils::is_zero(sumWeight-1.0, 1E-6 ) )
        {
            cerr << "\nERROR: Sum of weights (" << sumWeight << ") != 1!" << endl;
            //exit(1);
        }               
    }
    // -------------------------------------------------------------------------
        
    void VJCascadeLearner::resetWeights(InputData* pData)
    {
        // this weighting corresponds to the sharepoint one
        int numOfClasses = pData->getNumClasses();
        vector< int > numPerClasses(numOfClasses);
        int numOfSamples = pData->getNumExamples();
        vector< AlphaReal > wi( numOfClasses );
        vector< AlphaReal > wic( numOfClasses );
                
        fill(numPerClasses.begin(),numPerClasses.end(), 0 );
        for (int i = 0; i < numOfSamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                if ( lIt->y > 0 )
                    numPerClasses[lIt->idx]++;
            }
                        
        }                       
                
                
        // we assume pl = 1/K
        for( int i = 0; i < numOfClasses; i ++ ) {
            wi[i] =  1.0  / (4.0 * numPerClasses[i]);
            wic[i] = 1.0 / (4.0 * ( numOfSamples - numPerClasses[i] ));
        }
        //cout << endl;
                
                
                
                
        //this->_nExamplesPerClass
        // for each example
        for (int i = 0; i < numOfSamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            // first find the sum of the weights                                                            
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                if (lIt->y>0) lIt->weight = wi[lIt->idx];
                else lIt->weight=wic[lIt->idx];
            }
                        
        }
        // check for the sum of weights!
        AlphaReal sumWeight = 0;
        for (int i = 0; i < numOfSamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            // first find the sum of the weights
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                sumWeight += lIt->weight;
        }
                
        if ( !nor_utils::is_zero(sumWeight-1.0, 1E-6 ) )
        {
            cerr << "\nERROR: Sum of weights (" << sumWeight << ") != 1!" << endl;
            cerr << "Try a different weight policy (--weightpolicy under 'Basic Algorithm Options')!" << endl;
            //exit(1);
        }
                
                
    }
    // -------------------------------------------------------------------------
    void VJCascadeLearner::getTPRandFPR( InputData* pData, vector<AlphaReal>& posteriors, AlphaReal& TPR, AlphaReal& FPR, const FeatureReal threshold )
    {
        const int numOfExamples = pData->getNumExamples();
        int TP=0,FP=0;
        int P=0,N=0;
                
                
        for(int i=0; i<numOfExamples; ++i )
        {
            int forecast = 0;
            if (posteriors[i] >= threshold ) forecast = 1;
                        
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);

            if (labelY>0)
                //if (pData->hasLabel(_positiveLabelIndex, i) ) //positive element
            {
                P++;
                if (forecast==1) TP++;
            } else {
                N++;
                if (forecast==1) FP++;
            }                       
        }       
        TPR = TP / (AlphaReal)P;
        FPR = FP / (AlphaReal)N;
    }
        
    // -------------------------------------------------------------------------
    FeatureReal VJCascadeLearner::getThresholdBasedOnTPR( InputData* pData, vector<AlphaReal>& posteriors, const AlphaReal expectedTPR, AlphaReal& TPR, AlphaReal& FPR )
    {
        const int numOfExamples = pData->getNumExamples();
        vector<int> numPerClasses(2);
        int TP=0,FP=0;          
        FeatureReal threshold=numeric_limits<FeatureReal>::max();
                
        vector<pair<AlphaReal,int> > sortedPosteriors(posteriors.size());
                
        for(int i=0; i<numOfExamples; ++i )
        {
            //cout << posteriors[i] << " ";
            sortedPosteriors[i].first = posteriors[i];
                        
            const Example& example = pData->getExample(i);
            int labelY = example.getLabelY(_positiveLabelIndex);

            if (labelY>0)                           
                //if (pData->hasLabel(_positiveLabelIndex, i) ) //positive element
            {
                sortedPosteriors[i].second=1;
            } else {
                sortedPosteriors[i].second=0;
            }                                               
        }
                
        sort( sortedPosteriors.begin(), sortedPosteriors.end(), 
              nor_utils::comparePair< 1, AlphaReal, int, greater<AlphaReal> >() );
                
                
                
        fill(numPerClasses.begin(),numPerClasses.end(), 0 );
        for (int i = 0; i < numOfExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                if ( lIt->y > 0 )
                    numPerClasses[lIt->idx]++;
            }
                        
        }                       
                
                
                
        //FP=numPerClasses[1-_positiveLabelIndex];              
        FP=TP=0;
        threshold=static_cast<FeatureReal>(sortedPosteriors[0].first)+numeric_limits<FeatureReal>::min();
                
        for(int i=0; i<numOfExamples; ++i )
        {
                        
            if (sortedPosteriors[i].second) TP++;
            else FP++;
                        
                        
            AlphaReal currentTPR = TP / static_cast<AlphaReal>(numPerClasses[_positiveLabelIndex]);
            //double currentFPR = FP / (double)numPerClasses[1-_positiveLabelIndex];
                        
            //cout << currentTPR <<  "\t" << currentFPR << endl << flush;
                        
            if ((i>0)&&(sortedPosteriors[i-1].first!=sortedPosteriors[i].first))
            {
                threshold=(sortedPosteriors[i-1].first+sortedPosteriors[i].first)/2;
                if ( currentTPR > expectedTPR ) {
                    break;
                }
            }
                        
        }       
                
        TPR = TP/static_cast<AlphaReal>(numPerClasses[_positiveLabelIndex]);
        FPR = FP/static_cast<AlphaReal>(numPerClasses[1-_positiveLabelIndex]);
                
        return threshold;
    }
    // -------------------------------------------------------------------------
    void VJCascadeLearner::forecastOverAllCascade( InputData* pData, vector< AlphaReal >& posteriors, vector<CascadeOutputInformation>& cascadeData, const FeatureReal threshold )
    {
        const int numOfExamples = pData->getNumExamples();
        bool isPos;
        int sumOfWeakClassifier=0;
        int stagei = _foundHypotheses.size();
        //collect cascade information
        for(int i=0; i<_foundHypotheses.size(); ++i)
        {
            sumOfWeakClassifier += _foundHypotheses[i].size();
        }
                
        AlphaReal sumalphas = 0.0;
        for(int i=0; i<_foundHypotheses[stagei-1].size(); ++i)
        {
            sumalphas += _foundHypotheses[stagei-1][i]->getAlpha();
        }
                                
        for(int i=0; i<numOfExamples; ++i )
        {
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
                
                // Here starts the new method
                cascadeData[i].score += stagei - 1 ;
                
                if (posteriors[i]<threshold)
                {
                    cascadeData[i].active = false; // classified
                    cascadeData[i].forecast=0;
                } else {
                    cascadeData[i].active = true; // continue
                    cascadeData[i].forecast=1;                                      
                }
                
                
// PREVIOUS METHOD
//                              if (posteriors[i]<threshold)
//                              {
//                                      cascadeData[i].active = false; // classified
//                                      cascadeData[i].forecast=0;
//                                      cascadeData[i].score+=(2.0*(stagei-1));//the coefficient 2.0 is used because the stagewise posteriors should not
//                                      cascadeData[i].score/=((2.0*stagei)+1);
//                              } else {
//                                      cascadeData[i].active = true; // continue
//                                      cascadeData[i].forecast=1;                                      
//                                      cascadeData[i].score+=(2.0*stagei);
//                                      cascadeData[i].score/=((2.0*stagei)+1);
//                              }
                                
                cascadeData[i].classifiedInStage=stagei;
                cascadeData[i].numberOfUsedClassifier=sumOfWeakClassifier;                                                              
            }                       
        }                               
    }
        
    // -------------------------------------------------------------------------
    void VJCascadeLearner::outputHeader()
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
                
        _output << "Valid. FPR\t";
        _output << "Valid. TPR\t";              
        _output << "Valid. ROC\t";
                
        _output << "Test FPR\t";
        _output << "Test TPR\t";                
        _output << "Test ROC\t";        
                
        _output << "Training Dataset\t";
        _output << "Positives\t";
        _output << "Negatives\t";

        _output << "Valid. Dataset\t";
        _output << "Positives\t";
        _output << "Negatives\t";
                
        _output << endl << flush;
    }
        
        
    // -------------------------------------------------------------------------
    void VJCascadeLearner::outputCascadeResult( InputData* pData, vector<CascadeOutputInformation>& cascadeData )
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
                
        /*
          int sumStage=0;
          for(int i=0; i<numOfExamples; ++i )
          {
          sumStage += cascadeData[i].classifiedInStage;
          }               
          _output << sumStage / ((double)numOfExamples) << "\t";
                                
          int sumWeakHyp=0;
          for(int i=0; i<numOfExamples; ++i )
          {
          sumWeakHyp += cascadeData[i].numberOfUsedClassifier;
          }               
          _output << sumWeakHyp / ((double)numOfExamples) << "\t";
        */
        //_output << endl << flush;
                
        if (0)
        {       
            _output << "origLabs";
            for(int i=0; i<numOfExamples; ++i )
            {
                const Example& example = pData->getExample(i);
                int labelY = example.getLabelY(_positiveLabelIndex);

                if (labelY>0)
                {
                    _output << ",1";
                } else { 
                    _output << ",0";
                }
            }               
            _output << endl;
                        
            _output << "forecast";
            for(int i=0; i<numOfExamples; ++i )
            {
                _output << "," << cascadeData[i].forecast;
            }               
            _output << endl;
                        
            _output << "classifiedInStage"; 
            for(int i=0; i<numOfExamples; ++i )
            {
                _output << "," << cascadeData[i].classifiedInStage;
            }               
            _output << endl;
                        
            _output << "numberOfUsedClassifier";    
            for(int i=0; i<numOfExamples; ++i )
            {
                _output << "," << cascadeData[i].numberOfUsedClassifier;
            }               
            _output << endl;
                        
            _output << "score";     
            for(int i=0; i<numOfExamples; ++i )
            {
                //double s = (cascadeData[i].score / (alphas[cascadeData[i].classifiedInStage-1]));
                _output << "," << cascadeData[i].score;
                //_output << "," << s;
                                
            }               
                        
        }
                
    }
        
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
        
} // end of namespace MultiBoost

