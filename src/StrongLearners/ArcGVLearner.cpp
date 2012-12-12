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
#include "ArcGVLearner.h"
#include "Classifiers/AdaBoostMHClassifier.h"

namespace MultiBoost {
    // -----------------------------------------------------------------------------------
        
    void ArcGVLearner::getArgs(const nor_utils::Args& args)
    {
        AdaBoostMHLearner::getArgs( args );
                
        // Set the minimal value of margin. Below this margin
        // the alphas are not regularized
        if ( args.hasArgument("minmarginthreshold") )
            args.getValue("minmarginthreshold", 0, _minMarginThreshold);  
                
    }               
        
    // -----------------------------------------------------------------------------------
        
    void ArcGVLearner::run(const nor_utils::Args& args)
    {
        // load the arguments
        this->getArgs(args);
                
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
                
        // get the testing input data, and load it
        InputData* pTestData = NULL;
        if ( !_testFileName.empty() )
        {
            pTestData = pWeakHypothesisSource->createInputData();
            pTestData->initOptions(args);
            pTestData->load(_testFileName, IT_TEST, _verbose);
        }
                
        // The output information object
        OutputInfo* pOutInfo = NULL;            
                
        if ( !_outputInfoFile.empty() ) 
        {
            pOutInfo = new OutputInfo(args);
            pOutInfo->initialize(pTrainingData);
            
            if (pTestData)
                pOutInfo->initialize(pTestData);
            pOutInfo->outputHeader(pTrainingData->getClassMap());
            
            if ( ! args.hasArgument("resume") )
            {
                // Baseline: constant classifier - goes into 0th iteration
                
                BaseLearner* pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
                pConstantWeakHypothesis->initLearningOptions(args);
                pConstantWeakHypothesis->setTrainingData(pTrainingData);
                pConstantWeakHypothesis->run();
                
                pOutInfo->outputIteration(-1);
                pOutInfo->outputCustom(pTrainingData, pConstantWeakHypothesis);
                if (pTestData != NULL)
                {
                    pOutInfo->separator();
                    pOutInfo->outputCustom(pTestData, pConstantWeakHypothesis);
                }
                pOutInfo->outputCurrentTime();
                pOutInfo->endLine();
                
                pOutInfo->initialize(pTrainingData);
                if (pTestData)
                    pOutInfo->initialize(pTestData);
            }
        } else {
            cout << "Output file is empty!!!!" << endl;
            exit(-1);
        }

        //cout << "Before serialization" << endl;
        // reload the previously found weak learners if -resume is set. 
        // otherwise just return 0
        int startingIteration = resumeWeakLearners(pTrainingData);
                
        Serialization ss(_shypFileName, _isShypCompressed );
        ss.writeHeader(_baseLearnerName); // this must go after resumeProcess has been called
                
        // perform the resuming if necessary. If not it will just return
        resumeProcess(ss, pTrainingData, pTestData, pOutInfo);
                
                
        if (_verbose == 1)
            cout << "Learning in progress..." << endl;
                
        //I put here the starting time, but it may take very long time to load the saved model
        time_t startTime, currentTime;
        time(&startTime);
                
        ///////////////////////////////////////////////////////////////////////
        // Starting the AdaBoost main loop
        ///////////////////////////////////////////////////////////////////////
        for (int t = startingIteration; t < _numIterations; ++t)
        {
            if (_verbose > 1)
                cout << "------- WORKING ON ITERATION " << (t+1) << " -------" << endl;
                        
            BaseLearner* pWeakHypothesis = pWeakHypothesisSource->create();
            pWeakHypothesis->initLearningOptions(args);
            //pTrainingData->clearIndexSet();
                        
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
                                
                if ( (constantEnergy <= energy) || ( energy != energy ) ) {
                    delete pWeakHypothesis;
                    pWeakHypothesis = pConstantWeakHypothesis;
                }
            }
                        
            // find the min margin on training data
            table& g = pOutInfo->getMargins( pTrainingData );                        
            AlphaReal alphaSum = pOutInfo->getSumOfAlphas(pTrainingData);
            if (nor_utils::is_zero(alphaSum)) alphaSum = 1.0;
                        
            AlphaReal minMargin = numeric_limits<AlphaReal>::max();
            for (int i=0; i<pTrainingData->getNumExamples(); ++i )
            {
                const vector<Label>& labels = pTrainingData->getLabels(i);
                vector<Label>::const_iterator lIt;
                                
                for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {                                       
                    AlphaReal tmpMargin = g[i][lIt->idx] / alphaSum;
                    if ( minMargin > tmpMargin) minMargin = tmpMargin;
                }
            }
                        
            if (_verbose>2)
            {
                cout << "---> Min margin: " << minMargin << endl;
            }

            // add the arc-gv term to the coefficient of base classifier
            // if the minMargin is smaller than a threshold, it is set to a user-defined value
            if (minMargin< _minMarginThreshold) minMargin = _minMarginThreshold;

            // update alpha
            AlphaReal alpha = pWeakHypothesis->getAlpha();
            AlphaReal newAlpha = alpha - 0.5 * log ( ( 1.0 + minMargin ) / ( 1.0 - minMargin ) ); 
            pWeakHypothesis->setAlpha(newAlpha);

            // update the sum of coefficients
            _alphaSum+=newAlpha;

            if (_verbose>2)
            {
                cout << "---> Alpha (based on AdaBoost.MH):  " << alpha << endl;
                cout << "---> Alpha (based on ARC-GV):       " << newAlpha << endl << flush;
            }
                        
                        
            if (_verbose > 1)
                cout << "Weak learner: " << pWeakHypothesis->getName()<< endl;

                        
            // Output the step-by-step information
            printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);
                        
            // Updates the weights and returns the edge, and update the alpha
            AlphaReal gamma = updateWeights(pTrainingData, pWeakHypothesis);
                        
                        
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
            if (gamma <= _theta)
            {
                if (_verbose > 0)
                {
                    cout << "Can't train any further: edge = " << gamma 
                         << " (with and edge offset (theta)=" << _theta << ")" << endl;
                }
                                
                //          delete pWeakHypothesis;
                //          break; 
            }
                        
            // append the current weak learner to strong hypothesis file,
            // that is, serialize it.
            ss.appendHypothesis(t, pWeakHypothesis);
                        
            // Add it to the internal list of weak hypotheses
            _foundHypotheses.push_back(pWeakHypothesis); 
                        
            // check if the time limit has been reached
            if (_maxTime > 0)
            {
                time( &currentTime );
                float diff = difftime(currentTime, startTime); // difftime is in seconds
                diff /= 60; // = minutes
                                
                if (diff > _maxTime)
                {
                    if (_verbose > 0)
                        cout << "Time limit of " << _maxTime 
                             << " minutes has been reached!" << endl;
                    break;     
                }
            } // check for maxtime
            delete pWeakHypothesis;
        }  // loop on iterations
        /////////////////////////////////////////////////////////
                
        // write the footer of the strong hypothesis file
        ss.writeFooter();
                
        // write the weights of the instances if the name of weights file isn't empty
        printOutWeights( pTrainingData );
                
                
        // Free the two input data objects
        if (pTrainingData)
            delete pTrainingData;
        if (pTestData)
            delete pTestData;
                
        if (pOutInfo)
            delete pOutInfo;
                
        if (_verbose > 0)
            cout << "Learning completed." << endl;
    }
        
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
} // end of namespace ArcGV
