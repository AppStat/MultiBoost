#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <limits>
#include <iomanip> // setprecision
#include <typeinfo> // for bad_cast

#include "Utils/Utils.h" // for addAndCheckExtension
#include "Defaults.h" // for defaultLearner
#include "IO/OutputInfo.h"
#include "Others/Rates.h"
#include "IO/InputData.h"
#include "IO/Serialization.h" // to save the found strong hypothesis

#include "WeakLearners/BaseLearner.h"
#include "StrongLearners/FilterBoostLearner.h"
#include "WeakLearners/StochasticLearner.h"

#include "Classifiers/AdaBoostMHClassifier.h"

namespace MultiBoost {

    // -----------------------------------------------------------------------------------

    void FilterBoostLearner::getArgs(const nor_utils::Args& args)
    {
        AdaBoostMHLearner::getArgs( args );
        // Set the value of the sample size
        if ( args.hasArgument("Cn") )
        {
            args.getValue("Cn", 0, _Cn);
            if (_verbose > 1)
                cout << "--> Resampling size: " << _Cn << endl;
        }

        if ( args.hasArgument("onlinetraining") )
        {
            _onlineWeakLearning = true;                     
        }                                       
    }

    // -----------------------------------------------------------------------------------

    void FilterBoostLearner::run(const nor_utils::Args& args)
    {
        // load the arguments
        this->getArgs(args);

        time_t startTime, currentTime;
        time(&startTime);

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

        const int numClasses = pTrainingData->getNumClasses();
        const int numExamples = pTrainingData->getNumExamples();
                
        //initialize the margins variable
        _margins.resize( numExamples );
        for( int i=0; i<numExamples; i++ )
        {
            _margins[i].resize( numClasses );
            fill( _margins[i].begin(), _margins[i].end(), 0.0 );
        }


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
            // Baseline: constant classifier - goes into 0th iteration

            BaseLearner* pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
            pConstantWeakHypothesis->initLearningOptions(args);
            pConstantWeakHypothesis->setTrainingData(pTrainingData);
            AlphaReal constantEnergy = pConstantWeakHypothesis->run();

            pOutInfo = new OutputInfo(args);
            pOutInfo->initialize(pTrainingData);

            updateMargins( pTrainingData, pConstantWeakHypothesis );

            if (pTestData)
                pOutInfo->initialize(pTestData);
            pOutInfo->outputHeader(pTrainingData->getClassMap() );

            pOutInfo->outputIteration(-1);
            pOutInfo->outputCustom(pTrainingData, pConstantWeakHypothesis);

            if (pTestData)
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
        // reload the previously found weak learners if -resume is set. 
        // otherwise just return 0
        int startingIteration = resumeWeakLearners(pTrainingData);


        Serialization ss(_shypFileName, _isShypCompressed );
        ss.writeHeader(_baseLearnerName); // this must go after resumeProcess has been called

        // perform the resuming if necessary. If not it will just return
        resumeProcess(ss, pTrainingData, pTestData, pOutInfo);

        if (_verbose == 1)
            cout << "Learning in progress..." << endl;
                                
        ///////////////////////////////////////////////////////////////////////
        // Starting the AdaBoost main loop
        ///////////////////////////////////////////////////////////////////////
        for (int t = startingIteration; t < _numIterations; ++t)
        {                       
            if (_verbose > 1)
                cout << "------- WORKING ON ITERATION " << (t+1) << " -------" << endl;
                
            // create the weak learner
            BaseLearner* pWeakHypothesis;
            BaseLearner* pConstantWeakHypothesis;
            pWeakHypothesis = pWeakHypothesisSource->create();
            pWeakHypothesis->initLearningOptions(args);
            //pTrainingData->clearIndexSet();
            pWeakHypothesis->setTrainingData(pTrainingData);
            AlphaReal edge, energy=0.0;
                        
            // create the constant learner
            pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
            pConstantWeakHypothesis->initLearningOptions(args);
            pConstantWeakHypothesis->setTrainingData(pTrainingData);
            AlphaReal constantEdge = -numeric_limits<AlphaReal>::max();
                        
            int currentNumberOfUsedData = static_cast<int>(_Cn * log(t+3.0));
                        
            if ( _onlineWeakLearning )
            {
                //check whether the weak learner is a ScalarLeaerner
                try {
                    StochasticLearner* pStochasticLearner = dynamic_cast<StochasticLearner*>(pWeakHypothesis);
                    StochasticLearner* pStochasticConstantWeakHypothesis = dynamic_cast<StochasticLearner*> (pConstantWeakHypothesis);
                                        
                    pStochasticLearner->initLearning();
                    pStochasticConstantWeakHypothesis->initLearning();                                                                              
                                        
                    if (_verbose>1)
                        cout << "Number of random instances: \t" << currentNumberOfUsedData << endl;
                                        
                    // set the weights
                    setWeightToMargins(pTrainingData);
                                        
                    //learning
                    for (int i=0; i<currentNumberOfUsedData; ++i )
                    {
                        int randomIndex = (rand() % pTrainingData->getNumExamples());   
                        //int randomIndex = getRandomIndex();
                        pStochasticLearner->update(randomIndex);
                        pStochasticConstantWeakHypothesis->update(randomIndex);
                    }                                       
                    pStochasticLearner->finishLearning();           
                    pStochasticConstantWeakHypothesis->finishLearning();
                }
                catch (bad_cast& e) {
                    cerr << "The weak learner must be a StochasticLearner!!!" << endl;
                    exit(-1);
                }                                                                                               
            }
            else
            {
                filter( pTrainingData, currentNumberOfUsedData );
                if ( pTrainingData->getNumExamples() < 2 ) 
                {
                    filter( pTrainingData, currentNumberOfUsedData, false );
                }
                                
                if (_verbose > 1)
                {
                    cout << "--> Size of training data = " << pTrainingData->getNumExamples() << endl;
                }
                                
                energy = pWeakHypothesis->run();                                                                
                pConstantWeakHypothesis->run(); 
            }                       

            //estimate edge
            filter( pTrainingData, currentNumberOfUsedData, false );
            edge = pWeakHypothesis->getEdge(true) / 2.0;                                            
            constantEdge = pConstantWeakHypothesis->getEdge() / 2.0;
                        
                        
            if ( constantEdge > edge )
            {
                delete pWeakHypothesis;
                pWeakHypothesis = pConstantWeakHypothesis;
                edge = constantEdge;
            } else {
                delete pConstantWeakHypothesis;
            }
                                                                        
            // calculate alpha
            AlphaReal alpha = 0.0;
            alpha = 0.5 * log( ( 1 + edge ) / ( 1 - edge ) );
            pWeakHypothesis->setAlpha( alpha );
            _sumAlpha += alpha;
                        
            if (_verbose > 1)
                cout << "Weak learner: " << pWeakHypothesis->getName()<< endl;
            // Output the step-by-step information
            pTrainingData->clearIndexSet();
            printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);

            // Updates the weights and returns the edge
            //AlphaReal gamma = updateWeights(pTrainingData, pWeakHypothesis);

            if (_verbose > 1)
            {
                cout << setprecision(5)
                     << "--> Alpha = " << pWeakHypothesis->getAlpha() << endl
                     << "--> Edge  = " << edge << endl
                     << "--> Energy  = " << energy << endl
                    //            << "--> ConstantEnergy  = " << constantEnergy << endl
                    //            << "--> difference  = " << (energy - constantEnergy) << endl
                    ;
            }

            // update the margins
            //saveMargins();
            updateMargins( pTrainingData, pWeakHypothesis );
                        
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

    void FilterBoostLearner::resumeProcess(Serialization& ss, 
                                           InputData* pTrainingData, InputData* pTestData, 
                                           OutputInfo* pOutInfo)
    {

        if (_resumeShypFileName.empty())
            return;

        if (_verbose > 0)
            cout << "Resuming up to iteration " << _foundHypotheses.size() - 1 << ": 0%." << flush;

        vector<BaseLearner*>::iterator it;
        int t;

        // rebuild the new strong hypothesis file
        for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
        {
            BaseLearner* pWeakHypothesis = *it;

            // append the current weak learner to strong hypothesis file,
            ss.appendHypothesis(t, pWeakHypothesis);
        }

        const int numIters = static_cast<int>(_foundHypotheses.size());
        const int step = numIters < 5 ? 1 : numIters / 5;

        // simulate the AdaBoost algorithm for the weak learners already found
        for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
        {
            BaseLearner* pWeakHypothesis = *it;

            // Output the step-by-step information
            printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);

            // Updates the weights and returns the edge
            updateMargins(pTrainingData, pWeakHypothesis);

            //updateFxs( pTrainingData, pWeakHypothesis );
                
            if (_verbose > 1 && (t + 1) % step == 0)
            {
                float progress = static_cast<float>(t) / static_cast<float>(numIters) * 100.0;                             
                cout << "." << setprecision(2) << progress << "%." << flush;
            }

            // If gamma <= theta there is something really wrong.
            /*
              if (gamma <= _theta)
              {
              cerr << "ERROR!" <<  setprecision(4) << endl
              << "At iteration <" << t << ">, edge smaller than the edge offset (theta). Something must be wrong!" << endl
              << "[Edge: " << gamma << " < Offset: " << _theta << "]" << endl
              << "Is the data file the same one used during the original training?" << endl;
              //          exit(1);
              }
            */
        }  // loop on iterations

        if (_verbose > 0)
            cout << "Done!" << endl;

    }

    // -------------------------------------------------------------------------

    void FilterBoostLearner::filter( InputData* pData, int size, bool rejection )
    {
        pData->clearIndexSet();
        const int numExamples = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();          

        set<int> indexSet;
        if (size<numExamples) // not whole dataset will be used
        {
            //random permutation
            vector< pair<int,int> > tmpRandomArr( numExamples );
            for( int i=0; i < numExamples; i++ ) 
            {
                tmpRandomArr[i].first = rand();
                tmpRandomArr[i].second = i;
            }
                        
            sort( tmpRandomArr.begin(), tmpRandomArr.end(), nor_utils::comparePair<1, int, int, less<int> >() );
                        
            vector< int > randPerm( numExamples );
            for( int i=0; i<numExamples; i++ )
            {
                randPerm[i] = tmpRandomArr[i].second;
            }
            //end: random permutation
                        
            int iter = 0;
            int maxIter = 5 * size;
            int wholeIter = 0;
                        
            indexSet.clear();
            while (1)
            {
                if ( size<=indexSet.size() ) break;
                if ( wholeIter > 5 ) rejection = false;
                if ( numExamples <= iter ) {
                    iter = 0;
                    wholeIter++;
                }
                                
                if ( rejection )
                {                               
                    const vector<Label>& labels = pData->getLabels( randPerm[iter] );
                    vector<Label>::const_iterator lIt;
                                        
                    AlphaReal scalar = 0.0;
                    //float scalar = numeric_limits<float>::max();
                    for ( lIt = labels.begin(); lIt != labels.end(); ++lIt ) 
                    {
                        //if ( scalar > _margins[ randPerm[iter] ][lIt->idx] ) scalar = _margins[ randPerm[iter] ][lIt->idx];
                        //if ( _margins[ randPerm[iter] ][lIt->idx] < 0.0 ) scalar += _margins[ randPerm[iter] ][lIt->idx];
                        scalar += (1 / ( 1 + exp(_margins[ randPerm[iter] ][lIt->idx])));
                    }
                                        
                    AlphaReal qValue = scalar / (AlphaReal) numClasses;
                    //AlphaReal qValue = 1 / ( 1 + exp( scalar ) );                                                            
                    AlphaReal randNum = (AlphaReal)rand() / RAND_MAX;
                                        
                                        
                    if ( randNum < qValue ) indexSet.insert( randPerm[iter] );
                }
                else
                {
                    indexSet.insert( randPerm[iter] );
                }
                iter++;
            }
            // normalize the weights of the labels
            set<int>::iterator sIt;
            AlphaReal sum = 0.0;
            // for each example are in use
            for ( sIt = indexSet.begin(); sIt != indexSet.end(); sIt++ )
            {
                vector<Label>& labels = pData->getLabels(*sIt);
                vector<Label>::iterator lIt;
                                
                for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {
                    lIt->weight = 1 /( 1+exp( _margins[ *sIt ][lIt->idx] ) );
                    sum += lIt->weight;
                }
            }
                        
            for ( sIt = indexSet.begin(); sIt != indexSet.end(); sIt++ )
            {
                vector<Label>& labels = pData->getLabels(*sIt);
                vector<Label>::iterator lIt;
                                
                for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {
                    lIt->weight /= sum;
                }
            }
                        
            pData->loadIndexSet( indexSet );
        } else { // the whole dataset is used
            AlphaReal sum = 0.0;
            // for each example are in use
            for ( int i = 0; i < pData->getNumExamples(); ++i )
            {
                vector<Label>& labels = pData->getLabels(i);
                vector<Label>::iterator lIt;
                                
                for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {
                    lIt->weight = 1 /( 1+exp( _margins[ i ][lIt->idx] ) );
                    sum += lIt->weight;
                }
            }
                        
            for ( int i = 0; i < pData->getNumExamples(); ++i )
            {
                vector<Label>& labels = pData->getLabels( i );
                vector<Label>::iterator lIt;
                                
                for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {
                    lIt->weight /= sum;
                }
            }                       
        }

        /*
          sum = 0.0;
          for ( int i=0; i < pData->getNumExamples(); i++ )
          {
          vector<Label>& labels = pData->getLabels(i);
          vector<Label>::iterator lIt;

          for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
          {
          sum += lIt->weight;
          }
          }
          cout << "The size of the dataset: " << pData->getNumExamples() << endl;
          cout << "Sum: " << sum << endl;
        */
    }

    // -------------------------------------------------------------------------
    void FilterBoostLearner::setWeightToMargins( InputData* pData )
    {
        pData->clearIndexSet();         
        const int numExamples = pData->getNumExamples();
                
        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;
                        
            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                lIt->weight = 1 /( 1+exp( _margins[ i ][lIt->idx] ) );
            }                                               
        }
    }
        
        
    // -------------------------------------------------------------------------
    void FilterBoostLearner::updateMargins( InputData* pData, BaseLearner* pWeakHypothesis )
    {
        pData->clearIndexSet();
        const int numExamples = pData->getNumExamples();

        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;

            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                AlphaReal hy =  pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
                    lIt->y; // y

                // compute the margin
                _margins[i][lIt->idx] += pWeakHypothesis->getAlpha() * hy;
            }
        }
    }
        
    // -------------------------------------------------------------------------
    // temporary function
    void FilterBoostLearner::saveMargins() 
    {
        // open the stream
        ofstream outStream("margins.txt");
                
        // for each example
        for (int i = 0; i < _margins.size(); ++i)
        {
            //outStream << _weights[i+1]-_weights[i] << " ";
            for (int j = 0; j < _margins[i].size(); ++j )
            {
                outStream << _margins[i][j] << " ";
            }
            outStream << endl;
        }
                
        outStream.close();
    }
    // -------------------------------------------------------------------------
        
} // end of namespace MultiBoost

