/*
* This file is part of MultiBoost, a multi-class 
* AdaBoost learner/classifier
*
* Copyright (C) 2005-2006 Norman Casagrande
* For informations write to nova77@gmail.com
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*
*/

#include "EnumLearnerKNN.h"
#include <limits>

#include "IO/Serialization.h"

namespace MultiBoost {

	//REGISTER_LEARNER_NAME(SingleStump, EnumLearnerKNN)
	REGISTER_LEARNER(EnumLearnerKNN)

	KNNGraph EnumLearnerKNN::_kNN;
		// -----------------------------------------------------------------------

		void EnumLearnerKNN::declareArguments(nor_utils::Args& args)
	{
		BaseLearner::declareArguments(args);

		args.declareArgument("uoffset", 
			"The offset of u\n",
			1, "<offset>");

		args.declareArgument("knn", 
			"k-nn graph (k)\n",
			1, "<k>");

		args.declareArgument("knnfile", 
			"k-nn graph filename\n",
			1, "<knngraphfile>");

	}

	// ------------------------------------------------------------------------------

	void EnumLearnerKNN::initLearningOptions(const nor_utils::Args& args)
	{
		BaseLearner::initLearningOptions(args);

		if ( args.hasArgument( "uoffset" ) )  
			args.getValue("uoffset", 0, _uOffset);   
		
		
		if ( args.hasArgument( "knn" ) ) { 
			int knn;
			args.getValue("knn", 0, knn);
			_kNN.setK( knn );
		}

		if ( args.hasArgument( "knnfile" ) ) { 
			string fn;
			args.getValue("knnfile", 0, fn);
			_kNN.setName( fn );
		}

	}


		// ------------------------------------------------------------------------------

		float EnumLearnerKNN::run()
	{
		if ( ! _kNN.isReady() ) { 
			_kNN.setTrainingData( _pTrainingData );
			_kNN.calculagteKNNGraph();
		}

		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		const int numExamples = _pTrainingData->getNumExamples();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> vMu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions
		vector<float> previousTmpV(numClasses); // The class-wise votes/abstentions
		vector< int > neighbors;

		float tmpAlpha,previousTmpAlpha, previousEnergy;
		float bestEnergy = numeric_limits<float>::max();

		int numOfDimensions = _maxNumOfDimensions;
		for (int j = 0; j < numColumns; ++j)
		{
			// Tricky way to select numOfDimensions columns randomly out of numColumns
			int rest = numColumns - j;
			float r = rand()/static_cast<float>(RAND_MAX);

			if ( static_cast<float>(numOfDimensions) / rest > r ) 
			{
				--numOfDimensions;

				if (_verbose > 2)
					cout << "    --> trying attribute = "
					<<_pTrainingData->getAttributeNameMap().getNameFromIdx(j)
					<< endl << flush;

				const int numIdxs = _pTrainingData->getEnumMap(j).getNumNames();

				// Create and initialize the numIdxs x numClasses gamma matrix
				vector<vector<float> > tmpGammasPls(numIdxs);
				vector<vector<float> > tmpGammasMin(numIdxs);
				for (int io = 0; io < numIdxs; ++io) {
					vector<float> tmpGammaPls(numClasses);
					vector<float> tmpGammaMin(numClasses);
					fill(tmpGammaPls.begin(), tmpGammaPls.end(), 0.0);
					fill(tmpGammaMin.begin(), tmpGammaMin.end(), 0.0);
					tmpGammasPls[io] = tmpGammaPls;
					tmpGammasMin[io] = tmpGammaMin;
				}

				// Compute the elements of the gamma plus and minus matrices
				float entry;
				for (int i = 0; i < numExamples; ++i) {
					const vector<Label>& labels = _pTrainingData->getLabels(i);
					int io = static_cast<int>(_pTrainingData->getValue(i,j));	    
					for (int l = 0; l < numClasses; ++l) {
						entry = labels[l].weight * labels[l].y;
						if (entry > 0)
							tmpGammasPls[io][l] += entry;
						else if (entry < 0)
							tmpGammasMin[io][l] += -entry;
					}
				}

				// Initialize the u vector to random +-1
				vector<sRates> uMu(numIdxs); // The idx-wise rates
				vector<float> tmpU(numIdxs);// The idx-wise votes/abstentions
				vector<float> previousTmpU(numIdxs);// The idx-wise votes/abstentions
				for (int io = 0; io < numIdxs; ++io) {
					uMu[io].classIdx = io;	    
					if ( rand()/static_cast<float>(RAND_MAX) > 0.5 )
						tmpU[io] = +1;
					else
						tmpU[io] = -1;
				}

				vector<sRates> vMu(numClasses); // The label-wise rates
				for (int l = 0; l < numClasses; ++l)
					vMu[l].classIdx = l;
				vector<float> tmpV(numClasses); // The label-wise votes/abstentions

				float tmpEnergy = numeric_limits<float>::max();
				float tmpVal;
				tmpAlpha = 0.0;

				while (1) {
					previousEnergy = tmpEnergy;
					previousTmpV = tmpV;
					previousTmpAlpha = tmpAlpha;

					//filling out tmpV and vMu
					for (int l = 0; l < numClasses; ++l) {
						vMu[l].rPls = vMu[l].rMin = vMu[l].rZero = 0; 
						for (int io = 0; io < numIdxs; ++io) {
							if (tmpU[io] > 0) {
								vMu[l].rPls += tmpGammasPls[io][l];
								vMu[l].rMin += tmpGammasMin[io][l];
							}
							else if (tmpU[io] < 0) {
								vMu[l].rPls += tmpGammasMin[io][l];
								vMu[l].rMin += tmpGammasPls[io][l];
							}
						}
						if (vMu[l].rPls >= vMu[l].rMin) {
							tmpV[l] = +1;
						}
						else {
							tmpV[l] = -1;
							tmpVal = vMu[l].rPls;
							vMu[l].rPls = vMu[l].rMin;
							vMu[l].rMin = tmpVal;
						}
					}

					tmpEnergy = AbstainableLearner::getEnergy(vMu, tmpAlpha, tmpV);

					if (_verbose > 2)
						cout << "        --> energy V = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

					if (tmpEnergy >= previousEnergy) {
						tmpV = previousTmpV;
						break;
					}

					previousEnergy = tmpEnergy;
					previousTmpU = tmpU;
					previousTmpAlpha = tmpAlpha;

					//filling out tmpU and uMu
					for (int io = 0; io < numIdxs; ++io) {
						uMu[io].rPls = uMu[io].rMin = uMu[io].rZero = 0; 
						for (int l = 0; l < numClasses; ++l) {
							if (tmpV[l] > 0) {
								uMu[io].rPls += tmpGammasPls[io][l];
								uMu[io].rMin += tmpGammasMin[io][l];
							}
							else if (tmpV[l] < 0) {
								uMu[io].rPls += tmpGammasMin[io][l];
								uMu[io].rMin += tmpGammasPls[io][l];
							}
						}
					}

					for (int io = 0; io < numIdxs; ++io) {
						// sparse U 
						EnumLearnerKNN::_kNN.getjthFeatureithExampleNeighborhood( j, io, neighbors );
						//if ( (uMu[io].rPls >= uMu[io].rMin ) && ( rand()/static_cast<float>(RAND_MAX) > _uOffset ) ) {
						//if ( uMu[io].rPls >= (uMu[io].rMin + _uOffset) ) {
						if ( uMu[io].rPls >= uMu[io].rMin ) {
							int numberOfPositiveNeighbors = 0;
							for( int i3 = 0; i3 < neighbors.size(); i3++ ) {
								if ( uMu[ neighbors[ i3 ] ].rPls >= uMu[ neighbors[ i3 ] ].rMin ) {
									numberOfPositiveNeighbors++;
								}
							}
							if ( numberOfPositiveNeighbors >= ( neighbors.size() / 2 ) ) {
								tmpU[io] = +1;
							} else {
								tmpU[io] = -1;
							}
						}
						else {
							tmpU[io] = -1;
							tmpVal = uMu[io].rPls;
							uMu[io].rPls = uMu[io].rMin;
							uMu[io].rMin = tmpVal;
						}
					}

					tmpEnergy = AbstainableLearner::getEnergy(uMu, tmpAlpha, tmpU);

					if (_verbose > 2)
						cout << "        --> energy U = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

					if (tmpEnergy >= previousEnergy) {
						tmpU = previousTmpU;
						break;
					}
				}

				if ( previousEnergy < bestEnergy && previousTmpAlpha > 0 ) {
					_alpha = previousTmpAlpha;
					_v = tmpV;
					_u = tmpU;
					_selectedColumn = j;
					bestEnergy = previousEnergy;
				}
			}
		}


		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn);
		return bestEnergy;

	}

		// ------------------------------------------------------------------------------

		float EnumLearnerKNN::run( int colIdx )
	{
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		const int numExamples = _pTrainingData->getNumExamples();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> vMu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions
		vector<float> previousTmpV(numClasses); // The class-wise votes/abstentions

		float tmpAlpha,previousTmpAlpha, previousEnergy;
		float bestEnergy = numeric_limits<float>::max();

		int numOfDimensions = _maxNumOfDimensions;

		// Tricky way to select numOfDimensions columns randomly out of numColumns
		int j = colIdx;


		if (_verbose > 2)
			cout << "    --> trying attribute = "
			<<_pTrainingData->getAttributeNameMap().getNameFromIdx(j)
			<< endl << flush;

		const int numIdxs = _pTrainingData->getEnumMap(j).getNumNames();

		// Create and initialize the numIdxs x numClasses gamma matrix
		vector<vector<float> > tmpGammasPls(numIdxs);
		vector<vector<float> > tmpGammasMin(numIdxs);
		for (int io = 0; io < numIdxs; ++io) {
			vector<float> tmpGammaPls(numClasses);
			vector<float> tmpGammaMin(numClasses);
			fill(tmpGammaPls.begin(), tmpGammaPls.end(), 0.0);
			fill(tmpGammaMin.begin(), tmpGammaMin.end(), 0.0);
			tmpGammasPls[io] = tmpGammaPls;
			tmpGammasMin[io] = tmpGammaMin;
		}

		// Compute the elements of the gamma plus and minus matrices
		float entry;
		for (int i = 0; i < numExamples; ++i) {
			const vector<Label>& labels = _pTrainingData->getLabels(i);
			int io = static_cast<int>(_pTrainingData->getValue(i,j));	    
			for (int l = 0; l < numClasses; ++l) {
				entry = labels[l].weight * labels[l].y;
				if (entry > 0)
					tmpGammasPls[io][l] += entry;
				else if (entry < 0)
					tmpGammasMin[io][l] += -entry;
			}
		}

		// Initialize the u vector to random +-1
		vector<sRates> uMu(numIdxs); // The idx-wise rates
		vector<float> tmpU(numIdxs);// The idx-wise votes/abstentions
		vector<float> previousTmpU(numIdxs);// The idx-wise votes/abstentions
		for (int io = 0; io < numIdxs; ++io) {
			uMu[io].classIdx = io;	    
			if ( rand()/static_cast<float>(RAND_MAX) > 0.5 )
				tmpU[io] = +1;
			else
				tmpU[io] = -1;
		}

		//vector<sRates> vMu(numClasses); // The label-wise rates
		for (int l = 0; l < numClasses; ++l)
			vMu[l].classIdx = l;
		//vector<float> tmpV(numClasses); // The label-wise votes/abstentions

		float tmpEnergy = numeric_limits<float>::max();
		float tmpVal;
		tmpAlpha = 0.0;

		while (1) {
			previousEnergy = tmpEnergy;
			previousTmpV = tmpV;
			previousTmpAlpha = tmpAlpha;

			//filling out tmpV and vMu
			for (int l = 0; l < numClasses; ++l) {
				vMu[l].rPls = vMu[l].rMin = vMu[l].rZero = 0; 
				for (int io = 0; io < numIdxs; ++io) {
					if (tmpU[io] > 0) {
						vMu[l].rPls += tmpGammasPls[io][l];
						vMu[l].rMin += tmpGammasMin[io][l];
					}
					else if (tmpU[io] < 0) {
						vMu[l].rPls += tmpGammasMin[io][l];
						vMu[l].rMin += tmpGammasPls[io][l];
					}
				}
				if (vMu[l].rPls >= vMu[l].rMin) {
					tmpV[l] = +1;
				}
				else {
					tmpV[l] = -1;
					tmpVal = vMu[l].rPls;
					vMu[l].rPls = vMu[l].rMin;
					vMu[l].rMin = tmpVal;
				}
			}

			tmpEnergy = AbstainableLearner::getEnergy(vMu, tmpAlpha, tmpV);

			if (_verbose > 2)
				cout << "        --> energy V = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

			if (tmpEnergy >= previousEnergy) {
				tmpV = previousTmpV;
				break;
			}

			previousEnergy = tmpEnergy;
			previousTmpU = tmpU;
			previousTmpAlpha = tmpAlpha;

			//filling out tmpU and uMu
			for (int io = 0; io < numIdxs; ++io) {
				uMu[io].rPls = uMu[io].rMin = uMu[io].rZero = 0; 
				for (int l = 0; l < numClasses; ++l) {
					if (tmpV[l] > 0) {
						uMu[io].rPls += tmpGammasPls[io][l];
						uMu[io].rMin += tmpGammasMin[io][l];
					}
					else if (tmpV[l] < 0) {
						uMu[io].rPls += tmpGammasMin[io][l];
						uMu[io].rMin += tmpGammasPls[io][l];
					}
				}
				if (uMu[io].rPls >= uMu[io].rMin) {
					tmpU[io] = +1;
				}
				else {
					tmpU[io] = -1;
					tmpVal = uMu[io].rPls;
					uMu[io].rPls = uMu[io].rMin;
					uMu[io].rMin = tmpVal;
				}
			}

			tmpEnergy = AbstainableLearner::getEnergy(uMu, tmpAlpha, tmpU);

			if (_verbose > 2)
				cout << "        --> energy U = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

			if (tmpEnergy >= previousEnergy) {
				tmpU = previousTmpU;
				break;
			}

			if ( previousEnergy < bestEnergy && previousTmpAlpha > 0 ) {
				_alpha = previousTmpAlpha;
				_v = tmpV;
				_u = tmpU;
				_selectedColumn = j;
				bestEnergy = previousEnergy;
			}
		}
		

		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn);
		return bestEnergy;

	}


	// ------------------------------------------------------------------------------

	float EnumLearnerKNN::phi(float val, int /*classIdx*/) const
	{
		return _u[static_cast<int>(val)];
	}

	// -----------------------------------------------------------------------

	void EnumLearnerKNN::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		FeaturewiseLearner::save(outputStream, numTabs);

		// save the _u vector
		outputStream << Serialization::vectorTag("uArray", _u, 
			_pTrainingData->getEnumMap(_selectedColumn), 
			"idx", (float) 0.0, numTabs) << endl;
	}

	// -----------------------------------------------------------------------

	void EnumLearnerKNN::load(nor_utils::StreamTokenizer& st)
	{
		// Calling the super-class method
		FeaturewiseLearner::load(st);

		// load phiArray data
		UnSerialization::seekAndParseVectorTag(st, "uArray", _pTrainingData->getEnumMap(_selectedColumn), 
			"idx", _u);
	}

	// -----------------------------------------------------------------------

	void EnumLearnerKNN::subCopyState(BaseLearner *pBaseLearner)
	{
		FeaturewiseLearner::subCopyState(pBaseLearner);

		EnumLearnerKNN* pEnumLearnerKNN =
			dynamic_cast<EnumLearnerKNN*>(pBaseLearner);

		pEnumLearnerKNN->_u = _u;
		pEnumLearnerKNN->_uOffset = _uOffset;
	}

} // end of namespace MultiBoost
