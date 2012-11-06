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
 * \file SingleSparseStump.h A single threshold decision stump learner. 
 */

#ifndef __SPARSE_STUMP_LEARNER_H
#define __SPARSE_STUMP_LEARNER_H

#include "WeakLearners/FeaturewiseLearner.h"
#include "WeakLearners/ScalarLearner.h"

#include "Utils/Args.h"
#include "IO/InputData.h"
#include "IO/SortedData.h"

#include <vector>
#include <fstream>
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
	
	/**
	 * A \b double thresholds decision stump learner. 
	 * There are TWO thresholds, between the thresholds it returns with 0.
	 */
	class SingleSparseStump : virtual public FeaturewiseLearner, virtual public ScalarLearner
	{
	public:
		
		/**
		 * The destructor. Must be declared (virtual) for the proper destruction of 
		 * the object.
		 */
		virtual ~SingleSparseStump() {}
		
		/**
		 * Returns itself as object.
		 * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
		 * for the auto-registering classes.
		 * \date 14/11/2005
		 */
		virtual BaseLearner* subCreate() { return new SingleSparseStump(); }
		
		/**
		 * Creates an InputData object that it is good for the
		 * weak learner. Overridden to return SortedData.
		 * \see InputData
		 * \see BaseLearner::createInputData()
		 * \see SortedData
		 * \warning The object \b must be destroyed by the caller.
		 * \date 21/11/2005
		 */
		virtual InputData* createInputData() { return new SortedData(); }
		
		/**
		 * Run the learner to build the classifier on the given data.
		 * \param pData The pointer to the data.
		 * \see BaseLearner::run
		 * \date 11/11/2005
		 */
		virtual AlphaReal run();
		
		/**
		 * Run the learner to build the classifier using only one single feature from the given data.		 
		 * \see FeaturewiseLearner::run
		 * \date 21/05/2009
		 */		
		virtual AlphaReal run( int colIdx ) { return 0.0; }
				
		/**
		 * Save the current object information needed for classification,
		 * that is the single threshold.
		 * \param outputStream The stream where the data will be saved
		 * \param numTabs The number of tabs before the tag. Useful for indentation
		 * \remark To fully save the object it is \b very \b important to call
		 * also the super-class method.
		 * \see StumpLearner::save()
		 * \date 13/11/2005
		 */
		virtual void save(ofstream& outputStream, int numTabs = 0);
		
		/**
		 * Load the xml file that contains the serialized information
		 * needed for the classification and that belongs to this class.
		 * \param st The stream tokenizer that returns tags and values as tokens
		 * \see save()
		 * \date 13/11/2005
		 */
		virtual void load(nor_utils::StreamTokenizer& st);
		
		/**
		 * Copy all the info we need in classify().
		 * pBaseLearner was created by subCreate so it has the correct (sub) type.
		 * Usually one must copy the same fields that are loaded and saved. Don't 
		 * forget to call the parent's subCopyState().
		 * \param pBaseLearner The sub type pointer into which we copy.
		 * \see save
		 * \see load
		 * \see classify
		 * \see ProductLearner::run()
		 * \date 25/05/2007
		 */
		virtual void subCopyState(BaseLearner *pBaseLearner);
		
		/**
		 * Returns a vector of float holding any data that the specific weak learner can generate
		 * using the given input dataset. Right now just a single case is contemplated, therefore
		 * "reason" is not used. The returned vector of data correspond to:
		 * \f[
		 * \left\{ v^{(t)}_1, v^{(t)}_2, \cdots, v^{(t)}_K, \phi^{(t)}({\bf x}_1), 
		 * \phi^{(t)}({\bf x}_2), \cdots, \phi^{(t)}({\bf x}_n) \right\}
		 * \f]
		 * where \f$v\f$ is the alignment vector, \f$\phi\f$ is the discriminative function and
		 * \f$t\f$ is the current iteration (to which this instantiation of weak learner belongs to).
		 * \param data The vector with the returned data.
		 * \param pData The pointer to the data points (the examples) previously loaded.
		 * \remark This particular method has been created for analysis purposes (research). If you
		 * need to get similar analytical information from this weak learner, add it to the function
		 * and uncomment the parameter "reason".
		 * \see BaseLearner::getStateData
		 * \see Classifier::saveSingleStumpFeatureData
		 * \date 10/2/2006
		 * \remark TEMPORARLY OFF!!
		 */
		//virtual void getStateData( vector<float>& data, const string& /*reason = ""*/, InputData* pData = 0 );
		
		/**
		 * The same discriminative function as below, but called with a data point. 
		 * Called only from HierarchicalStumpLearner::phi
		 * \param pData The input data.
		 * \param pointIdx The index of the data point.
		 * \return +1 if \a pData[pointIdx][_selectedColumn] is on one side of the 
		 * border for \a classIdx and -1 otherwise.
		 * \date 17/02/2006
		 */
		virtual AlphaReal phi(InputData* pData, int pointIdx) const;
		
	protected:
		
		/**
		 * A discriminative function. 
		 * \remarks Positive or negative do NOT refer to positive or negative classification.
		 * This function is equivalent to the phi function in my thesis.
		 * \param val The value to discriminate
		 * \param classIdx The class index, used by MultiStumpLearner when phi depends on the class
		 * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
		 * \date 11/11/2005
		 * \see classify
		 */
		virtual AlphaReal phi( FeatureReal val ) const;
		
		/**
		 * A discriminative function. 
		 * \remarks Positive or negative do NOT refer to positive or negative classification.
		 * This function is equivalent to the phi function in my thesis.
		 * \param val The value to discriminate
		 * \param classIdx The class index, used by MultiStumpLearner when phi depends on the class
		 * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
		 * \date 11/11/2005
		 * \see classify
		 */		
		virtual AlphaReal phi(FeatureReal val, int /*classIdx*/) const { return phi(val); }
		
		/**
		 * A simple cut on a feature. This simple function is needed for TreeLearner.
		 * \param pData The input data.
		 * \param idx The index of the data point.
		 * \return +1 if \a val is on one side of the border for \a classIdx and -1 otherwise
		 * \date 11/11/2005
		 */		
		virtual AlphaReal cut( InputData* pData, int idx ) const
		{
			return phi( pData->getValue( idx, _selectedColumn) );
		}
		
		
		vector<FeatureReal> _threshold; //!< for both thresholds of the decision stump, agnostic case
	};
	
	//////////////////////////////////////////////////////////////////////////
	
} // end of namespace MultiBoost

#endif // __SINGLE_STUMP_LEARNER_H
