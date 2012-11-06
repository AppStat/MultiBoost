#ifndef __BANDIT_PRODUCT_LEARNER_H
#define __BANDIT_PRODUCT_LEARNER_H

#include "BaseLearner.h"
#include "Utils/Args.h"
#include "IO/InputData.h"
#include "BanditsLS/GenericBanditAlgorithmLS.h"
#include "WeakLearners/BanditTreeLearner.h"
#include "WeakLearners/BanditLearner.h"

#include <vector>
#include <fstream>
#include <string>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* A learner that loads a set of base learners, and boosts on the top of them. 
*/
class BanditProductLearner : public BanditLearner
{
public:


   /**
   * The constructor. It initializes _numBaseLearners to -1
   * \date 26/05/2007
   */
   BanditProductLearner() : BanditLearner(), _numBaseLearners(-1) { }

   /**
   * The destructor. Must be declared (virtual) for the proper destruction of 
   * the object.
   */
   virtual ~BanditProductLearner() {
      for( int ib = 0; ib < _numBaseLearners; ++ib )
	 delete _baseLearners[ib];
   }

   /**
   * Creates an InputData object using the base learner's createInputData.
   * \see InputData
   * \date 21/11/2005
   */
   virtual InputData* createInputData() { return _baseLearners[0]->createInputData(); }

   /**
   * Declare weak-learner-specific arguments.
   * adding --baselearnertype
   * \param args The Args class reference which can be used to declare
   * additional arguments.
   * \date 24/04/2007
   */
   virtual void declareArguments(nor_utils::Args& args);

   /**
   * Set the arguments of the algorithm using the standard interface
   * of the arguments. Call this to set the arguments asked by the user.
   * \param args The arguments defined by the user in the command line.
   * \date 24/04/2007
   */
   virtual void initLearningOptions(const nor_utils::Args& args);

   /**
   * Returns itself as object.
   * \remark It uses the trick described in http://www.parashift.com/c++-faq-lite/serialization.html#faq-36.8
   * for the auto-registering classes.
   * \date 14/11/2005
   */
   virtual BaseLearner* subCreate() { 
		BaseLearner* retLearner = new BanditProductLearner();
		static_cast< BanditProductLearner* >(retLearner)->setBanditAlgoObject( static_cast< BanditProductLearner* >(this)->getBanditAlgoObject() );
		return retLearner;  
   }

   /**
   * Run the learner to build the classifier on the given data.
   * \see BaseLearner::run
   * \date 24/04/2007
   */
   virtual float run();

   /**
   * Return the classification using the learned classifier.
   * \param pData The pointer to the data
   * \param idx The index of the example to classify
   * \param classIdx The index of the class
   * \remark Passing the data and the index to the example is not nice at all.
   * This will soon be replace with the passing of the example itself in some
   * form (probably a structure to the example).
   * \return the classification using the learned classifier.
   * \date 24/04/2007
   */
   virtual float classify(InputData* pData, int idx, int classIdx);

   /**
   * Save the current object information needed for classification,
   * that is the single threshold.
   * \param outputStream The stream where the data will be saved
   * \param numTabs The number of tabs before the tag. Useful for indentation
   * \remark To fully save the object it is \b very \b important to call
   * also the super-class method.
   * \see BaseLearner::save()
   * \date 24/04/2007
   */
   virtual void save(ofstream& outputStream, int numTabs = 0);

   /**
   * Load the xml file that contains the serialized information
   * needed for the classification and that belongs to this class.
   * \param st The stream tokenizer that returns tags and values as tokens
   * \see save()
   * \date 24/04/2007
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
   * \see BanditProductLearner::run()
   * \date 25/05/2007
   */
   virtual void subCopyState(BaseLearner *pBaseLearner);


protected:

   vector<BaseLearner*> _baseLearners; //!< the learners of the product
   int _numBaseLearners;

private:
   vector< vector<char> > _savedLabels; //!< original labels saved before run

};

//////////////////////////////////////////////////////////////////////////

} // end of namespace MultiBoost

#endif // __PRODUCT_LEARNER_H
