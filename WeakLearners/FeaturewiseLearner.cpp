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

#include <cassert>
#include <limits> // for numeric_limits<>
#include <cmath>

#include "StrongLearners/AdaBoostMHLearner.h"
#include "WeakLearners/FeaturewiseLearner.h"

#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/SortedData.h"

namespace MultiBoost {

// ------------------------------------------------------------------------------

void FeaturewiseLearner::declareArguments(nor_utils::Args& args)
{
   AbstainableLearner::declareArguments(args);

   args.declareArgument("rsample",
                        "Instead of searching for a featurewise in all the possible dimensions (features), select a set of "
                        " size <num> of random dimensions. "
                        "Example: -rsample 50 -> Search over only 50 dimensions"
                        "(Turned off for Haar: use -csample instead)",
                        1, "<num>");

}

// ------------------------------------------------------------------------------

void FeaturewiseLearner::initLearningOptions(const nor_utils::Args& args)
{
   AbstainableLearner::initLearningOptions(args);
   _maxNumOfDimensions = numeric_limits<int>::max();

   // If the sampling is required
   if ( args.hasArgument("rsample") )
      _maxNumOfDimensions = args.getValue<int>("rsample", 0);
}

// -----------------------------------------------------------------------

void FeaturewiseLearner::save(ofstream& outputStream, int numTabs)
{
   // Calling the super-class method
   AbstainableLearner::save(outputStream, numTabs);

   // save selectedColumn
	string selectedColumnName;
   if (_selectedColumn > -1)
      selectedColumnName = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn);
   else
      // This is because of Haar: it calls this save() but doesn't use _selectedColumn
      selectedColumnName = "unused";
   outputStream << Serialization::standardTag("column", selectedColumnName, numTabs) << endl;
}

// -----------------------------------------------------------------------

void FeaturewiseLearner::load(nor_utils::StreamTokenizer& st)
{
   // Calling the super-class method
   AbstainableLearner::load(st);

   // load selectedColumn
   string selectedColumnName = UnSerialization::seekAndParseEnclosedValue<string>(st, "column");
   if (selectedColumnName == "unused")
      // This is because of Haar: it calls this load() but doesn't use _selectedColumn
      _selectedColumn = -1;
   else 
      _selectedColumn = _pTrainingData->getAttributeNameMap().getIdxFromName(selectedColumnName);
   _id = selectedColumnName;
}

// -----------------------------------------------------------------------

void FeaturewiseLearner::subCopyState(BaseLearner *pBaseLearner)
{
   AbstainableLearner::subCopyState(pBaseLearner);

   FeaturewiseLearner* pFeaturewiseLearner =
      dynamic_cast<FeaturewiseLearner*>(pBaseLearner);

   pFeaturewiseLearner->_selectedColumn = _selectedColumn;
   pFeaturewiseLearner->_maxNumOfDimensions = _maxNumOfDimensions;
}

// -----------------------------------------------------------------------

float FeaturewiseLearner::phi(InputData* pData, int idx, int classIdx) const
{
   return phi( pData->getValue(idx, _selectedColumn), classIdx );
}

// -----------------------------------------------------------------------

} // end of namespace MultiBoost
