/*
$Author: nova77 $
$Date: 2007/02/12 16:13:17 $
$Revision: 1.7 $
*/

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

/**
* \file GenericStrongLearner.h An abstract class for a generic meta-learner.
*/

#ifndef __GENERIC_STRONG_LEARNER_H
#define __GENERIC_STRONG_LEARNER_H

#include "Utils/Args.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {

/**
* An abstract class for a generic meta-learner.
* \see AdaBoostMHLearner
* \date 20/3/2006
*/
class GenericStrongLearner
{
public:

   /**
   * Performs the learning process.
   * \param args The arguments provided by the command line.
   */
   virtual void run(const nor_utils::Args& args) = 0;

   /**
   * Performs the classification.
   * \param args The arguments provided by the command line.
   */
   virtual void classify(const nor_utils::Args& args) = 0;

   /**
   * Print to stdout (or to file) a confusion matrix.
   * \param args The arguments provided by the command line.
   * \date 20/3/2006
   */
   virtual void doConfusionMatrix(const nor_utils::Args& args) = 0;

   /**
   * Print to stdout (or to file) the posteriors of each class.
   * \param args The arguments provided by the command line.
   */
   virtual void doPosteriors(const nor_utils::Args& args) = 0;

   /**
   * Using the Platt calibration this function calibrate the posterior
   * probalities using a dataset which is used at the calibration.
   */

   virtual void doCalibratedPosteriors(const nor_utils::Args& args) = 0;


   virtual void doLikelihoods(const nor_utils::Args& args) = 0;

   virtual void doROC(const nor_utils::Args& args) = 0;

   virtual ~GenericStrongLearner(){}
};

} // end of namespace MultiBoost

#endif // __GENERIC_MODEL_H
