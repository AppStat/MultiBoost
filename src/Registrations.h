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

#pragma once

#include "WeakLearners/AdaLineLearner.h"
#include "WeakLearners/BanditSingleSparseStump.h"
#include "WeakLearners/BanditSingleStumpLearner.h"
#include "WeakLearners/ConstantLearner.h"
#include "WeakLearners/HaarMultiStumpLearner.h"
#include "WeakLearners/HaarSingleStumpLearner.h"
#include "WeakLearners/IndicatorLearner.h"
#include "WeakLearners/MultiStumpLearner.h"
#include "WeakLearners/MultiThresholdStumpLearner.h"
#include "WeakLearners/OneClassStumpLearner.h"
#include "WeakLearners/ParasiteLearner.h"
#include "WeakLearners/ProductLearner.h"
#include "WeakLearners/ProductLearnerUCT.h"
#include "WeakLearners/SelectorLearner.h"
#include "WeakLearners/SigmoidSingleStumpLearner.h"
#include "WeakLearners/SingleRegressionStumpLearner.h"
#include "WeakLearners/SingleSparseStump.h"
#include "WeakLearners/SingleSparseStumpLearner.h"
#include "WeakLearners/SingleStumpLearner.h"
#include "WeakLearners/TreeLearner.h"
#include "WeakLearners/TreeLearnerUCT.h"
#include "WeakLearners/UCBVHaarSingleStumpLearner.h"

#include "WeakLearners/Haar/HaarFeatures.h"

namespace MultiBoost {
    REGISTER_LEARNER(AdaLineLearner)                
        REGISTER_LEARNER(BanditSingleSparseStump)
        REGISTER_LEARNER(BanditSingleStumpLearner)
        REGISTER_LEARNER(ConstantLearner)
        REGISTER_LEARNER_NAME(HaarMultiStump, HaarMultiStumpLearner)
        REGISTER_LEARNER(HaarSingleStumpLearner)
        REGISTER_LEARNER(IndicatorLearner)
        REGISTER_LEARNER(MultiStumpLearner)
        REGISTER_LEARNER(MultiThresholdStumpLearner)
        REGISTER_LEARNER(OneClassStumpLearner)
        REGISTER_LEARNER(ParasiteLearner)
        REGISTER_LEARNER(ProductLearner)
        REGISTER_LEARNER(ProductLearnerUCT)
        REGISTER_LEARNER(SelectorLearner)
        REGISTER_LEARNER(SigmoidSingleStumpLearner)
        REGISTER_LEARNER(SingleRegressionStumpLearner)
        REGISTER_LEARNER(SingleSparseStump)
        REGISTER_LEARNER(SingleSparseStumpLearner)
        REGISTER_LEARNER(SingleStumpLearner)
        REGISTER_LEARNER(TreeLearner)
        REGISTER_LEARNER(TreeLearnerUCT)
        REGISTER_LEARNER_NAME(UCBVHaarSingleStump, UCBVHaarSingleStumpLearner)

        // Register the haar features
        REGISTER_HAAR_FEATURE(2h, HaarFeature_2H);
    REGISTER_HAAR_FEATURE(2v, HaarFeature_2V);
    REGISTER_HAAR_FEATURE(3h, HaarFeature_3H);
    REGISTER_HAAR_FEATURE(3v, HaarFeature_3V);
    REGISTER_HAAR_FEATURE(4q, HaarFeature_4SQ);

}
