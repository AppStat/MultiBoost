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


#ifndef __CONSTANT_ALGORITHM_LSHTC_H
#define __CONSTANT_ALGORITHM_LSHTC_H

#include <vector>
#include <cassert>

#include "IO/InputData.h"
#include "Others/Rates.h"

using namespace std;

namespace MultiBoost {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

    class ConstantAlgorithmLSHTC
    {
    public:

        /**
         * Computes the half of the classwise weights and half of the
         * classwise edges.
         * @param pData The pointer to the data.
         * @param pHalfWeightsPerClass The pointer to the half of the classwise weights.
         * @param pHalfEdges The pointer to the half of the classwise edges.
         * @date 30/06/2006
         */
        void findConstantWeightsEdges( InputData* pData, 
                                       vector<AlphaReal>& pHalfWeightsPerClass,
                                       vector<AlphaReal>& pHalfEdges);
        /**
         * Computes the classwise mus and votes (alignment) of the constant classifier
         * @param pData The pointer to the data.
         * @param pMu The The class-wise rates to update.
         * @param pV The alignment vector to update.
         * @return The edge.
         * @date 20/07/2006
         */
        AlphaReal findConstant( InputData* pData, 
                                vector<sRates>* pMu, vector<AlphaReal>* pV);
    };

} // end of namespace MultiBoost

#endif // __CONSTANT_ALGORITHM_H
