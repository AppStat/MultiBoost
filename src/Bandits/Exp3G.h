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



#ifndef _Exp3G_H
#define _Exp3G_H

#include <list> 
#include <functional>
#include <math.h> //for pow
#include "GenericBanditAlgorithm.h"
#include "Utils/Utils.h"

/*
  The Exp3G Algorithm was published in: 
  Kocsis and Szepesvari:
  Reduced-Variance Payoff Estimation in Adversarial Bandit Problems
  (it isn't available the journal where it was appeared, but it was written that it is published in the proceedings of ECML05)
*/


using namespace std;
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {


    class Exp3G : public GenericBanditAlgorithm
    {
    protected:
        AlphaReal _eta;
        AlphaReal _gamma;
        vector< AlphaReal > _p;
        vector< AlphaReal > _w;
        vector< AlphaReal > _tmpW;
        vector< vector< int > > _sideInformation;
        vector< int > _actions;
        //Hedge _hedge;
    public:
        Exp3G(void);
        virtual ~Exp3G(void) 
        {
        }

        //----------------------------------------------------------------
        //----------------------------------------------------------------
        // getters and setters 
        //----------------------------------------------------------------
        //----------------------------------------------------------------
        double getEta() { return _eta; }
        void setEta( AlphaReal eta ) { _eta = eta; }
        double getGamma() { return _gamma; }
        void setGamma( double gamma ) { _gamma = gamma; }

        virtual void receiveReward( int armNum, AlphaReal reward );

        virtual void initialize( vector< AlphaReal >& vals );
        virtual int getNextAction();

        virtual void initLearningOptions(const nor_utils::Args& args );
    protected:
        virtual void updateithValue( int arm ); 
    };


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


} // end of namespace MultiBoost

#endif

