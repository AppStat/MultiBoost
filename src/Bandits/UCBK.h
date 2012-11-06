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



#ifndef _UCBK_H
#define _UCBK_H

#include <list> 
#include <functional>
#include <math.h> //for log
#include "GenericBanditAlgorithm.h"
#include "Utils/Utils.h"
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace MultiBoost {
/*
class StorageElement{
protected:
	int		_index;
	double	_value;
public:
	int getIndex() { return _index; }
	double getValue() { return _value; }
	void setIndex( int index ) { _index = index; }
	void setValue( double value ) { _value = value; }
};
*/

class UCBK : public GenericBanditAlgorithm
{
protected:
   list< pair<AlphaReal, int >* > _valuesList;
   vector< pair<AlphaReal, int >* > _valueRecord;
public:
	UCBK(void);
	virtual ~UCBK(void) 
	{
		list< pair< AlphaReal, int >* >::iterator it;
		
		for( it = _valuesList.begin(); it != _valuesList.end(); it ++ )
		{
			delete *it;
		}

		_valuesList.clear();
		_valueRecord.clear();
	}

	virtual void initialize( vector< AlphaReal >& vals );

	virtual void getKBestAction( const int k, vector<int>& bestArms );
	virtual int getNextAction();

	virtual void initLearningOptions(const nor_utils::Args& args) {}
protected:
	virtual void updateithValue( int i );	
};

} // end of namespace MultiBoost

#endif

