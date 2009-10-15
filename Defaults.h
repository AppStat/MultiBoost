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
* \file Defaults.h
* Defines macros and default values.
* This file just holds some macros and static strings that define
* some default values.
* \date 11/11/2005
*/

#ifndef __DEFAULTS_H
#define __DEFAULTS_H

static const char SHYP_NAME[] = "shyp"; //!< The default strong hypothesis file name
static const char SHYP_EXTENSION[] = "xml"; //!< The default strong hypothesis file name extension

static const char COMMENT[] = "Research code"; //!< Comment to put in the executable 

static const char defaultLearner[] = "SingleStumpLearner"; //!< The default weak learner

static const char CURRENT_VERSION[] = "0.85dev";

/**
* Defines the sorting method.
* Columns needs to be sorted when using the decision stumps algorithm;
* Generally conservativeness is not an issue with this type of algorithm
* but because I want to get \b exactly the same results on win32 an unix, 
* if I set non conservative sort, the outcome is different on the two.
* \remarks The valid values are:
* - 1 = slow but stable
* - 0 = fast but non stable
* \date 16/11/2005
*/
#define STABLE_SORT 0

/**
* Debug level. If there is a problem try activating this.
* \date 16/11/2005
*/ 
#define MB_DEBUG 0

#define S_(X) #X //!< Used with STRINGIZE can stringize a macro name
#define STRINGIZE(X) S_(X) //!< Stringize a macro name! :)

#endif // __DEFAULTS_H
