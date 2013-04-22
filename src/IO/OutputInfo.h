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
 * \file OutputInfo.h Outputs the step-by-step information.
 */

#ifndef __OUTPUT_INFO_H
#define __OUTPUT_INFO_H

#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "IO/InputData.h"
#include "Defaults.h" //for the default output

using namespace std;

namespace MultiBoost {
        
    // forward declaration to avoid an include
    class BaseLearner;
    class BaseOutputInfoType;
    
    /**
     * A table representing the votes for each example.
     * Example:
     * \verbatim
     Ex_1:  Class 0, Class 1, Class 2, .. , Class k
     Ex_2:  Class 0, Class 1, Class 2, .. , Class k
     ..
     Ex_n:  Class 0, Class 1, Class 2, .. , Class k \endverbatim
     * \date 16/11/2005
     */
    typedef vector< vector<AlphaReal> > table;
    typedef map<string, BaseOutputInfoType*>::iterator OutInfIt;
    
//    template<typename T>
//        struct SpecificInfo
//        {
//            typedef map<string, vector<T> > Type;
//        };
    
        
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
        
    /**
     * Format and output step-by-step information.
     * With this class it is possible to output and update
     * the error rates, margins and the edge.
     * These function must be called at each iteration with the
     * newly found weak hypothesis, but \b before the update of the
     * weights.
     * \warning Don't forget to begin the list of information
     * printed with outputIteration(), and close it with
     * a call to endLine()!
     * \date 16/11/2005
     */
    class OutputInfo
    {
    public:
                        
        /**
         * The constructor. Create the object and open the output file.
         * \param args The arguments passed through command line
         * \param clArg The command line argument that gives the output file name
         * \date 17/06/2011
         */
        explicit OutputInfo(const nor_utils::Args& args, bool customUpdate = false, const string & clArg = "outputinfo");
        
        
        /**
         * Forces the choice of the output information 
         * \param list The list of the output names (eg. err, auc etc.)
         * \param append If true, "list" is added to the other outputs, otherwise, it 
         * replace them.
         * \date 04/07/2011
         */
        void setOutputList(const string& list, const nor_utils::Args* args = NULL);
        
        /**
         * Just output the iteration number.
         * \param t The iteration number.
         * \date 14/11/2005
         */
        void outputIteration(int t);
        void initialize(InputData* pData);
                
        /**
         * Just output the current time.
         * \param 
         * \date 14/11/2005
         */
        void outputCurrentTime( void );
                
        /**
         * Output the column names
         * Note that this method must be called after the 
         * initialization of all the datasets
         * \param namemap The structure that holds the class information
         * \date 04/08/2006
         */
        void outputHeader(const NameMap& namemap, bool outputIterations = true, bool outputTime = true, bool endline = true);
        /**
         * Just return the sum of alphas
         * \param pData pointer of the input data
         * \date 04/04/2012
         */             
        AlphaReal getSumOfAlphas( InputData* pData )
        {
            return _alphaSums[pData];
        }
                

        /**
         * Output the information the user wants
         * This "wish-list" is specified either through 
         * the command line or directly through the constructor
         * \param pData The input data.
         * \param pWeakHypothesis The current weak hypothesis.
         * \date 17/06/2011
         */
        void outputCustom(InputData* pData, BaseLearner* pWeakHypothesis = 0);
                
        /**
         * End of line in the file stream.
         * Call it when all the needed information has been outputted.
         * \date 16/11/2005
         */
        void endLine() { _outStream << endl; }
        void headerEndLine() { _headerOutStream << endl; }

        
        /**
         * Separator in the file stream.
         * \date 20/06/2011
         */
        void separator() { _outStream << OUTPUT_SEPARATOR << OUTPUT_SEPARATOR; }

        void outputUserData( float data )
        {
            _outStream << data;
        }
        
        void outputUserHeader( const string& h )
        {
            _headerOutStream << h << OUTPUT_SEPARATOR;
        }
                
        table& getTable( InputData* pData )
        {
            table& g = _gTableMap[pData];
            return g;
        }
                
        void setTable( InputData* pData, table& tmpTable )
        {
            table& g = _gTableMap[pData];
            
            // in case the dimensions are different
            const int newDimension = (int)tmpTable.size();
            const int numClasses = pData->getNumClasses();
            const int oldDimension = (int)g.size();
            
            g.resize(newDimension);
            for (int i = oldDimension; i < newDimension; ++i) {
                g[i].resize(numClasses, 0.);
            }

            for( int i=0; i<g.size(); i++ )
                copy( tmpTable[i].begin(), tmpTable[i].end(), g[i].begin() ); 
        }
                
        table& getMargins( InputData* pData )
        {
            table& g = _margins[pData];
            return g;
        }
        
        /*
         * Updates the G and Margin tables and alphaSums vector
         * \date 17/06/2011
         */
        void updateTables(InputData* pData, BaseLearner* pWeakHypothesis);
        
        /**
         * Calls the updateSpecificInfo method for each OutputInfoType subclass.
         * The subclasses that implement this method must check whether the update is targetted to them or not 
         * through the prefix of the param type.
         * \param type The name of the info to be updated, it is highly recommanded to prefix this name by an abreviation of the class name.
         * \param value The value to be added/updated
         * \date 05/07/2011
         */
//        void updateSpecificInfo(const string& type, AlphaReal value);
        
        BaseOutputInfoType* getOutputInfoObject(const string& type);

        /**
         * Return the specific metric output for a given
         * dataset and a given iteration. 
         * If the iteration is -1, it returns the last value.
         * \param pData The dataset on which the metric was computed.
         * \param outputName The three caracters code of the output (the same as for --outputinfo argument).
         * \param iteration The iteration at which the metric was computed.
         * \date 05/04/2013
         */
        AlphaReal getOutputHistory(InputData *pData, const string& outputName, int iteration = -1);

        /**
         * Indicate whether a given Output information
         * is activated.
         * \param outputName The three caracters code of the output (the same as for --outputinfo argument).
         * \date 05/04/2013
         */        
        bool outputIsActivated(const string& outputName);
        
        void setStartingIteration(unsigned int i) {_historyStartingIteration = i;}

    protected:
                
                
        fstream                _outStream; //!< The output stream 
        
        time_t                  _beginingTime;
        
        time_t                  _timeBias;
        
        
        // TODO: refactoring : replace the general 
        // map<InputData*, table> _gTableMap 
        // by 
        // map<InputData*, table*> whithin BaseOutputInfoType subclasses
        // so that they can sharing tables through pointers.
                
        /**
         * Maps the data to its g(x) table.
         * It is needed to keep this information saved from iteration to
         * iteration.
         * \see table
         * \see outputError()
         * \date 16/11/2005
         */
        map<InputData*, table> _gTableMap; 
                
        /**
         * Maps the data to the margins table.
         * It is needed to keep this information saved from iteration to
         * iteration.
         * \see table
         * \see outputMargins()
         * \date 16/11/2005
         */
        map<InputData*, table>  _margins;
                
        /**
         * Maps the data to the sum of the alpha.
         * It is needed to keep this information saved from iteration to
         * iteration.
         * \see outputMargins()
         * \date 16/11/2005
         */
        map<InputData*, AlphaReal> _alphaSums;
                
                
        //double getROC( vector< pair< int, AlphaReal > > data );
        
        /* 
         * Keeps the list of the output types the user wants to be output
         * \date 17/06/2011
         */
        map<string, BaseOutputInfoType*> _outputList;
                
        /**
         * Creates the OutputInfoType instances
         * from a string
         * \date 04/07/2011
         */
        void getOutputListFromString(const string& list,  const nor_utils::Args* args = NULL);
        
        /* 
         * Indicates whether we systematically update the internal structures after each
         * call of outputCustom(). Set to true if the learner wants to manage them itself.
         * \date 04/07/2011
         */
        bool _customTablesUpdate;
        
        /**
         * The header output stream
         */
        fstream _headerOutStream;
      
        unsigned int _historyStartingIteration; //to handle fastResume case
    };
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The elementary output information. For the most basic one is OuputErrorInfo.
     * This class is a factory, it creates the different outputs the user specifies
     * in the command line. So adding a new type of output only necessitate to 
     * inherit from this class, implement the pure virtual methods and to update 
     * the creation method (see further). 
     * Please note that the subclasses can use the information on the posteriors
     * and the margin that OutputInfo handle and they __dont__ need to update them.
     * Moreover, they can implement specific structures which they will responsible to update.
     * 
     * \date 17/06/2011
     */
    class BaseOutputInfoType
    {
    protected:
        map<InputData*, vector<AlphaReal> >       _outputHistory;
        
    public:
        
        BaseOutputInfoType() {};
        BaseOutputInfoType(const nor_utils::Args& args) {};
        
        /*
          Compute the output it is specialized in and print it.
          * \param outStream The stream where the output is directed to
          * \param pData The input data.
          * \param pWeakHypothesis The current weak hypothesis.
          * \see table
          * \see _gTableMap
          * \see _alphaSums
          * \date 17/06/2011
          */
        virtual void computeAndOutput(ostream& outStream, InputData* pData, 
                                      map<InputData*, table>& gTableMap, 
                                      map<InputData*, table>& marginsTableMap, 
                                      map<InputData*, AlphaReal>& alphaSums,
                                      BaseLearner* pWeakHypothesis = 0) = 0;
        
        /**
         * Print the header 
         * \param outStream The stream where the output is directed to
         * \param namemap The structure that contains the class information ( \see NameMap )
         * \date 17/06/2011
         */
        virtual void outputHeader(ostream& outStream, const NameMap& namemap) = 0;

        /**
         * Print a detailed description of the output.
         * \param outStream The stream where the output is directed to
         * \date 12/12/2012
         */
        virtual void outputDescription(ostream& outStream) {};

        /*
         * The creation method of the factory
         */
        static BaseOutputInfoType* createOutput(string type, const nor_utils::Args* args = NULL);
        
//        /**
//         * For specific cases (like cascades), the output info needs to keep extra information.
//         * This method updates the internal structure _specificInfo. It is left very generic on purpose.
//         * \param type The name of the info to be updated, it is highly recommanded to prefix this name by an abreviation of the class name.
//         * \param value The value to be added/updated
//         * \date 05/07/2011
//         */
//        virtual void updateSpecificInfo(const string& type, AlphaReal value) = 0;
        
        AlphaReal getOutputHistory(InputData *pData, int iteration)
        { return iteration < 0 ? _outputHistory[pData].back() : _outputHistory[pData].at(iteration); }
    };
        
//    template<class T = AlphaReal>
//    class OutputInfoType : public BaseOutputInfoType {
//    public:
//        /**
//         * For specific cases (like cascades), the output info needs to keep extra information.
//         * This method updates the internal structure _specificInfo. It is left very generic on purpose.
//         * \param type The name of the info to be updated, it is highly recommanded to prefix this name by an abreviation of the class name.
//         * \param value The value to be added/updated
//         * \date 05/07/2011
//         */
//        virtual void updateSpecificInfo(const string& type, AlphaReal value) {}
//        
//    protected:
//        
//        typename SpecificInfo<T>::Type _specificInfo;
//
//    };
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    //Subclasses for the factory
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The default error output
     * Outputs the error of the given data.
     * The error is computed by holding the information on the previous
     * weak hypotheses. In AdaBoost, the discriminant function is computed with the formula
     * \f[
     * {\bf g}(x) = \sum_{t=1}^T \alpha^{(t)} {\bf h}^{(t)}(x),
     * \f]
     * we therefore update the \f${\bf g}(x)\f$ vector (for each example)
     * each time this method is called:
     * \f[
     * {\bf g} = {\bf g} + \alpha^{(t)} {\bf h}^{(t)}(x).
     * \f]
     * \remark There can be any number of data to have the gTable. Each one is
     * mapped into a map that uses the pointer of the data as key.

     * \date 17/06/2011
     */
    class RestrictedZeroOneError : public BaseOutputInfoType {
        
    public:
        
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "r01" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "r01: Restricted Zero-One Error (error = min positive class score < max negative class score)";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The 0-1 error. 
     * \date 19/07/2011
     */
    class ZeroOneErrorOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "e01" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "e01: Zero-One Error (error = max positive class score < max negative class score)";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };
        
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The weighted 0-1 error. 
     * \date 19/07/2011
     */
    class WeightedZeroOneErrorOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "w01" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "w01: Weighted Zero-One Error (error = max positive class score < max negative class score)";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The hamming error. 
     * \date 19/07/2011
     */
    class HammingErrorOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "ham" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "ham: Hamming Error";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The weighted hamming error. 
     * \date 19/07/2011
     */
    class WeightedHammingErrorOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "wha" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "wha: Weighted Hamming Error";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };
        
        
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The weighted (restricted) error 
     * \date 20/06/2011
     */
    class WeightedErrorOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "werr" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "werr: Weighted Restricted Error (error = min positive class score < max negative class score)";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The balanced error 
     * \ref http://www.kddcup-orange.com/evaluation.php
     * \date 20/06/2011
     */
    class BalancedErrorOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) 
        { 
            outStream << "balerr" ;
            
            const int numClasses = namemap.getNumNames();
            for (int i = 0; i < numClasses; ++i ) {
                outStream << OUTPUT_SEPARATOR <<  "balerr[" << namemap.getNameFromIdx(i) << "]" ;}
        }
        
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "balerr: Balanced Error (see http://www.kddcup-orange.com/evaluation.php)";} ;
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    // TODO: lack of documentation below (especially, comment how the output is computed)
    /**
     * MAE
     * \date 20/06/2011
     */
    class MAEOuput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream <<  "mae" << OUTPUT_SEPARATOR << HEADER_FIELD_LENGTH << "mse";}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "mae: MAE Error";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * Output the minimum margin the sum of below zero margins.
     * These two elements are useful for an analysis of the training process.
     *
     * The margins are represent the per-class weighted correct rate, that is
     * \f[
     * \rho_{i, \ell} = \sum_{t=1}^T \alpha^{(t)} h_\ell^{(t)}(x_i) y_i
     * \f]
     * The \b fist \b value that this method outputs is the minimum margin, that is
     * \f[
     * \rho_{min} = \mathop{\rm arg\, min}_{i, \ell} \rho_{i, \ell},
     * \f]
     * which is normalized by the sum of alpha
     * \f[
     * \frac{\rho_{min}}{\sum_{t=1}^T \alpha^{(t)}}.
     * \f]
     * This can give a useful measure of the size of the functional margin.
     *
     * The \b second \b value which this method outputs is simply the sum of the
     * margins below zero.
     * \date 20/06/2011
     */
    class MarginsOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "min_mar" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "min_mar: Minimum Margin";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * Output the edge. It is the measure of the accuracy of the current 
     * weak hypothesis relative to random guessing, and is defined as
     * \f[
     * \gamma = \sum_{i=1}^n \sum_{\ell=1}^k w_{i, \ell}^{(t)} h_\ell^{(t)}(x_i)
     * \f]
     * \date 20/06/2011
     */
    class EdgeOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) { outStream << "edge" ;}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
//        void outputDescription(ostream& outStream) { outStream << "edge: Edge";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * The Area Under the ROC curve
     * (see \link http://sites.google.com/a/lesoliveira.net/luiz-eduardo-s-oliveira/Reconhecimento-de-Padr%C3%B5es--SS-/Artigos/ROCintro.pdf )
     * \date 20/06/2011
     */
    class AUCOutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap)
        { 
            outStream << "auc" ;
            const int numClasses = namemap.getNumNames();
            for (int i = 0; i < numClasses; ++i ) {
                outStream << OUTPUT_SEPARATOR << "auc[" << namemap.getNameFromIdx(i) << "]" ;}

        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "auc: Area Under The ROC Curve";} ;

        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * True positive and False positive rates (ROC curve coodinates)
     * (see \link http://sites.google.com/a/lesoliveira.net/luiz-eduardo-s-oliveira/Reconhecimento-de-Padr%C3%B5es--SS-/Artigos/ROCintro.pdf )
     * \date 20/06/2011
     */
    class TPRFPROutput : public BaseOutputInfoType {
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) 
        {
            outStream << "tpr" ;
            
            const int numClasses = namemap.getNumNames();
            for (int i = 0; i < numClasses; ++i ) {
                outStream << OUTPUT_SEPARATOR << "tpr[" << namemap.getNameFromIdx(i) << "]" ;}
            
            outStream << OUTPUT_SEPARATOR << "fpr" ;
            for (int i = 0; i < numClasses; ++i ) {
                outStream << OUTPUT_SEPARATOR << "fpr[" << namemap.getNameFromIdx(i) << "]" ;}

        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) { outStream << "tpr/fpr: True and False Positive Rates";} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
    };
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * SoftCascade (embedded-like) performance.
     * Outputs the error, auc, tpr, fpr.
     * The individual outputs should be refactored if needed.
     * \date 04/07/2011
     */
    class SoftCascadeOutput : public BaseOutputInfoType {
        
    protected:
        
        string  _positiveLabelName;
//        vector<vector<AlphaReal> > _posteriors;
        
        vector<BaseLearner*> _calibratedWeakHypotheses;
        vector<AlphaReal> _rejectionThresholds;
        
        vector<char> _forecast;
        
    public:
        
        SoftCascadeOutput(const nor_utils::Args& args) 
        {
            string _positiveLabelName; //!< The name of the positive class, to be used for some kinds of binary classification, otherwise set to "".
            if ( args.hasArgument("positivelabel") )
            {
                args.getValue("positivelabel", 0, _positiveLabelName);
            }
            else
            {
                cout << "Error : Positive class name must be provided.\n";
                exit(-1);
            }
        }
                        
        
        void outputHeader(ostream& outStream, const NameMap& namemap) 
        {
            outStream   << "err" << OUTPUT_SEPARATOR
                        << "auc" << OUTPUT_SEPARATOR
                        << "fpr" << OUTPUT_SEPARATOR
                        << "tpr" << OUTPUT_SEPARATOR
                        << "nbeval" ;
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) {} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
        
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void appendRejectionThreshold(AlphaReal value) 
        {
            _rejectionThresholds.push_back(value);
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        vector<char> & getForcastVector() {
            return _forecast;
        }

    };

    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * SoftCascade (embedded-like) performance.
     * Outputs the error, auc, tpr, fpr.
     * The individual outputs should be refactored if needed.
     * \date 04/07/2011
     */
    class VJCascadeOutput : public BaseOutputInfoType {
        
    protected:
        
        string  _positiveLabelName;
        //        vector<vector<AlphaReal> > _posteriors;
        
        vector<BaseLearner*> _currentBaseLearners;
        AlphaReal _rejectionThreshold;
        
    public:
        
        VJCascadeOutput(const nor_utils::Args& args) 
        {
            string _positiveLabelName; //!< The name of the positive class, to be used for some kinds of binary classification, otherwise set to "".
            if ( args.hasArgument("positivelabel") )
            {
                args.getValue("positivelabel", 0, _positiveLabelName);
            }
            else
            {
                cout << "Error : Positive class name must be provided.\n";
                exit(-1);
            }
        }
                
        
        void outputHeader(ostream& outStream, const NameMap& namemap) 
        {
            outStream   << "err" << OUTPUT_SEPARATOR
                        << "auc" << OUTPUT_SEPARATOR
                        << "fpr" << OUTPUT_SEPARATOR
                        << "tpr" << OUTPUT_SEPARATOR
                        << "nbeval" ;
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) {} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
        
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void appendRejectionThreshold(AlphaReal value) 
        {
            _rejectionThreshold = value;
        }
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void setCurrentStageWeakLearners(vector<BaseLearner*>& vBaseLearners ) 
        {
            _currentBaseLearners = vBaseLearners;
        }
                
    };
    
        
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * Output the predictor function (called also the posteriors).
     * One example per line. Only the specified class IdX (through \see updateSpecificInfo)
     * will be output.
     * \date 06/07/2011
     */
    class PosteriorsOutput : public BaseOutputInfoType {
        
    protected:
        vector<int> _classIdx;
        
    public:
        
        void outputHeader(ostream& outStream, const NameMap& namemap) {}
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void outputDescription(ostream& outStream) {} ;

        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void computeAndOutput(ostream& outStream, InputData* pData, 
                              map<InputData*, table>& gTableMap, 
                              map<InputData*, table>& marginsTableMap, 
                              map<InputData*, AlphaReal>& alphaSums,
                              BaseLearner* pWeakHypothesis = 0);
        
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        void addClassIndex(int value) 
        {
            _classIdx.push_back(value);
        }

    };

    
} // end of namespace MultiBoost

#endif // __OUTPUT_INFO_H
