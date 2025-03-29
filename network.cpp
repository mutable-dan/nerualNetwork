/**********************************************************
      File:  Network..cpp
         Project - Network.cpp       Member functions
                   Network.h         Class definitions for exe
                   NetworkD.h        Class definitions for dll
 
     -- G. Dan
**********************************************************/


#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <stdarg.h>

#include "network.h"


/******************  Member functions of Neuron   ******************/


/* Neuron:
      id - for internal accounting
      next - next in list
      _numberOfAxons - also # of neurons in next layer since fully connected between layers
      _numberOfDendrites - also # of neurons in previous layer since fully connected between layers
*/
neuron::neuron( neuron *_next, const int &_numberOfAxons, const int &_numberOfDendrites, const LayerType &_type,
       SIGNAL_T &sig, SIGNAL_T &_slope ) {
   type = _type;
   slope = _slope;
   input = 0;
   output = 0;
   _error = 0;
   learningFlag = 0;
   axonSignal = 0.0f;
   nextl = _next;
   if( type > INPUT_LAYER )
      bias = new biasSignal( sig, this );   // if it is not an input layer add a bias and set bias signal to sig
   else
      bias = 0;   // no bias on input layer
   numberOfAxons = _numberOfAxons;
   numberOfDendrites = _numberOfDendrites;
   if( _numberOfAxons ) {      // hidden layer
      axon = new synapse*[_numberOfAxons];
      for( int i=0; i<_numberOfAxons; i++ )
         *(axon+i) = 0;
      output = 0;
   }
   else                        // this is !output layer
      axon = 0;

   if( _numberOfDendrites ) {  // hidden layer
      dendrite = new synapse*[_numberOfDendrites];
      for( int i=0; i<_numberOfDendrites; i++ )
         *(dendrite+i) = 0;
      input = 0;
   }
   else       // then it is an input layer
      dendrite = 0;
}



/* Connect axon of current neuron to synapse and dendrite of n2 
 */
void neuron::connect( neuron *n2 ) {
synapse *s;
   s = new synapse( this, n2 );    // this synapse is the 'go between' for neurons this and n2
   if( s == 0 ) {
      std::cerr << "Memory allocation: neuron::connection" << std::endl;
      exit(1);
   }
   this->setAxon(s);    // sets pointer array of axons to point to synapses
   n2->setDendrite(s);  // sets pointer array of dendrites to point to synapse s
}



/* first empty pointer in the axon list will be used to point to the synapse
   Axon carries signal away from neuron
 */
void neuron::setAxon( synapse *s ) {
   for( int i=0; i<numberOfAxons; i++ )
      if( axon[i] == 0 ) {
         axon[i] = s;
         break;
      }
}



/* first empty pointer in the dendrite list will be used to point to the synapse
   Dendrite carries signal to neuron
 */
void neuron::setDendrite( synapse *s ) {
   for( int i=0; i<numberOfDendrites; i++ )
      if( dendrite[i] == 0 ) {
         dendrite[i] = s;
         break;
      }
}




// generalizedOutputlayerError - Polls synapses for signal, error product and returns sum of products
//    This is used to keep a value of the output layer error in the network administrator (network::)
//    so that the hidden layers can compute an error based on the error of the ouptut
//    NOTE:  May only be applied to output layer
SIGNAL_T neuron::generalizedOutputlayerError() {
   SIGNAL_T sum = 0.0;
   SIGNAL_T e = (*desiredOutput - axonSignal)*(1.0 - axonSignal)*axonSignal;         // unipolar activation
   //SIGNAL_T e = (*desiredOutput - axonSignal)*(1.0 - axonSignal*axonSignal)/2.0;   // bipolar activation
   for( int i=0; i<numberOfDendrites; i++ )
         sum += dendrite[i]->getWeight()*e;
   sum += bias->getWeight()*e;
   return sum;
}




/* Learn - goes through list of synapses and tells each synapse to learn (adjust weights) something 
                     Caluculates the error of a neuron.
                     If output layer:  Takes the desired output vector and compares to the actual signal to calc error
                     input  layer:  Gets the sum of product of errors of the next connected neuron and output
.                    W'ij = EiOouti + Wij,   where  Ei is the error of neuron i and Oouti is the output of neuron i.

 */
SIGNAL_T neuron::learn( SIGNAL_T &learningConstant, SIGNAL_T &weightFactor, SIGNAL_T &outputLayerWeightedError ) {
SIGNAL_T ret = 0.0;     // used to find error wrt target output

   if( type < OUTPUT_LAYER )         // then it's a hidden layer
      _error = (1 - axonSignal)*axonSignal*outputLayerWeightedError;          // unipolar activation
      //_error = (1.0 - axonSignal*axonSignal)*outputLayerWeightedError/2.0;  //bipolar activation
   else {
      _error = (*desiredOutput - axonSignal)*(1 -  axonSignal)*axonSignal;         // unipolar activation
      //_error = (*desiredOutput - axonSignal)*(1 -  axonSignal*axonSignal)/2.0;   // Bipolar activation
      ret = *desiredOutput - axonSignal;     // used to find error wrt target output
      ret *= ret;
   } 
   for( int i=0; i<numberOfDendrites; i++ ) {
      dendrite[i]->learn( learningConstant, weightFactor, _error );   // adjusts weights of the synapses connected to this neuron
   }
   bias->learn( learningConstant, _error );    // adjust weight of bias term
   return ret;
}



/* Returns the weight of synapse i of this neuron.  Declared inline and outside of class for scope reaosns
 */
const SIGNAL_T neuron::getWeight( const int &i ) {
       return ( (i < numberOfAxons) ? axon[i]->getWeight() : 0 );
}



/* Activate - Instructs a neuron to compute output
 */
void neuron::reflect() {
SIGNAL_T sum = 0;
   if( type != INPUT_LAYER ) {
      for( int i=0; i<numberOfDendrites; i++ ) 
         sum += dendrite[i]->weightedSignal();   // get the weighted sum of synapses
      sum += bias->weightedSignal();
      axonSignal = xferFn( sum );           // send synapse data to transfer (activation, threshold)function
      if( type == OUTPUT_LAYER )
         *output = axonSignal;
   }
   else
      axonSignal = *input;         // take input pointer to member of input vector and use as neural input

   for( int i=0; i<numberOfAxons; i++ )
      axon[i]->setSignal( axonSignal );  // place the neuron signal in each synapse of next neuron via axons
                                         // signal is constant for the bias term
}


 


/* Operator <<  - show neuron  info
 */
std::ostream &operator <<( std::ostream &os,  const neuron &n ) {
   os << "Neuron output: " << n.axonSignal;
   if( n.numberOfDendrites != 0 )
      os << "  -------> Weights: "; 
   for( int i=0; i<n.numberOfDendrites; i++ )
      os << (n.dendrite[i])->getWeight() << ", ";
   if( n.bias )
      os << "    Bias " << n.bias->getSignal() << "  Weight " << n.bias->getWeight();
   os << std::endl;
   return os;

}



/******************  Member functions of Synapse ******************/

/* Synapse:
      _prevLayer - pointer to neuron sending signal, the this pointer
      _nextLayer - pointer to neuron recieving signal, neuron in next layer
      prev layer neuron axon points to synapse and next layer dendrite points to synapse
 */
synapse::synapse( neuron *_prevLayer, neuron *_nextLayer )  {
SIGNAL_T sign, test;
   prevLayer = _prevLayer;
   nextLayer = _nextLayer;
   oldWeight = 0.0;
   signal = 0.0;
   if ( _prevLayer == 0 )      // if prev layer (or this pointer) is NULL then it is a bias input
      return;
   if ( _nextLayer->whatType() == HIDDEN_LAYER ) {               // make hidden layer between +- (1 to 1.5)
      weight = 1.0 + ( (SIGNAL_T)getrandom(1,50) )/100.0;        // random value from 1 - 1.5
      sign = (test=getrandom(1, 100)) < 50? -1.0: 1.0;  // randomly change sign
      weight *= sign;
   } else {                                       // make output layer between -.1 and .1
      weight = ( (SIGNAL_T)(getrandom(1,101) - 1.0) )/1000.00;   // random value from 0 and .1
      sign = (test=getrandom(1, 100)) < 50? -1.0: 1.0;           // randomly change sign
      weight *= sign;
   } /* endif */
  // weight = (SIGNAL_T)(rand() % 100) /100.0;   // random value from 0 - .99
   signal = 0.0f;
}



// Assumes error has been properly
// set in connecting neuron and calcs
// new weight
void synapse::learn(const SIGNAL_T &lc, const SIGNAL_T wf, const SIGNAL_T &e ) {
SIGNAL_T delta = 0.0;
   delta = lc*e*signal + wf*oldWeight;
   oldWeight = delta;
   weight += delta;
}



/******************  Member functions of Layer ******************/

/* Layer:
      _layerNumber - a number from 0...n-1
      _numberOfNeurons - pointer to array of ints cooresponding to neuron per layer
      _numberOfNeuronsNextLayer -  used by neuron to determine # of axons
      type - type of layer, input, hidden or output
      sig - value of bias signal
 */

layer::layer( int _layerNumber, int *_numberOfNeurons, int _numberOfNeuronsNextLayer, const LayerType &type,
      SIGNAL_T sig, SIGNAL_T &slope ) {
int neuronPrevLayer;      // used by neuron to determine # of dendrites
   // For:: *numberOfNeurons,  _layerNumber-1 is current layer pointer
   layerNumber = _layerNumber;
   numberOfNeurons = _numberOfNeurons[_layerNumber-1];
   neuronPrevLayer =  _layerNumber-2<0 ? 0 : _numberOfNeurons[_layerNumber-2];  // use array of neurons per layer to index # of neurons
   nextl = 0;
   prevl = 0;
   firstNeuron = 0;
   // set up list of objects, 1st created is last on list
   for( int i=1; i<=numberOfNeurons; i++ )  // build list of neurons
      firstNeuron = new neuron( firstNeuron, _numberOfNeuronsNextLayer, neuronPrevLayer, 
         type, sig, slope );
}



// Delete neurons in a layer
layer::~layer() {
neuron *p = firstNeuron;
neuron *t = firstNeuron;
   while( p ) {
      t = p->next();
      delete p;
      p = t;
   }
}



/* generalizedOutputlayerError- Goes thu list of neurons  and computes S(error*wij)
      Should only be used in the output layer
*/
SIGNAL_T layer::generalizedOutputlayerError() {
neuron *nextNeuron = firstNeuron;
SIGNAL_T outputError = 0.0;
   while( nextNeuron ) {
      outputError += nextNeuron->generalizedOutputlayerError();
      nextNeuron = nextNeuron->next();
   }
   return outputError;
}



/* numberOfNeuronsNextLayer - returns the value xor zero
 */
inline int layer::numberOfNeuronsNextLayer() {
   if( nextl == 0 )
      return 0;
   else
      return nextl->numberOfNeuronsInLayer();
}


/* numberOfNeuronsPrevLayer - returns the value xor zero
 */
inline int layer::numberOfNeuronsPrevLayer() {
   if( prevl == 0 )
      return 0;
   else
      return prevl->numberOfNeuronsInLayer();
}



/* Learn - goes through each neuron and tells it to  learn something
 */
SIGNAL_T layer::learn( SIGNAL_T &learningConstant, SIGNAL_T &weightFactor, SIGNAL_T &errors ) {
neuron *nextNeuron = firstNeuron;
SIGNAL_T outputError = 0.0;
   while( nextNeuron ) {
      outputError += nextNeuron->learn( learningConstant, weightFactor, errors );      // square of target - output is returned
      nextNeuron = nextNeuron->next();
   }
   if( outputError == 0.0 )
      return 0.0;
   else
      return( outputError/2.0 );
}




/* Activate - cause of layer of neurons to fire
 */
void layer::reflect() {
neuron *nextNeuron = firstNeuron;
   while( nextNeuron ) {
      nextNeuron->reflect();
      nextNeuron = nextNeuron->next();
   }
}





/******************  Member functions of Network:: ******************/

/* Network:
      _numberOfLayers - as stated
      _nuronsPerLayer - input string to be converted to int array
      layerList - Array of neurons per layerwhere element 0 is the input layer and n-1 is the output layuer
         Used to pass a function ptr as:  SIGNAL_T (*_xferFn)(const SIGNAL_T &x=0)
 */                                         
network::network( const SIGNAL_T slope, const SIGNAL_T
_learningConstant, const SIGNAL_T _weightFactor,
   const SIGNAL_T sig, const int _numberOfLayers, int *layerList ) {


   inputLayer = 0;
   outputLayer = 0;
   numberOfLayers = _numberOfLayers;
   numberOfNeuronsInLayer = new int[numberOfLayers];
   learningConstant = _learningConstant;
   weightFactor = _weightFactor;
   if ( weightFactor < 0.0 || weightFactor >= 1.0 ) 
      std::cout << "WARNING:  Weight momentum factors should be in the range of 0 >= m < 1.  For m = 0, Momentum is disabled\n" << std::endl;
   sumOfErrors = 0.0;

   int i = 0;
   while ( i <= _numberOfLayers ) {
      numberOfNeuronsInLayer[i] = layerList[i];
      i++;
   }

   inputVector = new SIGNAL_T [numberOfNeuronsInLayer[0]];                          // allocate room for input vector
   outputVector = new SIGNAL_T [ numberOfNeuronsInLayer[numberOfLayers-1] ];        // output layer
   desiredOutputVector = new SIGNAL_T [ numberOfNeuronsInLayer[numberOfLayers-1] ]; // output layer

   for( i=1; i<=numberOfLayers; i++ ) // build network
      insertLayer( i, sig, slope );
   connect();   // make network fully connected between layers
   setIO_Vector();   // connect the vectors to the input and output layers, as appropriate
}




// Destroy the layers and net
network::~network() {
layer *p = inputLayer;
layer *t = inputLayer;
   delete [] numberOfNeuronsInLayer;
   delete [] inputVector;
   delete [] outputVector;
   delete [] desiredOutputVector;
   while( p ) {
      t = p->next();
      delete p;
      p = t;
   }
}


/* Connect: each neuron on current layer to every neuron on next layer.  If
    layer is for output, then ther is no connection
 */
void network::connect() {
neuron *nptr;
neuronIterator nI1, nI2;
layerIterator lI;

   lI = inputLayer;       // layer ptr
   if( lI.address() == 0 ) return; // there is no network
   while( lI.next() ) {
      for( nI1 = lI.getFirstNeuron(); nI1.address() != 0; nI1++ )  // for each neuron of current layer
         for( nI2 = ( lI.next() )->getFirstNeuron(); nI2.address() != 0; nI2++ ) {   // connect to each of these
            nptr = nI1.address();
            nptr->connect( nI2.address() );      // connect neuron npt1 to neuron npt2
         }
      lI++;          // goto next layer
   }
}




/* insertLayer: add layer to list of layers
      layerNumber - of layer being built from 0..n-1
 */
int network::insertLayer( const int &layerNumber, SIGNAL_T sig, SIGNAL_T slope ) {
int numberOfNeuronsNextLayer = layerNumber < numberOfLayers ? numberOfNeuronsInLayer[layerNumber] : 0;  // safe array indexing
   // for above, if an output layer set to 0, else set to neurons in next layer
   // remember id is from 1..n, !0..n
LayerType type;
   if( layerNumber == 1 )
      type = INPUT_LAYER;
   else
      if( layerNumber < numberOfLayers )
         type = HIDDEN_LAYER;
   else
      if( layerNumber == numberOfLayers )
         type = OUTPUT_LAYER;

   layer *tmp = new layer( layerNumber, numberOfNeuronsInLayer, numberOfNeuronsNextLayer, type, sig, slope );
                 // id=layer #, neurons in current layer, neurons in next layer
   if( tmp == 0 )    // build double linked list
      return 0;        // failure
   if( outputLayer == 0 ) {     // 1st element
      outputLayer = tmp;
      inputLayer = tmp;    // note pev and next ptr of tmp are set on construction
   }
   else {
      tmp->setPrev(outputLayer);   // link to head of list
      outputLayer->setNext(tmp);
      tmp->setNext(0);
      outputLayer = tmp;
   }
   return 1;
}



/* Set the input layer neurons to point to the input vector.  For example, neuron 1
   would point to the first position on the input vector and so on.  Same for output layer
 */
void network::setIO_Vector() {             
neuron *n;
   n = inputLayer->getFirstNeuron();
   int limit = inputLayer->numberOfNeuronsInLayer();
   for( int i=0; i<limit; i++ ) {                    // read from 1st o last neuron in list
      n->setInput( inputVector, i );                 // attach neuron input to input vector
      n = n->next();
   }

   n = outputLayer->getFirstNeuron();
   limit = outputLayer->numberOfNeuronsInLayer();
   for( int i=0; i<limit; i++ ) {                   // want the output layer to point to the output vector and an desired output for learning
      n->setOutput( outputVector, i );          // the same for the ouput layer
      n->setDesiredOutput( desiredOutputVector, i );
      n = n->next();
   }
}




/* LearnSE - Learn mean square error - Starting at the output layer - do Back-Propagation
      But, return the sum of square errors
 */
SIGNAL_T network::learnSE() {
layer *p = outputLayer;
SIGNAL_T outputError = 0;
 
   reflect();
   sumOfErrors = outputLayer->generalizedOutputlayerError();
   while( p != inputLayer ) {           // input layer cannot learn since it has no weighted inputs
      outputError += p->learn( learningConstant, weightFactor, sumOfErrors );   // if not an output layer than zero is returned
      p = p->prev();    // notice that the above is a programming trick wrt outputError
   }
   return outputError;
}



// Learn with no parameters, do one pass
void network::learn() {
layer *p = outputLayer;

   reflect();
   sumOfErrors = outputLayer->generalizedOutputlayerError();
   while( p != inputLayer ) {           // input layer cannot learn since it has no weighted inputs
      p->learn( learningConstant, weightFactor, sumOfErrors );   // if not an output layer than zero is returned
      p = p->prev();    // notice that the above is a programming trick wrt outputError
   }
}



/* Reflect - will do an iteration of the network.  After the network was taught, let it reflect, ie use
             the network.
 */
void network::reflect() {
layer *p = inputLayer;
   while( p ) {
      p->reflect();
      p = p->next();
   }
}




/* Show network:
 */
std::ostream &operator <<( std::ostream &os, const network &n ) {
neuronIterator nI;
layerIterator lI;
int i = 0;

/*   limit = n.numberOfNeuronsInLayer[n.numberOfLayers-1];
   os << "Actual output vector: ";
   for( int j=0; j<limit; j++ )
      os << n.outputVector[j] << "  ";
   cout << endl;

   os << "Target output vector: ";
   for( j=0; j<limit; j++ )
      os << n.desiredOutputVector[j] << "  ";
   cout << endl;
//   return os;
*/

   lI = n.inputLayer;
   while( lI.address() ) {
      os << "\nLayer " << i++ << std::endl;
      for( nI = lI.address()->getFirstNeuron(); nI.address() != 0;
        nI++ )
         os << *nI.address();
      lI++;
   }
   return os;

}
