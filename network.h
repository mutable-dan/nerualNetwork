/**********************************************************
Neural Network:

      File:  Network.h     -- G. Dan
         Project - Network.cpp      Member functions
                   Network.h        Class definitions


**********************************************************/

#include <math.h>


#define SIGNAL_T double    // type for signal transmission
enum LayerType { INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER };
#define getrandom( min, max ) ((rand() % (int)(((max)+1) - (min))) + (min))
     /*  Random integer within a range from 1 .. max <= sizeof(int) */

class synapse;
class biasSignal;


/* Basic unit of network:  Neuron connect to other neurons thru axons and dendrites.
     The axon connects to a synapse which is connected to the next neuron (in next layer)
     via the dendrite.  Please see notes on specific members for details.
*/
class neuron {
   friend class neuronIterator;
   protected:
      LayerType type;                  // 0 - input, 1 - inner, 2 - output layer GET RID OF, SEE BELOW
      int      learningFlag;           // on if neuron was used in this learning trial
      int      numberOfAxons;          // # of neurons this neuron is connected to, 0 means output layer
      int      numberOfDendrites;      // # of neurons that connect to this neuron, 0 means input layer
      SIGNAL_T axonSignal;             // signal that neuron generates
      SIGNAL_T _error;                 // error value of neuron,  for output layer error = desired - actual, for inner
                                       // layer error = errorNext*weight
      SIGNAL_T slope;                       // the transfer (activation, threshold) function slope factor
      synapse  **axon;                      // carries signal away from neuron to, each axon is connected to one neuron
                                            // in next level or output
      synapse  **dendrite;                  // carries signal into neuron, leads to synapse associated with this neuron
      biasSignal *bias;                     // special fixed input to neuron
      SIGNAL_T *input;                      // normalized preprocessed input on 1st layer,  This points to a member of vector
                                            //, do NOT index, this points to an array element.
      SIGNAL_T *output;                     // output to send to postprocessor,  This points to a member of vector,
                                            // do NOT index, this points to an array element.
      SIGNAL_T *desiredOutput;              // assign vector to output layer only, This points to a member of vector,
                                            //  do NOT index.
      neuron   *nextl;                      // part of a linked list
      SIGNAL_T xferFn( const SIGNAL_T &x=0 ) { return ( 1.0/(1.0+exp(-slope*(double)x) ) ); }
                                            // Unipolar activaation or transfer function
   public:
      neuron( neuron *_next, const int &_numberOfAxons, const int &_numberOfDendrites, const LayerType &type,
            SIGNAL_T &sig, SIGNAL_T &slope );   // set up axons & dendrites
      ~neuron() { delete [] axon; delete [] dendrite; /*delete bias;*/ }
      LayerType whatType() { return type; }
      neuron *next() { return nextl; }
      void connect( neuron *n2 );   // connect 'this' neuron to neuron n2
      void setAxon( synapse *s );
      void setDendrite( synapse *s );                                           //For the below:  Apply ONLY to INPUT
                                                                                // and OUTPUT layers
      void setInput( SIGNAL_T *vector, const int &i ) { input = (vector+i); }   //Given the input vector and an offset,
                                                                                // attach the neuron input to that vector offset
      void setOutput( SIGNAL_T *vector, const int &i ) { output = (vector+i); } //Given the input vector and an offset,
                                                                                // attach the neuron input to that vector offset
      void setDesiredOutput( SIGNAL_T *vector, const int &i ) { desiredOutput = (vector+i); } //Given the input vector
                                                                                // and an offset, attach the neuron input
                                                                                // to that vector offset
      const SIGNAL_T &errorSignal();
      const SIGNAL_T &getError() { return _error; }
      const SIGNAL_T getWeight( const int &i );
      const SIGNAL_T &getNeuronOuput() { return axonSignal; }
      SIGNAL_T generalizedOutputlayerError();            // sums weighted signals from synapses
      SIGNAL_T learn( SIGNAL_T &learningConstant, SIGNAL_T &weightFactor, SIGNAL_T &outputLayerWeightedError );
      void reflect();                                    // neuron is activated, and part of an active NN
      friend std::ostream &operator <<( std::ostream &os, const neuron &n );
};


/* Forms connection between neurons and facilitates learning
 */
class synapse {
   protected:
      SIGNAL_T weight;                // weight of signal
      SIGNAL_T oldWeight;             // weight mometum term
      SIGNAL_T signal;                // neuron signal being sent
      neuron  *nextLayer;             // who is recieving signal
      neuron  *prevLayer;             // who is sending signal
   public:
      synapse( neuron *_prevLayer, neuron  *_nextLayer );
      ~synapse() {}
      SIGNAL_T weightedSignal() { return weight*signal; }
      void setWeight( const SIGNAL_T &w ) { weight = w; }
      void setSignal( const SIGNAL_T &s ) { signal = s; }
      SIGNAL_T getWeight() { return weight; }
      SIGNAL_T getErrorOfNeuron() { return nextLayer->getError(); }
      void learn(const SIGNAL_T &lc, const SIGNAL_T wf, const SIGNAL_T &e );
         // lc - learning const, wf - weight factor for weight momentum, e - error, delta

};



/* Bias  - Defines the bias term for each neuron 
      The bias connects to one neuron only, there is no connection to layer n-1
 */
class biasSignal: public synapse {
   protected:
      const SIGNAL_T signal;
   public:
      biasSignal( SIGNAL_T sig, neuron * n ) : synapse( 0, n ), signal(sig) {
         synapse:: weight = 0.0; }
      ~biasSignal() {};
      SIGNAL_T weightedSignal() { return weight*signal;}
      SIGNAL_T getSignal() { return signal; }
      void learn(const SIGNAL_T &lc, const SIGNAL_T &e ) { weight += lc*e*signal; }
};




/* Basic iterator:
 */
class neuronIterator {
   protected:
      neuron *current;
   public:
      neuronIterator() { current = 0; }
      ~neuronIterator() {}
      neuron *&operator =( neuron *n ) { current = n; return current; };   // setup
      neuron *address() { return current; }        // address of object being pointed to
      void operator ++(int) { current =  current->nextl!=0 ? current->nextl : 0; } // incr pointer, dummy arg int states postfix
};



/* A collection of neurons forms a layer:  The neurons know who is input and who is output
      Although a layer where prevl = 0 is an input and where nextl = 0 is an output layer
 */
class layer {
   friend class layerIterator;
   private:
      int     layerNumber;   // for layers 1..n, where 1 is input and n is output, all others are hidden
      neuron *firstNeuron;
      int     numberOfNeurons;
      layer  *nextl;    // double LL
      layer  *prevl;
   public:
      layer( int _layerNumber, int *_numberOfNeurons, int _numberOfNeuronsNextLayer, const LayerType &type,
            SIGNAL_T sig, SIGNAL_T &slope );
      ~layer();
      int numberOfNeuronsInLayer() { return numberOfNeurons; }
      inline int numberOfNeuronsNextLayer();
      inline int numberOfNeuronsPrevLayer();
      neuron *getFirstNeuron() { return firstNeuron; }
      SIGNAL_T generalizedOutputlayerError();
      void setPrev( layer *p ) { prevl = p; }
      void setNext( layer *p ) { nextl = p; }
      layer *next() { return nextl; }
      layer *prev() { return prevl; }
      SIGNAL_T learn( SIGNAL_T &learningConstant, SIGNAL_T &weightFactor, SIGNAL_T &errors );
      void reflect();              // tell neurons in layer cause layer to activate
};


/* Basic iterator  */
class layerIterator {
   protected:
      layer *current;
   public:
      layerIterator() { current = 0; }
      ~layerIterator() {}
      layer *&operator =( layer *l ) { current = l; return current; }  //setup
      neuron *getFirstNeuron() { return current->firstNeuron; }
      layer *next() { return current->nextl; }
      layer *address() { return current; }   // get address of object pointed to
      void operator ++(int) { current = current->nextl; }  // dummy arg int states postfix
      void operator --(int) { current = current->prevl; }
};



/* Network object contains simulated Neural Network
 */
class network {
   private:
      int      numberOfLayers;           // How many layers this network contains
      char    *nuronsPerLayer;           // char string to be converted to an int array, see next member...
      int     *numberOfNeuronsInLayer;   // n dim array where n=numberOfLayers.
      SIGNAL_T learningConstant;         // factor to speed learning rate.  Used in synapse
      SIGNAL_T weightFactor;             // Factor to add momentum to weight change
      SIGNAL_T sumOfErrors;              // Sum of error signals in output layer, computed once for each learning run
      layer    *inputLayer;              // head of list
      layer    *outputLayer;             // rail of list

      void setIO_Vector();
      int insertLayer( const int &id, SIGNAL_T sig, SIGNAL_T slope );   //add a layer to network
      void connect();                    // connect current layer to next layer and connect neurons in layers
      void outputLayerErrorSignal();
   public:
      SIGNAL_T *inputVector;             // input vector for input layer
      SIGNAL_T *outputVector;            // output vector for output layer
      SIGNAL_T *desiredOutputVector;     // vector of desired output, used for learning

      network( const SIGNAL_T slope, const SIGNAL_T learningConstant, const SIGNAL_T weightFactor,
         const SIGNAL_T BiasSignal, const int _numberOfLayers, int *layerList );
      ~network();
      SIGNAL_T learnSE();                // learn and return the square error
      void learn();                      // learn with no error return
      void reflect();                    // let the network ponder what was learned.  Use the NN
      friend std::ostream &operator <<( std::ostream &os, const network &n );
};

