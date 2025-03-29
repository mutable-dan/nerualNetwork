//***************** DEMO OF NEURAL NETWORK

                                 
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "network.h"


int main() {
std::ofstream fo("network Output.network");
double momentum, ls, bias;
SIGNAL_T err = 0;
int i = 0;

   std::cout << "Learning speed:";
   std::cin >> ls;
   std::cout << "\nMomentum:";
   std::cin >> momentum;
   std::cout << "\nBias:";
   std::cin >> bias;

   int neurons[] = { 2, 3, 1 };
   network Laura3( 1.0, ls, momentum, bias, 3, neurons );

   fo << "-------------LAURA3--------------" << std::endl;

   i = 0;
    do {
        Laura3.inputVector[0] = 0;
        Laura3.inputVector[1] = 1;
        Laura3.desiredOutputVector[0] = 1;
        err = Laura3.learnSE();

        Laura3.inputVector[0] = 1;
        Laura3.inputVector[1] = 0;
        Laura3.desiredOutputVector[0] = 1;
        err += Laura3.learnSE();

        Laura3.inputVector[0] = 1;
        Laura3.inputVector[1] = 1;
        Laura3.desiredOutputVector[0] = 0;
        err += Laura3.learnSE();

        Laura3.inputVector[0] = 0;
        Laura3.inputVector[1] = 0;
        Laura3.desiredOutputVector[0] = 0;
        err += Laura3.learnSE();
//        fo << "Run #" << i+1 << " Error = " << err << endl;

        if ( i++ > 5000 )
            break;
        } while ( err > 0.03 ); /* enddo */
    
   fo << " # of itereations: " << i << std::endl;


   fo << "Reflect Laura 4 ********************" << std::endl;
   fo << "-------reflec    11 = 0    tXXXXXXXXXXXXX ---------------" << std::endl;
   Laura3.inputVector[0] = 1;
   Laura3.inputVector[1] = 1;
   Laura3.reflect();
   fo << "Net output is: " << Laura3.outputVector[0] << std::endl;
   fo << Laura3 << std::endl;

   fo << "-------reflec  01 = 1      tXXXXXXXXXXXXX ---------------" << std::endl;
   Laura3.inputVector[0] = 0;
   Laura3.inputVector[1] = 1;
   Laura3.reflect();
   fo << "Net output is: " << Laura3.outputVector[0] << std::endl;
   fo << Laura3 << std::endl;

   fo << "-------reflec   10 = 1     tXXXXXXXXXXXXX ---------------" << std::endl;
   Laura3.inputVector[0] = 1;
   Laura3.inputVector[1] = 0;
   Laura3.reflect();
   fo << "Net output is: " << Laura3.outputVector[0] << std::endl;
   fo << Laura3 << std::endl;


   fo << "-------reflec    00 = 0    tXXXXXXXXXXXXX ---------------" << std::endl;
   Laura3.inputVector[0] = 0;
   Laura3.inputVector[1] = 0;
   Laura3.reflect();
   fo << "Net output is: " << Laura3.outputVector[0] << std::endl;
   fo << Laura3 << std::endl;

   fo << "----------------------" << std::endl;

   fo.close();
   return 0;
}
