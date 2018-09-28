#include "rd_2d_karb_nogfx_doms.h"

#include <iostream>
#include <vector>
#include <string>

#ifdef __ICC__
#include <ittnotify.h>
#endif

using namespace std;

int main (int argc, char **argv)
{
#ifdef __ICC__
    __itt_pause();
#endif
    int rtn = 1;
    if (argc < 1) {
        cerr << "\nUsage: ./build/sim/process\n\n";
        cerr << "Be sure to run from the base NeoArealize source directory.\n";
        return rtn;
    }

    // Set RNG seed
    int rseed = 1;
    srand(rseed);

    // Instantiate the model object
    RD_2D_Karb RD;
    RD.domainMode = true;
    RD.hextohex_d = 0.005;
    try {
        RD.init();
    } catch (const exception& e) {
        cerr << "Exception initialising RD_2D_Karb object: " << e.what() << endl;
        return rtn;
    }

    // Start the loop
    unsigned int maxSteps = 2000; // 2000 for speed comparisons
    bool finished = false;
#ifdef __ICC__
    __itt_resume();
#endif
    while (finished == false) {
        // Step the model
        try {
            RD.step();
        } catch (const exception& e) {
            cerr << "Caught exception calling RD.step(): " << e.what() << endl;
            finished = true;
        }

        if (RD.stepCount > maxSteps) {
            rtn = 0;
            finished = true;
        }
    }

    RD.saveC();

    return rtn;
};
