#ifdef __ICC__
#include <ittnotify.h>
#endif

#include "rd_2d_karb_nogfx_subp.h"

#include <iostream>
#include <vector>
#include <string>

// Provides the max steps parameter, which can be shared across
// process_nogfx_*.cpp programs for easy comparisons.
#include "progparams.h"

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
    RD.svgpath = BOUNDARY_SVG;
    RD.domainMode = false;
    RD.hextohex_d = PROCESS_HEXTOHEX_D;
    try {
        RD.init();
    } catch (const exception& e) {
        cerr << "Exception initialising RD_2D_Karb object: " << e.what() << endl;
        return rtn;
    }

#ifdef __ICC__
    __itt_resume();
#endif
    // Do requisite number of steps
    for (unsigned int st = 0; st < PROCESS_MAXSTEPS; ++st) {
        RD.step();
    }
#ifdef __ICC__
    __itt_pause();
#endif

    RD.saveC();

    return rtn;
};
