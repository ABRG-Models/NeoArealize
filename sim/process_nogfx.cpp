#define MANUFACTURE_GUIDANCE_MOLECULES 1
#include "rd_2d_karb_nogfx.h"

#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main (int argc, char **argv)
{
    if (argc < 1) {
        cerr << "\nUsage: ./build/sim/process\n\n";
        cerr << "Be sure to run from the base NeoArealize source directory.\n";
        return -1;
    }

    // Set RNG seed
    int rseed = 1;
    srand(rseed);

    // Instantiate the model object
    RD_2D_Karb RD;
    try {
        RD.init();
    } catch (const exception& e) {
        cerr << "Exception initialising RD_2D_Karb object: " << e.what() << endl;
    }

    // Start the loop
    unsigned int maxSteps = 2000;
    bool finished = false;
    while (finished == false) {
        // Step the model
        try {
            RD.step();
        } catch (const exception& e) {
            cerr << "Caught exception calling RD.step(): " << e.what() << endl;
            finished = true;
        }

        if (RD.stepCount > maxSteps) {
            finished = true;
        }
    }

    return 0;
};
