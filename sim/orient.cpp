#include "rd_orientpref.h"

#include "morph/display.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main (int argc, char **argv)
{
    if (argc < 1) {
        cerr << "\nUsage: ./build/sim/orient";
        cerr << "Be sure to run from the base NeoArealize source directory.\n";
        return -1;
    }

    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    vector<double> rot(3, 0.0);

    double rhoInit = 1.4;
    string worldName(argv[1]);
    string winTitle = worldName + ": n";
    displays.push_back (morph::Gdisplay (500, 500, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();


    // Set RNG seed
    int rseed = 1;
    srand(rseed);

    cout << "First rand num for seed " << rseed << " is: " << morph::Tools::randDouble() << endl;

    // Instantiate the model object
    RD_OrientPref RD;

    // Do any modifications to RD parameters here.
    // RD.myparam = 3.4;

    // Call the init function, which can allocate variables and run
    // any pre-stepping computations.
    try {
        RD.init();
    } catch (const exception& e) {
        cerr << "Exception initialising RD object: " << e.what() << endl;
    }

    // How many iterations to compute?
    unsigned int T = 100000; // Care - while debugging h5 files are being created, with T=10000 30 GB of logs will be created!

    // Start the loop
    bool finished = false;
    while (!finished) {
        // Step the model
        try {
            RD.step();
        } catch (const exception& e) {
            cerr << "Caught exception calling RD.step(): " << e.what() << endl;
            finished = true;
        }

        // Plot every 100 steps
        if (RD.stepCount % 100 == 0) {
            displays[0].resetDisplay (fix, eye, rot);
            try {
                RD.plot (displays);
            } catch (const exception& e) {
                cerr << "Caught exception calling M.plot(): " << e.what() << endl;
                //doing = false;
            }
        }


        if (RD.stepCount > T) {
            finished = true;
        }
    }

    // Save final state of RD system
    RD.saveState();

    return 0;
};
