#include "rd_2d_erm.h"

#include "morph/display.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main (int argc, char **argv)
{
    if (argc < 3) {
        cerr << "\nUsage: " << argv[0] << " w0 Dn\n\n";
        cerr << "Be sure to run from the base source directory.\n";
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

    rhoInit = 1.4;
    winTitle = worldName + ": c";
    displays.push_back (morph::Gdisplay (500, 500, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    // How long to run the sim:
    unsigned int maxSteps = 50000;

    // Instantiate the model object
    RD_2D_Erm M;

    // Modify any parameters before calling M.init()
    M.setLogpath (string("./logs/") + worldName);
    M.Dn = stod(argv[2]);
    M.chi = M.Dn;
    M.Dc = 0.3*M.Dn;
    M.N = 3; // For three chemo attractant molecules (i.e. not using
             // this in the sense of the original Ermentrout system)

    try {
        M.init (displays);
    } catch (const exception& e) {
        cerr << "Exception initialising RD_2D_Karb object: " << e.what() << endl;
    }

    // Start the loop
    bool doing = true;
    while (doing) {

        try {
            // Step the model:
            M.step();
            // Plot every 100 steps:
            if (M.stepCount % 100 == 0) {
                displays[0].resetDisplay (fix, eye, rot);
                M.plot (displays);
            }
            // After a while, stop:
            if (M.stepCount > maxSteps) {
                doing = false;
            }

        } catch (const exception& e) {
            cerr << "Caught exception: " << e.what() << endl;
            doing = false;
        }
    }

    // Before exit, save data
    M.saveState();

    return 0;
};
