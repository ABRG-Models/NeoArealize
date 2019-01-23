#define N_TC 2
#include "rd_james.h"

#include <iostream>
#include <vector>
#include <string>

// Choose whether to plot or not.
#define PLOT_STUFF 1

#ifdef PLOT_STUFF
#include "morph/display.h"
#include "rd_plot.h"
#endif

using namespace std;

// This will be passed as the template argument for RD_plot and RD.
#define FLOATTYPE double

int main (int argc, char **argv)
{
    if (argc < 2) {
        cerr << "\nUsage: " << argv[0] << " w0\n\n";
        cerr << "Be sure to run from the base NeoArealize source directory.\n";
        return -1;
    }

    // Set RNG seed
    int rseed = 1;
    srand(rseed);

#ifdef PLOT_STUFF
    // Create some displays
    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    eye[2] = -0.4;
    vector<double> rot(3, 0.0);

    // A plot object.
    RD_plot<FLOATTYPE> plt(fix, eye, rot);

    double rhoInit = 1.5;
    string worldName(argv[1]);

    string winTitle = worldName + ": rhoA_rhoB_rhoC";
    displays.push_back (morph::Gdisplay (1020, 300, 100, 300, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": a[0] to a[N]";
    displays.push_back (morph::Gdisplay (340*N_TC, 300, 100, 900, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": c[0] to c[N]";
    displays.push_back (morph::Gdisplay (340*N_TC, 300, 100, 900, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    // SW - Contours
    winTitle = worldName + ": contours";
    displays.push_back (morph::Gdisplay (500, 500, 100, 900, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    // Final Contours
    winTitle = worldName + ": final contours";
    displays.push_back (morph::Gdisplay (500, 500, 100, 900, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();
#endif

    // Instantiate the model object
    RD_James<FLOATTYPE> RD;
    RD.N = 3; // Number of TC populations
    RD.M = 1; // Number of guidance molecules that are sculpted

    // Choose and parameterise the guidance molecules
    RD.rhoMethod = GuidanceMoleculeMethod::Sigmoid1D;
    // Set up guidance molecule method parameters
    RD.guidance_gain.push_back (1.0);
    RD.guidance_phi.push_back (0.0);
    RD.guidance_width.push_back (1.0);
    RD.guidance_offset.push_back (0.5);

    try {
        RD.init();
#ifdef PLOT_STUFF
        plt.scalarfields (displays[1], RD.hg, RD.rho);
        // Save pngs of the factors and guidance expressions.
        string logpath = "logs";
        displays[0].saveImage (logpath + "/guidance.png");
#endif
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

#ifdef PLOT_STUFF
        try {
            if (RD.stepCount % 10 == 0) {
                // Do a final plot of the ctrs as found.
                vector<list<Hex> > ctrs = plt.get_contours (RD.hg, RD.c, RD.contour_threshold);
                plt.plot_contour (displays[3], RD.hg, ctrs);
                plt.scalarfields (displays[1], RD.hg, RD.a);
                plt.scalarfields (displays[2], RD.hg, RD.c);
                // If required:
                //RD.save();
            }
            // Save some frames ('c' variable only for now)
            if (RD.stepCount % 100 == 0) {
                // Once template has been successfully specialised:
                RD.saveC();
            }

        } catch (const exception& e) {
            cerr << "Caught exception calling RD.plot(): " << e.what() << endl;
            finished = true;
        }
#endif
        if (RD.stepCount > maxSteps) {
            finished = true;
        }
    }

#ifdef PLOT_STUFF
    // Extract contours
    vector<list<Hex> > ctrs = RD.get_contours (0.6);
    // Create new HexGrids from the contours
    HexGrid* hg1 = new HexGrid (RD.hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
    hg1->setBoundary (ctrs[0]);
    HexGrid* hg2 = new HexGrid (RD.hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
    hg2->setBoundary (ctrs[1]);
    // Do a final plot of the ctrs as found.
    plt.plot_contour (displays[4], RD.hg, ctrs);
    // Output information about the contours
    cout << "Sizes: countour 0: " << hg1->num() << ", contour 1: " << hg2->num() << endl;
    cout << "Ratio: " << ((double)hg1->num()/(double)hg2->num()) << endl;
    // Allow for a keypress so that images can be studied
    int a;
    cout << "Press any key[return] to exit.\n";
    cin >> a;
#endif

    return 0;
};
