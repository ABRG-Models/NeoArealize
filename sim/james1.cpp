#define N_TC 2
#include "rd_james.h"

#include "morph/display.h"
#include <iostream>
#include <vector>
#include <string>

// Choose whether to plot or not.
//#define PLOT_STUFF 1

#if defined PLOT_STUFF
#include "rd_plot.h"
#endif

using namespace std;

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

#if defined PLOT_STUFF
    // Create some displays
    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    eye[2] = -0.4;
    vector<double> rot(3, 0.0);

    // A plot object.
    RD_plot plt(fix, eye, rot);

    double rhoInit = 1.5;
    string worldName(argv[1]);
    string winTitle = worldName + ": emx_pax_fgf";
    displays.push_back (morph::Gdisplay (1020, 300, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": rhoA_rhoB_rhoC";
    displays.push_back (morph::Gdisplay (1020, 300, 100, 300, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
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
    RD_James RD;
    RD.N = 2; // Number of TC populations
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
#if defined PLOT_STUFF
        plt.plot_chemo (displays, RD.chemo);
        // Save pngs of the factors and guidance expressions.
        displays[0].saveImage (this->logpath + "/factors.png");
        displays[1].saveImage (this->logpath + "/guidance.png");
#endif
    } catch (const exception& e) {
        cerr << "Exception initialising RD_2D_Karb object: " << e.what() << endl;
    }

    // A threshold chosen for defining contours in the RD system. SHould be parameter of RD? Probably.
    double contour_threshold = 0.5;

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

#if defined PLOT_STUFF
        try {
            displays[0].resetDisplay (fix, eye, rot);
            if (RD.stepCount % 10 == 0) {
                // Do a final plot of the ctrs as found.
                vector<list<Hex> > ctrs = plt.get_contours (RD.hg, RD.c, contour_threshold);
                plt.plot_contour (displays[4], RD.hg, ctrs);
                plt.scalarfields (RD.hg, RD.a, displays[2]);
                plt.scalarfields (RD.hg, RD.c, displays[3]);
                // If required:
                //RD.save();
            }
            // Save some frames ('c' variable only for now)
            if (RD.stepCount % 100 == 0) {
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

#if defined PLOT_STUFF
    vector<list<Hex> > ctrs = RD.get_contours (0.6);

    // Create a HexGrid
    HexGrid* hg1 = new HexGrid (RD.hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
    hg1->setBoundary (ctrs[0]);
    HexGrid* hg2 = new HexGrid (RD.hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
    hg2->setBoundary (ctrs[1]);

    // Do a final plot of the ctrs as found.
    plt.plot_contour (displays[5], RD.hg, ctrs);

    cout << "Sizes: countour 0: " << hg1->num() << ", contour 1: " << hg2->num() << endl;
    cout << "Ratio: " << ((double)hg1->num()/(double)hg2->num()) << endl;
    int a;
    cout << "Press any key[return] to exit.\n";
    cin >> a;
#endif

    return 0;
};
