#include <iostream>
#include <vector>
#include <string>

/*!
 * This will be passed as the template argument for RD_plot and RD.
 */
#define FLOATTYPE double

/*!
 * The reaction diffusion computation class.
 */
#include "rd_2d_erm.h"

/*!
 * Choose whether to plot or not. Comment out to only compute. The
 * code could be changed so that the decision to plot or not was
 * selected via a command line argument.
 */
#define PLOT_SIM 1

#ifdef PLOT_SIM
/*!
 * Include display and plotting code
 */
# include "morph/display.h"
# include "rd_plot.h"
#endif

using namespace std;

int main (int argc, char **argv)
{
    if (argc < 3) {
        cerr << "\nUsage: " << argv[0] << " w0 Dn\n\n";
        cerr << "Be sure to run from the base source directory.\n";
        return -1;
    }
    string worldName(argv[1]);

#ifdef PLOT_SIM
    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    eye[2] = 0.3; // Acts as a zoom. +ve and larger to zoom out, negative and larger to zoom in.
    vector<double> rot(3, 0.0);

    // A plotting object.
    RD_plot<FLOATTYPE> plt(fix, eye, rot);

    double rhoInit = 1;
    string winTitle = worldName + ": n";
    displays.push_back (morph::Gdisplay (500, 500, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": c";
    displays.push_back (morph::Gdisplay (500, 500, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();
#endif

    // How long to run the sim:
    unsigned int maxSteps = 50000;

    // Instantiate the model object
    RD_2D_Erm<FLOATTYPE> M;

    // Modify any parameters before calling M.init()
    M.setLogpath (string("./logs/") + worldName);
    M.Dn = stod(argv[2]);
    M.chi = M.Dn;
    M.Dc = 0.3*M.Dn;
    M.N = 1;

    // Initialise the model
    M.init();

    // Start the loop
    bool doing = true;
    while (doing) {
        // Step the model:
        M.step();
#ifdef PLOT_SIM
        // Plot every 100 steps:
        if (M.stepCount % 100 == 0) {
            plt.scalarfields (displays[0], M.hg, M.n);
            plt.scalarfields (displays[1], M.hg, M.c);

#ifdef SAVE_PNGS // Or maybe put something into rd_plot.h?
            std::stringstream frameFile1;
            frameFile1<<"logs/tmp/demo";
            frameFile1<<setw(5)<<setfill('0')<<frameN;
            frameFile1<<".png";
            disps[0].saveImage(frameFile1.str());
            frameN++;
#endif

        }
#endif
        // After a while, stop:
        if (M.stepCount > maxSteps) {
            doing = false;
        }
    }

    // Before exit, save data
    M.saveState();

    return 0;
};
