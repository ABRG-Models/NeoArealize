/*
 * Reproduce approximate behaviour of Fig 6 of the Karbowski 2004
 * paper in our 2D system of arbitrary boundary shape. Pax6 turned off.
 */

#include "rd_2d_karb.h"

#include "morph/display.h"
#include "morph/tools.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main (void)
{
    // Set RNG seed
    int rseed = 1;
    srand(rseed);

    string worldName("2dkarb_f6_pax6_off");

    // Create some displays
    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    eye[2] = -0.4;
    vector<double> rot(3, 0.0);

    double rhoInit = 1.5;
    string winTitle = worldName + ": emx_pax_fgf";
    displays.push_back (morph::Gdisplay (1020, 300, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": rhoA_rhoB_rhoC";
    displays.push_back (morph::Gdisplay (1020, 300, 100, 300, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": a[0] to a[4]";
    displays.push_back (morph::Gdisplay (1700, 300, 100, 600, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": c[0] to c[4]";
    displays.push_back (morph::Gdisplay (1700, 300, 100, 900, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    // SW - Contours
    winTitle = worldName + ": contours";
    displays.push_back (morph::Gdisplay (500, 500, 100, 900, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    // Save PNGs and turn them into movies?
    bool makeMovies = true;

    // How many frames to simulate before finishing.
    unsigned int endFrame = 1500;

    // Instantiate the model object
    RD_2D_Karb M;
    M.rhoMethod = GuidanceMoleculeMethod::KarbowskiOriginal;

    // Turn pax6 off like this:
    M.Apax = 0;

    // Set the log path (also creates that directory if required)
    M.setLogpath ("logs/" + worldName);
    // Initialise variables and do any pre-computation
    try {
        M.init (displays);
    } catch (const exception& e) {
        cerr << "Exception initialising RD_2D_Karb object: " << e.what() << endl;
    }

    // Start the loop
    bool finished = false;
    while (!finished) {
        // Step the model
        try {
            M.step();
        } catch (const exception& e) {
            cerr << "Caught exception calling M.step(): " << e.what() << endl;
            finished = true;
        }

        displays[0].resetDisplay (fix, eye, rot);
        try {
            M.plot (displays, makeMovies);
            // Save some frames ('c' variable only for now)
            // FIXME: Think about this.
            //if (M.stepCount % 100 == 0) {
            //    M.saveC();
            //}

        } catch (const exception& e) {
            cerr << "Caught exception calling M.plot(): " << e.what() << endl;
            finished = true;
        }

        if (M.stepCount > endFrame) {
            //cout << "Press any key (then return) to make the movies." << endl;
            //int a;
            //cin >> a;
            finished = true;
        }
    }

    // Last job is to make movies and erase pngs
    if (makeMovies == true) {
        string cmd = "ffmpeg -i " + M.logpath + "/c_%05d.png -c:v libx264 -pix_fmt yuv420p " + M.logpath + "/c.mp4";
        system (cmd.c_str());
        cmd = "ffmpeg -i " + M.logpath + "/a_%05d.png -c:v libx264 -pix_fmt yuv420p " + M.logpath + "/a.mp4";
        system (cmd.c_str());
        cmd = "ffmpeg -i " + M.logpath + "/cntr_%05d.png -c:v libx264 -pix_fmt yuv420p " + M.logpath + "/cntr.mp4";
        system (cmd.c_str());
        cmd = "rm " + M.logpath + "/[ac]_*.png " + M.logpath + "/cntr_*.png";
        system (cmd.c_str());
    }

    return 0;
};
