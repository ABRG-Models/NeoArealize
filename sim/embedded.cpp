/*!
 * Wolf models embedded in contours determined by Karbowski model.
 */
#define MANUFACTURE_GUIDANCE_MOLECULES 1
#include "rd_2d_karb.h"
#include "rd_orientpref.h"

#include "morph/display.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

void plot_all (morph::Gdisplay& disp, RD_2D_Karb& rdkarb, array<list<Hex>, 5> ctrs, RD_OrientPref& rda, RD_OrientPref& rdb, RD_OrientPref& rdc)
{
    disp.resetDisplay (vector<double>(3, 0.0), vector<double>(3, 0.0), vector<double>(3, 0.0));
    //rdkarb.add_contour_plot (rdkarb.c, disp, rdkarb.contour_threshold);
    rdkarb.add_contour_plot (ctrs, disp);
    rda.add_to_plot (disp);
    rdb.add_to_plot (disp);
    rdc.add_to_plot (disp);
    disp.redrawDisplay();

    // Save pngs
    stringstream ff3;
    ff3 << rdkarb.logpath << "/cntr_";
    ff3 << std::setw(5) << std::setfill('0') << rdkarb.frameN;
    ff3 << ".png";
    disp.saveImage (ff3.str());
    rdkarb.frameN++;
}

int main (int argc, char **argv)
{
    if (argc < 2) {
        cerr << "\nUsage: ./build/sim/process w0\n\n";
        cerr << "Be sure to run from the base NeoArealize source directory.\n";
        return -1;
    }

    // Set RNG seed
    int rseed = 1;
    srand(rseed);

    // Create some displays
    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    eye[2] = -0.4;
    vector<double> rot(3, 0.0);

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

    // Final Contours
    winTitle = worldName + ": final contours";
    displays.push_back (morph::Gdisplay (500, 500, 100, 900, winTitle.c_str(), rhoInit, 0.0, 0.0, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    // Instantiate the model object
    RD_2D_Karb RD;
    RD.contour_threshold = 0.6;

    try {
        RD.init (displays);
    } catch (const exception& e) {
        cerr << "Exception initialising RD_2D_Karb object: " << e.what() << endl;
    }

    // Start the loop
    unsigned int maxSteps = 200;
    bool finished = false;
    while (finished == false) {
        // Step the model
        try {
            RD.step();
        } catch (const exception& e) {
            cerr << "Caught exception calling RD.step(): " << e.what() << endl;
            finished = true;
        }

#if 1
        try {

            if (RD.stepCount % 1 == 0) {
                RD.plot (displays, true); // true for savePngs
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

    array<list<Hex>, 5> ctrs = RD.get_contours (RD.contour_threshold);
    // Do a final plot of the ctrs as found.
    RD.plot_contour (ctrs, displays[5]);

    RD_OrientPref RD2_0;
    RD2_0.eta = 0.000001;
    RD_OrientPref RD2_2;
    RD2_2.eta = 0.0000002;
    RD_OrientPref RD2_4;
    RD2_4.eta = 0.0000005;
    try {
        RD2_0.init (ctrs[0]);
        RD2_2.init (ctrs[2]);
        RD2_4.init (ctrs[4]);
    } catch (const exception& e) {
        cerr << "Exception initialising RD_OreintPref object: " << e.what() << endl;
        return -1;
    }

    maxSteps = 65000;
    finished = false;
    while (!finished) {
        // Step the models
        try {
            RD2_0.step();
            RD2_2.step();
            RD2_4.step();
        } catch (const exception& e) {
            cerr << "Caught exception calling RD.step(): " << e.what() << endl;
            finished = true;
        }

        // Plot every 100 steps
        array<list<Hex>, 5> ctrs2;
        if (RD2_0.stepCount % 50 == 0) {
            displays[4].resetDisplay (fix, eye, rot);
            try {
                plot_all (displays[4], RD, ctrs, RD2_0, RD2_2, RD2_4);
            } catch (const exception& e) {
                cerr << "Caught exception calling plot(): " << e.what() << endl;
            }
        }


        if (RD2_0.stepCount > maxSteps) {
            finished = true;
        }
    }

    // Save final state of RD system
    RD2_0.saveState();

    int a;
    cout << "Press any key[return] to make movie.\n";
    cin >> a;

    // Make movie
    string cmd = "ffmpeg -i " + RD.logpath + "/cntr_%05d.png -c:v libx264 -pix_fmt yuv420p " + RD.logpath + "/cntr.mp4";
    system (cmd.c_str());
    cmd = "rm " + RD.logpath + "/[ac]_*.png " + RD.logpath + "/cntr_*.png";
    system (cmd.c_str());

    return 0;
};
