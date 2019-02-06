/*
 * This program is intended to facilite the study of a reaction
 * diffusion system which is guided by M guidance molecules, whose
 * expression gradients drive N thalamocortical axon types to make
 * connections in an elliptical region.
 */

/*!
 * This will be passed as the template argument for RD_plot and RD.
 */
#ifndef FLOATTYPE
// NB: This is just the default. Check CMakeLists.txt to try double vs. float
# define FLOATTYPE float
#endif

/*!
 * General STL includes
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <string>
#include <limits>

/*!
 * for clock() and getpid()
 */
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

/*!
 * Include the reaction diffusion class
 */
#include "rd_james.h"

#ifdef COMPILE_PLOTTING
/*!
 * Include display and plotting code
 */
# include "morph/display.h"
# include "rd_plot.h"
#endif

/*!
 * Included for directory manipulation code
 */
#include "morph/tools.h"

/*!
 * I'm using JSON to read in simulation parameters
 */
#include <json/json.h>

using namespace std;

/*!
 * For mixing up bits of three args; used to generate a good random
 * seed using time() getpid() and clock().
 */
unsigned int
mix (unsigned int a, unsigned int b, unsigned int c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

/*!
 * main(): Run a simulation, using parameters obtained from a JSON
 * file.
 *
 * Open and read a simple JSON file which contains the parameters for
 * the simulation, such as number of guidance molecules (M), guidance
 * parameters (probably get M from these) and so on.
 *
 * Sample JSON:
 * {
 *   // Overall parameters
 *   "steps":5000,                // Number of steps to simulate for
 *   "logevery":20,               // Log data every logevery steps.
 *   "svgpath":"./ellipse.svg",   // The boundary shape to use
 *   "hextohex_d":0.01,           // Hex to hex distance, determines num hexes
 *   "boundaryFalloffDist":0.01,
 *   "D":0.1,                     // Global diffusion constant
 *   // Array of parameters for N thalamocortical populations:
 *   "tc": [
 *     // The first TC population
 *     {
 *       "alpha":3,
 *       "beta":3
 *     },
 *     // The next TC population
 *     {
 *       "alpha":3,
 *       "beta":3
 *     } // and so on.
 *   ],
 *   // Array of parameters for the guidance molecules
 *   "guidance": [
 *     {
 *       "shape":"Sigmoid1D", // and so on
 *       "gain":0.5,
 *       "phi":0.8,
 *       "width":0.1,
 *       "offset":0.0,
 *     }
 *   ]
 * }
 *
 * A file containing JSON similar to the above should be saved and its
 * path provided as the only argument to the binary here.
 */
int main (int argc, char **argv)
{
    // Randomly set the RNG seed
    unsigned int rseed = mix(clock(), time(NULL), getpid());
    srand (rseed);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " /path/to/params.json [/path/to/logdir]" << endl;
        return 1;
    }
    string paramsfile (argv[1]);

    // Set up JSON code for reading the parameters

    // Test for existence of the file.
    ifstream jsonfile_test;
    int srtn = system ("pwd");
    if (srtn) {
        cerr << "system call returned " << srtn << endl;
    }
    jsonfile_test.open (paramsfile, ios::in);
    if (jsonfile_test.is_open()) {
        // Good, file exists.
        jsonfile_test.close();
    } else {
        cerr << "json config file " << paramsfile << " not found." << endl;
        return 1;
    }

    ifstream jsonfile (paramsfile, ifstream::binary);

    Json::Value root;
    string errs;
    Json::CharReaderBuilder rbuilder;
    rbuilder["collectComments"] = false;

    bool parsingSuccessful = Json::parseFromStream (rbuilder, jsonfile, &root, &errs);
    if (!parsingSuccessful) {
        // report to the user the failure and their locations in the document.
        cerr << "Failed to parse JSON: " << errs;
        return 1;
    }

    // Get simulation-wide parameters
    const unsigned int steps = root.get ("steps", 1000).asUInt();
    if (steps == 0) {
        cerr << "Not much point simulating 0 steps! Exiting." << endl;
        return 1;
    }
    const unsigned int logevery = root.get ("logevery", 100).asUInt();
    if (logevery == 0) {
        cerr << "Can't log every 0 steps. Exiting." << endl;
        return 1;
    }
    const float hextohex_d = root.get ("hextohex_d", 0.01).asFloat();
    const float boundaryFalloffDist = root.get ("boundaryFalloffDist", 0.01).asFloat();
    const string svgpath = root.get ("svgpath", "./ellipse.svg").asString();
    string logpath = root.get ("logpath", "logs/james1").asString();
    if (argc == 3) {
        string argpath(argv[2]);
        cerr << "Overriding the config-given logpath " << logpath << " with " << argpath << endl;
        logpath = argpath;
    }

    const double D = root.get ("D", 0.1).asDouble();
    const FLOATTYPE contour_threshold = root.get ("contour_threshold", 0.6).asDouble();
    const FLOATTYPE k = root.get ("k", 3).asDouble();

    cout << "steps to simulate: " << steps << endl;

    // Thalamocortical populations array of parameters:
    const Json::Value tcs = root["tc"];
    unsigned int N_TC = static_cast<unsigned int>(tcs.size());
    if (N_TC == 0) {
        cerr << "Zero thalamocortical populations makes no sense for this simulation. Exiting."
             << endl;
        return 1;
    }

    // Guidance molecule array of parameters:
    const Json::Value guid = root["guidance"];
    unsigned int M_GUID = static_cast<unsigned int>(guid.size());

#ifdef COMPILE_PLOTTING
    // Create some displays
    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    eye[2] = 0.12; // This also acts as a zoom. more +ve to zoom out, more -ve to zoom in.
    vector<double> rot(3, 0.0);

    // A plot object.
    RD_plot<FLOATTYPE> plt(fix, eye, rot);

    double rhoInit = 1; // This is effectively a zoom control. Increase to zoom out.
    double thetaInit = 0.0;
    double phiInit = 0.0;

    string worldName("j");

    string winTitle = worldName + ": Guidance molecules";
    displays.push_back (morph::Gdisplay (340 * (M_GUID>0?M_GUID:1), 300, 100, 300, winTitle.c_str(), rhoInit, thetaInit, phiInit));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": a[0] to a[N]";
    displays.push_back (morph::Gdisplay (340*N_TC, 300, 100, 900, winTitle.c_str(), rhoInit, thetaInit, phiInit, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": c[0] to c[N]";
    displays.push_back (morph::Gdisplay (340*N_TC, 300, 100, 1200, winTitle.c_str(), rhoInit, thetaInit, phiInit, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    // SW - Contours
    winTitle = worldName + ": contours (from c)";
    displays.push_back (morph::Gdisplay (360, 300, 100, 1500, winTitle.c_str(), rhoInit, thetaInit, phiInit, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": Guidance gradient (x)";
    displays.push_back (morph::Gdisplay (340*N_TC, 300, 100, 1800, winTitle.c_str(), rhoInit, thetaInit, phiInit, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": Guidance gradient (y)";
    displays.push_back (morph::Gdisplay (340*N_TC, 300, 100, 1800, winTitle.c_str(), rhoInit, thetaInit, phiInit, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = worldName + ": n";
    displays.push_back (morph::Gdisplay (340, 300, 100, 1800, winTitle.c_str(), rhoInit, thetaInit, phiInit, displays[0].win));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

#endif

    // Instantiate the model object
    RD_James<FLOATTYPE> RD;

    RD.svgpath = svgpath;
    RD.logpath = logpath;

    // NB: Set .N, .M BEFORE RD.allocate().
    RD.N = N_TC; // Number of TC populations
    RD.M = M_GUID; // Number of guidance molecules that are sculpted

    // Control the size of the hexes, and therefore the number of hexes in the grid
    RD.hextohex_d = hextohex_d;

    // Boundary fall-off distance
    RD.boundaryFalloffDist = boundaryFalloffDist;

    // After setting N and M, we can set up all the vectors in RD:
    RD.allocate();

    // After allocate(), we can set up parameters:
    RD.set_D (D);
    RD.contour_threshold = contour_threshold;
    RD.k = k;

    // Index through thalamocortical fields, setting params:
    for (unsigned int i = 0; i < tcs.size(); ++i) {
        Json::Value v = tcs[i];
        RD.alpha[i] = v.get("alpha", 0.0).asDouble();
        RD.beta[i] = v.get("beta", 0.0).asDouble();
    }

    // Index through guidance molecule parameters:
    for (unsigned int j = 0; j < guid.size(); ++j) {
        Json::Value v = guid[j];
        // What guidance molecule method will we use?
        string rmeth = v.get ("shape", "Sigmoid1D").asString();
        if (rmeth == "Sigmoid1D") {
            RD.rhoMethod = GuidanceMoleculeMethod::Sigmoid1D;
        } else if (rmeth == "Exponential1D") {
            RD.rhoMethod = GuidanceMoleculeMethod::Exponential1D;
        } else if (rmeth == "Gauss1D") {
            RD.rhoMethod = GuidanceMoleculeMethod::Gauss1D;
        } else if (rmeth == "Gauss2D") {
            RD.rhoMethod = GuidanceMoleculeMethod::Gauss2D;
        } else {
            RD.rhoMethod = GuidanceMoleculeMethod::Sigmoid1D;
        }
        // Set up guidance molecule method parameters
        RD.guidance_gain.push_back (v.get("gain", 1.0).asDouble());
        RD.guidance_phi.push_back (v.get("phi", 1.0).asDouble());
        RD.guidance_width.push_back (v.get("width", 1.0).asDouble());
        RD.guidance_offset.push_back (v.get("offset", 1.0).asDouble());
    }

    // Set up the interaction parameters between the different TC
    // populations and the guidance molecules (aka gamma).
    int paramRtn = 0;
    for (unsigned int i = 0; i < tcs.size(); ++i) {
        Json::Value tcv = tcs[i];
        Json::Value gamma = tcv["gamma"];
        for (unsigned int j = 0; j < guid.size(); ++j) {
            // Set up gamma values using a setter which checks we
            // don't set a value that's off the end of the gamma
            // container.
            cout << "Set gamma for guidance " << j << " over TC " << i << " = " << gamma[j] << endl;
            paramRtn += RD.setGamma (j, i, gamma[j].asDouble());
        }
    }

    if (paramRtn && M_GUID>0) {
        cerr << "Something went wrong setting gamma values" << endl;
        return paramRtn;
    }

    // Now have the guidance molecule densities and their gradients computed:
    RD.init();

    // Now is the time to create a log directory if necessary, and exit on any failures.
    if (morph::Tools::dirExists (logpath) == false) {
        morph::Tools::createDir (logpath);
        if (morph::Tools::dirExists (logpath) == false) {
            cerr << "Failed to create the logpath directory "
                 << logpath << " which does not exist."<< endl;
            return 1;
        }
    } else {
        // Directory DOES exist. See if it contains a previous run and
        // exit without overwriting to avoid confusion.
        if (morph::Tools::fileExists (logpath + "/params.json") == true
            || morph::Tools::fileExists (logpath + "/guidance.h5") == true
            || morph::Tools::fileExists (logpath + "/positions.h5") == true) {
            cerr << "Seems like a previous simulation was logged in " << logpath
                 << ". Please clean it out manually or choose another directory." << endl;
            return 1;
        }
    }

    // As RD.allocate() as been called, positions can be saved to file.
    RD.savePositions();
    // Save the guidance molecules now.
    RD.saveGuidance();

#ifdef COMPILE_PLOTTING
    plt.scalarfields (displays[0], RD.hg, RD.rho);
    vector<vector<FLOATTYPE> > gx = plt.separateVectorField (RD.g, 0);
    vector<vector<FLOATTYPE> > gy = plt.separateVectorField (RD.g, 1);
    plt.scalarfields (displays[4], RD.hg, gx);
    plt.scalarfields (displays[5], RD.hg, gy);
#endif

    // Start the loop
    bool finished = false;
    while (finished == false) {
        // Step the model
        RD.step();

#ifdef COMPILE_PLOTTING
        if (RD.stepCount % 10 == 0) {
            // Do a final plot of the ctrs as found.
            vector<list<Hex> > ctrs = plt.get_contours (RD.hg, RD.c, RD.contour_threshold);
            plt.plot_contour (displays[3], RD.hg, ctrs);
            plt.scalarfields (displays[1], RD.hg, RD.a);
            plt.scalarfields (displays[2], RD.hg, RD.c);
            plt.scalarfields (displays[6], RD.hg, RD.n);
            displays[4].redrawDisplay();
            displays[5].redrawDisplay();
        }
#endif
        // Save data every 'logevery' steps
        if ((RD.stepCount % logevery) == 0) {
            RD.save();
        }

        if (RD.stepCount > steps) {
            finished = true;
        }
    }

    // Before saving the json, we'll place any additional useful info
    // in there, such as the FLOATTYPE. If float_width is 4, then
    // results were computed with single precision, if 8, then double
    // precision was used.
    root["float_width"] = sizeof(FLOATTYPE);
    root["sim_ran_at_time"] = morph::Tools::timeNow();

    // We'll save a copy of the parameters for the simulation in the log directory as params.json
    const string paramsCopy = logpath + "/params.json";
    ofstream paramsConf;
    paramsConf.open (paramsCopy.c_str(), ios::out|ios::trunc);
    if (paramsConf.is_open()) {
        paramsConf << root;
        paramsConf.close();
    } else {
        cerr << "Warning: Failed to open file to write a copy of the params.json" << endl;
    }

#ifdef COMPILE_PLOTTING
# if 0
    // Extract contours
    vector<list<Hex> > ctrs = RD.get_contours (0.6);
    // Create new HexGrids from the contours
    cout << "Creating final hexgrids..." << endl;
    HexGrid* hg1 = new HexGrid (RD.hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
    hg1->setBoundary (ctrs[0]);
    HexGrid* hg2 = new HexGrid (RD.hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
    hg2->setBoundary (ctrs[1]);
    // Do a final plot of the ctrs as found.
    //plt.plot_contour (displays[4], RD.hg, ctrs);
    // Output information about the contours
    cout << "Sizes: countour 0: " << hg1->num() << ", contour 1: " << hg2->num() << endl;
    cout << "Ratio: " << ((double)hg1->num()/(double)hg2->num()) << endl;
# endif
    // Allow for a keypress so that images can be studied
    int a;
    cout << "Press any key[return] to exit.\n";
    cin >> a;
#endif

    return 0;
};
