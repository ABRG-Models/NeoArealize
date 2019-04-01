/*
 * Scan parameter space. Vary parameters within the C++ program
 * because that avoids having to re-parse the HexGrid each time. Write
 * data out into log directories. python can then process each log
 * directory in a post-processing step.
 */

/*!
 * This will be passed as the template argument for RD_plot and RD and
 * should be defined when compiling.
 */
#ifndef FLOATTYPE
// Check CMakeLists.txt to change to double or float
# error "Please define FLOATTYPE when compiling (hint: See CMakeLists.txt)"
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
 * Include the reaction diffusion class
 */
#include "rd_james.h"

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
 * main(): Run many simulations, using parameters obtained from a JSON
 * file.
 */
int main (int argc, char **argv)
{
    // Randomly set the RNG seed
    srand (morph::Tools::randomSeed());

    /*
     * Set simulation-wide parameters
     */
    const unsigned int steps = 5000;
    const unsigned int logevery = 5000;
    const float hextohex_d = 0.01;
    const float boundaryFalloffDist = 0.01;
    const string svgpath = "./ellipse.svg";
    bool overwrite_logs = true;
    string logpath = "";
    const double D = 0.1;
    unsigned int N_TC = 2;
    unsigned int M_GUID = 0;

    /*!
     * Variable parameters. Allow command line to specify range?
     */
    FLOATTYPE klo = 2.0;
    FLOATTYPE khi = 4.0;
    FLOATTYPE kinc = 0.1;

    FLOATTYPE alo = 2.0;
    FLOATTYPE ahi = 5.0;
    FLOATTYPE ainc = 0.15;

    FLOATTYPE blo = 2.0;
    FLOATTYPE bhi = 5.0;
    FLOATTYPE binc = 0.15;

    /*
     * Instantiate and set up the model object
     */
    RD_James<FLOATTYPE> RD;

    RD.svgpath = svgpath;

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

    //RD.contour_threshold = contour_threshold;

    unsigned int scannum = 0;
    for (FLOATTYPE kk = klo; kk < khi; kk += kinc) {
        for (FLOATTYPE aa = alo; aa < ahi; aa += ainc) {
            for (FLOATTYPE bb = blo; bb < bhi; bb += binc) {

                RD.k = kk;

                stringstream ss;
                ss << "logs/ps_2N0M_" << scannum++;
                logpath = ss.str();
                RD.logpath = logpath;

                // Index through thalamocortical fields, setting params:
                for (unsigned int i = 0; i < N_TC; ++i) {
                    RD.alpha[i] = aa;
                    RD.beta[i] = bb;
                }

                // Calling init() resets the various variables to 0 or initial noise
                RD.init();

                /*
                 * Now create a log directory if necessary, and exit on any
                 * failures.
                 */
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
                    if (overwrite_logs == false
                        && (morph::Tools::fileExists (logpath + "/params.json") == true
                            || morph::Tools::fileExists (logpath + "/guidance.h5") == true
                            || morph::Tools::fileExists (logpath + "/positions.h5") == true)) {
                        cerr << "Seems like a previous simulation was logged in " << logpath << ".\n"
                             << "Please clean it out manually, choose another directory or set\n"
                             << "overwrite_logs to true in your parameters config JSON file." << endl;
                        return 1;
                    }
                }

                // As RD.allocate() as been called (and log directory has been
                // created/verified ready), positions can be saved to file.
                RD.savePositions();
                // Save the guidance molecules now.
                RD.saveGuidance();

                // Start the loop
                bool finished = false;
                while (finished == false) {
                    // Step the model
                    RD.step();

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
                // precision was used. Also save various parameters from the RD system.
                Json::Value root;
                root["float_width"] = (unsigned int)sizeof(FLOATTYPE);
                string tnow = morph::Tools::timeNow();
                root["sim_ran_at_time"] = tnow.substr(0,tnow.size()-1);
                root["hextohex_d"] = RD.hextohex_d;
                root["D"] = RD.get_D();
                root["k"] = RD.k;
                root["dt"] = RD.get_dt();
                root["alpha"] = aa;
                root["beta"] = bb;

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

            } // end loops through aa, bb, kk
        }
    }

    cout << "Completed " << scannum << " scans.";

    return 0;
};
