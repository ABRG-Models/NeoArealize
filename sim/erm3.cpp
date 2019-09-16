/*
 * Run Ermentrout simulations inside each of the N different contours
 * determined by the Karbowski+competition simulation.
 */
#include <iostream>
#include <vector>
#include <string>

/*!
 * This will be passed as the template argument for RD_Plot and RD and
 * should be defined when compiling.
 */
#ifndef FLOATTYPE
// Check CMakeLists.txt to change to double or float
# error "Please define FLOATTYPE when compiling (hint: See CMakeLists.txt)"
#endif

#include "rd_erm2.h"
#include "morph/tools.h"
#include <json/json.h>

/*!
 * Choose whether to plot or not. Comment out to only compute. The
 * code could be changed so that the decision to plot or not was
 * selected via a command line argument.
 */
#define COMPILE_PLOTTING 1

#ifdef COMPILE_PLOTTING
// Include display and plotting code
# include "morph/display.h"
# include "morph/RD_Plot.h"
#endif

// for git processes
#include "morph/Process.h"
using morph::Process;
using morph::ProcessData;
using morph::ProcessCallbacks;

using namespace std;

int main (int argc, char **argv)
{
    // Randomly set the RNG seed
    srand (morph::Tools::randomSeed());

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " /path/to/params.json [/path/to/logdir]" << endl;
        return 1;
    }
    string paramsfile (argv[1]);

    // JSON setup
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
    // Parse the JSON
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

    /*
     * Get simulation-wide parameters from JSON
     */
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
    bool overwrite_logs = root.get ("overwrite_logs", false).asBool();
    string logpath = root.get ("logpath", "logs/erm2").asString();
    if (argc == 3) {
        string argpath(argv[2]);
        cerr << "Overriding the config-given logpath " << logpath << " with " << argpath << endl;
        logpath = argpath;
        if (overwrite_logs == true) {
            cerr << "WARNING: You set a command line log path.\n"
                 << "       : Note that the parameters config permits the program to OVERWRITE LOG\n"
                 << "       : FILES on each run (\"overwrite_logs\" is set to true)." << endl;
        }
    }

    const FLOATTYPE dt = static_cast<FLOATTYPE>(root.get ("dt", 0.00001).asDouble());
    const double Dn = root.get ("Dn", 0.3).asDouble();
    const double Dc = root.get ("Dc", 0.3*0.3).asDouble(); // 0.3 * Dn
    const double beta = root.get ("beta", 5.0).asDouble();
    const double a = root.get ("a", 1.0).asDouble();
    const double b = root.get ("b", 1.0).asDouble();
    const double mu = root.get ("mu", 1.0).asDouble();
    //const double chi = root.get ("chi", 0.3).asDouble(); // Dn

    cout << "steps to simulate: " << steps << endl;

#ifdef COMPILE_PLOTTING

    // Parameters from the config that apply only to plotting:
    const unsigned int plotevery = root.get ("plotevery", 10).asUInt();
    const bool vidframes = root.get ("vidframes", false).asBool();
    unsigned int framecount = 0;

    vector<morph::Gdisplay> displays;
    vector<double> fix(3, 0.0);
    vector<double> eye(3, 0.0);
    eye[2] = 0.3; // Acts as a zoom. +ve and larger to zoom out, negative and larger to zoom in.
    vector<double> rot(3, 0.0);

    // A plotting object.
    morph::RD_Plot<FLOATTYPE> plt(fix, eye, rot);

    double rhoInit = root.get ("rhoInit", 1.0).asDouble(); // This is effectively a zoom control. Increase to zoom out.
    const unsigned int win_width = root.get ("win_width", 340).asUInt();
    unsigned int win_height = static_cast<unsigned int>(0.8824f * (float)win_width);

    string winTitle = "n";
    displays.push_back (morph::Gdisplay (win_width, win_height, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();

    winTitle = "c";
    displays.push_back (morph::Gdisplay (win_width, win_height, 100, 0, winTitle.c_str(), rhoInit, 0.0, 0.0));
    displays.back().resetDisplay (fix, eye, rot);
    displays.back().redrawDisplay();
#endif

    // Load contours from contours.h5 here

    list<RD_Erm2<FLOATTYPE> > RDs;

    for each contour in contours:
    {
    // Instantiate the model object (list of RDs?)
    RD_Erm2<FLOATTYPE> RD;

    //RD.svgpath = svgpath; // not this
    RD.logpath = logpath;

    RD.set_dt(dt);
    RD.hextohex_d = hextohex_d;
    RD.boundaryFalloffDist = boundaryFalloffDist;

    RD.N = 1;
    RD.Dn = Dn;
    RD.Dc = Dc;
    RD.beta = beta;
    RD.a = a;
    RD.b = b;
    RD.mu = mu;
    // Set chi to Dn, as in the paper (see linear analysis)
    RD.chi = RD.Dn;

    // Allocate and initialise the model
    RD.allocate (contour); // Need to add this overload to base class.
    RD.init();

    // Now create a log directory if necessary, and exit on any failures.
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

    // Start the loop
    bool doing = true;
    while (doing) {
        // Step the model:
        // For each in list,
        RD.step();

#ifdef COMPILE_PLOTTING
        // Special plotting code...
#endif
         // Save data every 'logevery' steps
        if ((RD.stepCount % logevery) == 0) {
            cout << "Logging data at step " << RD.stepCount << endl;
            // For each in list,
            RD.saveState();
        }

        // After a while, stop:
        if (RD.stepCount > steps) {
            doing = false;
        }
    }

    // Before exit, save data
    // For each in list,
    RD.saveState();

    // Before saving the json, we'll place any additional useful info
    // in there, such as the FLOATTYPE. If float_width is 4, then
    // results were computed with single precision, if 8, then double
    // precision was used. Also save various parameters from the RD system.
    root["float_width"] = (unsigned int)sizeof(FLOATTYPE);
    string tnow = morph::Tools::timeNow();
    root["sim_ran_at_time"] = tnow.substr(0,tnow.size()-1);
    root["hextohex_d"] = RD.hextohex_d;
    //root["Dn"] = RD.Dn;
    root["dt"] = RD.get_dt();
    // Call our function to place git information into root.
    morph::Tools::insertGitInfo (root, "sim/");
    // Store the binary name and command argument into root, too.
    if (argc > 0) { root["argv0"] = argv[0]; }
    if (argc > 1) { root["argv1"] = argv[1]; }

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
    // Ask for a keypress before exiting so that the final images can be studied
    int key;
    cout << "Press any key[return] to exit.\n";
    cin >> key;
#endif // COMPILE_PLOTTING

    return 0;
};
