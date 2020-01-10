/*
 * This is a version of erm2 which will use HexGridVisuals and morph::Visual for
 * realtime visualization.
 */
#include <iostream>
#include <vector>
#include <string>

/*!
 * This will be passed as the template argument for RD_Plot and RD and
 * should be defined when compiling.
 */
#ifndef FLT
// Check CMakeLists.txt to change to double or float
# error "Please define FLT when compiling (hint: See CMakeLists.txt)"
#endif

#include "rd_erm2.h"
#include "morph/tools.h"
#include "morph/Config.h"

#include <morph/MathAlgo.h>
using morph::MathAlgo;

/*!
 * Choose whether to plot or not. Comment out to only compute. The
 * code could be changed so that the decision to plot or not was
 * selected via a command line argument.
 */
#define COMPILE_PLOTTING 1

#ifdef COMPILE_PLOTTING
# include "morph/Visual.h"
using morph::Visual;
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

    // Set up a morph::Config object for reading configuration
    morph::Config conf(paramsfile);
    if (!conf.ready) {
        cerr << "Error setting up JSON config: " << conf.emsg << endl;
        return 1;
    }

    /*
     * Get simulation-wide parameters from JSON
     */
    const unsigned int steps = conf.getUInt ("steps", 1000);
    if (steps == 0) {
        cerr << "Not much point simulating 0 steps! Exiting." << endl;
        return 1;
    }
    const unsigned int logevery = conf.getUInt ("logevery", 100UL);
    if (logevery == 0) {
        cerr << "Can't log every 0 steps. Exiting." << endl;
        return 1;
    }
    const float hextohex_d = conf.getFloat ("hextohex_d", 0.01f);
    const float boundaryFalloffDist = conf.getFloat ("boundaryFalloffDist", 0.01f);

    const string svgpath = conf.getString ("svgpath", "./ellipse.svg");
    bool overwrite_logs = conf.getBool ("overwrite_logs", false);
    string logpath = conf.getString ("logpath", "fromfilename");
    string logbase = "";
    if (logpath == "fromfilename") {
        // Using json filename as logpath
        string justfile = paramsfile;
        // Remove trailing .json and leading directories
        vector<string> pth = morph::Tools::stringToVector (justfile, "/");
        justfile = pth.back();
        morph::Tools::searchReplace (".json", "", justfile);
        // Use logbase as the subdirectory into which this should go
        logbase = conf.getString ("logbase", "logs/");
        if (logbase.back() != '/') {
            logbase += '/';
        }
        logpath = logbase + justfile;
    }
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

    const FLT dt = static_cast<FLT>(conf.getDouble ("dt", 0.00001));
    const double Dn = conf.getDouble ("Dn", 0.3);
    const double Dc = conf.getDouble ("Dc", 0.3*0.3); // 0.3 * Dn
    const double beta = conf.getDouble ("beta", 5.0);
    const double a = conf.getDouble ("a", 1.0);
    const double b = conf.getDouble ("b", 1.0);
    const double mu = conf.getDouble ("mu", 1.0);

    cout << "steps to simulate: " << steps << endl;

#ifdef COMPILE_PLOTTING
    // Parameters from the config that apply only to plotting:
    const unsigned int plotevery = conf.getUInt ("plotevery", 10UL);
    const bool vidframes = conf.getBool ("vidframes", false);
    unsigned int framecount = 0;

    const unsigned int win_width = conf.getUInt ("win_width", 600UL);
    unsigned int win_height = static_cast<unsigned int>(0.8824f * (float)win_width);
    Visual plt (win_width, win_height, "Ermentrout 2009 simulation");
    // Note: Want to plot 'n' and 'c'
#endif

    // Instantiate the model object
    RD_Erm2<FLT> RD;

    RD.svgpath = svgpath;
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
    RD.allocate();
    RD.init();

#ifdef COMPILE_PLOTTING
    array<float, 3> offset = { 0.0, 0.0, 0.0 };
    float _m = 0.3;
    float _c = 0.0;
    const array<float, 4> scaling = { _m/10, _c/10, _m, _c };
    pair<float, float> mm = MathAlgo<float>::maxmin (RD.c[0]);
    cout << "Max n: " << mm.first << ", min n: " << mm.second << endl;

    unsigned int ngrid = plt.addHexGridVisual (RD.hg, offset, RD.n[0], scaling);
    offset[0] += RD.hg->width()*1.1;
    unsigned int cgrid = plt.addHexGridVisual (RD.hg, offset, RD.c[0], scaling);
    cout << "Added HexGridVisual with grid IDs " << ngrid << "(n) and " << cgrid << "(c)" << endl;
#endif

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
        RD.step();
#ifdef COMPILE_PLOTTING
        if ((RD.stepCount % plotevery) == 0) {
            plt.updateHexGridVisual (ngrid, RD.n[0], scaling);
            plt.updateHexGridVisual (cgrid, RD.c[0], scaling);
        }
        // Render more often than the hex grid is updated with data, to keep it
        // responsive, but not too often, lest too much performance be used up with
        // rendering the graphics.
        if ((RD.stepCount % 500) == 0) {
            glfwPollEvents();
            plt.render();
        }
#endif
         // Save data every 'logevery' steps
        if ((RD.stepCount % logevery) == 0) {
            cout << "Logging data at step " << RD.stepCount << endl;
            RD.saveState();
        }

        // After a while, stop:
        if (RD.stepCount > steps) {
            doing = false;
        }
    }

    // Before exit, save data
    RD.saveState();

    // Before saving the json, we'll place any additional useful info
    // in there, such as the FLT. If float_width is 4, then
    // results were computed with single precision, if 8, then double
    // precision was used. Also save various parameters from the RD system.
    conf.set ("float_width", (unsigned int)sizeof(FLT));
    string tnow = morph::Tools::timeNow();
    conf.set ("sim_ran_at_time", tnow.substr(0,tnow.size()-1));
    conf.set ("hextohex_d", RD.hextohex_d);
    conf.set ("dt", RD.get_dt());
    // Call our function to place git information into root.
    conf.insertGitInfo ("sim/");
    // Store the binary name and command argument into root, too.
    if (argc > 0) { conf.set("argv0", argv[0]); }
    if (argc > 1) { conf.set("argv1", argv[1]); }

    // We'll save a copy of the parameters for the simulation in the log directory as params.json
    const string paramsCopy = logpath + "/params.json";
    conf.write (paramsCopy);
    if (conf.ready == false) {
        cerr << "Warning: Something went wrong writing a copy of the params.json: " << conf.emsg << endl;
    }

#ifdef COMPILE_PLOTTING
    // Keep window open & active until user exits.
    cout << "Press x to exit.\n";
    plt.keepOpen();
#endif // COMPILE_PLOTTING

    return 0;
};
