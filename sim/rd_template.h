/*!
 *
 * CURRENTLY OUT OF DATE! Please see james1.cpp and rd_james.h!!!
 *
 * A template for starting a new morphologica-based reaction diffusion
 * system. Look at this for the core requirements, then consult the
 * most recent real RD system for the current best practice.
 */
#include "morph/display.h"
#include "morph/tools.h"
#include "morph/ReadCurves.h"
#include "morph/HexGrid.h"
#include "morph/HdfData.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <iomanip>
#include <cmath>
#ifdef __GLN__
#include <omp.h>
#endif
#include <unistd.h>

#define DEBUG 1
#define DBGSTREAM std::cout
#include <morph/MorphDbg.h>

using std::vector;
using std::array;
using std::string;
using std::stringstream;
using std::cerr;
using std::endl;
using std::runtime_error;

using morph::HexGrid;
using morph::ReadCurves;
using morph::HdfData;

/*!
 * Reaction diffusion system; A dummy template system.
 */
class RD_Template
{
public:

    /*!
     * Constants
     */
    //@{
    //! Square root of 3 over 2
    const double R3_OVER_2 = 0.866025403784439;
    //! Square root of 3
    const double ROOT3 = 1.73205080756888;
    //@}

    /*!
     * The logpath for this model. Used when saving data out.
     */
    string logpath = "logs";

    /*!
     * Setter which attempts to ensure the path exists.
     */
    void setLogpath (const string p) {
        this->logpath = p;
        // Ensure log directory exists
        morph::Tools::createDir (this->logpath);
    }

    /*!
     * Holds the number of hexes in the populated HexGrid
     */
    unsigned int nhex = 0;

    /*!
     * May not be necessary in this model - number of fields to compute.
     */
    unsigned int N = 1;

    /*!
     * A vector of N vectors of doubles called c.
     */
    //vector<vector<double> > c;

    /*!
     * Do gradients like this. This holds the two components of the
     * gradient field of the N scalar fields c[i](x,t) (where this x
     * is a vector in two-space)
     */
    //vector<array<vector<double>, 2> > grad_c;

    /*!
     * Our choice of dt.
     */
    double dt = 0.0001;

    /*!
     * Compute half and sixth dt in constructor.
     */
    //@{
    double halfdt = 0.0;
    double sixthdt = 0.0;
    //@}

    /*!
     * The HexGrid "background" for the Reaction Diffusion system.
     */
    HexGrid* hg;

    /*!
     * Store Hex positions for saving.
     */
    vector<float> hgvx;
    vector<float> hgvy;

    /*!
     * Hex to hex distance. Populate this from hg.d after hg has been
     * initialised.
     */
    double d = 1.0;

    /*!
     * Track the number of computational steps that we've carried
     * out, so that we can stop after some number of steps.
     */
    unsigned int stepCount = 0;

    /*!
     * A frame number, incremented when an image is plotted to a PNG
     * file.
     */
    unsigned int frameN = 0;

    /*!
     * Simple constructor; no arguments.
     */
    RD_Template (void) {
        this->halfdt = this->dt/2.0;
        this->sixthdt = this->dt/6.0;
    }

    /*!
     * Destructor required to free up HexGrid memory
     */
    ~RD_Template (void) {
        delete (this->hg);
    }

    /*!
     * A utility function to resize the vector-vectors that hold a
     * variable for the N different thalamo-cortical axon types.
     */
    void resize_vector_vector (vector<vector<double> >& vv) {
        vv.resize (this->N);
        for (unsigned int i=0; i<this->N; ++i) {
            vv[i].resize (this->nhex, 0.0);
        }
    }

    /*!
     * Resize a variable that'll be nhex elements long
     */
    void resize_vector_variable (vector<double>& v) {
        v.resize (this->nhex, 0.0);
    }

    /*!
     * Resize a parameter that'll be N elements long
     */
    void resize_vector_param (vector<double>& p) {
        p.resize (this->N, 0.0);
    }

    /*!
     * Resize a gradient field
     */
    void resize_gradient_field (array<vector<double>, 2>& g) {
        g[0].resize (this->nhex, 0.0);
        g[1].resize (this->nhex, 0.0);
    }

    /*!
     * Resize a vector (over TC types i) of an array of two
     * vector<double>s which are the x and y components of a
     * (mathematical) vector field.
     */
    void resize_vector_array_vector (vector<array<vector<double>, 2> >& vav) {
        vav.resize (this->N);
        for (unsigned int i = 0; i<this->N; ++i) {
            this->resize_gradient_field (vav[i]);
        }
    }

    /*!
     * Initialise this vector of vectors with noise. This is a
     * model-specific function.
     *
     * I apply a sigmoid to the boundary hexes, so that the noise
     * drops away towards the edge of the domain.
     */
    void noiseify_vector_vector (vector<vector<double> >& vv) {
        double randNoiseOffset = 0.8;
        double randNoiseGain = 0.1;
        for (unsigned int i = 0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                // boundarySigmoid. Jumps sharply (100, larger is
                // sharper) over length scale 0.05 to 1. So if
                // distance from boundary > 0.05, noise has normal
                // value. Close to boundary, noise is less.
                vv[i][h.vi] = morph::Tools::randDouble() * randNoiseGain + randNoiseOffset;
                if (h.distToBoundary > -0.5) { // It's possible that distToBoundary is set to -1.0
                    double bSig = 1.0 / ( 1.0 + exp (-100.0*(h.distToBoundary-0.02)) );
                    vv[i][h.vi] = vv[i][h.vi] * bSig;
                }
            }
        }
    }

    /*!
     * Initialise HexGrid, variables and parameters. Carry out
     * one-time computations of the model.
     */
    void init (vector<morph::Gdisplay>& displays, bool useSavedGenetics = false) {

        DBG ("called");
        // Create a HexGrid
        this->hg = new HexGrid (0.01, 3, 0, morph::HexDomainShape::Boundary);
        // Read the curves which make a boundary
        ReadCurves r("./trial.svg");
        // Set the boundary in the HexGrid
        this->hg->setBoundary (r.getCorticalPath());
        // Compute the distances from the boundary
        this->hg->computeDistanceToBoundary();
        // Vector size comes from number of Hexes in the HexGrid
        this->nhex = this->hg->num();
        // Spatial d comes from the HexGrid, too.
        this->d = this->hg->getd();
        // Save hex positions in vectors for datafile saving
        for (auto h : this->hg->hexen) {
            this->hgvx.push_back (h.x);
            this->hgvy.push_back (h.y);
        }

        // Resize and zero-initialise the various containers
        //this->resize_vector_vector (this->c);

        //this->resize_vector_variable (this->n);

        //this->resize_vector_param (this->alpha);

        //this->resize_gradient_field (this->grad_rhoA);

        // Resize grad_a and other vector-array-vectors
        //this->resize_vector_array_vector (this->grad_a);

        // Initialise a with noise
        //this->noiseify_vector_vector (this->a);

        // Now initialise the model's variables and parameters.
    }

    /*!
     * HDF5 file saving/loading methods
     */
    //@{
    void saveHexPositions (HdfData& dat) {
        dat.add_contained_vals ("/x", this->hgvx);
        dat.add_contained_vals ("/y", this->hgvy);
        // And hex to hex distance:
        dat.add_val ("/d", this->d);
    }

    /*!
     * Save some data like this.
     */
    void saveStuff (void) {
        string fname = this->logpath + "/factorexpression.h5";
        cout << "Saving to file " << fname << endl;
        HdfData data (fname);
        // Save some variables
        //data.add_val ("/Aemx", this->Aemx);
        //data.add_contained_vals ("/emx", this->emx);
        this->saveHexPositions (data);
    }
    //@} // HDF5

    /*!
     * Computation methods
     */
    //@{

    /*!
     * Do a step through the model.
     */
    void step (void) {

        this->stepCount++;

        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
        }

        // 1. Do some computes
#if 0
        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Hex* h = this->hg->vhexen[hi];
        }
#endif
        // 2. Do integration of some variable.
#if 0
        // Runge-Kutta integration:
        #pragma omp parallel for
        for (unsigned int i=0; i<this->N; ++i) {

            DBG2 ("(a) alpha_c_beta_na["<<i<<"][0] = " << this->alpha_c_beta_na[i][0]);

            // Runge-Kutta integration for A
            vector<double> q(this->nhex, 0.0);
            this->compute_divJ (a[i], i); // populates divJ[i]
            DBG2 ("Computing divJ, divJ[0][0]: " << divJ[0][0]);
            vector<double> k1(this->nhex, 0.0);
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k1[h] * halfdt;
            }
            DBG2 ("(a) After RK stage 1, q[0]: " << q[0]);

            vector<double> k2(this->nhex, 0.0);
            this->compute_divJ (q, i);
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k2[h] * halfdt; // Kaboom!
            }
            DBG2 ("(a) After RK stage 2, q[0]:" << q[0] << " from a["<<i<<"][0]:" << a[i][0] << " divj["<<i<<"][0]:" << divJ[i][0] << " k2[0]:" << k2[0]);

            vector<double> k3(this->nhex, 0.0);
            this->compute_divJ (q, i);
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k3[h] * dt;
            }
            DBG2 ("(a) After RK stage 3, q[0]: " << q[0]);

            vector<double> k4(this->nhex, 0.0);
            this->compute_divJ (q, i);
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * sixthdt;
            }
            DBG2 ("(a) After RK stage 4, a[" << i << "][0]: " << a[i][0]);

            DBG2("(a) Debug a["<<i<<"]");
        }
#endif
    }

    /*!
     * Examine the value in each Hex of the hexgrid of the scalar
     * field f. If abs(f[h]) exceeds the size of dangerThresh, then
     * output debugging information.
     */
    void debug_values (vector<double>& f, double dangerThresh) {
        for (auto h : this->hg->hexen) {
            if (abs(f[h.vi]) > dangerThresh) {
                DBG ("Blow-up threshold exceeded at Hex.vi=" << h.vi << " ("<< h.ri <<","<< h.gi <<")" <<  ": " << f[h.vi]);
                unsigned int wait = 0;
                while (wait++ < 120) {
                    usleep (1000000);
                }
            }
        }
    }

    /*!
     * 2D spatial integration of the function f. Result placed in gradf.
     *
     * For each Hex, work out the gradient in x and y directions
     * using whatever neighbours can contribute to an estimate.
     */
    void spacegrad2D (vector<double>& f, array<vector<double>, 2>& gradf) {

        // Note - East is positive x; North is positive y. Does this match how it's drawn in the display??
        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Hex* h = this->hg->vhexen[hi];

            gradf[0][h->vi] = 0.0;
            gradf[1][h->vi] = 0.0;

            DBG2 ("(h->ri,h->gi): (" << h->ri << "," << h->gi << ")");
            // Find x gradient
            if (h->has_ne() && h->has_nw()) {
                DBG2 ("x case 1 f[h->ne]: " << f[h->ne->vi] << " - f[h->nw]" << f[h->nw->vi] << "/ h->d*2: " << (double)h->d * 2.0);
                gradf[0][h->vi] = (f[h->ne->vi] - f[h->nw->vi]) / ((double)h->d * 2.0);
            } else if (h->has_ne()) {
                DBG2 ("x case 2 f[h->ne]: " << f[h->ne->vi] << " - f[h]" << f[h->vi] << "/ h->d: " << (double)h->d);
                gradf[0][h->vi] = (f[h->ne->vi] - f[h->vi]) / (double)h->d;
            } else if (h->has_nw()) {
                DBG2 ("x case 3 f[h]: " << f[h->vi] << " - f[h->nw]" << f[h->nw->vi] << "/ h->d: " << (double)h->d);
                gradf[0][h->vi] = (f[h->vi] - f[h->nw->vi]) / (double)h->d;
            } else {
                // zero gradient in x direction as no neighbours in
                // those directions? Or possibly use the average of
                // the gradient between the nw,ne and sw,se neighbours
            }

            // Find y gradient
            // Note: Now that I can access flags in h, can get the flags and achieve this in a single test:
            // hflags & (HEX_HAS_NNW | HAS_HAS_NNE | HAS_HAS_NSW | HAS_HAS_NSE) == (HEX_HAS_NNW | HAS_HAS_NNE | HAS_HAS_NSW | HAS_HAS_NSE)
            // or perhaps hflags & HEX_NEIGHB_DIAGS == HEX_NEIGHB_DIAGS
            if (h->has_nnw() && h->has_nne() && h->has_nsw() && h->has_nse()) {
                // Full complement. Compute the mean of the nse->nne and nsw->nnw gradients
#ifdef DEBUG2
                if (h->vi == 0) {
                    double _d = (double)h->getV();
                    double _td = (double)h->getTwoV();
                    DBG2 ("y case 1. getV: " << _d << " getTwoV: " << _td);
                }
#endif
                gradf[1][h->vi] = ((f[h->nne->vi] - f[h->nse->vi]) + (f[h->nnw->vi] - f[h->nsw->vi])) / (double)h->getV();

            } else if (h->has_nnw() && h->has_nne() ) {
                //if (h->vi == 0) { DBG ("y case 2"); }
                gradf[1][h->vi] = ( (f[h->nne->vi] + f[h->nnw->vi]) / 2.0 - f[h->vi]) / (double)h->getV();

            } else if (h->has_nsw() && h->has_nse()) {
                //if (h->vi == 0) { DBG ("y case 3"); }
                gradf[1][h->vi] = (f[h->vi] - (f[h->nse->vi] + f[h->nsw->vi]) / 2.0) / (double)h->getV();

            } else if (h->has_nnw() && h->has_nsw()) {
                //if (h->vi == 0) { DBG ("y case 4"); }
                gradf[1][h->vi] = (f[h->nnw->vi] - f[h->nsw->vi]) / (double)h->getTwoV();

            } else if (h->has_nne() && h->has_nse()) {
                //if (h->vi == 0) { DBG ("y case 5"); }
                gradf[1][h->vi] = (f[h->nne->vi] - f[h->nse->vi]) / (double)h->getTwoV();
            } else {
                // Leave grady at 0
            }

            //if (h->vi == 0) {
            //    DBG ("gradf[0/1][0]: " << gradf[0][0] << "," << gradf[1][0]);
            //}
        }
    }

    /*!
     * Computes the "flux of axonal branches" term, J_i(x) (Eq 4)
     *
     * Inputs: this->g, fa (which is this->a[i] or a q in the RK
     * algorithm), this->D, @a i, the TC type.  Helper functions:
     * spacegrad2D().  Output: this->divJ
     *
     * Stable with dt = 0.0001;
     */
#if 0
    void compute_divJ (vector<double>& fa, unsigned int i) {

        // Three terms to compute; see Eq. 14 in methods_notes.pdf

        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa, this->grad_a[i]);

        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            Hex* h = this->hg->vhexen[hi];
            // 1. The D Del^2 a_i term
            // Compute the sum around the neighbours
            double thesum = -6 * fa[h->vi];
            if (h->has_ne()) {
                thesum += fa[h->ne->vi];
            } else {
                // Apply boundary condition
            }
            if (h->has_nne()) {
                thesum += fa[h->nne->vi];
            } else {
                thesum += fa[h->vi]; // A ghost neighbour-east with same value as Hex_0
            }
            if (h->has_nnw()) {
                thesum += fa[h->nnw->vi];
            } else {
                thesum += fa[h->vi];
            }
            if (h->has_nw()) {
                thesum += fa[h->nw->vi];
            } else {
                thesum += fa[h->vi];
            }
            if (h->has_nsw()) {
                thesum += fa[h->nsw->vi];
            } else {
                thesum += fa[h->vi];
            }
            if (h->has_nse()) {
                thesum += fa[h->nse->vi];
            } else {
                thesum += fa[h->vi];
            }
            // Multiply bu 2D/3d^2
            double term1 = (this->D * 2) / (3 * this->d * this->d) * thesum;

            this->divJ[i][h->vi] = term1;
        }
    }
#endif

    /*!
     * Plotting code
     */
    //@{
    /*!
     * Plot the system on @a disps
     */
    void plot (vector<morph::Gdisplay>& disps, bool savePngs = false) {

        //this->plot_f (this->a, disps[2]);
        //this->plot_contour (this->c, disps[4], 0.75);

#if 0
        if (savePngs) {
            // a
            stringstream ff1;
            ff1 << this->logpath << "/a_";
            ff1 << std::setw(5) << std::setfill('0') << frameN;
            ff1 << ".png";
            disps[2].saveImage (ff1.str());
            // contours
            stringstream ff3;
            ff3 << this->logpath << "/cntr_";
            ff3 << std::setw(5) << std::setfill('0') << frameN;
            ff3 << ".png";
            disps[4].saveImage (ff3.str());

            frameN++;
        }
#endif
    }

    /*!
     * Plot a field. If individual_scaling is true, then each map is
     * normalised individually, otherwise the group is normalised so
     * that maps can be compared alongside each other.
     */
    void plot_f (vector<vector<double> >& f, morph::Gdisplay& disp, bool individual_scaling=false) {
        vector<double> fix(3, 0.0);
        vector<double> eye(3, 0.0);
        eye[2] = -0.4;
        vector<double> rot(3, 0.0);

        vector<vector<double> > norm_a;
        this->resize_vector_vector (norm_a);
        if (individual_scaling) {
            // Copies data to plot out of the model
            vector<double> maxa (5, -1e7);
            vector<double> mina (5, +1e7);
            // Determines min and max
            for (auto h : this->hg->hexen) {
                if (h.onBoundary() == false) {
                    for (unsigned int i = 0; i<this->N; ++i) {
                        if (f[i][h.vi]>maxa[i]) { maxa[i] = f[i][h.vi]; }
                        if (f[i][h.vi]<mina[i]) { mina[i] = f[i][h.vi]; }
                    }
                }
            }
            vector<double> scalea (5, 0);
            for (unsigned int i = 0; i<this->N; ++i) {
                scalea[i] = 1.0 / (maxa[i]-mina[i]);
            }

            // Determine a colour from min, max and current value
            for (unsigned int i = 0; i<this->N; ++i) {
                for (unsigned int h=0; h<this->nhex; h++) {
                    norm_a[i][h] = fmin (fmax (((f[i][h]) - mina[i]) * scalea[i], 0.0), 1.0);
                }
            }
        } else {
            // Copies data to plot out of the model
            double maxa = -1e7;
            double mina = +1e7;
            // Determines min and max

#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex; ++hi) {
                Hex* h = this->hg->vhexen[hi];
                if (h->onBoundary() == false) {
                    for (unsigned int i = 0; i<this->N; ++i) {
                        if (f[i][h->vi]>maxa) { maxa = f[i][h->vi]; }
                        if (f[i][h->vi]<mina) { mina = f[i][h->vi]; }
                    }
                }
            }
            double scalea = 1.0 / (maxa-mina);

            // Determine a colour from min, max and current value
            for (unsigned int i = 0; i<this->N; ++i) {
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; h++) {
                    norm_a[i][h] = fmin (fmax (((f[i][h]) - mina) * scalea, 0.0), 1.0);
                }
            }
        }

        // Create an offset which we'll increment by the width of the
        // map, starting from the left-most map (f[0])
        float hgwidth = this->hg->getXmax()-this->hg->getXmin();
        array<float,3> offset = { 2*(-hgwidth-(hgwidth/20)), 0.0f, 0.0f };

        // Draw
        disp.resetDisplay (fix, eye, rot);
        for (unsigned int i = 0; i<this->N; ++i) {
            // Note: OpenGL isn't thread-safe, so no omp parallel for here.
            for (auto h : this->hg->hexen) {
                array<float,3> cl_a = morph::Tools::HSVtoRGB ((float)i/(float)this->N,
                                                              norm_a[i][h.vi], 1.0);
                disp.drawHex (h.position(), offset, (h.d/2.0f), cl_a);
            }
            offset[0] += hgwidth + (hgwidth/20);
        }
        disp.redrawDisplay();
    }

    void plot_contour (vector<vector<double> >& f, morph::Gdisplay& disp, double threshold) {

        vector<double> fix(3, 0.0);
        vector<double> eye(3, 0.0);
        vector<double> rot(3, 0.0);

        // Copies data to plot out of the model
        vector<double> maxa (5, -1e7);
        vector<double> mina (5, +1e7);

        // Determines min and max
        for (auto h : this->hg->hexen) {
            if (h.onBoundary() == false) {
                for (unsigned int i = 0; i<this->N; ++i) {
                    if (f[i][h.vi]>maxa[i]) { maxa[i] = f[i][h.vi]; }
                    if (f[i][h.vi]<mina[i]) { mina[i] = f[i][h.vi]; }
                }
            }
        }

        vector<double> scalea (5, 0);
        for (unsigned int i = 0; i<this->N; ++i) {
            scalea[i] = 1.0 / (maxa[i]-mina[i]);
        }

        // Re-normalize
        vector<vector<double> > norm_a;
        this->resize_vector_vector (norm_a);
        for (unsigned int i = 0; i<this->N; ++i) {
            for (unsigned int h=0; h<this->nhex; h++) {
                norm_a[i][h] = fmin (fmax (((f[i][h]) - mina[i]) * scalea[i], 0.0), 1.0);
            }
        }

        // Draw
        double c = threshold;
        disp.resetDisplay (fix, eye, rot);
        array<float,3> cl_blk = {0.0f, 0.0f, 0.0f};
        array<float,3> zero_offset = {0.0f, 0.0f, 0.0f};

        for (unsigned int i = 0; i<this->N; ++i) {
            array<float,3> cl_b = morph::Tools::HSVtoRGB ((double)i/(double)(this->N),1.,1.);
            for (auto h : this->hg->hexen) {
                if (h.onBoundary() == false) {
                    if (norm_a[i][h.vi]<c) {
#warning "FIXME: ordering here too"
                        if (norm_a[i][h.ne->vi]>c && h.has_ne()) {
                            disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_b, 0);
                        }
                        if (norm_a[i][h.nne->vi]>c && h.has_nne()) {
                            disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_b, 1);
                        }
                        if (norm_a[i][h.nnw->vi]>c && h.has_nnw()) {
                            disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_b, 2);
                        }
                        if (norm_a[i][h.nw->vi]>c && h.has_nw()) {
                            disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_b, 3);
                        }
                        if (norm_a[i][h.nsw->vi]>c && h.has_nsw()) {
                            disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_b, 4);
                        }
                        if (norm_a[i][h.nse->vi]>c && h.has_nse()) {
                            disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_b, 5);
                        }
                    }

                } else { // h.onBoundary() is true

                    if (!h.has_ne()) {
                        disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_blk, 0);
                    }
                    if (!h.has_nne()) {
                        disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_blk, 1);
                    }
                    if (!h.has_nnw()) {
                        disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_blk, 2);
                    }
                    if (!h.has_nw()) {
                        disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_blk, 3);
                    }
                    if (!h.has_nsw()) {
                        disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_blk, 4);
                    }
                    if (!h.has_nse()) {
                        disp.drawHexSeg (h.position(), zero_offset, (h.d/2.0f), cl_blk, 5);
                    }
                }
            }
        }
        disp.redrawDisplay();
    }
    //@} // Plotting code

}; // RD_Template
