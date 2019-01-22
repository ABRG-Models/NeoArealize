#include "morph/display.h"
#include "morph/tools.h"
#include "morph/ReadCurves.h"
#include "morph/HexGrid.h"
#include "morph/HdfData.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <iomanip>
#include <cmath>
#include <hdf5.h>
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
 * Macros for testing neighbours. The step along for neighbours on the
 * rows above/below is given by:
 *
 * Dest  | step
 * ----------------------
 * NNE   | +rowlen
 * NNW   | +rowlen - 1
 * NSW   | -rowlen
 * NSE   | -rowlen + 1
 */
//@{
#define NE(hi) (this->hg->d_ne[hi])
#define HAS_NE(hi) (this->hg->d_ne[hi] == -1 ? false : true)

#define NW(hi) (this->hg->d_nw[hi])
#define HAS_NW(hi) (this->hg->d_nw[hi] == -1 ? false : true)

#define NNE(hi) (this->hg->d_nne[hi])
#define HAS_NNE(hi) (this->hg->d_nne[hi] == -1 ? false : true)

#define NNW(hi) (this->hg->d_nnw[hi])
#define HAS_NNW(hi) (this->hg->d_nnw[hi] == -1 ? false : true)

#define NSE(hi) (this->hg->d_nse[hi])
#define HAS_NSE(hi) (this->hg->d_nse[hi] == -1 ? false : true)

#define NSW(hi) (this->hg->d_nsw[hi])
#define HAS_NSW(hi) (this->hg->d_nsw[hi] == -1 ? false : true)
//@}

#define IF_HAS_NE(hi, yesval, noval)  (HAS_NE(hi)  ? yesval : noval)
#define IF_HAS_NNE(hi, yesval, noval) (HAS_NNE(hi) ? yesval : noval)
#define IF_HAS_NNW(hi, yesval, noval) (HAS_NNW(hi) ? yesval : noval)
#define IF_HAS_NW(hi, yesval, noval)  (HAS_NW(hi)  ? yesval : noval)
#define IF_HAS_NSW(hi, yesval, noval) (HAS_NSW(hi) ? yesval : noval)
#define IF_HAS_NSE(hi, yesval, noval) (HAS_NSE(hi) ? yesval : noval)

/*!
 * Enumerates the way that the guidance molecules are set up
 */
enum class GuidanceMoleculeMethod {
    Gauss1D,
    Gauss2D,
    Exponential1D,
    Sigmoid1D
};

/*!
 * Reaction diffusion system. Based on Karbowski 2004, but with a
 * removal of the Fgf8, Pax6, Emx2 system, and instead an option to
 * define several guidance molecules and
 */
class RD_James
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
    //! Passed to HdfData constructor to say we want to read the data
    const bool READ_DATA = true;
    //@}

    /*!
     * Hex to hex d for the grid. Make smaller to increase the number
     * of Hexes being computed.
     */
    alignas(4) float hextohex_d = 0.01;

    /*!
     * Holds the number of hexes in the populated HexGrid
     */
    alignas(4) unsigned int nhex = 0;

    /*!
     * how many thalamo-cortical axon types are there? Denoted by N in
     * the paper, and so we use N here too.
     */
    alignas(4) unsigned int N = 5;

    /*!
     * M is the number of guidance molecules to use.
     */
    alignas(4) unsigned int M = 3;

    /*!
     * These are the c_i(x,t) variables from the Karb2004 paper. x is
     * a vector in two-space.
     */
    alignas(8) vector<vector<double> > c;

    /*!
     * These are the a_i(x,t) variables from the Karb2004 paper. x is
     * a vector in two-space. The first vector is over the different
     * TC axon types, enumerated by i, the second vector are the a_i
     * values, indexed by the vi in the Hexes in HexGrid.
     */
    alignas(8) vector<vector<double> > a;

    /*!
     * For each TC axon type, this holds the two components of the
     * gradient field of the scalar value a(x,t) (where this x is a
     * vector in two-space)
     */
    alignas(8) vector<array<vector<double>, 2> > grad_a;

    /*!
     * Contains the chemo-attractant modifiers which are applied to
     * a_i(x,t) in Eq 4.
     */
    alignas(8) vector<array<vector<double>, 2> > g;

    /*!
     * n(x,t) variable from the Karb2004 paper.
     */
    alignas(8) vector<double> n;

    /*!
     * J_i(x,t) variables - the "flux current of axonal branches of
     * type i". This is a vector field.
     */
    alignas(8) vector<array<vector<double>, 2> > J;

    /*!
     * Holds the divergence of the J_i(x)s
     */
    alignas(8) vector<vector<double> > divJ;

    /*!
     * Our choice of dt.
     */
    alignas(8) double dt = 0.0001;

    /*!
     * Compute half and sixth dt in constructor.
     */
    //@{
    alignas(8) double halfdt = 0.0;
    alignas(8) double sixthdt = 0.0;
    //@}

    /*!
     * The power to which a_i(x,t) is raised in Eqs 1 and 2 in the
     * paper.
     */
    alignas(8) double k = 3.0;

    /*!
     * The diffusion parameter.
     */
    alignas(8) double D = 0.1;

    /*!
     * alpha_i parameters
     */
    alignas(8) vector<double> alpha;

    /*!
     * beta_i parameters
     */
    alignas(8) vector<double> beta;

    /*!
     * gamma_A/B/C_i (etc) parameters from Eq 4. There are M vectors
     * of doubles in here.
     */
    //@{
    alignas(8) vector<vector<double> > gamma;
    //@}

    /*!
     * A vector of parameters for the direction of the guidance
     * molecules
     */
    alignas(8) vector<double> guidance_phi;

    /*!
     * Guidance molecule parameters for the width of the function
     */
    alignas(8) vector<double> guidance_width;

    /*!
     * Width in orthogonal direction, for 2D fields.
     */
    alignas(8) vector<double> guidance_width_ortho;

    /*!
     * Guidance molecule parameters for the offset of the function
     */
    alignas(8) vector<double> guidance_offset;

    /*!
     * Guidance molecule parameters to be the gains of the functions
     */
    alignas(8) vector<double> guidance_gain;

    /*!
     * Rho variables in Eq 4 - the concentrations of axon guidance
     * molecules A, B, C, etc. In Karbowski 2004, these are time
     * independent and we will treat them as such, populating them at
     * initialisation.
     *
     * There are M vector<doubles> in rho.
     */
    //@{
    alignas(8) vector<vector<double> > rho;
    //@}

    /*!
     * Into grad_rho put the two components of the gradient of
     * rho computed across the HexGrid surface.
     *
     * There are M gradient fields stored in this variable.
     */
    //@{
    alignas(8) vector<array<vector<double>, 2> > grad_rho;
    //@}

    /*!
     * Store Hex positions for saving.
     */
    alignas(8) vector<float> hgvx;
    alignas(8) vector<float> hgvy;

    /*!
     * Rowlen and numrows in the parallelogram domain.
     */
    //@{
    alignas(4) unsigned int rl = 0;
    alignas(4) unsigned int nr = 0;
    //@}

    /*!
     * Hex to hex distance. Populate this from hg.d after hg has been
     * initialised.
     */
    alignas(8) double d = 1.0;

    /*!
     * Various other parameters
     */
    //@{
    alignas(8) double oneoverd = 1.0/this->d;
    alignas(8) double v = 1;
    alignas(8) double oneoverv = 1.0/this->v;
    alignas(8) double twov = this->v+this->v;
    alignas(8) double oneover2v = 1.0/this->twov;
    alignas(8) double oneover2d = 1.0/(this->d+this->d);
    alignas(8) double twoDover3dd = 0.0;
    //@}

    /*!
     * Memory to hold an intermediate result
     */
    alignas(8) vector<vector<double> > betaterm;

    /*!
     * Holds an intermediate value for the computation of Eqs 1 and 2.
     */
    alignas(8) vector<vector<double> > alpha_c_beta_na;

    /*!
     * Track the number of computational steps that we've carried
     * out. Only to show a message saying "100 steps done...", but
     * that's reason enough.
     */
    alignas(4) unsigned int stepCount = 0;

    /*!
     * A frame number, incremented when an image is plotted to a PNG file.
     */
    alignas(4) unsigned int frameN = 0;

    /*!
     * The contour threshold. For contour plotting [see
     * plot_contour()], the field is normalised, then the contour is
     * plotted where the field crosses this threshold.
     */
    alignas(8) double contour_threshold = 0.5;

    /*!
     * Used by plotting functions
     */
    //@{
    alignas(8) vector<double> fix = {3, 0.0};
    alignas(8) vector<double> eye = {3, 0.0};
    alignas(8) vector<double> rot = {3, 0.0};
    //@}

    /*!
     * ALIGNAS REGION ENDS.
     *
     * Below here, there's no need to worry about alignas keywords.
     */

    /*!
     * The HexGrid "background" for the Reaction Diffusion system.
     */
    HexGrid* hg;

    /*!
     * Sets the function of the guidance molecule method
     */
    GuidanceMoleculeMethod rhoMethod = GuidanceMoleculeMethod::Gauss1D;

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
     * Make the svgpath something that can be set by client code...
     */
    string svgpath = "./trial.svg";

    /*!
     * Simple constructor; no arguments.
     */
    RD_James (void) {
        this->halfdt = this->dt/2.0;
        this->sixthdt = this->dt/6.0;
    }

    /*!
     * Destructor required to free up HexGrid memory
     */
    ~RD_James (void) {
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
     * M members, each nhex elements long
     */
    void resize_guidance_variable (vector<vector<double> >& v) {
        v.resize (this->M);
        for (unsigned int m = 0; m<this->M; ++m) {
            v[m].resize (this->nhex, 0.0);
        }
    }

    /*!
     * Resize a parameter that'll be N elements long
     */
    void resize_vector_param (vector<double>& p) {
        p.resize (this->N, 0.0);
    }

    /*!
     * Resize a vector of M vectors of parameters that'll each be N
     * elements long
     */
    void resize_vector_vector_param (vector<vector<double> >& vp) {
        vp.resize (this->M);
        for (unsigned int m = 0; m<this->M; ++m) {
            vp[m].resize (this->N, 0.0);
        }
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

    void resize_guidance_gradient_field (vector<array<vector<double>, 2> >& vav) {
        vav.resize (this->M);
        for (unsigned int m = 0; m<this->M; ++m) {
            this->resize_gradient_field (vav[m]);
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
    void init (void) {

        DBG ("called");
        // Create a HexGrid
        this->hg = new HexGrid (this->hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
        // Read the curves which make a boundary
        ReadCurves r("./ellipse.svg");
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
        this->resize_vector_vector (this->c);
        this->resize_vector_vector (this->a);
        this->resize_vector_vector (this->betaterm);
        this->resize_vector_vector (this->alpha_c_beta_na);
        this->resize_vector_vector (this->divJ);

        this->resize_vector_variable (this->n);
        this->resize_guidance_variable (this->rho);

        this->resize_vector_param (this->alpha);
        this->resize_vector_param (this->beta);
        this->resize_vector_vector_param (this->gamma);

        this->resize_guidance_gradient_field (this->grad_rho);
        //this->resize_gradient_field (this->grad_rhoB);
        //this->resize_gradient_field (this->grad_rhoC);

        // Resize grad_a and other vector-array-vectors
        this->resize_vector_array_vector (this->grad_a);
        this->resize_vector_array_vector (this->g);
        this->resize_vector_array_vector (this->J);

        // Initialise a with noise
        this->noiseify_vector_vector (this->a);

        // The gamma values - notice the symmetry here.
        // red
#if 0
        this->gamma[0][0] = -2.0;
        this->gamma[1][0] =  0.5;
        this->gamma[2][0] =  0.5;
        // yellow
        this->gamma[0][1] = -2.0;
        this->gamma[1][1] = -2.0;
        this->gamma[2][1] =  0.5;
        // green
        this->gamma[0][2] =  0.5;
        this->gamma[1][2] = -2.0;
        this->gamma[2][2] =  0.5;
        // blue
        this->gamma[0][3] =  0.5;
        this->gamma[1][3] = -2.0;
        this->gamma[2][3] = -2.0;
        // magenta
        this->gamma[0][4] =  0.5;
        this->gamma[1][4] =  0.5;
        this->gamma[2][4] = -2.0;
#endif

#if 1 // Set up parameters somewhere else?
        this->gamma[0][0] = -2.0;
        //this->gamma[1][0] =  0.5;
        //this->gamma[2][0] =  0.5;

        this->gamma[0][1] =  0.5;
        //this->gamma[1][1] =  0.5;
        //this->gamma[2][1] = -2.0;
#endif
        for (unsigned int i=0; i<this->N; ++i) {
            this->alpha[i] = 3;
            this->beta[i] = 3;
        }

        // If client code didn't initialise the guidance molecules, then do so
        if (this->guidance_phi.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_phi.push_back(0.0);
            }
        }
        if (this->guidance_width.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_width.push_back(1.0);
            }
        }
        if (this->guidance_width_ortho.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_width_ortho.push_back(1.0);
            }
        }
        if (this->guidance_offset.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_offset.push_back(0.0);
            }
        }
        if (this->guidance_gain.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_gain.push_back(1.0);
            }
        }

        // Make one axon branching decay parameter stronger here, or better in client code.
        //this->beta[0] = 5;

        if (this->rhoMethod == GuidanceMoleculeMethod::Gauss1D) {
            // Construct Gaussian-waves rather than doing the full-Karbowski shebang.
            this->gaussian1D_guidance();

        } else if (this->rhoMethod == GuidanceMoleculeMethod::Gauss2D) {
            // Construct 2 dimensional gradients
            this->gaussian2D_guidance();

        } else if (this->rhoMethod == GuidanceMoleculeMethod::Sigmoid1D) {
            this->sigmoid_guidance();
        } // etc - linear, exponential etc. also need a scheme to set
          // the parameters for the guidance methods.

        // Compute gradients of guidance molecule concentrations once only
        for (unsigned int m = 0; m<this->M; ++m) {
            this->spacegrad2D (this->rho[m], this->grad_rho[m]);
        }

        // Having computed gradients, build this->g; has
        // to be done once only. Note that a sigmoid is applied so
        // that g(x) drops to zero around the boundary of the domain.
        for (unsigned int i=0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                // Sigmoid/logistic fn params: 100 sharpness, 0.02 dist offset from boundary
                double bSig = 1.0 / ( 1.0 + exp (-100.0*(h.distToBoundary-0.02)) );
                for (unsigned int m = 0; m<this->M; ++m) {
                    this->g[i][0][h.vi] += (this->gamma[m][i] * this->grad_rho[m][0][h.vi]) * bSig;
                    this->g[i][1][h.vi] += (this->gamma[m][i] * this->grad_rho[m][1][h.vi]) * bSig;
                }
            }
        }
    }

    /*!
     * HDF5 file saving/loading methods
     */
    //@{

    /*!
     * Save the c variable.
     */
    void saveC (void) {
        stringstream fname;
        fname << this->logpath << "/c_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        HdfData data(fname.str());
        for (unsigned int i = 0; i<this->N; ++i) {
            stringstream path;
            path << "/c" << i;
            data.add_double_vector (path.str().c_str(), this->c[i]);
        }
        this->saveHexPositions (data);
    }

    /*!
     * Save positions of the hexes - note using two vector<floats>
     * that have been populated with the positions from the HexGrid,
     * to fit in with the HDF API.
     */
    void saveHexPositions (HdfData& dat) {
        dat.add_float_vector ("/x", this->hgvx);
        dat.add_float_vector ("/y", this->hgvy);
        // And hex to hex distance:
        dat.add_double ("/d", this->d);
    }

    //@} // HDF5

    /*!
     * Computation methods
     */
    //@{

    /*!
     * Normalise the vector of doubles f.
     */
    void normalise (vector<double>& f) {

        double maxf = -1e7;
        double minf = +1e7;

        // Determines min and max
        for (auto val : f) {
            if (val>maxf) { maxf = val; }
            if (val<minf) { minf = val; }
        }
        double scalef = 1.0 /(maxf - minf);

        vector<vector<double> > norm_a;
        this->resize_vector_vector (norm_a);
        for (unsigned int fi = 0; fi < f.size(); ++fi) {
            f[fi] = fmin (fmax (((f[fi]) - minf) * scalef, 0.0), 1.0);
        }
    }

    /*!
     * Do a step through the model.
     */
    void step (void) {

        this->stepCount++;

        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
        }

        // 1. Compute Karb2004 Eq 3. (coupling between connections made by each TC type)
        double nsum = 0.0;
        double csum = 0.0;
#pragma omp parallel for reduction(+:nsum,csum)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            n[hi] = 0;
            for (unsigned int i=0; i<N; ++i) {
                n[hi] += c[i][hi];
            }
            csum += c[0][hi];
            n[hi] = 1. - n[hi];
            nsum += n[hi];
        }
        if (this->stepCount % 100 == 0) {
            DBG ("sum of all n is " << nsum);
            DBG ("sum of all c for i=0 is " << csum);
        }

        // 2. Do integration of a (RK in the 1D model). Involves computing axon branching flux.

        // Pre-compute intermediate val:
        for (unsigned int i=0; i<this->N; ++i) {
#pragma omp parallel for shared(i,k)
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->alpha_c_beta_na[i][h] = alpha[i] * c[i][h] - beta[i] * n[h] * pow (a[i][h], k);
            }
        }

        // Runge-Kutta:
        // No OMP here - there are only N(<10) loops, which isn't
        // enough to load the threads up.
        for (unsigned int i=0; i<this->N; ++i) {

            // Runge-Kutta integration for A
            vector<double> q(this->nhex, 0.0);
            this->compute_divJ (a[i], i); // populates divJ[i]

            vector<double> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k1[h] * halfdt;
            }

            vector<double> k2(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k2[h] * halfdt;
            }

            vector<double> k3(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k3[h] * dt;
            }

            vector<double> k4(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * sixthdt;
            }
        }

        // 3. Do integration of c
        for (unsigned int i=0; i<this->N; ++i) {

#pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                this->betaterm[i][h] = beta[i] * n[h] * pow (a[i][h], k);
            }

            // Runge-Kutta integration for C (or ci)
            vector<double> q(nhex,0.);
            vector<double> k1 = compute_dci_dt (c[i], i);
#pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                q[h] = c[i][h] + k1[h] * halfdt;
            }

            vector<double> k2 = compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                q[h] = c[i][h] + k2[h] * halfdt;
            }

            vector<double> k3 = compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                q[h] = c[i][h] + k3[h] * dt;
            }

            vector<double> k4 = compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                c[i][h] += (k1[h]+2. * (k2[h] + k3[h]) + k4[h]) * sixthdt;
            }
        }
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
#pragma omp parallel for schedule(static)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            // Find x gradient
            if (HAS_NE(hi) && HAS_NW(hi)) {
                gradf[0][hi] = (f[NE(hi)] - f[NW(hi)]) * oneover2d;
            } else if (HAS_NE(hi)) {
                gradf[0][hi] = (f[NE(hi)] - f[hi]) * oneoverd;
            } else if (HAS_NW(hi)) {
                gradf[0][hi] = (f[hi] - f[NW(hi)]) * oneoverd;
            } else {
                // zero gradient in x direction as no neighbours in
                // those directions? Or possibly use the average of
                // the gradient between the nw,ne and sw,se neighbours
                gradf[0][hi] = 0.0;
            }

            // Find y gradient
            if (HAS_NNW(hi) && HAS_NNE(hi) && HAS_NSW(hi) && HAS_NSE(hi)) {
                // Full complement. Compute the mean of the nse->nne and nsw->nnw gradients
                gradf[1][hi] = ((f[NNE(hi)] - f[NSE(hi)]) + (f[NNW(hi)] - f[NSW(hi)])) * oneoverv;

            } else if (HAS_NNW(hi) && HAS_NNE(hi)) {
                gradf[1][hi] = ( (f[NNE(hi)] + f[NNW(hi)]) * 0.5 - f[hi]) * oneoverv;

            } else if (HAS_NSW(hi) && HAS_NSE(hi)) {
                gradf[1][hi] = (f[hi] - (f[NSE(hi)] + f[NSW(hi)]) * 0.5) * oneoverv;

            } else if (HAS_NNW(hi) && HAS_NSW(hi)) {
                gradf[1][hi] = (f[NNW(hi)] - f[NSW(hi)]) * oneover2v;

            } else if (HAS_NNE(hi) && HAS_NSE(hi)) {
                gradf[1][hi] = (f[NNE(hi)] - f[NSE(hi)]) * oneover2v;
            } else {
                // Leave grady at 0
                gradf[1][hi] = 0.0;
            }
        }
    }

    /*!
     * Does: f = (alpha * f) + betaterm. c.f. Karb2004, Eq 1. f is
     * c[i] or q from the RK algorithm.
     */
    vector<double> compute_dci_dt (vector<double>& f, unsigned int i) {
        vector<double> dci_dt (this->nhex, 0.0);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; h++) {
            dci_dt[h] = (this->betaterm[i][h] - this->alpha[i] * f[h]);
        }
        return dci_dt;
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
    void compute_divJ (vector<double>& fa, unsigned int i) {

        // Three terms to compute; see Eq. 14 in methods_notes.pdf

        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa, this->grad_a[i]);

        #pragma omp parallel for schedule(dynamic,50)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            // 1. The D Del^2 a_i term
            // Compute the sum around the neighbours
            double thesum = -6 * fa[hi];

            thesum += fa[(HAS_NE(hi)?NE(hi):hi)];
            thesum += fa[(HAS_NNE(hi)?NNE(hi):hi)];
            thesum += fa[(HAS_NNW(hi)?NNW(hi):hi)];
            thesum += fa[(HAS_NW(hi)?NW(hi):hi)];
            thesum += fa[(HAS_NSW(hi)?NSW(hi):hi)];
            thesum += fa[(HAS_NSE(hi)?NSE(hi):hi)];

            // Multiply bu 2D/3d^2
            double term1 = this->twoDover3dd * thesum;

            // 2. The a div(g) term. Two sums for this.
            double term2 = 0.0;
            // First sum
            if (HAS_NE(hi)) {
                term2 += /*cos (0)*/ (this->g[i][0][NE(hi)] + this->g[i][0][hi]);
            } else {
                // Boundary condition _should_ be satisfied by
                // sigmoidal roll-off of g towards the boundary, so
                // add only g[i][0][hi]
                term2 += /*cos (0)*/ (this->g[i][0][hi]);
            }
            if (HAS_NNE(hi)) {
                term2 += /*cos (60)*/ 0.5 * (this->g[i][0][NNE(hi)] + this->g[i][0][hi])
                    + /*sin (60)*/ R3_OVER_2 * (this->g[i][1][NNE(hi)] + this->g[i][1][hi]);
            } else {
                term2 += /*cos (60)*/ 0.5 * (this->g[i][0][hi])
                    + /*sin (60)*/ R3_OVER_2 * (this->g[i][1][hi]);
            }
            if (HAS_NNW(hi)) {
                term2 += -(/*cos (120)*/ 0.5 * (this->g[i][0][NNW(hi)] + this->g[i][0][hi]))
                    + /*sin (120)*/ R3_OVER_2 * (this->g[i][1][NNW(hi)] + this->g[i][1][hi]);
            } else {
                term2 += -(/*cos (120)*/ 0.5 * (this->g[i][0][hi]))
                    + /*sin (120)*/ R3_OVER_2 * (this->g[i][1][hi]);
            }
            if (HAS_NW(hi)) {
                term2 -= /*cos (180)*/ (this->g[i][0][NW(hi)] + this->g[i][0][hi]);
            } else {
                term2 -= /*cos (180)*/ (this->g[i][0][hi]);
            }
            if (HAS_NSW(hi)) {
                term2 -= /*cos (240)*/ 0.5 * (this->g[i][0][NSW(hi)] + this->g[i][0][hi])
                    - (/*sin (240)*/ R3_OVER_2 * (this->g[i][1][NSW(hi)] + this->g[i][1][hi]));
            } else {
                term2 -= /*cos (240)*/ 0.5 * (this->g[i][0][hi])
                    - (/*sin (240)*/ R3_OVER_2 * (this->g[i][1][hi]));
            }
            if (HAS_NSE(hi)) {
                term2 += /*cos (300)*/ 0.5 * (this->g[i][0][NSE(hi)] + this->g[i][0][hi])
                    - (/*sin (300)*/ R3_OVER_2 * (this->g[i][1][NSE(hi)] + this->g[i][1][hi]));
            } else {
                term2 += /*cos (300)*/ 0.5 * (this->g[i][0][hi])      // 1st sum
                    - (/*sin (300)*/ R3_OVER_2 * (this->g[i][1][hi])); // 2nd sum
            }

            term2 /= (3.0 * this->d);
            term2 *= fa[hi];

            // 3. Third term is this->g . grad a_i. Should not
            // contribute to J, as g(x) decays towards boundary.
            double term3 = this->g[i][0][hi] * this->grad_a[i][0][hi]
                + this->g[i][1][hi] * this->grad_a[i][1][hi];

            this->divJ[i][hi] = term1 + term2 + term3;
        }
    }

    /*!
     * Generate Gaussian profiles for the chemo-attractants.
     *
     * Instead of using the Karbowski equations, just make some
     * gaussian 'waves'
     */
    void gaussian1D_guidance (void) {
#if 0
        double xoffA = this->guidance_offset[0];
        double xoffC = this->guidance_offset[2];
        double xoffB = (xoffA + xoffC) / 2.0;
#endif
        for (auto h : this->hg->hexen) {
            for (unsigned int m = 0; m<this->M; ++m) {
                double cosphi = (double) cos (guidance_phi[m]);
                double sinphi = (double) sin (guidance_phi[m]);
                DBG ("phi: " << guidance_phi[m]);
                double x_ = (h.x * cosphi) + (h.y * sinphi);
                this->rho[m][h.vi] = guidance_gain[m] * exp(-((x_-guidance_offset[m])*(x_-guidance_offset[m])) / guidance_width[m]);
            }
        }
    }

    void gaussian2D_guidance (void) {
    }

    void sigmoid_guidance (void) {
        for (auto h : this->hg->hexen) {
            for (unsigned int m = 0; m<this->M; ++m) {
                double cosphi = (double) cos (this->guidance_phi[m]); // FIXME:
                double sinphi = (double) sin (this->guidance_phi[m]);
                double x_ = (h.x * cosphi) + (h.y * sinphi);
                this->rho[m][h.vi] = guidance_gain[m] / (1.0 + exp(-(x_-guidance_offset[m])/this->guidance_width[m]));
            }
        }
    }

    /*!
     * Data saving code
     */
    //@{

    /*!
     * Get the c contours in this case.
     */
    vector<list<Hex> > get_contours (double threshold) {
        return this->get_contours (this->c, threshold);
    }

    /*!
     * Obtain the contours (as list<Hex>?) in the scalar fields f,
     * where threshold is crossed. Is some sort of list of hexes
     * right, or would list of locations (r,g,b or x,y) be better?
     */
    vector<list<Hex> > get_contours (vector<vector<double> >& f, double threshold) {

        vector<list<Hex> > rtn;
        // Initialise
        for (unsigned int li = 0; li < this->N; ++li) {
            list<Hex> lh;
            rtn.push_back (lh);
        }

        // Determine min and max
        vector<double> maxf (this->N, -1e7);
        vector<double> minf (this->N, +1e7);
        for (auto h : this->hg->hexen) {
            if (h.onBoundary() == false) {
                for (unsigned int i = 0; i<this->N; ++i) {
                    if (f[i][h.vi] > maxf[i]) { maxf[i] = f[i][h.vi]; }
                    if (f[i][h.vi] < minf[i]) { minf[i] = f[i][h.vi]; }
                }
            }
        }

        vector<double> scalef (5, 0);
        #pragma omp parallel for
        for (unsigned int i = 0; i<this->N; ++i) {
            scalef[i] = 1.0 / (maxf[i]-minf[i]);
        }

        // Re-normalize
        vector<vector<double> > norm_f;
        this->resize_vector_vector (norm_f);
        for (unsigned int i = 0; i<this->N; ++i) {
            #pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                norm_f[i][h] = fmin (fmax (((f[i][h]) - minf[i]) * scalef[i], 0.0), 1.0);
            }
        }

        // Collate
        #pragma omp parallel for
        for (unsigned int i = 0; i<this->N; ++i) {

            for (auto h : this->hg->hexen) {
                if (h.onBoundary() == false) {
#ifdef DEBUG__
                    if (!i) {
                        DBG("Hex r,g: "<< h.ri << "," << h.gi << " OFF boundary with value: " << norm_f[i][h.vi]);
                    }
#endif
                    if (norm_f[i][h.vi] > threshold) {
#ifdef DEBUG__
                        if (!i) {
                            DBG("Value over threshold...");
                        }
#endif
                        if ( (h.has_ne && norm_f[i][h.ne->vi] < threshold)
                             || (h.has_nne && norm_f[i][h.nne->vi] < threshold)
                             || (h.has_nnw && norm_f[i][h.nnw->vi] < threshold)
                             || (h.has_nw && norm_f[i][h.nw->vi] < threshold)
                             || (h.has_nsw && norm_f[i][h.nsw->vi] < threshold)
                             || (h.has_nse && norm_f[i][h.nse->vi] < threshold) ) {
#ifdef DEBUG__
                            if (!i) {
                                DBG("...with neighbour under threshold (push_back)");
                            }
#endif
                            rtn[i].push_back (h);
                        }
                    }
                } else { // h.onBoundary() is true
#ifdef DEBUG__
                    if (!i) {
                        DBG("Hex r,g: "<< h.ri << "," << h.gi << " ON boundary with value: " << norm_f[i][h.vi]);
                    }
#endif
                    if (norm_f[i][h.vi] > threshold) {
#ifdef DEBUG__
                        if (!i) {
                            DBG("...Value over threshold (push_back)");
                        }
#endif
                        rtn[i].push_back (h);
                    }
                }
            }
        }

        return rtn;
    }

#ifdef GAUSSIAN_CODE_NEEDED
    /*!
     * Create a symmetric, 1D Gaussian hill centred at coordinate (x) with
     * width sigma and height gain. Place result into @a result.
     */
    void createGaussian1D (float x, float phi, double gain, double sigma, vector<double>& result) {

        // Once-only parts of the calculation of the Gaussian.
        double root_2_pi = 2.506628275;
        double one_over_sigma_root_2_pi = 1 / sigma * root_2_pi;
        double two_sigma_sq = 2 * sigma * sigma;

        // Gaussian dist. result, and a running sum of the results:
        double gauss = 0.0;

        double cosphi = (double) cos (phi);
        double sinphi = (double) sin (phi);

        // x and y components of the vector from (x,y) to any given Hex.
        float rx = 0.0f, ry = 0.0f;

        // Calculate each element of the kernel:
        for (auto h : this->hg->hexen) {
            rx = x - h.x;
            ry = 0 - h.y;
            double x_ = (rx * cosphi) + (ry * sinphi);
            gauss = gain * (one_over_sigma_root_2_pi
                            * exp ( static_cast<double>(-(x_*x_))
                                    / two_sigma_sq ));
            result[h.vi] = gauss;
            ++k;
        }
    }

    /*!
     * Create a symmetric, 2D Gaussian hill centred at coordinate (x,y) with
     * width sigma and height gain. Place result into @a result.
     */
    void createGaussian (float x, float y, double gain, double sigma, vector<double>& result) {

        // Once-only parts of the calculation of the Gaussian.
        double root_2_pi = 2.506628275;
        double one_over_sigma_root_2_pi = 1 / sigma * root_2_pi;
        double two_sigma_sq = 2 * sigma * sigma;

        // Gaussian dist. result, and a running sum of the results:
        double gauss = 0.0;
        double sum = 0.0;

        // x and y components of the vector from (x,y) to any given Hex.
        float rx = 0.0f, ry = 0.0f;
        // distance from any Hex to (x,y)
        float r = 0.0f;

        // Calculate each element of the kernel:
        for (auto h : this->hg->hexen) {
            rx = x - h.x;
            ry = y - h.y;
            r = sqrt (rx*rx + ry*ry);
            gauss = gain * (one_over_sigma_root_2_pi
                            * exp ( static_cast<double>(-(r*r))
                                    / two_sigma_sq ));
            result[h.vi] = gauss;
            sum += gauss;
            ++k;
        }

        // Normalise the kernel to 1 by dividing by the sum:
        unsigned int j = this->nhex;
        while (j > 0) {
            --j;
            result[j] = result[j] / sum;
        }
    }
#endif

}; // RD_James
