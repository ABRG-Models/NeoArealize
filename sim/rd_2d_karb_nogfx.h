#include "tools.h"
#include "ReadCurves.h"
#include "HexGrid.h"
#include "HdfData.h"
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
#include "MorphDbg.h"

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
 * Define this to use manufactured guidance molecules, A, B and C, in
 * contrast to what Karbowski et al did.
 */
//#define MANUFACTURE_GUIDANCE_MOLECULES 1

/*!
 * Enumerates the way that the guidance molecules are set up
 */
enum class GuidanceMoleculeMethod {
    GaussWaves,
    LoadToRhoDirect,
    LoadToInitialConc,
    KarbowskiOriginal
};

/*!
 * Reaction diffusion system; 2-D Karbowski 2004.
 */
class alignas(8) RD_2D_Karb
{
public:

    /*!
     * Constants
     */
    //@{
    //! Square root of 3 over 2
    const alignas(8) double R3_OVER_2 = 0.866025403784439;
    //! Square root of 3
    const alignas(8) double ROOT3 = 1.73205080756888;
    //! Passed to HdfData constructor to say we want to read the data
    const alignas(4) bool READ_DATA = true;
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
    static const alignas(4) unsigned int N = 5;

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
     * gamma_A/B/C_i parameters from Eq 4
     */
    //@{
    alignas(8) vector<double> gammaA;
    alignas(8) vector<double> gammaB;
    alignas(8) vector<double> gammaC;
    //@}

    /*!
     * Variables for factor expression dynamics (Eqs 5-7)
     */
    //@{
    /*!
     * Uncoupled concentrations of the factors - i.e. where they start.
     */
    //@{
    alignas(8) vector<double> eta_emx;
    alignas(8) vector<double> eta_pax;
    alignas(8) vector<double> eta_fgf;
    //@}

    /*!
     * These are s(x), r(x) and f(x) in Karb2004.
     */
    //@}
    alignas(8) vector<double> emx;
    alignas(8) vector<double> pax;
    alignas(8) vector<double> fgf;
    //@}
    //@}

    /*!
     * Parameters for factor expression dynamics (Eqs 5-7)
     */
    //@{
    alignas(8) double Aemx = 1; // 1.34 in paper
    alignas(8) double Apax = 1; // 1.4 in paper
    alignas(8) double Afgf = 1; // 0.9 in paper

    // These are scaled rougly in proportion with the values in
    // Karb2004. I have about a 1mm long cortex, so their Chis are
    // divided by 40 to get these values.
    alignas(8) double Chiemx = 0.64; // 25.6/40
    alignas(8) double Chipax = 0.68; // 27.3/40
    alignas(8) double Chifgf = 0.66; // 26.4/40

    // See Karb2004 Fig 8 - the arealization duplication figure
    //@{
    alignas(4) bool useSecondFgfSource = false;
    alignas(8) double Afgfprime = 1.5;
    alignas(8) double Chifgfprime = 0.3; // 12/40
    //@}

    alignas(8) double v1 = 2.6;
    alignas(8) double v2 = 2.7;
    alignas(8) double w1 = 2.4;
    alignas(8) double w2 = 2.1;

    /*!
     * Note: Using tau_emx, tau_pax, tau_fgf in place of tau_s, tau_r, tau_f
     */
    //@{
    alignas(8) double tau_emx = 0.0001;
    alignas(8) double tau_pax = 0.0001;
    alignas(8) double tau_fgf = 0.0001;
    //@}

    /*!
     * The directions of the change (in radians) in uncoupled factor
     * concentrations
     */
    //@{
    alignas(4) float diremx = 3.141593;
    alignas(4) float dirpax = 0;
    alignas(4) float dirfgf = 0;
    //@}

    //@} end factor expression dynamics parameters

    /*!
     * Params used in the calculation of rhoA, rhoB and rhoC from the
     * final eta, pax and fgf expression levels.
     */
    //@{
    alignas(8) double sigmaA = 0.2;
    alignas(8) double sigmaB = 0.3;
    alignas(8) double sigmaC = 0.2;

    alignas(8) double kA = 0.34;
    alignas(8) double kB = 0.9;
    alignas(8) double kC = 0.3;

    alignas(8) double theta1 = 0.4;  // 0.77 originally, in the paper
    alignas(8) double theta2 = 0.5;  // 0.5
    alignas(8) double theta3 = 0.39; // 0.39
    alignas(8) double theta4 = 0.3;  // 0.08
    //@}

    /*!
     * Rho_A/B/C variables in Eq 4 - the concentrations of axon
     * guidance molecules A, B and C. In Karbowski 2004, these are
     * time independent and we will treat time as such, populating
     * them at initialisation.
     */
    //@{
    alignas(8) vector<double> rhoA;
    alignas(8) vector<double> rhoB;
    alignas(8) vector<double> rhoC;
    //@}

    /*!
     * Into grad_rhoA/B/C put the two components of the gradient of
     * rhoA/B/C computed across the HexGrid surface.
     */
    //@{
    alignas(8) array<vector<double>, 2> grad_rhoA;
    alignas(8) array<vector<double>, 2> grad_rhoB;
    alignas(8) array<vector<double>, 2> grad_rhoC;
    //@}

    /*!
     * Hex to hex distance. Populate this from hg.d after hg has been
     * initialised.
     */
    alignas(8) double d = 1.0;

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
     * Members for which alignment is not important
     */
    //@{

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
     * How to load the guidance molecules?
     */
    GuidanceMoleculeMethod rhoMethod = GuidanceMoleculeMethod::GaussWaves;

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
    //@}

    /*!
     * Simple constructor; no arguments.
     */
    RD_2D_Karb (void) {
        this->halfdt = this->dt/2.0;
        this->sixthdt = this->dt/6.0;
    }

    /*!
     * Destructor required to free up HexGrid memory
     */
    ~RD_2D_Karb (void) {
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
    void init (void) {

        DBG ("called");
        // Create a HexGrid
        this->hg = new HexGrid (this->hextohex_d, 3);
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
        this->resize_vector_vector (this->c);
        this->resize_vector_vector (this->a);
        this->resize_vector_vector (this->betaterm);
        this->resize_vector_vector (this->alpha_c_beta_na);
        this->resize_vector_vector (this->divJ);

        this->resize_vector_variable (this->n);
        this->resize_vector_variable (this->rhoA);
        this->resize_vector_variable (this->rhoB);
        this->resize_vector_variable (this->rhoC);

        this->resize_vector_variable (this->eta_emx);
        this->resize_vector_variable (this->eta_pax);
        this->resize_vector_variable (this->eta_fgf);

        this->resize_vector_variable (this->emx);
        this->resize_vector_variable (this->pax);
        this->resize_vector_variable (this->fgf);

        this->resize_vector_param (this->alpha);
        this->resize_vector_param (this->beta);
        this->resize_vector_param (this->gammaA);
        this->resize_vector_param (this->gammaB);
        this->resize_vector_param (this->gammaC);

        this->resize_gradient_field (this->grad_rhoA);
        this->resize_gradient_field (this->grad_rhoB);
        this->resize_gradient_field (this->grad_rhoC);

        // Resize grad_a and other vector-array-vectors
        this->resize_vector_array_vector (this->grad_a);
        this->resize_vector_array_vector (this->g);
        this->resize_vector_array_vector (this->J);

        // Initialise a with noise
        this->noiseify_vector_vector (this->a);

        // The gamma values - notice the symmetry here.
        // red
        this->gammaA[0] = -2.0;
        this->gammaB[0] =  0.5;
        this->gammaC[0] =  0.5;
        // yellow
        this->gammaA[1] = -2.0;
        this->gammaB[1] = -2.0;
        this->gammaC[1] =  0.5;
        // green
        this->gammaA[2] =  0.5;
        this->gammaB[2] = -2.0;
        this->gammaC[2] =  0.5;
        // blue
        this->gammaA[3] =  0.5;
        this->gammaB[3] = -2.0;
        this->gammaC[3] = -2.0;
        // magenta
        this->gammaA[4] =  0.5;
        this->gammaB[4] =  0.5;
        this->gammaC[4] = -2.0;

        for (unsigned int i=0; i<this->N; ++i) {
            this->alpha[i] = 3;
            this->beta[i] = 3;
        }

        if (this->rhoMethod == GuidanceMoleculeMethod::GaussWaves) {
            // Construct Gaussian-waves rather than doing the full-Karbowski shebang.
            this->makeupChemoAttractants();

        } else if (this->rhoMethod == GuidanceMoleculeMethod::LoadToRhoDirect) {
            // Then load up result of a RD system generating the ephrin molecular distributions
            // Load the data from files - Right now loads only rhoA/B/B
            this->loadFactorExpression();

        } else if (this->rhoMethod == GuidanceMoleculeMethod::LoadToInitialConc) {
            // Load into eta_emx, eta_pax and eta_fgf
            this->loadToInitialConc();
            // Run the expression dynamics, showing images as we go.
            this->runExpressionDynamics();
            // Can now populate rhoA, rhoB and rhoC according to the paper.
            this->populateChemoAttractants();

        } else if (this->rhoMethod == GuidanceMoleculeMethod::KarbowskiOriginal) {
            // Generate the assumed uncoupled concentrations of growth/transcription factors
            this->createFactorInitialConc (this->diremx, this->Aemx, this->Chiemx, this->eta_emx);
            this->createFactorInitialConc (this->dirpax, this->Apax, this->Chipax, this->eta_pax);

            // Should we use a second Fgf source, as in the Karbowski paper (Fig 8)?
            if (this->useSecondFgfSource) {
                this->createFactorInitialConc (this->dirfgf, this->Afgf, this->Afgfprime,
                                               this->Chifgf, this->Chifgfprime, this->eta_fgf);
            } else {
                this->createFactorInitialConc (this->dirfgf, this->Afgf, this->Chifgf, this->eta_fgf);
            }
            // Run the expression dynamics, showing images as we go.
            this->runExpressionDynamics();
            // Can now populate rhoA, rhoB and rhoC according to the paper.
            this->populateChemoAttractants();
        }

        // Compute gradients of guidance molecule concentrations once only
        this->spacegrad2D (this->rhoA, this->grad_rhoA);
        this->spacegrad2D (this->rhoB, this->grad_rhoB);
        this->spacegrad2D (this->rhoC, this->grad_rhoC);

        // Having computed gradients, build this->g; has
        // to be done once only. Note that a sigmoid is applied so
        // that g(x) drops to zero around the boundary of the domain.
        for (unsigned int i=0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                // Sigmoid/logistic fn params: 100 sharpness, 0.02 dist offset from boundary
                double bSig = 1.0 / ( 1.0 + exp (-100.0*(h.distToBoundary-0.02)) );
                this->g[i][0][h.vi] = (this->gammaA[i] * this->grad_rhoA[0][h.vi]
                                       + this->gammaB[i] * this->grad_rhoB[0][h.vi]
                                       + this->gammaC[i] * this->grad_rhoC[0][h.vi]) * bSig;
                this->g[i][1][h.vi] = (this->gammaA[i] * this->grad_rhoA[1][h.vi]
                                       + this->gammaB[i] * this->grad_rhoB[1][h.vi]
                                       + this->gammaC[i] * this->grad_rhoC[1][h.vi]) * bSig;
            }
        }

        // Save that data out
        this->saveFactorExpression();
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

    /*!
     * Save the results of running createFactorInitialConc(),
     * runExpressionDynamics() and populateChemoAttractants().
     */
    void saveFactorExpression (void) {
        string fname = this->logpath + "/factorexpression.h5";
        cout << "Saving to file " << fname << endl;
        HdfData data (fname);

        // Save initial factor calculations - required in
        // createFactorInitialConc() and runExpressionDynamics()
        data.add_double ("/Aemx", this->Aemx);
        data.add_double ("/Apax", this->Apax);
        data.add_double ("/Afgf", this->Afgf);

        data.add_double ("/Chiemx", this->Chiemx);
        data.add_double ("/Chipax", this->Chipax);
        data.add_double ("/Chifgf", this->Chifgf);

        data.add_double ("/tau_emx", this->tau_emx);
        data.add_double ("/tau_pax", this->tau_pax);
        data.add_double ("/tau_fgf", this->tau_fgf);

        data.add_float ("/diremx", this->diremx);
        data.add_float ("/dirpax", this->dirpax);
        data.add_float ("/dirfgf", this->dirfgf);

        data.add_double ("/v1", this->v1);
        data.add_double ("/v2", this->v2);

        data.add_double ("/w1", this->w1);
        data.add_double ("/w2", this->w2);

        // Signalling molecule expression levels
        data.add_double_vector ("/emx", this->emx);
        data.add_double_vector ("/pax", this->pax);
        data.add_double_vector ("/fgf", this->fgf);

        data.add_double_vector ("/eta_emx", this->eta_emx);
        data.add_double_vector ("/eta_pax", this->eta_pax);
        data.add_double_vector ("/eta_fgf", this->eta_fgf);

        // parameters and vars for populateChemoAttractants
        data.add_double ("/sigmaA", this->sigmaA);
        data.add_double ("/sigmaB", this->sigmaB);
        data.add_double ("/sigmaC", this->sigmaC);

        data.add_double ("/kA", this->kA);
        data.add_double ("/kB", this->kB);
        data.add_double ("/kC", this->kC);

        data.add_double ("/theta1", this->theta1);
        data.add_double ("/theta2", this->theta2);
        data.add_double ("/theta3", this->theta3);
        data.add_double ("/theta4", this->theta4);

        // The axon guidance molecule expression levels
        data.add_double_vector ("/rhoA", this->rhoA);
        data.add_double_vector ("/rhoB", this->rhoB);
        data.add_double_vector ("/rhoC", this->rhoC);

        // And gradient thereof
        data.add_double_vector ("/grad_rhoA_x", this->grad_rhoA[0]);
        data.add_double_vector ("/grad_rhoA_y", this->grad_rhoA[1]);
        data.add_double_vector ("/grad_rhoB_x", this->grad_rhoB[0]);
        data.add_double_vector ("/grad_rhoB_y", this->grad_rhoB[1]);
        data.add_double_vector ("/grad_rhoC_x", this->grad_rhoC[0]);
        data.add_double_vector ("/grad_rhoC_y", this->grad_rhoC[1]);

        // g - the guidance molecular modifier on a.
        data.add_double_vector ("/g_0_x", this->g[0][0]);
        data.add_double_vector ("/g_0_y", this->g[0][1]);
        data.add_double_vector ("/g_1_x", this->g[1][0]);
        data.add_double_vector ("/g_1_y", this->g[1][1]);
        data.add_double_vector ("/g_2_x", this->g[2][0]);
        data.add_double_vector ("/g_2_y", this->g[2][1]);
        data.add_double_vector ("/g_3_x", this->g[3][0]);
        data.add_double_vector ("/g_3_y", this->g[3][1]);
        data.add_double_vector ("/g_4_x", this->g[4][0]);
        data.add_double_vector ("/g_4_y", this->g[4][1]);

        this->saveHexPositions (data);
    }

    /*!
     * Load the results of running createFactorInitialConc(),
     * runExpressionDynamics() and populateChemoAttractants().
     */
    void loadFactorExpression (void) {
        // The only thing I'll load for now is the rhoA/B/C values.
        HdfData data ("./logs/e0/2Derm.h5", READ_DATA);
        data.read_double_vector ("/c_0", this->rhoA);
        if (rhoA.size() != this->nhex) {
            throw runtime_error ("Guidance molecules came from HexGrid with different size from this one...");
        }
        DBG ("rhoA now has size " << this->rhoA.size() << " with first two values: " << rhoA[0] << "," << rhoA[1]);
        data.read_double_vector ("/c_1", this->rhoB);
        data.read_double_vector ("/c_2", this->rhoC);

        this->normalise (this->rhoA);
        this->normalise (this->rhoB);
        this->normalise (this->rhoC);
    }

    void loadToInitialConc (void) {
        HdfData data ("./logs/e0/2Derm.h5", READ_DATA);
        data.read_double_vector ("/c_0", this->eta_pax);
        if (eta_pax.size() != this->nhex) {
            throw runtime_error ("Guidance molecules came from HexGrid with different size from this one...");
        }
        data.read_double_vector ("/c_1", this->eta_emx);
        data.read_double_vector ("/c_2", this->eta_fgf);

        this->normalise (this->eta_pax);
        this->normalise (this->eta_emx);
        this->normalise (this->eta_fgf);
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
     *
     * T460s manages about 20 of these per second.
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
            Hex* h = this->hg->vhexen[hi];
            n[h->vi] = 0;
            for (unsigned int i=0; i<N; ++i) {
                n[h->vi] += c[i][h->vi];
            }
            csum += c[0][h->vi];
            n[h->vi] = 1. - n[h->vi];
            nsum += n[h->vi];
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
        // No OMP here - only N<10 loops
        for (unsigned int i=0; i<this->N; ++i) {

            // Runge-Kutta integration for A
            vector<double> q(this->nhex, 0.0);
            // Function call is bad already!
            this->compute_divJ (a[i], i); // populates divJ[i]

            vector<double> k1(this->nhex, 0.0);
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k1[h] * halfdt;
            }

            vector<double> k2(this->nhex, 0.0);
            this->compute_divJ (q, i);
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k2[h] * halfdt;
            }

            vector<double> k3(this->nhex, 0.0);
            this->compute_divJ (q, i);
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] + this->alpha_c_beta_na[i][h];
                q[h] = this->a[i][h] + k3[h] * dt;
            }

            vector<double> k4(this->nhex, 0.0);
            this->compute_divJ (q, i);
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
            DBG2 ("(c) betaterm[" << i << "][0]: " << betaterm[i][0]);

            // Runge-Kutta integration for C (or ci)
            vector<double> q(nhex,0.);
            vector<double> k1 = compute_dci_dt (c[i], i);
            #pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                q[h] = c[i][h] + k1[h] * halfdt;
            }
            DBG2 ("(c) After RK stage 1, q[0]: " << q[0]);

            vector<double> k2 = compute_dci_dt (q, i);
            #pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                q[h] = c[i][h] + k2[h] * halfdt;
            }
            DBG2 ("(c) After RK stage 2, q[0]: " << q[0]);

            vector<double> k3 = compute_dci_dt (q, i);
            #pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                q[h] = c[i][h] + k3[h] * dt;
            }
            DBG2 ("(c) After RK stage 3, q[0]: " << q[0]);

            vector<double> k4 = compute_dci_dt (q, i);
            #pragma omp parallel for
            for (unsigned int h=0; h<nhex; h++) {
                c[i][h] += (k1[h]+2. * (k2[h] + k3[h]) + k4[h]) * sixthdt;
            }
            DBG2 ("(c) After RK stage 4, c["<<i<<"][0]: " << c[i][0]);

            DBG2("(c) Debug c["<<i<<"]");
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
            Hex* h = this->hg->vhexen[hi];

            gradf[0][h->vi] = 0.0;
            gradf[1][h->vi] = 0.0;

            // Find x gradient
            if (h->has_ne && h->has_nw) {
                gradf[0][h->vi] = (f[h->ne->vi] - f[h->nw->vi]) / ((double)h->d * 2.0);
            } else if (h->has_ne) {
                gradf[0][h->vi] = (f[h->ne->vi] - f[h->vi]) / (double)h->d;
            } else if (h->has_nw) {
                gradf[0][h->vi] = (f[h->vi] - f[h->nw->vi]) / (double)h->d;
            } else {
                // zero gradient in x direction as no neighbours in
                // those directions? Or possibly use the average of
                // the gradient between the nw,ne and sw,se neighbours
            }

            // Find y gradient
            if (h->has_nnw && h->has_nne && h->has_nsw && h->has_nse) {
                // Full complement. Compute the mean of the nse->nne and nsw->nnw gradients
                gradf[1][h->vi] = ((f[h->nne->vi] - f[h->nse->vi]) + (f[h->nnw->vi] - f[h->nsw->vi])) / (double)h->getV();

            } else if (h->has_nnw && h->has_nne ) {
                //if (h->vi == 0) { DBG ("y case 2"); }
                gradf[1][h->vi] = ( (f[h->nne->vi] + f[h->nnw->vi]) / 2.0 - f[h->vi]) / (double)h->getV();

            } else if (h->has_nsw && h->has_nse) {
                //if (h->vi == 0) { DBG ("y case 3"); }
                gradf[1][h->vi] = (f[h->vi] - (f[h->nse->vi] + f[h->nsw->vi]) / 2.0) / (double)h->getV();

            } else if (h->has_nnw && h->has_nsw) {
                //if (h->vi == 0) { DBG ("y case 4"); }
                gradf[1][h->vi] = (f[h->nnw->vi] - f[h->nsw->vi]) / (double)h->getTwoV();

            } else if (h->has_nne && h->has_nse) {
                //if (h->vi == 0) { DBG ("y case 5"); }
                gradf[1][h->vi] = (f[h->nne->vi] - f[h->nse->vi]) / (double)h->getTwoV();
            } else {
                // Leave grady at 0
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
            dci_dt[h] = this->betaterm[i][h] - this->alpha[i] * f[h];
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

            Hex* h = this->hg->vhexen[hi];
            // 1. The D Del^2 a_i term
            // Compute the sum around the neighbours
            double thesum = -6 * fa[h->vi];
            if (h->has_ne) {
                thesum += fa[h->ne->vi];
            } else {
                // Apply boundary condition
            }
            if (h->has_nne) {
                thesum += fa[h->nne->vi];
            } else {
                thesum += fa[h->vi]; // A ghost neighbour-east with same value as Hex_0
            }
            if (h->has_nnw) {
                thesum += fa[h->nnw->vi];
            } else {
                thesum += fa[h->vi];
            }
            if (h->has_nw) {
                thesum += fa[h->nw->vi];
            } else {
                thesum += fa[h->vi];
            }
            if (h->has_nsw) {
                thesum += fa[h->nsw->vi];
            } else {
                thesum += fa[h->vi];
            }
            if (h->has_nse) {
                thesum += fa[h->nse->vi];
            } else {
                thesum += fa[h->vi];
            }
            // Multiply bu 2D/3d^2
            double term1 = (this->D * 2) / (3 * this->d * this->d) * thesum;

            // 2. The a div(g) term. Two sums for this.
            double term2 = 0.0;
            // First sum
            if (h->has_ne) {
                term2 += /*cos (0)*/ (this->g[i][0][h->ne->vi] + this->g[i][0][h->vi]);
            } else {
                // Boundary condition _should_ be satisfied by
                // sigmoidal roll-off of g towards the boundary, so
                // add only g[i][0][h->vi]
                term2 += /*cos (0)*/ (this->g[i][0][h->vi]);
            }
            if (h->has_nne) {
                term2 += /*cos (60)*/ 0.5 * (this->g[i][0][h->nne->vi] + this->g[i][0][h->vi]);
            } else {
                term2 += /*cos (60)*/ 0.5 * (this->g[i][0][h->vi]);
            }
            if (h->has_nnw) {
                term2 -= /*cos (120)*/ 0.5 * (this->g[i][0][h->nnw->vi] + this->g[i][0][h->vi]);
            } else {
                term2 -= /*cos (120)*/ 0.5 * (this->g[i][0][h->vi]);
            }
            if (h->has_nw) {
                term2 -= /*cos (180)*/ (this->g[i][0][h->nw->vi] + this->g[i][0][h->vi]);
            } else {
                term2 -= /*cos (180)*/ (this->g[i][0][h->vi]);
            }
            if (h->has_nsw) {
                term2 -= /*cos (240)*/ 0.5 * (this->g[i][0][h->nsw->vi] + this->g[i][0][h->vi]);
            } else {
                term2 -= /*cos (240)*/ 0.5 * (this->g[i][0][h->vi]);
            }
            if (h->has_nse) {
                term2 += /*cos (300)*/ 0.5 * (this->g[i][0][h->nse->vi] + this->g[i][0][h->vi]);
            } else {
                term2 += /*cos (300)*/ 0.5 * (this->g[i][0][h->vi]);
            }
            // 2nd sum
            //term2 += sin (0) * (this->g[i][1][h->ne->vi] + this->g[i][1][h->vi]);
            if (h->has_nne) {
                term2 += /*sin (60)*/ R3_OVER_2 * (this->g[i][1][h->nne->vi] + this->g[i][1][h->vi]);
            } else {
                term2 += /*sin (60)*/ R3_OVER_2 * (this->g[i][1][h->vi]);
            }
            if (h->has_nnw) {
                term2 += /*sin (120)*/ R3_OVER_2 * (this->g[i][1][h->nnw->vi] + this->g[i][1][h->vi]);
            } else {
                term2 += /*sin (120)*/ R3_OVER_2 * (this->g[i][1][h->vi]);
            }
            //term2 += sin (180) * (this->g[i][1][h->nw->vi] + this->g[i][1][h->vi]);
            if (h->has_nsw) {
                term2 -= /*sin (240)*/ R3_OVER_2 * (this->g[i][1][h->nsw->vi] + this->g[i][1][h->vi]);
            } else {
                term2 -= /*sin (240)*/ R3_OVER_2 * (this->g[i][1][h->vi]);
            }
            if (h->has_nse) {
                term2 -= /*sin (300)*/ R3_OVER_2 * (this->g[i][1][h->nse->vi] + this->g[i][1][h->vi]);
            } else {
                term2 -= /*sin (300)*/ R3_OVER_2 * (this->g[i][1][h->vi]);
            }

            term2 /= (3.0 * this->d);
            term2 *= fa[h->vi];

            // 3. Third term is this->g . grad a_i. Should not
            // contribute to J, as g(x) decays towards boundary.
            double term3 = this->g[i][0][h->vi] * this->grad_a[i][0][h->vi]
                + this->g[i][1][h->vi] * this->grad_a[i][1][h->vi];

            this->divJ[i][h->vi] = term1 + term2 + term3;
        }
    }

    /*!
     * Create a 2-D scalar field which follows a curve along one
     * direction (at angle @a phi radians, anti-clockwise from East),
     * being constant in the orthogonal direction. Place result into
     * @a result.
     *
     * @param Afac 'A' parameter for factor fac. c.f. Aemx, Apax, etc
     * in Karb2004.
     *
     * @param chifac 'chi' parameter for factor fac. c.f. Chi_emx, Chi_pax, etc
     */
    void createFactorInitialConc (float phi, double Afac, double chifac, vector<double>& result) {

        // Work in a co-ordinate system rotated by phi radians, called x_, y_
        double x_ = 0.0;

        double cosphi = (double) cos (phi);
        double sinphi = (double) sin (phi);
        DBG2 ("cosphi: " << cosphi);
        // Get minimum x and maximum x in the rotated co-ordinate system.
        double x_min_ = this->hg->getXmin (phi);
        DBG2 ("x_min_: " << x_min_);

        for (auto h : this->hg->hexen) {
            // Rotate x, then offset by the minimum along that line
            x_ = (h.x * cosphi) + (h.y * sinphi) - x_min_;
            // x_ here is x from the Hex.
            result[h.vi] = Afac * exp (-(x_ * x_) / (chifac * chifac));
        }
    }

    /*!
     * A special version of createFactorInitialConc which introduces a
     * second source for the factor - see Karb2004 Fig 8.
     */
    void createFactorInitialConc (float phi,
                                  double Afac, double Afacprime,
                                  double chifac, double chifacprime,
                                  vector<double>& result) {

        // Work in a co-ordinate system rotated by phi radians, called x_, y_
        double x_ = 0.0;
        double x_rev = 0.0;

        double cosphi = (double) cos (phi);
        double sinphi = (double) sin (phi);

        // Get minimum x and maximum x in the rotated co-ordinate system.
        double x_min_ = this->hg->getXmin (phi);
        double x_max_ = this->hg->getXmax (phi);

        for (auto h : this->hg->hexen) {
            // Rotate x, then offset by the minimum along that line
            x_ = (h.x * cosphi) + (h.y * sinphi) - x_min_;
            x_rev = x_max_ - (h.x * cosphi) + (h.y * sinphi);
            // x_ here is x from the Hex.
            result[h.vi] = Afac * exp (-(x_ * x_) / (chifac * chifac)) + Afacprime * exp (-(x_rev * x_rev) / (chifacprime * chifacprime));
        }
    }

    /*!
     * Execute Eqs 5-7 of the Karbowski paper to find the steady state
     * of the growth/transcription factors after they have interacted
     * for a long time.
     */
    void runExpressionDynamics (void) {
        #pragma omp parallel for
        for (unsigned int t=0; t<300000; ++t) { // 300000 matches Stuart's 1D Karbowski model
            for (auto h : this->hg->hexen) {
                emx[h.vi] += tau_emx * (-emx[h.vi] + eta_emx[h.vi] / (1. + w2 * fgf[h.vi] + v2 * pax[h.vi]));
                pax[h.vi] += tau_pax * (-pax[h.vi] + eta_pax[h.vi] / (1. + v1 * emx[h.vi]));
                fgf[h.vi] += tau_fgf * (-fgf[h.vi] + eta_fgf[h.vi] / (1. + w1 * emx[h.vi]));
            }
        }
    }

    /*!
     * Using this->fgf and some hyperbolic tangents, populate rhoA/B/C
     */
    void populateChemoAttractants (void) {
        // chemo-attraction gradient. cf Fig 1 of Karb 2004
        #pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            this->rhoA[h] = (kA/2.)*(1.+tanh((fgf[h]-theta1)/sigmaA));
            this->rhoB[h] = (kB/2.)*(1.+tanh((theta2-fgf[h])/sigmaB))*(kB/2.)*(1.+tanh((fgf[h]-theta3)/sigmaB));
            this->rhoC[h] = (kC/2.)*(1.+tanh((theta4-fgf[h])/sigmaC));
        }
    }

    /*!
     * Generate Gaussian profiles for the chemo-attractants.
     *
     * Instead of using the Karbowski equations, just make some
     * gaussian 'waves'
     */
    void makeupChemoAttractants (void) {

        // Potentially alter angle of Gaussian wave:
        double phi = M_PI*0.0;

        // Gaussian params:
        double sigma = 0.1;
        double gain = 1;

        // Positions of the bumps:
        double xoffA = -0.2;
        double xoffC = 0.24;
        double xoffB = (xoffA + xoffC) / 2.0;

        double cosphi = (double) cos (phi);
        double sinphi = (double) sin (phi);

        for (auto h : this->hg->hexen) {
            double x_ = (h.x * cosphi) + (h.y * sinphi);
            this->rhoA[h.vi] = gain * exp(-((x_-xoffA)*(x_-xoffA)) / sigma);
            this->rhoB[h.vi] = gain * exp(-((x_-xoffB)*(x_-xoffB)) / sigma);
            this->rhoC[h.vi] = gain * exp(-((x_-xoffC)*(x_-xoffC)) / sigma);
        }
    }


}; // RD_2D_Karb
