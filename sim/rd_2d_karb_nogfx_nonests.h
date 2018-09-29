/*
 * sub-parallelograms without nested vectors version
 *
 * This version of rd_2d_karb is going to create some
 * sub-parallelograms inside the arbitrary boundary. Computing the
 * step within these parallelogram shaped regions should be fast, and
 * will avoid all the branching which I think is slowing down my
 * existing code.
 *
 * The sub-parallelogram setup will occur in HexGrid.
 *
 * This class will be substantially complicated, because the RD data
 * will be held in at least two vectors - one for the single
 * parallelogram, one for the rest of the hexes. In the case of
 * multiple parallelograms, there'll be a vector of vectors, one for
 * each parallelogram.
 *
 */

#include <morph/tools.h>
#include <morph/ReadCurves.h>
#include <morph/HexGrid.h>
#include <morph/HdfData.h>
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
 * Macros for d_ vector data
 */
//@{

/*!
 * Macros for testing neighbours for d_ vectors. These'll change
 * slightly for the case where a neighbour is in a subparallelogram.
 */
//@{
#define D_NE(hi) (this->hg->d_ne[hi])   // iterator into vector to find the NE neigbour
#define D_V_NE(hi) (this->hg->d_v_ne[hi]) // *which* vector to iterate into to find NE neighbour
#define D_HAS_NE(hi) (this->hg->d_ne[hi] == -1 ? false : true)

#define D_NW(hi) (this->hg->d_nw[hi])
#define D_V_NW(hi) (this->hg->d_v_nw[hi])
#define D_HAS_NW(hi) (this->hg->d_nw[hi] == -1 ? false : true)

#define D_NNE(hi) (this->hg->d_nne[hi])
#define D_V_NNE(hi) (this->hg->d_v_nne[hi])
#define D_HAS_NNE(hi) (this->hg->d_nne[hi] == -1 ? false : true)

#define D_NNW(hi) (this->hg->d_nnw[hi])
#define D_V_NNW(hi) (this->hg->d_v_nnw[hi])
#define D_HAS_NNW(hi) (this->hg->d_nnw[hi] == -1 ? false : true)

#define D_NSE(hi) (this->hg->d_nse[hi])
#define D_V_NSE(hi) (this->hg->d_v_nse[hi])
#define D_HAS_NSE(hi) (this->hg->d_nse[hi] == -1 ? false : true)

#define D_NSW(hi) (this->hg->d_nsw[hi])
#define D_V_NSW(hi) (this->hg->d_v_nsw[hi])
#define D_HAS_NSW(hi) (this->hg->d_nsw[hi] == -1 ? false : true)
//@}

//@}

/*!
 * Macros for the sub-parallelogram data regions
 */
//@{
/*!
 * Macros for testing neighbours. For a sub-parallelogram we
 * engineered its size such that:
 *
 * If it's *not* on the boundary of the sub-parallelogram, it
 * definitely has neighbours. So these tests are only used for the
 * hexes on the boundary of the subp, and in fact incur the usual
 * computational drawbacks.
 */
//@{
#define SP_NE(vi,hi) (this->hg->sp_ne[vi][hi])
#define SP_V_NE(vi,hi) (this->hg->sp_v_ne[vi][hi])
#define SP_HAS_NE(vi,hi) (this->hg->sp_nsw[vi][hi] == -1 ? false : true)

#define SP_NW(vi,hi) (this->hg->sp_nw[vi][hi])
#define SP_V_NW(vi,hi) (this->hg->sp_v_nw[vi][hi])
#define SP_HAS_NW(vi,hi) (this->hg->sp_nw[vi][hi] == -1 ? false : true)

#define SP_NNE(vi,hi) (this->hg->sp_nne[vi][hi])
#define SP_V_NNE(vi,hi) (this->hg->sp_v_nne[vi][hi])
#define SP_HAS_NNE(vi,hi) (this->hg->sp_nne[vi][hi] == -1 ? false : true)

#define SP_NNW(vi,hi) (this->hg->sp_nnw[vi][hi])
#define SP_V_NNW(vi,hi) (this->hg->sp_v_nnw[vi][hi])
#define SP_HAS_NNW(vi,hi) (this->hg->sp_nnw[vi][hi] == -1 ? false : true)

#define SP_NSW(vi,hi) (this->hg->sp_nsw[vi][hi])
#define SP_V_NSW(vi,hi) (this->hg->sp_v_nsw[vi][hi])
#define SP_HAS_NSW(vi,hi) (this->hg->sp_nsw[vi][hi] == -1 ? false : true)

#define SP_NSE(vi,hi) (this->hg->sp_nse[vi][hi])
#define SP_V_NSE(vi,hi) (this->hg->sp_v_nse[vi][hi])
#define SP_HAS_NSE(vi,hi) (this->hg->sp_nse[vi][hi] == -1 ? false : true)
//@}

/*!
 * Offsets to gain neighbours when within the body of a parallelogram
 * of rowlength _rl
 */
//@{
#define SP_OFFS_NE       (1)
#define SP_OFFS_NW       (-1)
#define SP_OFFS_NNE(_rl) (_rl)
#define SP_OFFS_NNW(_rl) (_rl-1)
#define SP_OFFS_NSW(_rl) (-_rl)
#define SP_OFFS_NSE(_rl) (1-_rl)
//@}

//@}

/*!
 * Enumerates the way that the guidance molecules are set up
 */
enum class GuidanceMoleculeMethod {
    GaussWaves,
    LoadToRhoDirect,
    LoadToInitialConc,
    KarbowskiOriginal
};

#define NL 5 // Num layers
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
     * Holds the number of hexes in the d_ vectors - i.e. the length
     * of any of the d vectors.
     */
    alignas(4) unsigned int nhex_d = 0;

    /*!
     * how many thalamo-cortical axon types are there? Denoted by N in
     * the paper, and so we use N here too. This gives relevant strides through vectors of pointers.
     */
    static const alignas(4) unsigned int N = NL;

    /*!
     * These are the c_i(x,t) variables from the Karb2004 paper. x is
     * a vector in two-space.
     */
    //alignas(8) vector       <vector<double> > c_d;
    alignas(8) array<double*, NL> c_d; // array of 5 pointers to double arrays

    /*!
     * The companion data container to c_d. The outer-most vector
     * covers the i fields. the inner vector<vector<double> > are the
     * vectors, one for each sub-parallelogram.
     */
    //alignas(8) vector<vector<vector<double> > > c_sp;
    alignas(8) vector<double*> c_sp; // vector of (5 * num_sub-parallelograms) pointers to double arrays

    /*!
     * These are the a_i(x,t) variables from the Karb2004 paper. x is
     * a vector in two-space. The first vector is over the different
     * TC axon types, enumerated by i, the second vector are the a_i
     * values, indexed by the vi in the Hexes in HexGrid.
     */
    //alignas(8) vector       <vector<double> > a_d;
    alignas(8) array<double*, NL> a_d;

    /*!
     * Here, the first vector STILL enumerates over TC axon type (i);
     * the second vector over the sub-parallelogram vector (vi) and
     * the third over hexes.
     *
     * General rule followed here: If, for an attribute 'attr' which
     * is split up into attr_d and attr_sp; if the outer vector of
     * attr_d enumerates over TC type, then the outer vector of
     * attr_sp also enumerates over TC type.
     *
     * NEW: Now this is a single vector of contiguous double
     * arrays. These are arranged grouped by TC type. That means that
     * if there are 5 TC types and 2 sub-parallelograms, then a_sp[0]
     * is vi==0, i==0; a_sp[1] is vi==1, i==0; a_sp[2] is vi==0, i==1,
     * and so on.
     */
    //alignas(8) vector<vector<vector<double> > > a_sp;
    alignas(8) vector<double*> a_sp;

    /*!
     * This holds the two components of the gradient field of the
     * scalar value a(x,t), interleaved in the double array. The
     * vector holds a double array for each combination of TC type i
     * and sub-parallelogram vector vi.
     */
    alignas(8) array<double*, NL> grad_a_d;
    /*!
     * Here, the first vector STILL enumerates over TC axon type (i);
     * the second vector over the sub-parallelogram vector
     * (vi). Grouped in the usual way, vi group/i=0, vi group/i=1, etc.
     */
    alignas(8) vector<double*> grad_a_sp;

    /*!
     * Contains the chemo-attractant modifiers which are applied to
     * a_i(x,t) in Eq 4. There are 5 elements here, and each element
     * is a vector field, with interleaved x and y values (x first).
     */
    //alignas(8) vector       <array<vector<double>, 2> > g_d;
    alignas(8) array<double*, NL> g_d;

    /*!
     * The vector iterates over vi and i, with vi grouped, so we have
     * i=0, vi=0 to n, i=1, vi=0 to n, etc. The double* is interleaved
     * x and y coordinates.
     *
     */
    alignas(8) vector<double*> g_sp;

    /*!
     * n(x,t) variable from the Karb2004 paper.
     */
    //alignas(8)        vector<double> n_d;
    alignas(8) double* n_d;

    /*!
     * Outer vector enumerates over subparallelogram vector vi.
     */
    alignas(8) vector<double*> n_sp;

    /*!
     * J_i(x,t) variables - the "flux current of axonal branches of
     * type i". This is a vector field.
     */
    alignas(8) array<double*, NL> J_d;
    /*!
     * Outer vector enumerates over TC type (i), second enumerates
     * over sub-parallelogram vector (vi).
     */
    alignas(8) vector<double*> J_sp;

    /*!
     * Holds the divergence of the J_i(x)s
     */
    //alignas(8) vector       <vector<double> > divJ_d;
    alignas(8) array<double*, NL> divJ_d;

    /*!
     * Outer vector enumerates over TC type (i), second enumerates
     * over sub-parallelogram vector (vi).
     */
    //alignas(8) vector<vector<vector<double> > > divJ_sp;
    alignas(8) vector<double*> divJ_sp;

    /*!
     * Intermediate terms computed for divJ
     */
    //@{
    alignas(8) double* term1_d;
    alignas(8) vector<double*> term1_sp;
    alignas(8) double* term2_d;
    alignas(8) vector<double*> term2_sp;
    alignas(8) double* term3_d;
    alignas(8) vector<double*> term3_sp;
    //@}

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
    alignas(8)        vector<double> eta_emx_d;
    /*!
     * Outer vector enumerates over subparallelogram (vi)
     */
    alignas(8) vector<vector<double> > eta_emx_sp;

    alignas(8)        vector<double> eta_pax_d;
    /*!
     * Outer vector enumerates over subparallelogram (vi)
     */
    alignas(8) vector<vector<double> > eta_pax_sp;

    alignas(8)        vector<double> eta_fgf_d;
    /*!
     * Outer vector enumerates over subparallelogram (vi)
     */
    alignas(8) vector<vector<double> > eta_fgf_sp;
    //@}

    /*!
     * These are s(x), r(x) and f(x) in Karb2004.
     */
    //@}
    alignas(8)        vector<double> emx_d;
    alignas(8) vector<vector<double> > emx_sp;
    alignas(8)        vector<double> pax_d;
    alignas(8) vector<vector<double> > pax_sp;
    alignas(8)        vector<double> fgf_d;
    alignas(8) vector<vector<double> > fgf_sp;
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
    alignas(8)        double* rhoA_d;
    alignas(8) vector<double*> rhoA_sp;
    alignas(8)        double* rhoB_d;
    alignas(8) vector<double*> rhoB_sp;
    alignas(8)        double* rhoC_d;
    alignas(8) vector<double*> rhoC_sp;
    //@}

    /*!
     * Into grad_rhoA/B/C put the two components of the gradient of
     * rhoA/B/C computed across the HexGrid surface.
     */
    //@{
    // Decision to make: two arrays or interleaved values in one
    // array? Probably interleaved, as x and y components are
    // read/written in nearby code. Yes. That.
    alignas(8)        double*  grad_rhoA_d;
    alignas(8) vector<double*> grad_rhoA_sp;
    alignas(8)        double*  grad_rhoB_d;
    alignas(8) vector<double*> grad_rhoB_sp;
    alignas(8)        double*  grad_rhoC_d;
    alignas(8) vector<double*> grad_rhoC_sp;
    //@

    /*!
     * Temporary variables used in step()
     */
    //@{
    alignas(8) double* q_d;
    alignas(8) vector<double*> q_sp;
    alignas(8) double* k1_d;
    alignas(8) vector<double*> k1_sp;
    alignas(8) double* k2_d;
    alignas(8) vector<double*> k2_sp;
    alignas(8) double* k3_d;
    alignas(8) vector<double*> k3_sp;
    alignas(8) double* k4_d;
    alignas(8) vector<double*> k4_sp;
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
     * Memory to hold an intermediate result. First vector enumerates
     * over TC type (i).
     */
    //alignas(8) vector       <vector<double> > betaterm_d;
    alignas(8) array<double*, NL> betaterm_d;

    /*!
     * Memory to hold an intermediate result. First vector enumerates
     * over TC type (i); second over subparallelogram vector (vi)
     */
    //alignas(8) vector<vector<vector<double> > > betaterm_sp;
    alignas(8) vector<double*> betaterm_sp;

    /*!
     * Holds an intermediate value for the computation of Eqs 1 and
     * 2. First vector is the TC type, 'i', second vector are values
     * for each element in the set 'd'.
     */
    //alignas(8) vector       <vector<double> > alpha_c_beta_na_d;
    alignas(8) array<double*, NL> alpha_c_beta_na_d;

    /*!
     * First vector is the TC type, 'i'. Second vector is the vector
     * identifier, third vector are values for each element in the set
     * 'd'.
     */
    //alignas(8) vector<vector<vector<double> > > alpha_c_beta_na_sp;
    alignas(8) vector<double*> alpha_c_beta_na_sp;

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
     * Members for which alignment is not important
     */
    //@{

    /*!
     * The HexGrid "background" for the Reaction Diffusion system.
     */
    HexGrid* hg;

    /*!
     * Sets domainMode for hg. Set up before calling init().
     */
    bool domainMode = false;

    /*!
     * Rowlen and numrows in the parallelogram domain.
     */
    //@{
    //unsigned int d_rl = 0;
    //unsigned int d_nr = 0;
    //@}

    /*!
     * How to load the guidance molecules?
     */
    GuidanceMoleculeMethod rhoMethod = GuidanceMoleculeMethod::GaussWaves;

    /*!
     * The logpath for this model. Used when saving data out.
     */
    string logpath = "logs";

    /*!
     * The number of subparallelograms in the boundary
     */
    unsigned int nv;

    /*!
     * Setter which attempts to ensure the path exists.
     */
    void setLogpath (const string p) {
        this->logpath = p;
        // Ensure log directory exists
        morph::Tools::createDir (this->logpath);
    }

    /*!
     * Which SVG to open for the boundary. Can be changed before
     * calling init().
     */
    string svgpath = "./trial.svg";

    //@} // align-unimportant members

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

    // Old version, will become deprecated...
    void resize_vector_vector (vector<vector<double> >& vv) {
        vv.resize (this->N);
        for (unsigned int i=0; i<this->N; ++i) {
            vv[i].resize (this->nhex, 0.0);
        }
    }

    /*!
     * A utility function to resize the vector-vectors that hold a
     * variable for the N different thalamo-cortical axon types.
     */
    void alloc_d_vector_vector (array<double*, NL>& vv) {
        for (unsigned int i=0; i<this->N; ++i) {
            vv[i] = (double*)_mm_malloc (this->hg->d_x.size()*sizeof(double), 64);
        }
    }
    void alloc_sp_vector_vector (vector<double*>& vvv) {
        vvv.resize(NL*this->nv);
        for (unsigned int i = 0; i < NL; ++i) {
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                vvv[vi+(i*this->nv)] = (double*)_mm_malloc (this->hg->sp_veclen[vi]*sizeof(double), 64);
            }
        }
    }

    /*!
     * Resize a variable that'll be nhex elements long
     */
    void alloc_d_vector_variable (double* v) {
        v = (double*)_mm_malloc (this->nhex_d*sizeof(double), 64);
    }
    void alloc_sp_vector_variable (vector<double*>& vv) {
        vv.resize (this->hg->sp_numvecs);
        for (unsigned int j=0; j<this->hg->sp_numvecs; ++j) {
            vv[j] = (double*)_mm_malloc (this->hg->sp_veclen[j]*sizeof(double), 64);
        }
    }

    void resize_d_vector_variable (vector<double>& v) {
        v.resize (this->hg->d_x.size(), 0.0);
    }
    void resize_sp_vector_variable (vector<vector<double> >& vv) {
        vv.resize (this->hg->sp_numvecs);
        for (unsigned int j=0; j<this->hg->sp_numvecs; ++j) {
            vv[j].resize (this->hg->sp_veclen[j], 0);
        }
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
    void resize_d_gradient_field (array<vector<double>, 2>& g) {
        g[0].resize (this->hg->d_x.size(), 0.0);
        g[1].resize (this->hg->d_x.size(), 0.0);
    }
    void resize_sp_gradient_field (vector<array<vector<double>, 2> >& g) {
        g.resize (this->hg->sp_numvecs);
        for (unsigned int j=0; j<this->hg->sp_numvecs; ++j) {
            g[j][0].resize (this->hg->sp_veclen[j], 0.0);
            g[j][1].resize (this->hg->sp_veclen[j], 0.0);
        }
    }
    /*!
     * A gradient field is just interleaved
     */
    void alloc_d_gradient_field (double* gf) {
        gf = (double*)_mm_malloc (2*this->nhex_d*sizeof(double), 64);
    }
    void alloc_sp_gradient_field (vector<double*>& gfs) {
        gfs.resize (this->hg->sp_numvecs);
        for (unsigned int j=0; j<this->hg->sp_numvecs; ++j) {
            gfs[j] = (double*)_mm_malloc (2*this->hg->sp_veclen[j]*sizeof(double), 64);
        }
    }

    /*!
     * Resize a vector (over TC types i) of an array of two
     * vector<double>s which are the x and y components of a
     * (mathematical) vector field.
     */
    void resize_d_vector_array_vector (vector<array<vector<double>, 2> >& vav) {
        vav.resize(this->N);
        for (unsigned int i = 0; i<this->N; ++i) {
            this->resize_d_gradient_field (vav[i]);
        }
    }
    void resize_sp_vector_array_vector (vector<vector<array<vector<double>, 2> > >& vvav) {
        vvav.resize (this->N);
        for (unsigned int i = 0; i<this->N; ++i) {
            this->resize_sp_gradient_field (vvav[i]);
        }
    }
    void alloc_d_vector_array_vector (array<double*, NL>& vav) {
        for (unsigned int i = 0; i<this->N; ++i) {
            vav[i] = (double*)_mm_malloc (2*this->nhex_d*sizeof(double), 64);
        }
    }
    void alloc_sp_vector_array_vector (vector<double*>& vvav) {
        vvav.resize (NL * this->nv);
        for (unsigned int i = 0; i<NL; ++i) {
            for (unsigned int j = 0; j<(this->nv); ++j) {
                vvav[(i*this->nv)+j] = (double*)_mm_malloc (2*this->hg->sp_veclen[j]*sizeof(double), 64);
            }
        }
    }

    /*!
     * Initialise this vector of vectors with noise. This is a
     * model-specific function.
     *
     * I apply a sigmoid to the boundary hexes, so that the noise
     * drops away towards the edge of the domain.
     */
    void noiseify_vector_vector (array<double*, 5>& vv_d, vector<double*>& vvv_sp) {
        double randNoiseOffset = 0.8;
        double randNoiseGain = 0.1;
        for (unsigned int i = 0; i<this->N; ++i) {
            //#pragma omp parallel for
            for (unsigned int hi=0; hi < this->hg->d_x.size(); ++hi) {
                // boundarySigmoid. Jumps sharply (100, larger is
                // sharper) over length scale 0.05 to 1. So if
                // distance from boundary > 0.05, noise has normal
                // value. Close to boundary, noise is less.
                vv_d[i][hi] = morph::Tools::randDouble() * randNoiseGain + randNoiseOffset;
                if (hg->d_distToBoundary[hi] > -0.5) { // It's possible that distToBoundary is set to -1.0
                    double bSig = 1.0 / ( 1.0 + exp (-100.0*(hg->d_distToBoundary[hi]-0.02)) );
                    vv_d[i][hi] = vv_d[i][hi] * bSig;
                }
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                DBG ("this->hg->sp_numvecs=" << this->hg->sp_numvecs << ", this->hg->sp_veclen[vi]=" << this->hg->sp_veclen[vi]);
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    unsigned int ivi = vi+(i*this->nv);
                    vvv_sp[ivi][hi] = morph::Tools::randDouble() * randNoiseGain + randNoiseOffset;
                    if (hg->sp_distToBoundary[vi][hi] > -0.5) { // It's possible that distToBoundary is set to -1.0
                        double bSig = 1.0 / ( 1.0 + exp (-100.0*(hg->sp_distToBoundary[vi][hi]-0.02)) );
                        vvv_sp[ivi][hi] = vvv_sp[ivi][hi] * bSig;
                    }
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
        this->hg = new HexGrid (this->hextohex_d, 3, 0, morph::HexDomainShape::SubParallelograms);
        // Read the curves which make a boundary
        ReadCurves r(this->svgpath);
        // Set the boundary in the HexGrid
        this->hg->setBoundary (r.getCorticalPath());
        // Total number of hexes comes from number of Hexes in the HexGrid
        this->nhex = this->hg->num();
        this->nhex_d = this->hg->d_x.size();
        DBG("Number of hexes: " << nhex);
        DBG("Number of non-subp hexes: " << nhex_d);
        DBG("Number of subp hexes: " << this->hg->sp_veclen[0]);
        DBG("These add up to: " << (nhex_d + this->hg->sp_veclen[0]));
        this->nv = this->hg->sp_numvecs;
        // row length and num rows
        //this->d_rl = this->hg->d_rowlen;
        //this->d_nr = this->hg->d_numrows;
        // Spatial d and v come from the HexGrid, too.
        this->d = this->hg->getd();
        this->v = this->hg->getv();
        // Various 1-overs and permutations computed once only
        this->oneoverd = 1.0/this->d;
        this->oneoverv = 1.0/this->v;
        this->twov = this->v+this->v;
        this->oneover2v = 1.0/this->twov;
        this->oneover2d = 1.0/(this->d+this->d);
        this->twoDover3dd = (this->D * 2) / (3 * this->d * this->d);

        // Intermediate terms used in compute_divJ()
        this->alloc_d_vector_variable (this->term1_d);
        this->alloc_d_vector_variable (this->term2_d);
        this->alloc_d_vector_variable (this->term3_d);

        // Resize and zero-initialise the various containers
        this->alloc_d_vector_vector (this->c_d);
        this->alloc_sp_vector_vector (this->c_sp);
        this->alloc_d_vector_vector (this->a_d);
        this->alloc_sp_vector_vector (this->a_sp);
        this->alloc_d_vector_vector (this->betaterm_d);
        this->alloc_sp_vector_vector (this->betaterm_sp);
        this->alloc_d_vector_vector (this->alpha_c_beta_na_d);
        this->alloc_sp_vector_vector (this->alpha_c_beta_na_sp);
        this->alloc_d_vector_vector (this->divJ_d);
        this->alloc_sp_vector_vector (this->divJ_sp);

        this->alloc_d_vector_variable (this->n_d);
        this->alloc_sp_vector_variable (this->n_sp);

        this->alloc_d_vector_variable (this->rhoA_d);
        this->alloc_sp_vector_variable (this->rhoA_sp);
        this->alloc_d_vector_variable (this->rhoB_d);
        this->alloc_sp_vector_variable (this->rhoB_sp);
        this->alloc_d_vector_variable (this->rhoC_d);
        this->alloc_sp_vector_variable (this->rhoC_sp);

        this->alloc_d_vector_variable (this->q_d);
        this->alloc_sp_vector_variable (this->q_sp);
        this->alloc_d_vector_variable (this->k1_d);
        this->alloc_sp_vector_variable (this->k1_sp);
        this->alloc_d_vector_variable (this->k2_d);
        this->alloc_sp_vector_variable (this->k2_sp);
        this->alloc_d_vector_variable (this->k3_d);
        this->alloc_sp_vector_variable (this->k3_sp);
        this->alloc_d_vector_variable (this->k4_d);
        this->alloc_sp_vector_variable (this->k4_sp);

        this->resize_d_vector_variable (this->eta_emx_d);
        this->resize_sp_vector_variable (this->eta_emx_sp);
        this->resize_d_vector_variable (this->eta_pax_d);
        this->resize_sp_vector_variable (this->eta_pax_sp);
        this->resize_d_vector_variable (this->eta_fgf_d);
        this->resize_sp_vector_variable (this->eta_fgf_sp);

        this->resize_d_vector_variable (this->emx_d);
        this->resize_sp_vector_variable (this->emx_sp);
        this->resize_d_vector_variable (this->pax_d);
        this->resize_sp_vector_variable (this->pax_sp);
        this->resize_d_vector_variable (this->fgf_d);
        this->resize_sp_vector_variable (this->fgf_sp);

        this->resize_vector_param (this->alpha);
        this->resize_vector_param (this->beta);
        this->resize_vector_param (this->gammaA);
        this->resize_vector_param (this->gammaB);
        this->resize_vector_param (this->gammaC);

        this->alloc_d_gradient_field (this->grad_rhoA_d);
        this->alloc_sp_gradient_field (this->grad_rhoA_sp);
        this->alloc_d_gradient_field (this->grad_rhoB_d);
        this->alloc_sp_gradient_field (this->grad_rhoB_sp);
        this->alloc_d_gradient_field (this->grad_rhoC_d);
        this->alloc_sp_gradient_field (this->grad_rhoC_sp);

        // Resize grad_a and other vector-array-vectors
        this->alloc_d_vector_array_vector (this->grad_a_d);
        this->alloc_sp_vector_array_vector (this->grad_a_sp);
        this->alloc_d_vector_array_vector (this->g_d);
        this->alloc_sp_vector_array_vector (this->g_sp);
        this->alloc_d_vector_array_vector (this->J_d);
        this->alloc_sp_vector_array_vector (this->J_sp);

        // Initialise a with noise
        this->noiseify_vector_vector (this->a_d, this->a_sp);

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
            //this->loadFactorExpression();

        } else if (this->rhoMethod == GuidanceMoleculeMethod::LoadToInitialConc) {
            // Load into eta_emx, eta_pax and eta_fgf
            ////this->loadToInitialConc();
            // Run the expression dynamics, showing images as we go.
            this->runExpressionDynamics();
            // Can now populate rhoA, rhoB and rhoC according to the paper.
            this->populateChemoAttractants();

        } else if (this->rhoMethod == GuidanceMoleculeMethod::KarbowskiOriginal) {
            // Generate the assumed uncoupled concentrations of growth/transcription factors
            this->createFactorInitialConc (this->diremx, this->Aemx, this->Chiemx, this->eta_emx_d);
            this->createFactorInitialConc (this->dirpax, this->Apax, this->Chipax, this->eta_pax_d);

            // Should we use a second Fgf source, as in the Karbowski paper (Fig 8)?
            if (this->useSecondFgfSource) {
                this->createFactorInitialConc (this->dirfgf, this->Afgf, this->Afgfprime,
                                               this->Chifgf, this->Chifgfprime, this->eta_fgf_d);
            } else {
                this->createFactorInitialConc (this->dirfgf, this->Afgf, this->Chifgf, this->eta_fgf_d);
            }
            // Run the expression dynamics, showing images as we go.
            this->runExpressionDynamics();
            // Can now populate rhoA, rhoB and rhoC according to the paper.
            this->populateChemoAttractants();
        }

        // Compute gradients of guidance molecule concentrations once only
        this->spacegrad2D (this->rhoA_d, this->rhoA_sp, 0, this->grad_rhoA_d, this->grad_rhoA_sp);
        this->spacegrad2D (this->rhoB_d, this->rhoB_sp, 0, this->grad_rhoB_d, this->grad_rhoB_sp);
        this->spacegrad2D (this->rhoC_d, this->rhoC_sp, 0, this->grad_rhoC_d, this->grad_rhoC_sp);

        // Having computed gradients, build this->g; has
        // to be done once only. Note that a sigmoid is applied so
        // that g(x) drops to zero around the boundary of the domain.
        for (unsigned int i=0; i<this->N; ++i) {

            // Sigmoid/logistic fn params: 100 sharpness, 0.02 dist offset from boundary
            //#pragma omp parallel for
            for (unsigned int gi=0; gi < 2*this->nhex_d; ++gi) {
                double bSig = 1.0 / ( 1.0 + exp (-100.0*(this->hg->d_distToBoundary[gi]-0.02)) );
                this->g_d[i][gi] = (this->gammaA[i] * this->grad_rhoA_d[gi]
                                       + this->gammaB[i] * this->grad_rhoB_d[gi]
                                       + this->gammaC[i] * this->grad_rhoC_d[gi]) * bSig;
                this->g_d[i][++gi] = (this->gammaA[i] * this->grad_rhoA_d[gi]
                                       + this->gammaB[i] * this->grad_rhoB_d[gi]
                                       + this->gammaC[i] * this->grad_rhoC_d[gi]) * bSig;
            }
            unsigned int inv = i * this->nv;
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                unsigned int ivi = inv + vi;
                //#pragma omp parallel for
                for (unsigned int gi=2, hi=1; gi < 2*this->hg->sp_veclen[vi]-2; ++gi, ++hi) {
                    double bSig = 1.0 / ( 1.0 + exp (-100.0*(this->hg->sp_distToBoundary[vi][hi]-0.02)) );
                    this->g_sp[ivi][gi] = (this->gammaA[i] * this->grad_rhoA_sp[vi][gi]
                                           + this->gammaB[i] * this->grad_rhoB_sp[vi][gi]
                                           + this->gammaC[i] * this->grad_rhoC_sp[vi][gi]) * bSig;
                    // NB: gi is incremented once here and once in the for clause, above to make for +2 per loop
                    this->g_sp[ivi][++gi] = (this->gammaA[i] * this->grad_rhoA_sp[vi][gi]
                                           + this->gammaB[i] * this->grad_rhoB_sp[vi][gi]
                                           + this->gammaC[i] * this->grad_rhoC_sp[vi][gi]) * bSig;
                }
            }
        }

        // Save that data out
        //this->saveFactorExpression();
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
#if 0
            data.add_double_vector (path.str().c_str(), this->c_d[i]);
#endif
        }
        //FIXME also save c_sp!
        this->saveHexPositions (data);
    }

    /*!
     * Save positions of the hexes - note using two vector<floats>
     * that have been populated with the positions from the HexGrid,
     * to fit in with the HDF API.
      */
    void saveHexPositions (HdfData& dat) {
        dat.add_float_vector ("/x", this->hg->d_x);
        dat.add_float_vector ("/y", this->hg->d_y);
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
        // FIXME Could add omp pragma here.
        for (unsigned int fi = 0; fi < f.size(); ++fi) {
            f[fi] = fmin (fmax (((f[fi]) - minf) * scalef, 0.0), 1.0);
        }
    }

    /*!
     * Do a step through the model.
     *
     * T460s manages about 20 of these per second.
     *
     * On Alienware, and at 23 Sept 2018, I'm doing 660 per second.
     */
    void step (void) {

        this->stepCount++;

        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
        }

        // 1. Compute Karb2004 Eq 3. (coupling between connections made by each TC type)
        double nsum = 0.0;
        double csum = 0.0;
        //#pragma omp parallel for reduction(+:nsum,csum)
        for (unsigned int hi=0; hi<this->nhex_d; ++hi) {
            n_d[hi] = 0;
            for (unsigned int i=0; i<N; ++i) {
                n_d[hi] += c_d[i][hi];
            }
            // Make csum, nsum operate only within boundary.
            csum += c_d[0][hi];
            n_d[hi] = 1. - n_d[hi];
            nsum += n_d[hi];
        }
        for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
            double nsum1 = 0.0;
            double csum1 = 0.0;
            //#pragma omp parallel for reduction(+:nsum1,csum1)
            for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {

                n_sp[vi][hi] = 0;
                for (unsigned int i=0; i<N; ++i) {
                    n_sp[vi][hi] += c_sp[vi+(i*this->nv)][hi];
                }
                // Make csum, nsum operate only within boundary.
                csum1 += c_sp[0+(NL*vi)][hi];
                n_sp[vi][hi] = 1. - n_sp[vi][hi];
                nsum1 += n_sp[vi][hi];
            }
            nsum += nsum1;
            csum += csum1;
        }

        if (this->stepCount % 100 == 0) {
            DBG ("sum of all n is " << nsum);
            DBG ("sum of all c for i=0 is " << csum);
        }

        // 2. Do integration of a (RK in the 1D model). Involves computing axon branching flux.

        // Pre-compute intermediate val:
        for (unsigned int i=0; i<this->N; ++i) {
            DBG2 ("After coupling compute, c["<<i<<"][4159]: " << c[i][4159]);
            DBG2 ("After coupling compute, n["<<i<<"][4159]: " << n[4159]);
#pragma omp simd
            for (unsigned int hi=0; hi<this->nhex_d; ++hi) {
                this->alpha_c_beta_na_d[i][hi] = (alpha[i] * c_d[i][hi] - beta[i] * n_d[hi] * pow (a_d[i][hi], k));
            }

            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
// NB In the next one, memory access is much worse. Nested vectors not a good idea?
#pragma omp simd
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    this->alpha_c_beta_na_sp[vi+(i*this->nv)][hi] = (alpha[i] * c_sp[vi+(i*this->nv)][hi] - beta[i] * n_sp[vi][hi] * pow (a_sp[vi+(i*this->nv)][hi], k));
                }
            }
        }

        // Runge-Kutta:
        // No OMP here - there are only N(<10) loops, which isn't
        // enough to load the threads up.
        for (unsigned int i=0; i<this->N; ++i) {

            // Runge-Kutta integration for A
            this->compute_divJ (a_d[i], a_sp, i); // populates divJ_d/sp[i]

            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex_d; ++hi) {
                k1_d[hi] = this->divJ_d[i][hi] + this->alpha_c_beta_na_d[i][hi];
                q_d[hi] = this->a_d[i][hi] + k1_d[hi] * halfdt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    k1_sp[vi][hi] = this->divJ_sp[vi+(i*this->nv)][hi] + this->alpha_c_beta_na_sp[vi+(i*this->nv)][hi];
                    q_sp[vi][hi] = this->a_sp[vi+(i*this->nv)][hi] + k1_sp[vi][hi] * halfdt;
                }
            }

            this->compute_divJ (q_d, q_sp, i);
            //#pragma omp parallel for
#pragma omp simd
            for (unsigned int hi=0; hi<this->nhex_d; ++hi) {
                k2_d[hi] = this->divJ_d[i][hi] + this->alpha_c_beta_na_d[i][hi];
                q_d[hi] = this->a_d[i][hi] + k2_d[hi] * halfdt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
#pragma omp simd
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    k2_sp[vi][hi] = this->divJ_sp[vi+(i*this->nv)][hi] + this->alpha_c_beta_na_sp[vi+(i*this->nv)][hi];
                    q_sp[vi][hi] = this->a_sp[vi+(i*this->nv)][hi] + k2_sp[vi][hi] * halfdt;
                }
            }

            this->compute_divJ (q_d, q_sp, i);
            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex_d; ++hi) {
                k3_d[hi] = this->divJ_d[i][hi] + this->alpha_c_beta_na_d[i][hi];
                q_d[hi] = this->a_d[i][hi] + k3_d[hi] * dt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    k3_sp[vi][hi] = this->divJ_sp[vi+(i*this->nv)][hi] + this->alpha_c_beta_na_sp[vi+(i*this->nv)][hi];
                    q_sp[vi][hi] = this->a_sp[vi+(i*this->nv)][hi] + k3_sp[vi][hi] * dt;
                }
            }

            this->compute_divJ (q_d, q_sp, i);
            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex_d; ++hi) {
                k4_d[hi] = this->divJ_d[i][hi] + this->alpha_c_beta_na_d[i][hi];
                a_d[i][hi] += (k1_d[hi] + 2.0 * (k2_d[hi] + k3_d[hi]) + k4_d[hi]) * sixthdt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    k4_sp[vi][hi] = this->divJ_sp[vi+(i*this->nv)][hi] + this->alpha_c_beta_na_sp[vi+(i*this->nv)][hi];
                    a_sp[vi+(i*this->nv)][hi] += (k1_sp[vi][hi] + 2.0 * (k2_sp[vi][hi] + k3_sp[vi][hi]) + k4_sp[vi][hi]) * sixthdt;
                }
            }
        } // end for each i in N

        // 3. Do integration of c
        for (unsigned int i=0; i<this->N; ++i) {

            // Pre-compute betaterm
            //#pragma omp parallel for
#pragma omp simd
            for (unsigned int hi=0; hi<this->nhex_d; hi++) {
                this->betaterm_d[i][hi] = (beta[i] * n_d[hi] * pow (a_d[i][hi], k));
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
#pragma omp simd
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    this->betaterm_sp[vi+(i*this->nv)][hi] = (beta[i] * n_sp[vi][hi] * pow (a_sp[vi+(i*this->nv)][hi], k));
                }
            }

            // Runge-Kutta integration for C (or ci)
            compute_dci_dt (c_d[i], c_sp, i, k1_d, k1_sp);
            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex_d; hi++) {
                q_d[hi] = c_d[i][hi] + k1_d[hi] * halfdt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    q_sp[vi][hi] = c_sp[vi+(i*this->nv)][hi] + k1_sp[vi][hi] * halfdt;
                }
            }
            DBG2 ("(c) After RK stage 1, q[4159]: " << q[4159]);

            compute_dci_dt (q_d, q_sp, i, k2_d, k2_sp);

            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex_d; hi++) {
                q_d[hi] = c_d[i][hi] + k2_d[hi] * halfdt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    q_sp[vi][hi] = c_sp[vi+(i*this->nv)][hi] + k2_sp[vi][hi] * halfdt;
                }
            }
            DBG2 ("(c) After RK stage 2, q[4159]: " << q[4159]);

            compute_dci_dt (q_d, q_sp, i, k3_d, k3_sp);

            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex_d; hi++) {
                q_d[hi] = c_d[i][hi] + k3_d[hi] * dt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    q_sp[vi][hi] = c_sp[vi+(i*this->nv)][hi] + k3_sp[vi][hi] * dt;
                }
            }
            DBG2 ("(c) After RK stage 3, q[4159]: " << q[4159]);

            compute_dci_dt (q_d, q_sp, i, k4_d, k4_sp);

            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex_d; hi++) {
                c_d[i][hi] += (k1_d[hi] + 2.0 * (k2_d[hi] + k3_d[hi]) + k4_d[hi]) * sixthdt;
            }
            for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
                //#pragma omp parallel for
                for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                    c_sp[vi+(i*this->nv)][hi] += (k1_sp[vi][hi] + 2.0 * (k2_sp[vi][hi] + k3_sp[vi][hi]) + k4_sp[vi][hi]) * sixthdt;
                }
            }
            DBG2 ("(c) After RK stage 4, c["<<i<<"][4159]: " << c[i][4159]);
            DBG2 ("(c) After RK stage 4, n["<<i<<"][4159]: " << n[4159]);

            DBG2("(c) Debug c["<<i<<"]");
        }
    }

    /*!
     * Does: f = (alpha * f) + betaterm. c.f. Karb2004, Eq 1. f is
     * c[i] or q from the RK algorithm.
     */
    void compute_dci_dt (double* f_d, vector<double*>& f_sp, unsigned int i,
                         double* dci_dt_d, vector<double*>& dci_dt_sp) {
        //#pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex_d; hi++) {
            dci_dt_d[hi] = (this->betaterm_d[i][hi] - this->alpha[i] * f_d[hi]);
        }
        unsigned int inv = i * this->nv;
        for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
            //#pragma omp parallel for
            for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                dci_dt_sp[vi][hi] = (this->betaterm_sp[inv+vi][hi] - this->alpha[i] * f_sp[vi][hi]);
            }
        }
    }

    /*!
     * 2D spatial integration of the function f, which is split across
     * f_d and f_sp according to the data layout in this system
     * (HexGrid's morph::HexDomainShape::SubParallelograms
     * layout). Result placed in gradf_d/gradf_sp.
     *
     * For each Hex, work out the gradient in x and y directions
     * using whatever neighbours can contribute to an estimate.
     */
    void spacegrad2D (double* f_d, vector<double*>& f_sp, unsigned int i,
                      double* gradf_d, vector<double*>& gradf_sp) {

        /*
         * Carry out the computations on the different data
         * containers. First up, the d_ vector, then the sp_ vectors.
         */
        unsigned int nhd = this->nhex_d;
        unsigned int inv = i * this->nv; // offset into f_sp.

        // Note - East is positive x; North is positive y. Does this match how it's drawn in the display??
        //#pragma omp parallel for schedule(static)
//#pragma omp simd // THIS LOOP DOESN'T BENEFIT FROM VECTORIZATION - (unavoidable memory access issues)
        for (unsigned int hi=0, gi=0; hi<nhd; ++hi, gi+=2) {

            // Find x gradient
            if (D_HAS_NE(hi) && D_HAS_NW(hi)) {
                // A neighbour hex may either be found in f_d *or* in f_sp!
                double _w = (this->hg->d_v_nw[hi] == -1) ? f_d[D_NW(hi)] : f_sp[this->hg->d_v_nw[hi]][D_NW(hi)];
                double _e = (this->hg->d_v_ne[hi] == -1) ? f_d[D_NE(hi)] : f_sp[this->hg->d_v_ne[hi]][D_NE(hi)];
                gradf_d[gi] = (_e - _w) * oneover2d;

            } else if (D_HAS_NE(hi)) {
                double _e = (this->hg->d_v_ne[hi] == -1) ? f_d[D_NE(hi)] : f_sp[this->hg->d_v_ne[hi]][D_NE(hi)];
                gradf_d[gi] = (_e - f_d[hi]) * oneoverd;

            } else if (D_HAS_NW(hi)) {
                double _w = (this->hg->d_v_nw[hi] == -1) ? f_d[D_NW(hi)] : f_sp[this->hg->d_v_nw[hi]][D_NW(hi)];
                gradf_d[gi] = (f_d[hi] - _w) * oneoverd;
            } else {
                // zero gradient in x direction as no neighbours in
                // those directions? Or possibly use the average of
                // the gradient between the nw,ne and sw,se neighbours
                gradf_d[gi] = 0.0;
            }
        }

//#pragma omp simd // No benefit from SIMD
        for (unsigned int hi=0, gi=1; hi<nhd; ++hi, gi+=2) {
            // Find y gradient
            if (D_HAS_NNW(hi) && D_HAS_NNE(hi) && D_HAS_NSW(hi) && D_HAS_NSE(hi)) {
                // Full complement. Compute the mean of the nse->nne and nsw->nnw gradients
                double _nw = (this->hg->d_v_nnw[hi] == -1) ? f_d[D_NNW(hi)] : f_sp[inv+this->hg->d_v_nnw[hi]][D_NNW(hi)];
                double _ne = (this->hg->d_v_nne[hi] == -1) ? f_d[D_NNE(hi)] : f_sp[inv+this->hg->d_v_nne[hi]][D_NNE(hi)];
                double _sw = (this->hg->d_v_nsw[hi] == -1) ? f_d[D_NSW(hi)] : f_sp[inv+this->hg->d_v_nsw[hi]][D_NSW(hi)];
                double _se = (this->hg->d_v_nse[hi] == -1) ? f_d[D_NSE(hi)] : f_sp[inv+this->hg->d_v_nse[hi]][D_NSE(hi)];
                gradf_d[gi] = ((_ne - _se) + (_nw - _sw)) * oneoverv;

            } else if (D_HAS_NNW(hi) && D_HAS_NNE(hi)) {
                double _nw = (this->hg->d_v_nnw[hi] == -1) ? f_d[D_NNW(hi)] : f_sp[inv+this->hg->d_v_nnw[hi]][D_NNW(hi)];
                double _ne = (this->hg->d_v_nne[hi] == -1) ? f_d[D_NNE(hi)] : f_sp[inv+this->hg->d_v_nne[hi]][D_NNE(hi)];
                gradf_d[gi] = ( (_ne + _nw) * 0.5 - f_d[hi]) * oneoverv;

            } else if (D_HAS_NSW(hi) && D_HAS_NSE(hi)) {
                double _sw = (this->hg->d_v_nsw[hi] == -1) ? f_d[D_NSW(hi)] : f_sp[inv+this->hg->d_v_nsw[hi]][D_NSW(hi)];
                double _se = (this->hg->d_v_nse[hi] == -1) ? f_d[D_NSE(hi)] : f_sp[inv+this->hg->d_v_nse[hi]][D_NSE(hi)];
                gradf_d[gi] = (f_d[hi] - (_se + _sw) * 0.5) * oneoverv;

            } else if (D_HAS_NNW(hi) && D_HAS_NSW(hi)) {
                double _nw = (this->hg->d_v_nnw[hi] == -1) ? f_d[D_NNW(hi)] : f_sp[inv+this->hg->d_v_nnw[hi]][D_NNW(hi)];
                double _sw = (this->hg->d_v_nsw[hi] == -1) ? f_d[D_NSW(hi)] : f_sp[inv+this->hg->d_v_nsw[hi]][D_NSW(hi)];
                gradf_d[gi] = (_nw - _sw) * oneover2v;

            } else if (D_HAS_NNE(hi) && D_HAS_NSE(hi)) {
                double _ne = (this->hg->d_v_nne[hi] == -1) ? f_d[D_NNE(hi)] : f_sp[inv+this->hg->d_v_nne[hi]][D_NNE(hi)];
                double _se = (this->hg->d_v_nse[hi] == -1) ? f_d[D_NSE(hi)] : f_sp[inv+this->hg->d_v_nse[hi]][D_NSE(hi)];
                gradf_d[gi] = (_ne - _se) * oneover2v;
            } else {
                // Leave grady at 0
                gradf_d[gi] = 0.0;
            }
        } // end for over d_ vectors

        /*
         * Compute over the sub-parallelogram vectors.
         */
        for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {

            /*
             * For each sub-parallelogram, compute gradient for the
             * hexes along the four edges, which will look a bit like
             * the code for the d_vectors because we have to make
             * tests of whether these edge hexes have neighbours in
             * the d_vectors or adjacent sub-parallelograms.
             */

            // Each sub-parallelogram vector has a row length and a vector length.
            unsigned int rl = this->hg->sp_rowlens[vi];
            unsigned int vl = this->hg->sp_veclen[vi];

            unsigned int ivi = inv + vi;

            /*
             * Compute bottom edge, for which most of the top
             * neighbours are guaranteed
             *
             * Do first and last hexes outside of loop, then also have
             * guarantee of left and right hexes.
             *
             * At hi==1, neighbour NE, NW & E guaranteed, and in the
             * same vector; neighbour W, SE, SW are not
             *
             * At hi==rl-1, neighbour NW, NE & W guaranteed, and in
             * the same vector; neighbour E, SW, SE are not
             */

            // x gradient for *Hex* at position 1
            gradf_sp[ivi][2] = SP_HAS_NW(vi,1) ?                                \
                ((f_sp[ivi][2] - f_sp[inv+SP_V_NW(vi,1)][SP_NW(vi,1)]) * oneover2d) \
                : ((f_sp[ivi][2] - f_sp[ivi][1]) * oneoverd);
            // x gradient for Hex at position rl-1
            gradf_sp[ivi][2*(rl-1)] = SP_HAS_NE(vi,rl-1) ?                                   \
                (f_sp[inv+SP_V_NE(vi,rl-1)][SP_NE(vi,rl-1)] - (f_sp[ivi][rl-2]) * oneover2d) \
                : ((f_sp[ivi][rl-1] - f_sp[ivi][rl-2]) * oneoverd);
            // y gradient for Hex at position 1
            if (SP_HAS_NSW(vi,1) && SP_HAS_NSE(vi,1)) {
                double _nw = f_sp[ivi][1+SP_OFFS_NNW(rl)];
                double _ne = f_sp[ivi][1+SP_OFFS_NNE(rl)];
                double _sw = (SP_V_NSW(vi,1) == -1) ? f_d[SP_NSW(vi,1)] : f_sp[inv+SP_V_NSW(vi,1)][SP_NSW(vi,1)];
                double _se = (SP_V_NSE(vi,1) == -1) ? f_d[SP_NSE(vi,1)] : f_sp[inv+SP_V_NSE(vi,1)][SP_NSE(vi,1)];
                gradf_sp[ivi][3] = ((_ne - _se) + (_nw - _sw)) * oneoverv;
            } else if (SP_HAS_NSW(vi,1)) {
                double _nw = f_sp[ivi][1+SP_OFFS_NNW(rl)];
                double _sw = (SP_V_NSW(vi,1) == -1) ? f_d[SP_NSW(vi,1)] : f_sp[inv+SP_V_NSW(vi,1)][SP_NSW(vi,1)];
                gradf_sp[ivi][3] = (_nw - _sw) * oneover2v;
            } else if (SP_HAS_NSE(vi,1)) {
                double _ne = f_sp[ivi][1+SP_OFFS_NNE(rl)];
                double _se = (SP_V_NSE(vi,1) == -1) ? f_d[SP_NSE(vi,1)] : f_sp[inv+SP_V_NSE(vi,1)][SP_NSE(vi,1)];
                gradf_sp[ivi][3] = (_ne - _se) * oneover2v;
            } else {
                gradf_sp[ivi][3] = 0.0;
            }
            // y gradient for Hex at position rl-1
            if (SP_HAS_NSW(vi,rl-1) && SP_HAS_NSE(vi,rl-1)) {
                double _nw = f_sp[ivi][rl-1+SP_OFFS_NNW(rl)];
                double _ne = f_sp[ivi][rl-1+SP_OFFS_NNE(rl)];
                double _sw = (SP_V_NSW(vi,rl-1) == -1) ? f_d[SP_NSW(vi,rl-1)] : f_sp[inv+SP_V_NSW(vi,rl-1)][SP_NSW(vi,rl-1)];
                double _se = (SP_V_NSE(vi,rl-1) == -1) ? f_d[SP_NSE(vi,rl-1)] : f_sp[inv+SP_V_NSE(vi,rl-1)][SP_NSE(vi,rl-1)];
                // [2*rl-1] == [2*(rl-1)-1]
                gradf_sp[ivi][2*rl-1] = ((_ne - _se) + (_nw - _sw)) * oneoverv;
            } else if (SP_HAS_NSW(vi,rl-1)) {
                double _nw = f_sp[vi+ivi][rl-1+SP_OFFS_NNW(rl)];
                double _sw = (SP_V_NSW(vi,rl-1) == -1) ? f_d[SP_NSW(vi,rl-1)] : f_sp[inv+SP_V_NSW(vi,rl-1)][SP_NSW(vi,rl-1)];
                gradf_sp[ivi][2*rl-1] = (_nw - _sw) * oneover2v;
            } else if (SP_HAS_NSE(vi,rl-1)) {
                double _ne = f_sp[vi+ivi][rl-1+SP_OFFS_NNE(rl)];
                double _se = (SP_V_NSE(vi,rl-1) == -1) ? f_d[SP_NSE(vi,rl-1)] : f_sp[inv+SP_V_NSE(vi,rl-1)][SP_NSE(vi,rl-1)];
                gradf_sp[ivi][2*rl-1] = (_ne - _se) * oneover2v;
            } else {
                gradf_sp[ivi][2*rl-1] = 0.0;
            }
            // Now loop along the main part of the bottom edge
            // Compute x gradient for main bottom edge
            //#pragma omp parallel for
            for (unsigned int hi=2, gi=4; hi < rl-1; ++hi, gi+=2) {
                gradf_sp[ivi][gi] = (f_sp[ivi][hi+1] - f_sp[ivi][hi-1]) * oneover2d;
            }
            // Compute y gradient for main bottom edge
            for (unsigned int hi=2, gi=5; hi < rl-1; ++hi, gi+=2) {
                // Find y gradient for hex in vector vi, position hi.
                if (SP_HAS_NSW(vi,hi) && SP_HAS_NSE(vi,hi)) {
                    // Full complement. Compute the mean of the nse->nne and nsw->nnw gradients
                    double _nw = (SP_V_NNW(vi,hi) == -1) ? f_d[SP_NNW(vi,hi)] : f_sp[inv+SP_V_NNW(vi,hi)][SP_NNW(vi,hi)];
                    double _ne = (SP_V_NNE(vi,hi) == -1) ? f_d[SP_NNE(vi,hi)] : f_sp[inv+SP_V_NNE(vi,hi)][SP_NNE(vi,hi)];
                    double _sw = (SP_V_NSW(vi,hi) == -1) ? f_d[SP_NSW(vi,hi)] : f_sp[inv+SP_V_NSW(vi,hi)][SP_NSW(vi,hi)];
                    double _se = (SP_V_NSE(vi,hi) == -1) ? f_d[SP_NSE(vi,hi)] : f_sp[inv+SP_V_NSE(vi,hi)][SP_NSE(vi,hi)];
                    gradf_sp[ivi][gi] = ((_ne - _se) + (_nw - _sw)) * oneoverv;

                } else if (SP_HAS_NSW(vi,hi)) {
                    double _nw = (SP_V_NNW(vi,hi) == -1) ? f_d[SP_NNW(vi,hi)] : f_sp[inv+SP_V_NNW(vi,hi)][SP_NNW(vi,hi)];
                    double _sw = (SP_V_NSW(vi,hi) == -1) ? f_d[SP_NSW(vi,hi)] : f_sp[inv+SP_V_NSW(vi,hi)][SP_NSW(vi,hi)];
                    gradf_sp[ivi][gi] = (_nw - _sw) * oneover2v;

                } else if (SP_HAS_NSE(vi,hi)) {
                    double _ne = (SP_V_NNE(vi,hi) == -1) ? f_d[SP_NNE(vi,hi)] : f_sp[inv+SP_V_NNE(vi,hi)][SP_NNE(vi,hi)];
                    double _se = (SP_V_NSE(vi,hi) == -1) ? f_d[SP_NSE(vi,hi)] : f_sp[inv+SP_V_NSE(vi,hi)][SP_NSE(vi,hi)];
                    gradf_sp[ivi][gi] = (_ne - _se) * oneover2v;

                } else if (SP_HAS_NNW(vi,hi) /*&& SP_HAS_NNE(hi)*/) { // Guaranteed neighbour NE
                    double _nw = (SP_V_NNW(vi,hi) == -1) ? f_d[SP_NNW(vi,hi)] : f_sp[inv+SP_V_NNW(vi,hi)][SP_NNW(vi,hi)];
                    double _ne = (SP_V_NNE(vi,hi) == -1) ? f_d[SP_NNE(vi,hi)] : f_sp[inv+SP_V_NNE(vi,hi)][SP_NNE(vi,hi)];
                    gradf_sp[ivi][gi] = ( (_ne + _nw) * 0.5 - f_sp[ivi][hi]) * oneoverv;

                } else {
                    // Leave grady at 0
                    gradf_sp[ivi][gi] = 0.0;
                }
            }

            /*
             * left edge
             */
            for (unsigned int hi=rl; hi <= vl-rl; hi += rl) {
                // WRITEME
                //throw runtime_error ("Writeme");
            }

            /*
             * right edge
             */
            for (unsigned int hi=rl-1; hi < vl-1; hi += rl) {
                // WRITEME
            }

            /*
             * top edge
             */
            for (unsigned int hi=vl-rl; hi < vl-1; ++hi) {
                // WRITEME
            }

            /*
             * ... then the fast bit - everything else. For this whole
             * scheme to be really worthwhile, this needs to be most
             * of the Hexes.
             */

#ifdef __ICC__
//    __itt_resume();
#endif
            //#pragma omp parallel for
            unsigned int sp_offs_nne = SP_OFFS_NNE(rl);
            unsigned int sp_offs_nse = SP_OFFS_NSE(rl);
            unsigned int sp_offs_nnw = SP_OFFS_NNW(rl);
            unsigned int sp_offs_nsw = SP_OFFS_NSW(rl);
#pragma omp simd
            for (unsigned int hi=rl+1, gi=2*(rl+1); hi < vl-rl-1; ++hi, ++gi) {
                // All neighbours guaranteed so:
                gradf_sp[ivi][gi] = (f_sp[ivi][hi+SP_OFFS_NE] - f_sp[ivi][hi+SP_OFFS_NW]) * oneover2d;
                gradf_sp[ivi][++gi] = ( (f_sp[ivi][hi+sp_offs_nne] - f_sp[ivi][hi+sp_offs_nse])
                                        + (f_sp[ivi][hi+sp_offs_nnw] - f_sp[ivi][hi+sp_offs_nsw]) ) * oneoverv;
            }

#ifdef __ICC__
//    __itt_pause();
#endif
        } // end for over sp_ vectors
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
    void compute_divJ (double* fa_d, vector<double*>& fa_sp, unsigned int i) {

        // Three terms to compute; see Eq. 14 in methods_notes.pdf

        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa_d, fa_sp, i, this->grad_a_d[i], this->grad_a_sp);

        unsigned int inv = i * this->nv;
        /*
         * Compute for fa_d...
         */
        ////#pragma omp parallel for schedule(dynamic,50)
//#pragma omp simd // Makes this loop *slower* due to vector register spilling
        for (unsigned int hi=0; hi<this->nhex_d; ++hi) {

            // 1. The D Del^2 a_i term
            // Compute the sum around the neighbours
            double thesum = -6 * fa_d[hi];

            thesum += (D_HAS_NE(hi))  ? ((D_V_NE(hi) == -1)  ? fa_d[D_NE(hi)]  : fa_sp[inv+D_V_NE(hi)][D_NE(hi)])   : fa_d[hi];
            thesum += (D_HAS_NNE(hi)) ? ((D_V_NNE(hi) == -1) ? fa_d[D_NNE(hi)] : fa_sp[inv+D_V_NNE(hi)][D_NNE(hi)]) : fa_d[hi];
            thesum += (D_HAS_NNW(hi)) ? ((D_V_NNW(hi) == -1) ? fa_d[D_NNW(hi)] : fa_sp[inv+D_V_NNW(hi)][D_NNW(hi)]) : fa_d[hi];
            thesum += (D_HAS_NW(hi))  ? ((D_V_NW(hi) == -1)  ? fa_d[D_NW(hi)]  : fa_sp[inv+D_V_NW(hi)][D_NW(hi)])   : fa_d[hi];
            thesum += (D_HAS_NSW(hi)) ? ((D_V_NSW(hi) == -1) ? fa_d[D_NSW(hi)] : fa_sp[inv+D_V_NSW(hi)][D_NSW(hi)]) : fa_d[hi];
            thesum += (D_HAS_NSE(hi)) ? ((D_V_NSE(hi) == -1) ? fa_d[D_NSE(hi)] : fa_sp[inv+D_V_NSE(hi)][D_NSE(hi)]) : fa_d[hi];

            // Multiply bu 2D/3d^2
            term1_d[hi] = this->twoDover3dd * thesum;

        }

//#pragma omp simd // No benefit from SIMD - unavoidable memory access
        for (unsigned int hi=0, gi=0, giy=1; hi<this->nhex_d; ++hi, gi+=2, giy+=2) {

            // 2. The a div(g) term. Two sums for this.
            // NB: g_d and g_sp are used here mostly without their this-> identifiers, to keep lines shorter.
            term2_d[hi] = 0.0;
            // First sum
            if (D_HAS_NE(hi)) {
                term2_d[hi] += /*cos (0)*/ ((D_V_NE(hi) == -1)  ? g_d[i][D_NE(hi)<<1]  : g_sp[inv+D_V_NE(hi)][D_NE(hi)<<1]) + g_d[i][gi];
            } else {
                // Boundary condition _should_ be satisfied by
                // sigmoidal roll-off of g towards the boundary, so
                // add only g[i][0][hi]
                term2_d[hi] += /*cos (0)*/ (this->g_d[i][hi<<1]);
            }
            if (D_HAS_NNE(hi)) {
                term2_d[hi] += /*cos (60)*/ 0.5 * ( ((D_V_NNE(hi) == -1) ? g_d[i][D_NNE(hi)<<1]    : g_sp[inv+D_V_NNE(hi)][D_NNE(hi)<<1])     + g_d[i][gi])
                    + /*sin (60)*/ R3_OVER_2 *    ( ((D_V_NNE(hi) == -1) ? g_d[i][1+(D_NNE(hi)<<1)]: g_sp[inv+D_V_NNE(hi)][1+(D_NNE(hi)<<1)]) + g_d[i][giy]);
            } else {
                term2_d[hi] += /*cos (60)*/ 0.5 * (this->g_d[i][gi])
                    + /*sin (60)*/ R3_OVER_2 * (this->g_d[i][giy]);
            }
            if (D_HAS_NNW(hi)) {
                term2_d[hi] += -(/*cos (120)*/ 0.5 *  ( ((D_V_NNW(hi) == -1)  ? g_d[i][D_NNW(hi)<<1]  : g_sp[inv+D_V_NNW(hi)][D_NNW(hi)<<1])  + g_d[i][gi]))
                    + /*sin (120)*/ R3_OVER_2 * ( ((D_V_NNW(hi) == -1)  ? g_d[i][1+(D_NNW(hi)<<1)]  : g_sp[inv+D_V_NNW(hi)][1+(D_NNW(hi)<<1)])  + g_d[i][giy]);
            } else {
                term2_d[hi] += -(/*cos (120)*/ 0.5 * (this->g_d[i][gi]))
                    + /*sin (120)*/ R3_OVER_2 * (this->g_d[i][giy]);
            }
            if (D_HAS_NW(hi)) {
                term2_d[hi] -= /*cos (180)*/ ( ((D_V_NW(hi) == -1)  ? g_d[i][D_NW(hi)<<1]  : g_sp[inv+D_V_NW(hi)][D_NW(hi)<<1]) + g_d[i][gi]);
            } else {
                term2_d[hi] -= /*cos (180)*/ (this->g_d[i][gi]);
            }
            if (D_HAS_NSW(hi)) {
                term2_d[hi] -= /*cos (240)*/ 0.5     * ( ((D_V_NSW(hi) == -1)  ? g_d[i][D_NSW(hi)<<1]  : g_sp[inv+D_V_NSW(hi)][D_NSW(hi)<<1])  + g_d[i][gi])
                    - (/*sin (240)*/ R3_OVER_2 * ( ((D_V_NSW(hi) == -1)  ? g_d[i][1+(D_NSW(hi)<<1)]  : g_sp[inv+D_V_NSW(hi)][1+(D_NSW(hi)<<1)])  + g_d[i][giy]));
            } else {
                term2_d[hi] -= /*cos (240)*/ 0.5 * (this->g_d[i][gi])
                    - (/*sin (240)*/ R3_OVER_2 * (this->g_d[i][giy]));
            }
            if (D_HAS_NSE(hi)) {
                term2_d[hi] += /*cos (300)*/ 0.5     * ( ((D_V_NSE(hi) == -1)  ? g_d[i][D_NSE(hi)<<1]  : g_sp[inv+D_V_NSE(hi)][D_NSE(hi)<<1])  + g_d[i][gi])
                    - (/*sin (300)*/ R3_OVER_2 * ( ((D_V_NSE(hi) == -1)  ? g_d[i][1+(D_NSE(hi)<<1)]  : g_sp[inv+D_V_NSE(hi)][1+(D_NSE(hi)<<1)])  + g_d[i][giy]));
            } else {
                term2_d[hi] += /*cos (300)*/ 0.5 * (this->g_d[i][gi])       // 1st sum
                    - (/*sin (300)*/ R3_OVER_2 * (this->g_d[i][giy])); // 2nd sum
            }

            term2_d[hi] /= (3.0 * this->d);
            term2_d[hi] *= fa_d[hi];
        }

#pragma omp simd
        for (unsigned int hi=0, gi=0, giy=1; hi<this->nhex_d; ++hi,gi+=2,giy+=2) {
            // 3. Third term is this->g . grad a_i. Should not
            // contribute to J, as g(x) decays towards boundary.
            term3_d[hi] = this->g_d[i][gi] * this->grad_a_d[i][gi]
                + this->g_d[i][giy] * this->grad_a_d[i][giy];

            this->divJ_d[i][hi] = term1_d[hi] + term2_d[hi] + term3_d[hi];
        }

        /*
         * Compute for fa_sp
         */
        for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {

            const unsigned int rl = this->hg->sp_rowlens[vi];
            const unsigned int vl = this->hg->sp_veclen[vi];

            /*
             * Bottom edge (excluding first and last hexes, handled separately)
             */
            ////#pragma omp parallel for
            for (unsigned int hi=2; hi < rl-1; ++hi) {
                // Compute stuff
            }
            /*
             * left edge
             */
            for (unsigned int hi=rl; hi <= vl-rl; hi += rl) {
                // WRITEME
                //throw runtime_error ("Writeme");
            }

            /*
             * right edge
             */
            for (unsigned int hi=rl-1; hi < vl-1; hi += rl) {
                // WRITEME
            }

            /*
             * top edge
             */
            for (unsigned int hi=vl-rl; hi < vl-1; ++hi) {
                // WRITEME
            }

            /*
             * Main body of parallelogram. Neighbours guaranteed.
             */

            unsigned int sp_offs_nne = SP_OFFS_NNE(rl);
            unsigned int sp_offs_nse = SP_OFFS_NSE(rl);
            unsigned int sp_offs_nnw = SP_OFFS_NNW(rl);
            unsigned int sp_offs_nsw = SP_OFFS_NSW(rl);
            // These ones are the offsets in a gradeint, and are hence*2
            unsigned int sp_offs_nne_2 = SP_OFFS_NNE(rl)<<1;
            unsigned int sp_offs_nse_2 = SP_OFFS_NSE(rl)<<1;
            unsigned int sp_offs_nnw_2 = SP_OFFS_NNW(rl)<<1;
            unsigned int sp_offs_nsw_2 = SP_OFFS_NSW(rl)<<1;
            unsigned int sp_offs_ne_2 = SP_OFFS_NE<<1;
            unsigned int sp_offs_nw_2 = SP_OFFS_NW<<1;

            unsigned int ivi = inv + vi;

            //#pragma omp parallel for
#pragma omp simd // Gives vectorization in this loop! ADDITIONAL SPEEDUP POTENTIAL - stuck between L1 and L2 cache.
            for (unsigned int hi=rl+1, gi=((rl+1)<<1), giy=(1+(rl+1)<<1); hi < vl-rl-1; ++hi,gi+=2,giy+=2) {
                // 1. The D Del^2 a_i term
                // Compute the sum around the neighbours
                double thesum = -6 * fa_sp[ivi][hi];

                thesum += fa_sp[ivi][hi+SP_OFFS_NE];
                thesum += fa_sp[ivi][hi+sp_offs_nne];
                thesum += fa_sp[ivi][hi+sp_offs_nnw];
                thesum += fa_sp[ivi][hi+SP_OFFS_NW];
                thesum += fa_sp[ivi][hi+sp_offs_nsw];
                thesum += fa_sp[ivi][hi+sp_offs_nse];

                // Multiply bu 2D/3d^2
                double term1 = this->twoDover3dd * thesum;

                // 2. The a div(g) term. Two sums for this.
                // NB: g_d and g_sp are used here mostly without their this-> identifiers, to keep lines shorter.
                double term2 = 0.0;
                // First and second sums together:
                term2 += /*cos (0)*/ g_sp[ivi][(hi+SP_OFFS_NE)<<1] + g_sp[ivi][gi];

                term2 += /*cos (60)*/ 0.5 *    (g_sp[ivi][gi+sp_offs_nne_2]   + g_sp[ivi][gi])
                    + /*sin (60)*/ R3_OVER_2 * (g_sp[ivi][giy+sp_offs_nne_2]   + g_sp[ivi][giy]);

                term2 += -(/*cos (120)*/ 0.5 *  (g_sp[ivi][gi+sp_offs_nnw_2]  + g_sp[ivi][gi]))
                    + /*sin (120)*/ R3_OVER_2 * (g_sp[ivi][giy+sp_offs_nnw_2]  + g_sp[ivi][giy]);

                term2 -= /*cos (180)*/ (g_sp[ivi][gi+sp_offs_nw_2] + g_d[i][gi]);

                term2 -= /*cos (240)*/ 0.5     * (g_sp[ivi][gi+sp_offs_nsw_2]  + g_sp[ivi][gi])
                    - (/*sin (240)*/ R3_OVER_2 * (g_sp[ivi][giy+sp_offs_nsw_2]  + g_sp[ivi][giy]));

                term2 += /*cos (300)*/ 0.5     * (g_sp[ivi][gi+sp_offs_nse_2]  + g_sp[ivi][gi])
                    - (/*sin (300)*/ R3_OVER_2 * (g_sp[ivi][giy+sp_offs_nse_2]  + g_sp[ivi][giy]));

                term2 /= (3.0 * this->d);
                term2 *= fa_sp[ivi][hi];

                // 3. Third term is this->g . grad a_i. Should not
                // contribute to J, as g(x) decays towards boundary.
                double term3 = this->g_sp[ivi][gi] * this->grad_a_sp[ivi][gi]
                    + this->g_sp[ivi][giy] * this->grad_a_sp[ivi][giy];

                this->divJ_sp[ivi][hi] = term1 + term2 + term3;
            }
        }
        /*
         * Done computing for fa_sp
         */
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
        for (unsigned int t=0; t<300000; ++t) { // 300000 matches Stuart's 1D Karbowski model
            // FIXME Can be converted to run in alternative way.
            //#pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex; ++hi) {
                emx_d[hi] += tau_emx * (-emx_d[hi] + eta_emx_d[hi] / (1. + w2 * fgf_d[hi] + v2 * pax_d[hi]));
                pax_d[hi] += tau_pax * (-pax_d[hi] + eta_pax_d[hi] / (1. + v1 * emx_d[hi]));
                fgf_d[hi] += tau_fgf * (-fgf_d[hi] + eta_fgf_d[hi] / (1. + w1 * emx_d[hi]));
            }
        }
        // FIXME: Add for emx_sp etc
    }

    /*!
     * Using this->fgf and some hyperbolic tangents, populate rhoA/B/C
     */
    void populateChemoAttractants (void) {
        // chemo-attraction gradient. cf Fig 1 of Karb 2004
        // NOTE: May parallelise fine. Init function in any case, so not important.
        //#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            this->rhoA_d[h] = (kA/2.)*(1.+tanh((fgf_d[h]-theta1)/sigmaA));
            this->rhoB_d[h] = (kB/2.)*(1.+tanh((theta2-fgf_d[h])/sigmaB))*(kB/2.)*(1.+tanh((fgf_d[h]-theta3)/sigmaB));
            this->rhoC_d[h] = (kC/2.)*(1.+tanh((theta4-fgf_d[h])/sigmaC));
        }
        // FIXME: Add for emx_sp etc
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

        //#pragma omp parallel for
        for (unsigned int hi=0; hi < this->nhex_d; ++hi) {
            // d vectors
            double x_ = (this->hg->d_x[hi] * cosphi) + (this->hg->d_y[hi] * sinphi);
            this->rhoA_d[hi] = gain * exp(-((x_-xoffA)*(x_-xoffA)) / sigma);
            this->rhoB_d[hi] = gain * exp(-((x_-xoffB)*(x_-xoffB)) / sigma);
            this->rhoC_d[hi] = gain * exp(-((x_-xoffC)*(x_-xoffC)) / sigma);
        }
        for (unsigned int vi=0; vi < this->hg->sp_numvecs; ++vi) {
            //#pragma omp parallel for
            for (unsigned int hi=1; hi < this->hg->sp_veclen[vi]-1; ++hi) {
                // sp vectors
                double x_ = (this->hg->sp_x[vi][hi] * cosphi) + (this->hg->sp_y[vi][hi] * sinphi);
                this->rhoA_sp[vi][hi] = gain * exp(-((x_-xoffA)*(x_-xoffA)) / sigma);
                this->rhoB_sp[vi][hi] = gain * exp(-((x_-xoffB)*(x_-xoffB)) / sigma);
                this->rhoC_sp[vi][hi] = gain * exp(-((x_-xoffC)*(x_-xoffC)) / sigma);
            }
        }
    }

}; // RD_2D_Karb
