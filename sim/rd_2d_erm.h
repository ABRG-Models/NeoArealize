#include "morph/tools.h"
#include "morph/HexGrid.h"
#include "morph/ReadCurves.h"
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
 * Reaction diffusion system; Ermentrout 2009.
 */
template <class Flt>
class RD_2D_Erm
{
public:

    /*!
     * Constants
     */
    //@{
    //! Square root of 3 over 2
    const Flt R3_OVER_2 = 0.866025403784439;
    //! Square root of 3
    const Flt ROOT3 = 1.73205080756888;
    //@}

    /*!
     * Set N>1 for maintaing multiple expression gradients
     */
    alignas(Flt) unsigned int N = 1;

    /*!
     * The c_i(x,t) variables from the Ermentrout paper (chemoattractant concentration)
     */
    alignas(Flt) vector<vector<Flt> > c;

    /*!
     * The n_i(x,t) variables from the Ermentrout paper (density of tc axons)
     */
    alignas(Flt) vector<vector<Flt> > n;

    /*!
     * Holds the Laplacian
     */
    alignas(Flt) vector<vector<Flt> > lapl;

    /*!
     * Holds the Poisson terms (final non-linear term in Ermentrout equation 1)
     */
    alignas(Flt) vector<vector<Flt> > poiss;

    /*!
     * Our choice of dt.
     */
    alignas(Flt) Flt dt = 0.0001;

    /*!
     * Compute half and sixth dt in constructor.
     */
    //@{
    alignas(Flt) Flt halfdt = 0.0;
    alignas(Flt) Flt sixthdt = 0.0;
    //@}

    /*!
     * Store Hex positions for saving.
     */
    //@{
    alignas(Flt) vector<float> hgvx;
    alignas(Flt) vector<float> hgvy;
    //@}

    /*!
     * Hex to hex distance. Populate this from hg.d after hg has been
     * initialised.
     */
    alignas(Flt) Flt d = 1.0;

    /*!
     * Parameters of the Ermentrout model - default values.
     */
    //@{
    //! Diffusion constant for n
    alignas(Flt) Flt Dn = 0.3;
    //! Diffusion constant for c
    alignas(Flt) Flt Dc = Dn * 0.3;
    //! saturation term in function for production of c
    alignas(Flt) Flt beta = 5.0;
    //! production of new axon branches
    alignas(Flt) Flt a = 1.0;
    //! pruning constant
    alignas(Flt) Flt b = 1.0;
    //! decay of chemoattractant constant
    alignas(Flt) Flt mu = 1.0;
    //! degree of attraction of chemoattractant
    alignas(Flt) Flt chi = Dn;
    //@}

    /*
     * Below this point, no more alignas() keywords.
     */

    /*!
     * Frame number, used when saving PNG movie frames.
     */
    unsigned int frameN = 0;

    /*!
     * Holds the number of hexes in the populated HexGrid
     */
    unsigned int nhex = 0;

    /*!
     * Track the number of computational steps that we've carried
     * out. Only to show a message saying "100 steps done...", but
     * that's reason enough.
     */
    unsigned int stepCount = 0;

    /*!
     * The HexGrid "background" for the Reaction Diffusion system.
     */
    HexGrid* hg;

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
     * Simple constructor; no arguments.
     */
    RD_2D_Erm (void) {
        this->halfdt = this->dt/2.0;
        this->sixthdt = this->dt/6.0;
    }

    /*!
     * Destructor required to free up HexGrid memory
     */
    ~RD_2D_Erm (void) {
        delete (this->hg);
    }

    /*!
     * A utility function to resize the vector-vectors that hold a
     * variable for the N different thalamo-cortical axon types.
     */
    void resize_vector_vector (vector<vector<Flt> >& vv) {
        vv.resize (this->N);
        for (unsigned int i =0; i<this->N; ++i) {
            vv[i].resize (this->nhex, 0.0);
        }
    }

    /*!
     * Resize a variable that'll be nhex elements long
     */
    void resize_vector_variable (vector<Flt>& v) {
        v.resize (this->nhex, 0.0);
    }

    /*!
     * Resize a parameter that'll be N elements long
     */
    void resize_vector_param (vector<Flt>& p) {
        p.resize (this->N, 0.0);
    }

    /*!
     * Initialise this vector of vectors with noise.
     */
    void noiseify_vector_vector (vector<vector<Flt> >& vv, Flt off, Flt sig) {
        for (unsigned int i = 0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                vv[i][h.vi] = morph::Tools::randF<Flt>() *sig + off;
            }
        }
    }

    /*!
     * Initialise HexGrid, variables. Carry out any one-time
     * computations of the model.
     */
    void init (void) {

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
        this->resize_vector_vector (this->c);
        this->resize_vector_vector (this->n);
        this->resize_vector_vector (this->lapl);
        this->resize_vector_vector (this->poiss);

        // Initialise a with noise
        this->noiseify_vector_vector (this->n, 1., 0.01);
        this->noiseify_vector_vector (this->c, beta*0.5, 0.01);
    }

    /*!
     * Computations
     */
    //@{

    /*!
     * Compute one step of the model
     */
    void step (void) {

        this->stepCount++;

        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
        }

        for (unsigned int i=0; i<this->N; ++i) {

            this->compute_poiss (n[i],c[i],i);  // compute the non-linear Poission term in Eq1
            this->compute_lapl (n[i], i);       // populate lapl[i] with laplacian of n

            // integrate n
            for (unsigned int h=0; h<this->nhex; ++h) {
                n[i][h] += (a - b*n[i][h] + Dn*lapl[i][h] - chi*poiss[i][h])*dt;
            }

            this->compute_lapl (c[i], i);       // populate lapl[i] with laplacian of c

            // integrate c
            Flt n2;
            for (unsigned int h=0; h<this->nhex; ++h) {
                n2 = n[i][h]*n[i][h];
                c[i][h] += (beta*n2/(1.+n2) - mu*c[i][h] +Dc*lapl[i][h])*dt;
            }
        }
    }

    /*!
     * Computes the Laplacian
     * Stable with dt = 0.0001;
     */
    void compute_lapl (vector<Flt>& fa, unsigned int i) {

        Flt norm  = (2) / (3 * this->d * this->d);

#pragma omp parallel for schedule(static)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            // 1. The D Del^2 term

            // Compute the sum around the neighbours
            Flt thesum = -6 * fa[hi];
            if (HAS_NE(hi)) {
                thesum += fa[NE(hi)];
            } else {
                thesum += fa[hi]; // A ghost neighbour-east with same value as Hex_0
            }
            if (HAS_NNE(hi)) {
                thesum += fa[NNE(hi)];
            } else {
                thesum += fa[hi];
            }
            if (HAS_NNW(hi)) {
                thesum += fa[NNW(hi)];
            } else {
                thesum += fa[hi];
            }
            if (HAS_NW(hi)) {
                thesum += fa[NW(hi)];
            } else {
                thesum += fa[hi];
            }
            if (HAS_NSW(hi)) {
                thesum += fa[NSW(hi)];
            } else {
                thesum += fa[hi];
            }
            if (HAS_NSE(hi)) {
                thesum += fa[NSE(hi)];
            } else {
                thesum += fa[hi];
            }

            this->lapl[i][hi] = norm * thesum;
        }
    }

    /*!
     * Computes the Poisson term
     *
     * Stable with dt = 0.0001;
     */
    void compute_poiss (vector<Flt>& fa1, vector<Flt>& fa2, unsigned int i) {

        // Compute non-linear term

#pragma omp parallel for schedule(static)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            vector<Flt> dum1(6,fa1[hi]);
            vector<Flt> dum2(6,fa2[hi]);

            if (HAS_NE(hi)) {
                dum1[0] = fa1[NE(hi)];
                dum2[0] = fa2[NE(hi)];
            }
            if (HAS_NNE(hi)) {
                dum1[1] = fa1[NNE(hi)];
                dum2[1] = fa2[NNE(hi)];
            }
            if (HAS_NNW(hi)) {
                dum1[2] = fa1[NNW(hi)];
                dum2[2] = fa2[NNW(hi)];
            }
            if (HAS_NW(hi)) {
                dum1[3] = fa1[NW(hi)];
                dum2[3] = fa2[NW(hi)];
            }
            if (HAS_NSW(hi)) {
                dum1[4] = fa1[NSW(hi)];
                dum2[4] = fa2[NSW(hi)];
            }
            if (HAS_NSE(hi)) {
                dum1[5] = fa1[NSE(hi)];
                dum2[5] = fa2[NSE(hi)];
            }

            // John Brooke's final thesis solution (based on 'finite volume method'
            // of Lee et al. https://doi.org/10.1080/00207160.2013.864392
            Flt val =
            (dum1[0]+fa1[hi])*(dum2[0]-fa2[hi])+
            (dum1[1]+fa1[hi])*(dum2[1]-fa2[hi])+
            (dum1[2]+fa1[hi])*(dum2[2]-fa2[hi])+
            (dum1[3]+fa1[hi])*(dum2[3]-fa2[hi])+
            (dum1[4]+fa1[hi])*(dum2[4]-fa2[hi])+
            (dum1[5]+fa1[hi])*(dum2[5]-fa2[hi]);
            this->poiss[i][hi] = val / (3 * this->d * this->d);
        }
    }
    //@} // computations

    /*!
     * HDF5 file saving/loading methods
     */
    //@{

    /*!
     * Save positions of the hexes - note using two vector<floats>
     * that have been populated with the positions from the HexGrid,
     * to fit in with the HDF API.
     */
    void saveHexPositions (HdfData& dat) {
        dat.add_contained_vals ("/x", this->hgvx);
        dat.add_contained_vals ("/y", this->hgvy);
        // And hex to hex distance:
        dat.add_val ("/d", this->d);
    }

    /*!
     * Save some data like this.
     */
    void saveState (void) {
        string fname = this->logpath + "/2Derm.h5";
        HdfData data (fname);

        // Save some variables
        for (unsigned int i = 0; i<this->N; ++i) {

            stringstream vss;
            vss << "c_" << i;
            string vname = vss.str();
            data.add_contained_vals (vname.c_str(), this->c[i]);
            vname[0] = 'n';
            data.add_contained_vals (vname.c_str(), this->n[i]);
        }

        // Parameters
        data.add_val ("/Dn", this->Dn);
        data.add_val ("/Dc", this->Dc);
        data.add_val ("/beta", this->beta);
        data.add_val ("/a", this->a);
        data.add_val ("/b", this->b);
        data.add_val ("/mu", this->mu);
        data.add_val ("/chi", this->chi);

        // HexGrid information
        this->saveHexPositions (data);
    }
    //@} // HDF5

}; // RD_2D_Erm
