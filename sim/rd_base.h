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
 * Base class for RD systems
 */
template <class Flt>
class RD_Base
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
    //! Passed to HdfData constructor to say we want to read the data
    const bool READ_DATA = true;
    //@}

    /*!
     * Hex to hex d for the grid. Make smaller to increase the number
     * of Hexes being computed.
     */
    alignas(float) float hextohex_d = 0.01;

    /*!
     * Holds the number of hexes in the populated HexGrid
     */
    alignas(Flt) unsigned int nhex = 0;

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
     * The power to which a_i(x,t) is raised in Eqs 1 and 2 in the
     * paper.
     */
    alignas(Flt) Flt k = 3.0;

    /*!
     * Over what length scale should some values fall off to zero
     * towards the boundary? Used in a couple of different locations.
     */
    alignas(Flt) Flt boundaryFalloffDist = 0.02; // 0.02 default

    /*!
     * k parameters
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> k;

protected:
    /*!
     * Hex to hex distance. Populate this from hg.d after hg has been
     * initialised.
     */
    alignas(Flt) Flt d = 1.0;
    alignas(Flt) Flt v = 1;

    /*!
     * Parameters that depend on d and v:
     */
    //@{
    alignas(Flt) Flt oneoverd = 1.0/this->d;
    alignas(Flt) Flt oneoverv = 1.0/this->v;
    alignas(Flt) Flt twov = this->v+this->v;
    alignas(Flt) Flt oneover2v = 1.0/this->twov;
    alignas(Flt) Flt oneover4v = 1.0/(this->twov+this->twov);
    alignas(Flt) Flt oneover2d = 1.0/(this->d+this->d);
    alignas(Flt) Flt oneover3d = 1.0/(3*this->d);
    alignas(Flt) Flt twoDover3dd = this->d+this->d / 3*this->d*this->d;
    //@}

public:

    /*!
     * Track the number of computational steps that we've carried
     * out. Only to show a message saying "100 steps done...", but
     * that's reason enough.
     */
    alignas(Flt) unsigned int stepCount = 0;

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
    RD_Base (void) {
        DBG("RD_Base constructor")
        this->halfdt = this->dt/2.0;
        this->sixthdt = this->dt/6.0;
    }

    /*!
     * Destructor required to free up HexGrid memory
     */
    ~RD_Base (void) {
        DBG("RD_Base deconstructor (deleting this->hg)")
        delete (this->hg);
    }

    /*!
     * A utility function to resize vector-vectors that hold N
     * different RD variables.
     */
    void resize_vector_vector (vector<vector<Flt> >& vv, unsigned int N) {
        vv.resize (N);
        for (unsigned int i=0; i<N; ++i) {
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
    void resize_vector_param (vector<Flt>& p, unsigned int N) {
        p.resize (N, 0.0);
    }

    /*!
     * Resize a vector of M vectors of parameters that'll each be N
     * elements long
     */
    void resize_vector_vector_param (vector<vector<Flt> >& vp, unsigned int N, unsigned int M) {
        vp.resize (M);
        for (unsigned int m = 0; m<M; ++m) {
            vp[m].resize (N, 0.0);
        }
    }

    /*!
     * Resize a gradient field
     */
    void resize_gradient_field (array<vector<Flt>, 2>& g) {
        g[0].resize (this->nhex, 0.0);
        g[1].resize (this->nhex, 0.0);
    }

    /*!
     * Resize a vector size N containing arrays of two vector<Flt>s
     * which are the x and y components of a (mathematical) vector
     * field.
     */
    void resize_vector_array_vector (vector<array<vector<Flt>, 2> >& vav, unsigned int N) {
        vav.resize (N);
        for (unsigned int n = 0; i<N; ++n) {
            this->resize_gradient_field (vav[n]);
        }
    }

    /*!
     * Initialise a vector with noise, but with sigmoidal roll-off to
     * zero at the boundary.
     *
     * I apply a sigmoid to the boundary hexes, so that the noise
     * drops away towards the edge of the domain.
     */
    void noiseify_vector (vector<Flt>& v) {
        Flt randNoiseOffset = 0.8;
        Flt randNoiseGain = 0.1;
        for (auto h : this->hg->hexen) {
            // boundarySigmoid. Jumps sharply (100, larger is
            // sharper) over length scale 0.05 to 1. So if
            // distance from boundary > 0.05, noise has normal
            // value. Close to boundary, noise is less.
            v[h.vi] = morph::Tools::randF<Flt>() * randNoiseGain + randNoiseOffset;
            if (h.distToBoundary > -0.5) { // It's possible that distToBoundary is set to -1.0
                Flt bSig = 1.0 / ( 1.0 + exp (-100.0*(h.distToBoundary-this->boundaryFalloffDist)) );
                v[h.vi] = v[h.vi] * bSig;
            }
        }
    }

    /*!
     * Perform memory allocations, vector resizes and so on.
     */
    virtual void allocate (void) {
        // Create a HexGrid. 3 is the 'x span' which determines how
        // many hexes are initially created. 0 is the z co-ordinate for the HexGrid.
        this->hg = new HexGrid (this->hextohex_d, 3, 0, morph::HexDomainShape::Boundary);
        // Read the curves which make a boundary
        ReadCurves r(this->svgpath);
        // Set the boundary in the HexGrid
        this->hg->setBoundary (r.getCorticalPath());
        // Compute the distances from the boundary
        this->hg->computeDistanceToBoundary();
        // Vector size comes from number of Hexes in the HexGrid
        this->nhex = this->hg->num();
        DBG ("HexGrid says num hexes = " << this->nhex);
        // Spatial d comes from the HexGrid, too.
        this->set_d(this->hg->getd());
        DBG ("HexGrid says d = " << this->d);
        this->set_v(this->hg->getv());
        DBG ("HexGrid says v = " << this->v);
    }

    /*!
     * Initialise variables and parameters. Carry out one-time
     * computations required of the model.
     */
    virtual void init (void) = 0;

protected:
    /*!
     * Require private setters for d and v as there are several other
     * members that have to be updated at the same time.
     */
    //@{
    virtual void set_d (Flt d_) {
        this->d = d_;
        this->oneoverd = 1.0/this->d;
        this->oneover2d = 1.0/(2*this->d);
        this->oneover3d = 1.0/(3*this->d);
    }

    virtual void set_v (Flt v_) {
        this->v = v_;
        this->oneoverv = 1.0/this->v;
        this->twov = this->v+this->v;
        this->oneover2v = 1.0/this->twov;
        this->oneover4v = 1.0/(this->twov+this->twov);
    }
    //@}

public:
    /*!
     * Public getters for d and v
     */
    //@{
    Flt get_d (void) {
        return this->d;
    }

    Flt get_v (void) {
        return this->v;
    }
    //@}

public:

    /*!
     * HDF5 file saving/loading methods.
     */
    //@{
    /*!
     * Save a data frame
     */
    virtual void save (void) { }

    /*!
     * Save position information
     */
    void savePositions (void) {
        stringstream fname;
        fname << this->logpath << "/positions.h5";
        HdfData data(fname.str());
        this->saveHexPositions (data);
    }

    /*!
     * Save positions of the hexes - note using two vector<float>s
     * that have been populated with the positions from the HexGrid,
     * to fit in with the HDF API.
     */
    void saveHexPositions (HdfData& dat) {
        dat.add_contained_vals ("/x", this->hg->d_x);
        dat.add_contained_vals ("/y", this->hg->d_y);

        // Add the neighbour information too.
        vector<float> x_ne = this->hg->d_x;
        vector<float> y_ne = this->hg->d_y;
        unsigned int count = 0;
        for (int i : this->hg->d_ne) {
            if (i >= 0) {
                x_ne[count] = this->hg->d_x[i];
                y_ne[count] = this->hg->d_y[i];
            }
            ++count;
        }
        dat.add_contained_vals ("/x_ne", x_ne);
        dat.add_contained_vals ("/y_ne", y_ne);

        vector<float> x_nne = this->hg->d_x;
        vector<float> y_nne = this->hg->d_y;
        count = 0;
        for (int i : this->hg->d_nne) {
            if (i >= 0) {
                x_nne[count] = this->hg->d_x[i];
                y_nne[count] = this->hg->d_y[i];
            }
            ++count;
        }
        dat.add_contained_vals ("/x_nne", x_nne);
        dat.add_contained_vals ("/y_nne", y_nne);

        vector<float> x_nnw = this->hg->d_x;
        vector<float> y_nnw = this->hg->d_y;
        count = 0;
        for (int i : this->hg->d_nnw) {
            if (i >= 0) {
                x_nnw[count] = this->hg->d_x[i];
                y_nnw[count] = this->hg->d_y[i];
            }
            ++count;
        }
        dat.add_contained_vals ("/x_nnw", x_nnw);
        dat.add_contained_vals ("/y_nnw", y_nnw);

        vector<float> x_nw = this->hg->d_x;
        vector<float> y_nw = this->hg->d_y;
        count = 0;
        for (int i : this->hg->d_nw) {
            if (i >= 0) {
                x_nw[count] = this->hg->d_x[i];
                y_nw[count] = this->hg->d_y[i];
            }
            ++count;
        }
        dat.add_contained_vals ("/x_nw", x_nw);
        dat.add_contained_vals ("/y_nw", y_nw);

        vector<float> x_nsw = this->hg->d_x;
        vector<float> y_nsw = this->hg->d_y;
        count = 0;
        for (int i : this->hg->d_nsw) {
            if (i >= 0) {
                x_nsw[count] = this->hg->d_x[i];
                y_nsw[count] = this->hg->d_y[i];
            }
            ++count;
        }
        dat.add_contained_vals ("/x_nsw", x_nsw);
        dat.add_contained_vals ("/y_nsw", y_nsw);

        vector<float> x_nse = this->hg->d_x;
        vector<float> y_nse = this->hg->d_y;
        count = 0;
        for (int i : this->hg->d_nse) {
            if (i >= 0) {
                x_nse[count] = this->hg->d_x[i];
                y_nse[count] = this->hg->d_y[i];
            }
            ++count;
        }
        dat.add_contained_vals ("/x_nse", x_nse);
        dat.add_contained_vals ("/y_nse", y_nse);

        // And hex to hex distance:
        dat.add_val ("/d", this->d);
    }
    //@} // HDF5

    /*!
     * Computation methods
     */
    //@{

    /*!
     * Normalise the vector of Flts f.
     */
    void normalise (vector<Flt>& f) {

        Flt maxf = -1e7;
        Flt minf = +1e7;

        // Determines min and max
        for (auto val : f) {
            if (val>maxf) { maxf = val; }
            if (val<minf) { minf = val; }
        }
        Flt scalef = 1.0 /(maxf - minf);

        vector<vector<Flt> > norm_a;
        this->resize_vector_vector (norm_a);
        for (unsigned int fi = 0; fi < f.size(); ++fi) {
            f[fi] = fmin (fmax (((f[fi]) - minf) * scalef, 0.0), 1.0);
        }
    }

    /*!
     * Do a single step through the model.
     */
    virtual void step (void) = 0;

    /*!
     * 2D spatial integration of the function f. Result placed in gradf.
     *
     * For each Hex, work out the gradient in x and y directions
     * using whatever neighbours can contribute to an estimate.
     */
    void spacegrad2D (vector<Flt>& f, array<vector<Flt>, 2>& gradf) {

        // Note - East is positive x; North is positive y.
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
                gradf[1][hi] = ( (f[NNE(hi)] - f[NSE(hi)]) + (f[NNW(hi)] - f[NSW(hi)]) ) * oneover4v;
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

#if 0
    /*!
     * Could become compute_divF (F, gradF, divF), for general use in the base
     * class.
     *
     * Computes the divergence term, J(x) (Eq 4). Probably too model
     * specific to go here in base class. What I really want is
     * compute Laplacian, which is simpler.
     */
    virtual void compute_divJ (vector<Flt>& Fa, array<vector<Flt>, 2>& gradFa, vector<Flt>& divFa) {

        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa, this->grad_a[i]);

        // Three terms to compute; see Eq. 17 in methods_notes.pdf
#pragma omp parallel for //schedule(static) // This was about 10% faster than schedule(dynamic,50).
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            // 1. The D Del^2 a_i term. Eq. 18.
            // Compute the sum around the neighbours
            Flt thesum = -6 * fa[hi];

            thesum += fa[(HAS_NE(hi)  ? NE(hi)  : hi)];
            thesum += fa[(HAS_NNE(hi) ? NNE(hi) : hi)];
            thesum += fa[(HAS_NNW(hi) ? NNW(hi) : hi)];
            thesum += fa[(HAS_NW(hi)  ? NW(hi)  : hi)];
            thesum += fa[(HAS_NSW(hi) ? NSW(hi) : hi)];
            thesum += fa[(HAS_NSE(hi) ? NSE(hi) : hi)];

            // Multiply sum by 2D/3d^2 to give term1
            Flt term1 = this->twoDover3dd * thesum;

            // 2. The (a div(g)) term.
            Flt term2 = fa[hi] * this->divg_over3d[i][hi];

            // 3. Third term is this->g . grad a_i. Should not contribute to J, as g(x) decays towards boundary.
            Flt term3 = this->g[i][0][hi] * this->grad_a[i][0][hi] + (this->g[i][1][hi] * this->grad_a[i][1][hi]);

            this->divJ[i][hi] = term1 - term2 - term3;
        }
    }
#endif

}; // RD_Base

/*!
 * A helper class, containing (at time of writing) get_contours()
 */
template <class Flt>
class RD_Help
{
public:
    /*!
     * Obtain the contours (as a vector of list<Hex>) in the scalar
     * fields f, where threshold is crossed.
     */
    static vector<list<Hex> > get_contours (HexGrid* hg,
                                            vector<vector<Flt> >& f,
                                            Flt threshold) {

        unsigned int nhex = hg->num();
        unsigned int N = f.size();

        vector<list<Hex> > rtn;
        // Initialise
        for (unsigned int li = 0; li < N; ++li) {
            list<Hex> lh;
            rtn.push_back (lh);
        }

        Flt maxf = -1e7;
        Flt minf = +1e7;
        for (auto h : hg->hexen) {
            if (h.onBoundary() == false) {
                for (unsigned int i = 0; i<N; ++i) {
                    if (f[i][h.vi] > maxf) { maxf = f[i][h.vi]; }
                    if (f[i][h.vi] < minf) { minf = f[i][h.vi]; }
                }
            }
        }
        Flt scalef = 1.0 / (maxf-minf);

        // Re-normalize
        vector<vector<Flt> > norm_f;
        norm_f.resize (N);
        for (unsigned int i=0; i<N; ++i) {
            norm_f[i].resize (nhex, 0.0);
        }

        for (unsigned int i = 0; i<N; ++i) {
            for (unsigned int h=0; h<nhex; h++) {
                norm_f[i][h] = (f[i][h] - minf) * scalef;
            }
        }

        // Collate
        for (unsigned int i = 0; i<N; ++i) {

            for (auto h : hg->hexen) {
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
}; // RD_Helper
