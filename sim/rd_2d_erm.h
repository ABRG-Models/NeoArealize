#include "morph/display.h"
#include "morph/tools.h"
#include "morph/ReadCurves.h"
#include "morph/HexGrid.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <iomanip>
#include <cmath>
#ifdef __GLN__ // Currently I've only tested OpenMP on Linux
#include <omp.h>
#endif
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


/*!
 * Reaction diffusion system; Ermentrout 2009.
 */
class RD_2D_Erm
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

    unsigned int frameN = 0;

    /*!
     * Holds the number of hexes in the populated HexGrid
     */
    unsigned int nhex = 0;

    /*!
     * Set N>1 for maintaing multiple expression gradients
     */
    unsigned int N = 1;

    /*!
     * The c_i(x,t) variables from the Ermentrout paper (chemoattractant concentration)
     */
    vector<vector<double> > c;

    /*!
     * The n_i(x,t) variables from the Ermentrout paper (density of tc axons)
     */
    vector<vector<double> > n;

    /*!
     * Holds the Laplacian
     */
    vector<vector<double> > lapl;

    /*!
     * Holds the Poisson terms (final non-linear term in Ermentrout equation 1)
     */
    vector<vector<double> > poiss;

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
     * Hex to hex distance. Populate this from hg.d after hg has been
     * initialised.
     */
    double d = 1.0;


    /*!
     * Parameters of the Ermentrout model
     */
    double Dn;
    double Dc;
    double beta;
    double a;
    double b;
    double mu;
    double chi;

    /*!
     * Track the number of computational steps that we've carried
     * out. Only to show a message saying "100 steps done...", but
     * that's reason enough.
     */
    unsigned int stepCount = 0;

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
    void resize_vector_vector (vector<vector<double> >& vv) {
        vv.resize (this->N);
        for (unsigned int i =0; i<this->N; ++i) {
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
     * Initialise this vector of vectors with noise.
     */
    void noiseify_vector_vector (vector<vector<double> >& vv, double off, double sig) {
        for (unsigned int i = 0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                vv[i][h.vi] = morph::Tools::randDouble() *sig + off;
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
        this->hg = new HexGrid (0.01, 3);
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

        // Resize and zero-initialise the various containers
        this->resize_vector_vector (this->c);
        this->resize_vector_vector (this->n);
        this->resize_vector_vector (this->lapl);
        this->resize_vector_vector (this->poiss);

        // Populate parameters

        this->Dn = 0.3;                 // Diffusion constant for n
        this->Dc = Dn*0.3;              // Diffusion constant for c
        this->beta = 5.;                // saturation term in function for production of c
        this->a = 1.;                   // production of new axon branches
        this->b = 1.;                   // pruning constant
        this->mu = 1.;                  // decay of chemoattractant constant
        this->chi = Dn;                 // degree of attraction of chemoattractant

        // Initialise a with noise
        this->noiseify_vector_vector (this->n, 1., 0.01);
        this->noiseify_vector_vector (this->c, beta*0.5, 0.01);
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


    void step (void) {

        this->stepCount++;

        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
        }

        #pragma omp parallel for
        for (unsigned int i=0; i<this->N; ++i) {

            this->compute_poiss (n[i],c[i],i);  // compute the non-linear Poission term in Eq1
            this->compute_lapl (n[i], i);       // populate lapl[i] with laplacian of n

            // integrate n
            for (unsigned int h=0; h<this->nhex; ++h) {
                n[i][h] += (a - b*n[i][h] + Dn*lapl[i][h] - chi*poiss[i][h])*dt;
            }

            this->compute_lapl (c[i], i);       // populate lapl[i] with laplacian of c

            // integrate c
            double n2;
            for (unsigned int h=0; h<this->nhex; ++h) {
                n2 = n[i][h]*n[i][h];
                c[i][h] += (beta*n2/(1.+n2) - mu*c[i][h] +Dc*lapl[i][h])*dt;
            }
        }

    }


    /*!
     * Plot the system on @a disps
     */
    void plot (vector<morph::Gdisplay>& disps) {
        this->plot_f (this->n, disps[0], true);
        this->plot_f (this->c, disps[1], true);

        /*
        std::stringstream frameFile1;
        frameFile1<<"logs/tmp/demo";
        frameFile1<<setw(5)<<setfill('0')<<frameN;
        frameFile1<<".png";
        //disps[0].saveImage(frameFile1.str());
        frameN++;
         */
    }

    /*!
     * Plot a or c
     */
    void plot_f (vector<vector<double> >& f, morph::Gdisplay& disp, bool combinedNorm) {
        vector<double> fix(3, 0.0);
        vector<double> eye(3, 0.0);
        vector<double> rot(3, 0.0);

        vector<double> maxa (this->N, -1e7);
        vector<double> mina (this->N, +1e7);
        // Copies data to plot out of the model
        if(combinedNorm){
            double maxb = -1e7;
            double minb = +1e7;
            for (auto h : this->hg->hexen) {
                if (h.onBoundary() == false) {
                    for (unsigned int i = 0; i<this->N; ++i) {
                        if (f[i][h.vi]>maxb) { maxb = f[i][h.vi]; }
                        if (f[i][h.vi]<minb) { minb = f[i][h.vi]; }
                    }
                }
            }
            for (unsigned int i = 0; i<this->N; ++i) {
                mina[i] = minb;
                maxa[i] = maxb;
            }
        } else {
            for (auto h : this->hg->hexen) {
                if (h.onBoundary() == false) {
                    for (unsigned int i = 0; i<this->N; ++i) {
                        if (f[i][h.vi]>maxa[i]) { maxa[i] = f[i][h.vi]; }
                        if (f[i][h.vi]<mina[i]) { mina[i] = f[i][h.vi]; }
                    }
                }
            }
        }
        vector<double> scalea (this->N, 0);
        for (unsigned int i = 0; i<this->N; ++i) {
            scalea[i] = 1.0 / (maxa[i]-mina[i]);
        }

        // Determine a colour from min, max and current value
        vector<vector<double> > norm_a;
        this->resize_vector_vector (norm_a);
        for (unsigned int i = 0; i<this->N; ++i) {
            for (unsigned int h=0; h<this->nhex; h++) {
                norm_a[i][h] = fmin (fmax (((f[i][h]) - mina[i]) * scalea[i], 0.0), 1.0);
            }
        }

        // Create an offset which we'll increment by the width of the
        // map, starting from the left-most map (f[0])
        float hgwidth = this->hg->getXmax()-this->hg->getXmin();
        array<float,3> offset = {{0.0f,0.0f,0.0f}};//{ 2*(-hgwidth-(hgwidth/20)), 0.0f, 0.0f };

        // Draw
        disp.resetDisplay (fix, eye, rot);
        for (unsigned int i = 0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                array<float,3> cl_a = morph::Tools::getJetColorF (norm_a[i][h.vi]);
                disp.drawHex (h.position(), offset, (h.d/2.0f), cl_a);
            }
            offset[0] += hgwidth + (hgwidth/20);
        }
        disp.redrawDisplay();
    }

    /*!
     * Computes the Laplacian
     * Stable with dt = 0.0001;
     */
    void compute_lapl (vector<double>& fa, unsigned int i) {

        double norm  = (2) / (3 * this->d * this->d);

#pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            Hex* h = this->hg->vhexen[hi];
            // 1. The D Del^2 term

            // Compute the sum around the neighbours
            double thesum = -6 * fa[h->vi];
            if (h->has_ne) {
                thesum += fa[h->ne->vi];
            } else {
                thesum += fa[h->vi]; // A ghost neighbour-east with same value as Hex_0
            }
            if (h->has_nne) {
                thesum += fa[h->nne->vi];
            } else {
                thesum += fa[h->vi];
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

            this->lapl[i][h->vi] = norm * thesum;
        }
    }

    /*!
     * Computes the Poisson term
     *
     * Stable with dt = 0.0001;
     */
    void compute_poiss (vector<double>& fa1, vector<double>& fa2, unsigned int i) {

        // Compute non-linear term

#pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            Hex* h = this->hg->vhexen[hi];

            vector<double> dum1(6,fa1[h->vi]);
            vector<double> dum2(6,fa2[h->vi]);


            if (h->has_ne) {
                dum1[0] = fa1[h->ne->vi];
                dum2[0] = fa2[h->ne->vi];
            }
            if (h->has_nne) {
                dum1[1] = fa1[h->nne->vi];
                dum2[1] = fa2[h->nne->vi];
            }
            if (h->has_nnw) {
                dum1[2] = fa1[h->nnw->vi];
                dum2[2] = fa2[h->nnw->vi];
            }
            if (h->has_nw) {
                dum1[3] = fa1[h->nw->vi];
                dum2[3] = fa2[h->nw->vi];
            }
            if (h->has_nsw) {
                dum1[4] = fa1[h->nsw->vi];
                dum2[4] = fa2[h->nsw->vi];
            }
            if (h->has_nse) {
                dum1[5] = fa1[h->nse->vi];
                dum2[5] = fa2[h->nse->vi];
            }

            /* // John Brooke's 'second interim report' solution
            double val =
            (dum1[0]+dum1[1])*(dum2[0]-fa2[h->vi])+
            (dum1[1]+dum1[2])*(dum2[1]-fa2[h->vi])+
            (dum1[2]+dum1[3])*(dum2[2]-fa2[h->vi])+
            (dum1[3]+dum1[4])*(dum2[3]-fa2[h->vi])+
            (dum1[4]+dum1[5])*(dum2[4]-fa2[h->vi])+
            (dum1[5]+dum1[0])*(dum2[5]-fa2[h->vi]);
            this->poiss[i][h->vi] = val / (ROOT3 * this->d * this->d);
             */

            // John Brooke's final thesis solution (based on 'finite volume method'
            // of Lee et al. https://doi.org/10.1080/00207160.2013.864392
            double val =
            (dum1[0]+fa1[h->vi])*(dum2[0]-fa2[h->vi])+
            (dum1[1]+fa1[h->vi])*(dum2[1]-fa2[h->vi])+
            (dum1[2]+fa1[h->vi])*(dum2[2]-fa2[h->vi])+
            (dum1[3]+fa1[h->vi])*(dum2[3]-fa2[h->vi])+
            (dum1[4]+fa1[h->vi])*(dum2[4]-fa2[h->vi])+
            (dum1[5]+fa1[h->vi])*(dum2[5]-fa2[h->vi]);
            this->poiss[i][h->vi] = val / (3 * this->d * this->d);
        }

    }


    }

}; // RD_2D_Erm
