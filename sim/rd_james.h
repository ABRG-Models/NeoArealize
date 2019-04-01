/*
 * Like RD_James, but derives from RD_Base
 */

#include "rd_base.h"

/*!
 * Enumerates the way that the guidance molecules are set up
 */
enum class GuidanceMoleculeMethod {
    Gauss1D,
    Gauss2D,
    Exponential1D,
    Sigmoid1D,
    Linear1D
};

/*!
 * Reaction diffusion system. Based on Karbowski 2004, but with a
 * removal of the Fgf8, Pax6, Emx2 system, and instead an option to
 * define several guidance molecules and
 *
 * Using 'Flt' for the float type, this will either be single precision
 * (float) or double precision (double).
 */
template <class Flt>
class RD_James : public RD_Base<Flt>
{
public:

    /*!
     * how many thalamo-cortical axon types are there? Denoted by N in
     * the paper, and so we use N here too.
     */
    alignas(Flt) unsigned int N = 5;

    /*!
     * M is the number of guidance molecules to use.
     */
    alignas(Flt) unsigned int M = 3;

    /*!
     * These are the c_i(x,t) variables from the Karb2004 paper. x is
     * a vector in two-space.
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > c;

    /*!
     * These are the a_i(x,t) variables from the Karb2004 paper. x is
     * a vector in two-space. The first vector is over the different
     * TC axon types, enumerated by i, the second vector are the a_i
     * values, indexed by the vi in the Hexes in HexGrid.
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > a;

    /*!
     * For each TC axon type, this holds the two components of the
     * gradient field of the scalar value a(x,t) (where this x is a
     * vector in two-space)
     */
    alignas(alignof(vector<array<vector<Flt>, 2> >))
    vector<array<vector<Flt>, 2> > grad_a;

    /*!
     * Contains the chemo-attractant modifiers which are applied to
     * a_i(x,t) in Eq 4.
     */
    alignas(alignof(vector<array<vector<Flt>, 2> >))
    vector<array<vector<Flt>, 2> > g;

    /*!
     * To hold div(g) / 3d, a static scalar field. There are M of
     * these vectors of Flts
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > divg_over3d;

    /*!
     * n(x,t) variable from the Karb2004 paper.
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> n;

    /*!
     * J_i(x,t) variables - the "flux current of axonal branches of
     * type i". This is a vector field.
     */
    alignas(alignof(vector<array<vector<Flt>, 2> >))
    vector<array<vector<Flt>, 2> > J;

    /*!
     * Holds the divergence of the J_i(x)s
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > divJ;

    /*!
     * The power to which a_i(x,t) is raised in Eqs 1 and 2 in the
     * paper.
     */
    alignas(Flt) Flt k = 3.0;

private:
    /*!
     * The diffusion parameter.
     */
    alignas(Flt) Flt D = 0.1;

    alignas(Flt) Flt twoDover3dd = this->D+this->D / 3*this->d*this->d;

public:

    /*!
     * alpha_i parameters
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> alpha;

    /*!
     * beta_i parameters
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> beta;

    /*!
     * epsilon_i parameters. axon competition parameter
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> epsilon;

private: // We have a setter for gamma.
    /*!
     * gamma_A/B/C_i (etc) parameters from Eq 4. There are M vectors
     * of Flts in here.
     */
    //@{
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > gamma;
    //@}

public:
    /*!
     * A vector of parameters for the direction of the guidance
     * molecules. This is an angle in Radians.
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> guidance_phi;

    /*!
     * Guidance molecule parameters for the width of the function
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> guidance_width;

    /*!
     * Width in orthogonal direction, for 2D fields.
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> guidance_width_ortho;

    /*!
     * Guidance molecule parameters for the offset of the function
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> guidance_offset;

    /*!
     * Guidance molecule parameters to be the gains of the functions
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> guidance_gain;

    /*!
     * Rho variables in Eq 4 - the concentrations of axon guidance
     * molecules A, B, C, etc. In Karbowski 2004, these are time
     * independent and we will treat them as such, populating them at
     * initialisation.
     *
     * There are M vector<Flts> in rho.
     */
    //@{
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > rho;
    //@}

    /*!
     * Into grad_rho put the two components of the gradient of
     * rho computed across the HexGrid surface.
     *
     * There are M gradient fields stored in this variable.
     */
    //@{
    alignas(alignof(vector<array<vector<Flt>, 2> >))
    vector<array<vector<Flt>, 2> > grad_rho;
    //@}

    /*!
     * Memory to hold an intermediate result
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > betaterm;

    /*!
     * Holds an intermediate value for the computation of Eqs 1 and 2.
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > alpha_c;

    /*!
     * The contour threshold. For contour plotting [see
     * plot_contour()], the field is normalised, then the contour is
     * plotted where the field crosses this threshold.
     */
    alignas(Flt) Flt contour_threshold = 0.5;

    /*!
     * ALIGNAS REGION ENDS.
     *
     * Below here, there's no need to worry about alignas keywords.
     */

    /*!
     * Sets the function of the guidance molecule method (FIXME: Make
     * this a vector)
     */
    vector<GuidanceMoleculeMethod> rhoMethod;

    /*!
     * Simple constructor; no arguments. Just calls RD_Base constructor
     */
    RD_James (void)
        : RD_Base<Flt>() {
    }

    /*!
     * Initialise this vector of vectors with noise. This is a
     * model-specific function.
     *
     * I apply a sigmoid to the boundary hexes, so that the noise
     * drops away towards the edge of the domain.
     */
    void noiseify_vector_vector (vector<vector<Flt> >& vv) {
        Flt randNoiseOffset = 0.8;
        Flt randNoiseGain = 0.1;
        for (unsigned int i = 0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                // boundarySigmoid. Jumps sharply (100, larger is
                // sharper) over length scale 0.05 to 1. So if
                // distance from boundary > 0.05, noise has normal
                // value. Close to boundary, noise is less.
                vv[i][h.vi] = morph::Tools::randF<Flt>() * randNoiseGain + randNoiseOffset;
                if (h.distToBoundary > -0.5) { // It's possible that distToBoundary is set to -1.0
                    Flt bSig = 1.0 / ( 1.0 + exp (-100.0*(h.distToBoundary-this->boundaryFalloffDist)) );
                    vv[i][h.vi] = vv[i][h.vi] * bSig;
                }
            }
        }
    }

    /*!
     * Perform memory allocations, vector resizes and so on.
     */
    void allocate (void) {

        RD_Base<Flt>::allocate();

        // Resize and zero-initialise the various containers
        this->resize_vector_vector (this->c, this->N);
        this->resize_vector_vector (this->a, this->N);
        this->resize_vector_vector (this->betaterm, this->N);
        this->resize_vector_vector (this->alpha_c, this->N);
        this->resize_vector_vector (this->divJ, this->N);
        this->resize_vector_vector (this->divg_over3d, this->N);

        this->resize_vector_variable (this->n);
        this->resize_vector_vector (this->rho, this->M);

        this->resize_vector_param (this->alpha, this->N);
        this->resize_vector_param (this->beta, this->N);
        this->resize_vector_param (this->epsilon, this->N);
        this->resize_vector_vector_param (this->gamma, this->N, this->M);

        this->resize_vector_array_vector (this->grad_rho, this->M);

        // Resize grad_a and other vector-array-vectors
        this->resize_vector_array_vector (this->grad_a, this->N);
        this->resize_vector_array_vector (this->g, this->N);
        this->resize_vector_array_vector (this->J, this->N);

        // rhomethod is a vector of size M
        this->rhoMethod.resize (this->M);
        for (unsigned int j=0; j<this->M; ++j) {
            // Set up with Sigmoid1D as default
            this->rhoMethod[j] = GuidanceMoleculeMethod::Sigmoid1D;
        }

        // Initialise alpha, beta and epsilon
        for (unsigned int i=0; i<this->N; ++i) {
            this->alpha[i] = 3;
            this->beta[i] = 3;
            this->epsilon[i] = 3;
        }
    }

    /*!
     * Initialise variables and parameters. Carry out one-time
     * computations required of the model. This should be able to
     * re-initialise a finished simulation as well as initialise the
     * first time.
     */
    void init (void) {

        this->stepCount = 0;

        // Zero c and n and other temporary variables
        this->zero_vector_vector (this->c, this->N);
        //this->zero_vector_vector (this->a); // gets noisified below
        this->zero_vector_vector (this->betaterm, this->N);
        this->zero_vector_vector (this->alpha_c, this->N);
        this->zero_vector_vector (this->divJ, this->N);
        this->zero_vector_vector (this->divg_over3d, this->N);

        this->zero_vector_variable (this->n);
        this->zero_vector_vector (this->rho, this->M);

        this->zero_vector_array_vector (this->grad_rho, this->M);

        // Resize grad_a and other vector-array-vectors
        this->zero_vector_array_vector (this->grad_a, this->N);
        this->zero_vector_array_vector (this->g, this->N);
        this->zero_vector_array_vector (this->J, this->N);

        // Initialise a with noise
        this->noiseify_vector_vector (this->a);

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

        for (unsigned int m=0; m<this->M; ++m) {
            if (this->rhoMethod[m] == GuidanceMoleculeMethod::Gauss1D) {
                // Construct Gaussian-waves rather than doing the full-Karbowski shebang.
                this->gaussian1D_guidance (m);

            } else if (this->rhoMethod[m] == GuidanceMoleculeMethod::Gauss2D) {
                // Construct 2 dimensional gradients
                this->gaussian2D_guidance (m);

            } else if (this->rhoMethod[m] == GuidanceMoleculeMethod::Sigmoid1D) {
                this->sigmoid_guidance (m);

            } else if (this->rhoMethod[m] == GuidanceMoleculeMethod::Linear1D) {
                this->linear_guidance (m);
            }
        }

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
                Flt bSig = 1.0 / ( 1.0 + exp (-100.0*(h.distToBoundary-this->boundaryFalloffDist)) );
                for (unsigned int m = 0; m<this->M; ++m) {
                    this->g[i][0][h.vi] += (this->gamma[m][i] * this->grad_rho[m][0][h.vi]) * bSig;
                    this->g[i][1][h.vi] += (this->gamma[m][i] * this->grad_rho[m][1][h.vi]) * bSig;
                }
            }
        }

        this->compute_divg_over3d();
    }

private:
    /*!
     * Require private setter for d. Slightly different from the base class version.
     */
    //@{
    void set_d (Flt d_) {
        RD_Base<Flt>::set_d (d_);
        /*
        this->d = d_;
        this->oneoverd = 1.0/this->d;
        this->oneover2d = 1.0/(2*this->d);
        this->oneover3d = 1.0/(3*this->d);
        */
        this->updateTwoDover3dd();
    }
    //@}

public:
    /*!
     * Public accessors for D, as it requires another attribute to be
     * updated at the same time.
     */
    //@{
    void set_D (Flt D_) {
        this->D = D_;
        this->updateTwoDover3dd();
    }
    Flt get_D (void) {
        return this->D;
    }
    //@}

private:
    /*!
     * Compute 2D/3d^2
     */
    void updateTwoDover3dd (void) {
        this->twoDover3dd = (this->D+this->D) / (3*this->d*this->d);
    }

public:
    /*!
     * Parameter setter methods
     */
    //@{
    int setGamma (unsigned int m_idx, unsigned int n_idx, Flt value) {
        if (gamma.size() > m_idx) {
            if (gamma[m_idx].size() > n_idx) {
                // Ok, we can set the value
                this->gamma[m_idx][n_idx] = value;
            } else {
                cerr << "WARNING: DID NOT SET GAMMA (too few TC axon types for n_idx=" << n_idx << ")" << endl;
                return 1;
            }
        } else {
            cerr << "WARNING: DID NOT SET GAMMA (too few guidance molecules for m_idx=" << m_idx << ")" << endl;
            return 2;
        }
        return 0;
    }
    //@}

    /*!
     * HDF5 file saving/loading methods.
     */
    //@{

    /*!
     * Save the c, a and n variables.
     */
    void save (void) {
        stringstream fname;
        fname << this->logpath << "/c_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        HdfData data(fname.str());
        for (unsigned int i = 0; i<this->N; ++i) {
            stringstream path;
            // The c variables
            path << "/c" << i;
            data.add_contained_vals (path.str().c_str(), this->c[i]);
            // The a variable
            path.str("");
            path.clear();
            path << "/a" << i;
            data.add_contained_vals (path.str().c_str(), this->a[i]);
            // An intermediate variable, for debugging.
            path.str("");
            path.clear();
            path << "/A" << i;
            data.add_contained_vals (path.str().c_str(), this->alpha_c[i]);
        }
        data.add_contained_vals ("/n", this->n);
    }

    /*!
     * Save the guidance molecules to a file (guidance.h5)
     */
    void saveGuidance (void) {
        stringstream fname;
        fname << this->logpath << "/guidance.h5";
        HdfData data(fname.str());
        for (unsigned int m = 0; m<this->M; ++m) {
            stringstream path;
            path << "/rh" << m;
            string pth(path.str());
            data.add_contained_vals (pth.c_str(), this->rho[m]);
            pth[1] = 'g'; pth[2] = 'x';
            data.add_contained_vals (pth.c_str(), this->grad_rho[m][0]);
            pth[2] = 'y';
            data.add_contained_vals (pth.c_str(), this->grad_rho[m][1]);
        }
        for (unsigned int i = 0; i<this->N; ++i) {
            stringstream path;
            path << "/divg_" << i;
            string pth(path.str());
            data.add_contained_vals (pth.c_str(), this->divg_over3d[i]);
        }
    }

    /*!
     * Computation methods
     */
    //@{

    /*!
     * Do a single step through the model.
     */
    void step (void) {

        this->stepCount++;

        // 1. Compute Karb2004 Eq 3. (coupling between connections made by each TC type)
        Flt nsum = 0.0;
        Flt csum = 0.0;
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

#ifdef DEBUG__
        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
            DBG ("sum of n+c is " << nsum+csum);
        }
#endif

        // 2. Do integration of a (RK in the 1D model). Involves computing axon branching flux.

        // Pre-compute intermediate val:
        for (unsigned int i=0; i<this->N; ++i) {
#pragma omp parallel for shared(i,k)
            for (unsigned int h=0; h<this->nhex; ++h) {
                // Fixme: I think the beta_n_a term needs to come OUT
                // of this, as it's the dependent variable in the RK
                // integration for A.
                //this->alpha_c_beta_na[i][h] = alpha[i] * c[i][h] - beta[i] * n[h] * static_cast<Flt>(pow (a[i][h], k));
                this->alpha_c[i][h] = alpha[i] * c[i][h];// -beta[i] * n[h] * static_cast<Flt>(pow (a[i][h], k));
            }
        }

        // Runge-Kutta:
        // No OMP here - there are only N(<10) loops, which isn't
        // enough to load the threads up.
        for (unsigned int i=0; i<this->N; ++i) {

            // Runge-Kutta integration for A
            vector<Flt> q(this->nhex, 0.0);
            this->compute_divJ (a[i], i); // populates divJ[i]

            vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->alpha_c[i][h] - beta[i] * n[h] * static_cast<Flt>(pow (a[i][h], k));
                q[h] = this->a[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->alpha_c[i][h] - beta[i] * n[h] * static_cast<Flt>(pow (q[h], k));
                q[h] = this->a[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] + this->alpha_c[i][h] - beta[i] * n[h] * static_cast<Flt>(pow (q[h], k));
                q[h] = this->a[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] + this->alpha_c[i][h] - beta[i] * n[h] * static_cast<Flt>(pow (q[h], k));
                a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
            }
        }

        // 3. Do integration of c
        for (unsigned int i=0; i<this->N; ++i) {

#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                // Note: betaterm used in compute_dci_dt()
                this->betaterm[i][h] = beta[i] * n[h] * static_cast<Flt>(pow (a[i][h], k));
            }

            // Runge-Kutta integration for C (or ci)
            vector<Flt> q(this->nhex,0.);
            vector<Flt> k1 = compute_dci_dt (c[i], i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                q[h] = c[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2 = compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                q[h] = c[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3 = compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                q[h] = c[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4 = compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                c[i][h] += (k1[h]+2. * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
            }
        }
    }

    /*!
     * Examine the value in each Hex of the hexgrid of the scalar
     * field f. If abs(f[h]) exceeds the size of dangerThresh, then
     * output debugging information.
     */
    void debug_values (vector<Flt>& f, Flt dangerThresh) {
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
     * Does: f = (alpha * f) + betaterm. c.f. Karb2004, Eq 1. f is
     * c[i] or q from the RK algorithm.
     */
    vector<Flt> compute_dci_dt (vector<Flt>& f, unsigned int i) {
        vector<Flt> dci_dt (this->nhex, 0.0);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; h++) {
            dci_dt[h] = (this->betaterm[i][h] - this->alpha[i] * f[h]);
        }
        return dci_dt;
    }

    /*!
     * Compute the divergence of g and divide by 3d. Used in
     * computation of term2 in compute_divJ().
     *
     * This computation is based on Gauss's theorem.
     */
    void compute_divg_over3d (void) {

        for (unsigned int i = 0; i < this->N; ++i) {

#pragma omp parallel for schedule(static)
            for (unsigned int hi=0; hi<this->nhex; ++hi) {

                Flt divg = 0.0;
                // First sum
                if (HAS_NE(hi)) {
                    divg += /*cos (0)*/ (this->g[i][0][NE(hi)] + this->g[i][0][hi]);
                } else {
                    // Boundary condition _should_ be satisfied by
                    // sigmoidal roll-off of g towards the boundary, so
                    // add only g[i][0][hi]
                    divg += /*cos (0)*/ (this->g[i][0][hi]);
                }
                if (HAS_NNE(hi)) {
                    divg += /*cos (60)*/ 0.5 * (this->g[i][0][NNE(hi)] + this->g[i][0][hi])
                        +  (/*sin (60)*/ this->R3_OVER_2 * (this->g[i][1][NNE(hi)] + this->g[i][1][hi]));
                } else {
                    //divg += /*cos (60)*/ (0.5 * (this->g[i][0][hi]))
                    //    +  (/*sin (60)*/ this->R3_OVER_2 * (this->g[i][1][hi]));
                }
                if (HAS_NNW(hi)) {
                    divg += -(/*cos (120)*/ 0.5 * (this->g[i][0][NNW(hi)] + this->g[i][0][hi]))
                        +    (/*sin (120)*/ this->R3_OVER_2 * (this->g[i][1][NNW(hi)] + this->g[i][1][hi]));
                } else {
                    //divg += -(/*cos (120)*/ 0.5 * (this->g[i][0][hi]))
                    //    +    (/*sin (120)*/ this->R3_OVER_2 * (this->g[i][1][hi]));
                }
                if (HAS_NW(hi)) {
                    divg -= /*cos (180)*/ (this->g[i][0][NW(hi)] + this->g[i][0][hi]);
                } else {
                    divg -= /*cos (180)*/ (this->g[i][0][hi]);
                }
                if (HAS_NSW(hi)) {
                    divg -= (/*cos (240)*/ 0.5 * (this->g[i][0][NSW(hi)] + this->g[i][0][hi])
                             + ( /*sin (240)*/ this->R3_OVER_2 * (this->g[i][1][NSW(hi)] + this->g[i][1][hi])));
                } else {
                    divg -= (/*cos (240)*/ 0.5 * (this->g[i][0][hi])
                             + (/*sin (240)*/ this->R3_OVER_2 * (this->g[i][1][hi])));
                }
                if (HAS_NSE(hi)) {
                    divg += /*cos (300)*/ 0.5 * (this->g[i][0][NSE(hi)] + this->g[i][0][hi])
                        - ( /*sin (300)*/ this->R3_OVER_2 * (this->g[i][1][NSE(hi)] + this->g[i][1][hi]));
                } else {
                    divg += /*cos (300)*/ 0.5 * (this->g[i][0][hi])
                        - ( /*sin (300)*/ this->R3_OVER_2 * (this->g[i][1][hi]));
                }

                this->divg_over3d[i][hi] = divg * this->oneover3d;
            }
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
    void compute_divJ (vector<Flt>& fa, unsigned int i) {

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

    /*!
     * Generate Gaussian profiles for the chemo-attractants.
     *
     * Instead of using the Karbowski equations, just make some
     * gaussian 'waves'
     */
    void gaussian1D_guidance (unsigned int m) {
        for (auto h : this->hg->hexen) {
            Flt cosphi = (Flt) cos (guidance_phi[m]);
            Flt sinphi = (Flt) sin (guidance_phi[m]);
            DBG2 ("phi: " << guidance_phi[m]);
            Flt x_ = (h.x * cosphi) + (h.y * sinphi);
            this->rho[m][h.vi] = guidance_gain[m] * exp(-((x_-guidance_offset[m])*(x_-guidance_offset[m])) / guidance_width[m]);
        }
    }

    void gaussian2D_guidance (unsigned int m) {
    }

    void sigmoid_guidance (unsigned int m) {
        for (auto h : this->hg->hexen) {
            Flt cosphi = (Flt) cos (this->guidance_phi[m]);
            Flt sinphi = (Flt) sin (this->guidance_phi[m]);
            //DBG("phi= " << this->guidance_phi[m] << ". cosphi: " << cosphi << " sinphi: " << sinphi);
            Flt x_ = (h.x * cosphi) + (h.y * sinphi);
            //DBG ("x_[" << h.vi << "] = " << x_);
            this->rho[m][h.vi] = guidance_gain[m] / (1.0 + exp(-(x_-guidance_offset[m])/this->guidance_width[m]));
        }
    }

    void linear_guidance (unsigned int m) {
        for (auto h : this->hg->hexen) {
            Flt cosphi = (Flt) cos (this->guidance_phi[m]);
            Flt sinphi = (Flt) sin (this->guidance_phi[m]);
            Flt x_ = (h.x * cosphi) + (h.y * sinphi);
            this->rho[m][h.vi] = (x_-guidance_offset[m]) * this->guidance_gain[m];
        }
    }

}; // RD_James
