/*
 * Competition method 7. Reduce diffusion multiplicatively.
 */

#include "rd_james.h"

template <class Flt>
class RD_James_comp7 : public RD_James<Flt>
{
public:
    /*!
     * The power to which a_j is raised for the inter-TC axon
     * competition term.
     */
    alignas(Flt) Flt l = 3.0;

    /*!
     * epsilon_i parameters. axon competition parameter
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> epsilon;

    /*!
     * Holds a copy of a[i] * epsilon / (1-N)
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > a_eps;

    /*!
     * \hat{a}_i. Recomputed for each new i, so doesn't need to be a
     * vector of vectors.
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> ahat;

    /*!
     * \lambda(\hat{a}_i)
     */
    //@{
    alignas(alignof(vector<Flt>))
    vector<Flt> lambda;
    alignas(alignof(array<vector<Flt>, 2>))
    array<vector<Flt>, 2> grad_lambda;
    //@}

    /*!
     * Sigmoid parameters
     */
    //@{
    alignas(Flt) Flt o = 5.0; // offset
    alignas(Flt) Flt s = 0.5; // sharpness
    //@}

    /*!
     * Simple constructor; no arguments. Just calls base constructor
     */
    RD_James_comp7 (void)
        : RD_James<Flt>() {
    }

    /*!
     * Override allocate() and init(), and add a couple of extra
     * resizes.
     */
    //@{
    virtual void allocate (void) {
        RD_James<Flt>::allocate();
        this->resize_vector_variable (this->ahat);
        this->resize_vector_variable (this->lambda);
        this->resize_gradient_field (this->grad_lambda);
        this->resize_vector_param (this->epsilon, this->N);
        this->resize_vector_vector (this->a_eps, this->N);
    }
    virtual void init (void) {
        RD_James<Flt>::init();
        this->zero_vector_variable (this->ahat);
        this->zero_vector_variable (this->lambda);
        this->zero_gradient_field (this->grad_lambda);
    }
    //@}
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
            this->n[hi] = 0;
            for (unsigned int i=0; i<this->N; ++i) {
                this->n[hi] += this->c[i][hi];
            }
            // Prevent sum of c being too large:
            this->n[hi] = (this->n[hi] > 1.0) ? 1.0 : this->n[hi];
            csum += this->c[0][hi];
            this->n[hi] = 1. - this->n[hi];
            nsum += this->n[hi];
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
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->alpha_c[i][h] = this->alpha[i] * this->c[i][h];
            }
        }

        // Runge-Kutta:
        // No OMP here - there are only N(<10) loops, which isn't
        // enough to load the threads up.
        for (unsigned int i=0; i<this->N; ++i) {

            // Compute epsilon * a_hat^l. a_hat is "the sum of all a_j
            // for which j!=i". Call the variable just 'eps'.
            vector<Flt> eps(this->nhex, 0.0);
            for (unsigned int j=0; j<this->N; ++j) {
                if (j==i) { continue; }
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    eps[h] += static_cast<Flt>(pow (this->a[j][h], this->l));
                }
            }

            // Multiply it by epsilon[i]/(N-1). Now it's ready to subtract from the solutions
            Flt eps_over_N = this->epsilon[i]/(this->N-1);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                eps[h] *= eps_over_N;
                // Store a[i][h] * eps[h] for analysis (it's also used 4 times below)
                this->a_eps[i][h] = this->a[i][h] * eps[h];
            }

            Flt ahatmax = 0.0;
            // Compute "the sum of all a_j for which j!=i"
            this->zero_vector_variable (this->ahat);
            for (unsigned int j=0; j<this->N; ++j) {
                if (j==i) { continue; }
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    this->ahat[h] += this->a[j][h];
                    ahatmax = (this->ahat[h] > ahatmax) ? this->ahat[h] : ahatmax;
                }
            }

            cout << "ahatmax = " << ahatmax << endl;
            cout << "lambda(ahatmax) = " << (1.0 - (1.0 / (1.0 + exp(this->o - this->s * ahatmax) ))) << endl;

            // Compute lambda(ahat) and grad lambda
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->lambda[h] = 1.0 - (1.0 / (1.0 + exp(o - s * this->ahat[h]) ));
                //cout << "ahat["<<h<<"]=" << this->ahat[h] << " lambda[h]=" << this->lambda[h] << endl;
            }
            this->spacegrad2D (this->lambda, this->grad_lambda);

            // Runge-Kutta integration for A
            vector<Flt> q(this->nhex, 0.0);
            this->compute_divJ (this->a[i], i); // populates divJ[i]

            vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a[i][h], this->k))  - this->a_eps[i][h];
                q[h] = this->a[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k)) - this->a_eps[i][h];
                q[h] = this->a[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k)) - this->a_eps[i][h];
                q[h] = this->a[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k)) - this->a_eps[i][h];
                this->a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
                // Prevent a from becoming negative:
                this->a[i][h] = (this->a[i][h] < 0.0) ? 0.0 : this->a[i][h];
            }
            //cout << "a[" << i << "][0] = " << this->a[i][0] << endl;
            if (isnan(this->a[i][0])) {
                cerr << "Exiting on a[i][0] == NaN" << endl;
                exit (1);
            }
        }

        // 3. Do integration of c
        for (unsigned int i=0; i<this->N; ++i) {

#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                // Note: betaterm used in compute_dci_dt()
                this->betaterm[i][h] = this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a[i][h], this->k));
            }

            // Runge-Kutta integration for C (or ci)
            vector<Flt> q(this->nhex,0.);
            vector<Flt> k1 = this->compute_dci_dt (this->c[i], i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                q[h] = this->c[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2 = this->compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                q[h] = this->c[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3 = this->compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                q[h] = this->c[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4 = this->compute_dci_dt (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                this->c[i][h] += (k1[h]+2. * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
                // Avoid over-saturating c_i:
                this->c[i][h] = (this->c[i][h] > 1.0) ? 1.0 : this->c[i][h];
            }
#if 0
            cout << "c[" << i << "][0] = " << this->c[i][0] << endl;
            if (isnan(this->c[i][0])) {
                cerr << "Exiting on c[i][0] == NaN" << endl;
                exit (2);
            }
#endif
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

        // _Five_ terms to compute; see Eq. 17 in methods_notes.pdf. Copy comp3.
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

            // Multiply sum by \lambda 2D/3d^2 to give term1
            Flt term1 = this->twoDover3dd * thesum * this->lambda[hi];
            if (isnan(term1)) {
                cerr << "term1 isnan" << endl;
                cerr << "thesum is " << thesum << " fa[hi=" << hi << "] = " << fa[hi] << endl;
                exit (21);
            }

            // Term 1.1 is D grad a dot grad lambda
            Flt term1_1 = this->D * (this->grad_a[i][0][hi] * this->grad_lambda[0][hi]
                                     + this->grad_a[i][1][hi] * this->grad_lambda[1][hi]);
            if (isnan(term1_1)) {
                cerr << "term1_1 isnan" << endl;
                cerr << "fa[hi="<<hi<<"] = " << fa[hi] << endl;
                exit (21);
            }

            // 2. The (a div(g)) term.
            Flt term2 = fa[hi] * this->divg_over3d[i][hi];

            if (isnan(term2)) {
                cerr << "term2 isnan" << endl;
                exit (21);
            }

            // 3. Third term is this->g . grad a_i. Should not contribute to J, as g(x) decays towards boundary.
            Flt term3 = this->g[i][0][hi] * this->grad_a[i][0][hi] + (this->g[i][1][hi] * this->grad_a[i][1][hi]);

            if (isnan(term3)) {
                cerr << "term3 isnan" << endl;
                exit (30);
            }

            this->divJ[i][hi] = term1 + term1_1 - term2 - term3;
        }
    }

}; // RD_James
