/*
 * Competition method 4, which combines method 1 and method 3.
 */

#include "rd_james.h"

template <class Flt>
class RD_James_comp4 : public RD_James<Flt>
{
public:
    /*!
     * Parameter which controls the strength of the contribution from
     * the gradient of n(x,t) to the flux current of axonal branching.
     */
    alignas(Flt) Flt E = 0.1;

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
     * This holds the two components of the gradient field of the
     * scalar value n(x,t)
     */
    alignas(alignof(array<vector<Flt>, 2>))
    array<vector<Flt>, 2> grad_n;

    /*!
     * divergence of n.
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> div_n;

    /*!
     * Simple constructor; no arguments. Just calls base constructor
     */
    RD_James_comp4 (void)
        : RD_James<Flt>() {
    }

    /*!
     * Override allocate() and init(), and add a couple of extra
     * resizes.
     */
    //@{
    virtual void allocate (void) {
        RD_James<Flt>::allocate();
        // Plus:
        this->resize_gradient_field (this->grad_n);
        this->resize_vector_variable (this->div_n);
        this->resize_vector_param (this->epsilon, this->N);
        this->resize_vector_vector (this->a_eps, this->N);
    }
    virtual void init (void) {
        RD_James<Flt>::init();
        this->zero_gradient_field (this->grad_n);
        this->zero_vector_variable (this->div_n);
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
        Flt asum = 0.0;
        Flt csum = 0.0;
#pragma omp parallel for reduction(+:nsum,csum)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->n[hi] = 0;
            for (unsigned int i=0; i<this->N; ++i) {
                this->n[hi] += this->c[i][hi];
                asum += this->a[i][hi];
            }
            // Prevent sum of c being too large:
            this->n[hi] = (this->n[hi] > 1.0) ? 1.0 : this->n[hi];
            csum += this->n[hi];
            this->n[hi] = 1. - this->n[hi];
            nsum += this->n[hi];
        }
        this->v_nsum.push_back(nsum);
        this->v_csum.push_back(csum);
        this->v_asum.push_back(asum);

        // 1.1 Compute divergence and gradient of n
        this->compute_divn();
        this->spacegrad2D (this->n, this->grad_n);

#define DEBUG_SUMS 1
#ifdef DEBUG_SUMS
        if (this->stepCount % 100 == 0) {
            //DBG ("System computed " << this->stepCount << " times so far...");
            DBG (this->stepCount << ": nsum = " << nsum << ", csum = " << csum << ", n+c = " << nsum + csum  << ", asum = " << asum);
        }
#endif

        // 2. Do integration of a (RK in the 1D model). Involves computing axon branching flux.

        // Pre-compute intermediate val:
        for (unsigned int i=0; i<this->N; ++i) {
//#pragma omp parallel for shared(i,k)
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


            // Runge-Kutta integration for A
            vector<Flt> q(this->nhex, 0.0);
            this->compute_divJ (this->a[i], i); // populates divJ[i]

            vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                //cout << "k1 = " << this->divJ[i][h] << "(divJ) + " << this->alpha_c[i][h] << "(alphac) - " << this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a[i][h], this->k)) << "(beta) - " << this->a_eps[i][h] << "(a_eps)" << endl;
                k1[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a[i][h], this->k)) - this->a_eps[i][h];
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
        }
    }

    /*!
     * Override save to additionally save a_eps.
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
            // divJ
            path.str("");
            path.clear();
            path << "/j" << i;
            data.add_contained_vals (path.str().c_str(), this->divJ[i]);
            // The a_eps variable
            path.str("");
            path.clear();
            path << "/a_eps_" << i;
            data.add_contained_vals (path.str().c_str(), this->a_eps[i]);

        }
        data.add_contained_vals ("/n", this->n);
    }

    /*!
     * Compute divergence of n
     */
    void compute_divn (void) {
#pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Flt thesum = -6 * this->n[hi];
            thesum += this->n[(HAS_NE(hi)  ? NE(hi)  : hi)];
            thesum += this->n[(HAS_NNE(hi) ? NNE(hi) : hi)];
            thesum += this->n[(HAS_NNW(hi) ? NNW(hi) : hi)];
            thesum += this->n[(HAS_NW(hi)  ? NW(hi)  : hi)];
            thesum += this->n[(HAS_NSW(hi) ? NSW(hi) : hi)];
            thesum += this->n[(HAS_NSE(hi) ? NSE(hi) : hi)];
            this->div_n[hi] = this->twoover3dd * thesum;
        }
    }

    /*!
     * Computes the "flux of axonal branches" term, J_i(x) (Eq 4)
     *
     * Inputs: this->g, @fa (which is this->a[i] or a q in the RK
     * algorithm), this->div_n, this->D, @i, the TC type.  Helper functions:
     * spacegrad2D().  Output: this->divJ
     *
     * Stable with dt = 0.0001;
     */
    void compute_divJ (vector<Flt>& fa, unsigned int i) {

        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa, this->grad_a[i]);

        // Three terms to compute; see Eq. 17 in methods_notes.pdf
#pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            // 1. The D Del^2 a_i term. Eq. 18.
            // 1a. Or D Del^2 Sum(a_i) (new)
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

            // Term 1.1 is E a div(n)
            Flt term1_1 = this->E * fa[hi] * this->div_n[hi];

            // Term 1.2 is E grad(n) . grad(a)
            Flt term1_2 = this->E * (this->grad_n[0][hi] * this->grad_a[i][0][hi]
                                     + this->grad_n[1][hi] * this->grad_a[i][1][hi]);

            // 2. The (a div(g)) term.
            Flt term2 = fa[hi] * this->divg_over3d[i][hi];

            // 3. Third term is this->g . grad a_i. Should not contribute to J, as g(x) decays towards boundary.
            Flt term3 = this->g[i][0][hi] * this->grad_a[i][0][hi] + (this->g[i][1][hi] * this->grad_a[i][1][hi]);

            this->divJ[i][hi] = term1 - term1_1 - term1_2 - term2 - term3;
        }
    }

}; // RD_James
