/*
 * Competition method 4, Grad n AND Grad a_hat contribute to J.
 * (rd_karbowski.pdf; label eq:Karb2D_J_NM_with_comp4)
 */

#include "rd_james.h"

/*!
 * Additional specialisation to add competition (by modifying the divJ
 * diffusion component)
 */
template <class Flt>
class RD_James_comp4 : public RD_James<Flt>
{
public:
    /*!
     * Parameter which controls the strength of diffusion away from
     * axon branching of other TC types.
     */
    alignas(Flt) Flt Dprime = 0.1;
    alignas(Flt) Flt DprimeOverNm1 = 0.0;

    /*!
     * Parameter which controls the strength of the contribution from
     * the gradient of n(x,t) to the flux current of axonal branching.
     */
    alignas(Flt) Flt Dn = 0.1;

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
     * This holds the two components of the gradient field of the
     * scalar value \hat{a}_i(x,t), which is the sum of the branching
     * densities of all axon types except i. See Eq 41. in
     * rd_karbowski.pdf
     */
    alignas(alignof(array<vector<Flt>, 2>))
    array<vector<Flt>, 2> grad_ahat;

    /*!
     * divergence of \hat{a}_i(x,t).
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> div_ahat;

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
        this->resize_gradient_field (this->grad_ahat);
        this->resize_vector_variable (this->div_ahat);
    }
    virtual void init (void) {
        RD_James<Flt>::init();
        this->zero_gradient_field (this->grad_n);
        this->zero_vector_variable (this->div_n);
        this->zero_gradient_field (this->grad_ahat);
        this->zero_vector_variable (this->div_ahat);
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
            csum += this->c[0][hi];
            this->n[hi] = 1. - this->n[hi];
            nsum += this->n[hi];
        }

        // 1.1 Compute divergence and gradient of n
        this->compute_divn();
        this->spacegrad2D (this->n, this->grad_n);

#ifdef DEBUG__
        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
            DBG ("sum of n+c is " << nsum+csum);
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

            // Compute a_hat, which is "the sum of all a_j for which j!=i"
            vector<Flt> a_hat(this->nhex, 0.0);
            for (unsigned int j=0; j<this->N; ++j) {
                if (j==i) { continue; }
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    a_hat[h] += this->a[j][h];
                }
            }
            // Compute divergence of a_hat
            this->compute_divf (a_hat, this->div_ahat);
            // And gradient of a_hat
            this->spacegrad2D (a_hat, this->grad_ahat);

            // Runge-Kutta integration for A
            vector<Flt> q(this->nhex, 0.0);
            this->compute_divJ (this->a[i], a_hat, i); // populates divJ[i]

            vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a[i][h], this->k));
                q[h] = this->a[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2(this->nhex, 0.0);
            this->compute_divJ (q, a_hat, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k));
                q[h] = this->a[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3(this->nhex, 0.0);
            this->compute_divJ (q, a_hat, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k));
                q[h] = this->a[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4(this->nhex, 0.0);
            this->compute_divJ (q, a_hat, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k));
                this->a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
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
            }
        }
    }

    /*!
     * Compute divergence of n
     */
    void compute_divn (void) {
#pragma omp parallel for //schedule(static) // This was about 10% faster than schedule(dynamic,50).
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

    void compute_divf (vector<Flt>& field, vector<Flt>& div_field) {
#pragma omp parallel for //schedule(static) // This was about 10% faster than schedule(dynamic,50).
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Flt thesum = -6 * field[hi];
            thesum += field[(HAS_NE(hi)  ? NE(hi)  : hi)];
            thesum += field[(HAS_NNE(hi) ? NNE(hi) : hi)];
            thesum += field[(HAS_NNW(hi) ? NNW(hi) : hi)];
            thesum += field[(HAS_NW(hi)  ? NW(hi)  : hi)];
            thesum += field[(HAS_NSW(hi) ? NSW(hi) : hi)];
            thesum += field[(HAS_NSE(hi) ? NSE(hi) : hi)];
            div_field[hi] = this->twoover3dd * thesum;
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
    void compute_divJ (vector<Flt>& fa, vector<Flt>& fa_hat, unsigned int i) {

        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa, this->grad_a[i]);

        if (this->N > 0) {
            this->DprimeOverNm1 = this->Dprime/(this->N-1);
        } else {
            this->DprimeOverNm1 = 0.0;
        }

        // Three terms to compute; see Eq. 17 in methods_notes.pdf
#pragma omp parallel for //schedule(static) // This was about 10% faster than schedule(dynamic,50).
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

            // Term 1.1 is Dn a div(n)
            Flt term1_1 = this->Dn * fa[hi] * this->div_n[hi];

            // Term 1.2 is Dn grad(n) . grad(a)
            Flt term1_2 = this->Dn * (this->grad_n[0][hi] * this->grad_a[i][0][hi]
                                      + this->grad_n[1][hi] * this->grad_a[i][1][hi]);

            // Term 1.3 is D' a div(a_hat)
            Flt term1_3 = this->DprimeOverNm1 * fa[hi] * this->div_ahat[hi];

            // Term 1.4 is D' grad(a_hat) . grad(a)
            Flt term1_4 = this->DprimeOverNm1 * (this->grad_ahat[0][hi] * this->grad_a[i][0][hi]
                                                 + this->grad_ahat[1][hi] * this->grad_a[i][1][hi]);

            // 2. The (a div(g)) term.
            Flt term2 = fa[hi] * this->divg_over3d[i][hi];

            // 3. Third term is this->g . grad a_i. Should not contribute to J, as g(x) decays towards boundary.
            Flt term3 = this->g[i][0][hi] * this->grad_a[i][0][hi] + (this->g[i][1][hi] * this->grad_a[i][1][hi]);

            this->divJ[i][hi] = term1 - term1_1 - term1_2 - term1_3 - term1_4 - term2 - term3;
        }
    }

}; // RD_James