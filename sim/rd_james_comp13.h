/*
 * 2D Karbowski system with *divisive* normalization of a_i and competition.
 */

#include "rd_james_comp8.h"

template <class Flt> //       RD_James_comp8 base class gives divisive normalization
class RD_James_comp13 : public RD_James_comp8<Flt>
{
public:

    //! Inter-TC-type competition
    //@{
    //! The power to which a_j is raised for the inter-TC axon competition term.
    alignas(Flt) Flt l = 3.0;
    //! epsilon_i parameters. axon competition parameter
    alignas(alignof(vector<Flt>))
    vector<Flt> epsilon;
    //! Holds a copy of a[i] * epsilon / (1-N)
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > a_eps;
    //@}

    //! comp2 params - flux current affected by gradient of other branching densities
    //@{
    //!n Parameter which controls the strength of diffusion away from axon branching of other TC types.
    alignas(Flt) Flt F = 0.2;
    alignas(Flt) Flt FOverNm1 = 0.0;
    //! \hat{a}_i. Recomputed for each new i, so doesn't need to be a vector of vectors.
    alignas(alignof(vector<Flt>)) vector<Flt> ahat;
    //! Gradient of \hat{a}_i(x,t)
    alignas(alignof(array<vector<Flt>, 2>)) array<vector<Flt>, 2> grad_ahat;
    //! divergence of \hat{a}_i(x,t).
    alignas(alignof(vector<Flt>)) vector<Flt> div_ahat;
    //@}

    RD_James_comp13 (void)
        : RD_James_comp8<Flt>() {
    }

    virtual void allocate (void) {
        RD_James_comp8<Flt>::allocate();

        // epsilon based competition
        this->resize_vector_param (this->epsilon, this->N);
        this->resize_vector_vector (this->a_eps, this->N);

        // Diffusion based on \hat{a}_i
        this->resize_vector_variable (this->ahat);
        this->resize_gradient_field (this->grad_ahat);
        this->resize_vector_variable (this->div_ahat);

    }

    virtual void init (void) {
        RD_James_comp8<Flt>::init();
        this->zero_vector_variable (this->ahat);
        this->zero_gradient_field (this->grad_ahat);
        this->zero_vector_variable (this->div_ahat);

    }

    //! Compute divergence of \hat{a}_i
    void compute_divahat (void) {
#pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Flt thesum = -6 * this->ahat[hi];
            thesum += this->ahat[(HAS_NE(hi)  ? NE(hi)  : hi)];
            thesum += this->ahat[(HAS_NNE(hi) ? NNE(hi) : hi)];
            thesum += this->ahat[(HAS_NNW(hi) ? NNW(hi) : hi)];
            thesum += this->ahat[(HAS_NW(hi)  ? NW(hi)  : hi)];
            thesum += this->ahat[(HAS_NSW(hi) ? NSW(hi) : hi)];
            thesum += this->ahat[(HAS_NSE(hi) ? NSE(hi) : hi)];
            this->div_ahat[hi] = this->twoover3dd * thesum;
            if (isnan(this->div_ahat[hi])) {
                cerr << "div ahat isnan" << endl;
                exit (3);
            }
        }
    }

    //! Compute the divergence of J
    void compute_divJ (vector<Flt>& fa, unsigned int i) {

        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa, this->grad_a[i]);

        if (this->N > 0) {
            this->FOverNm1 = this->F/(this->N-1);
        } else {
            this->FOverNm1 = 0.0;
        }

        // _Five_ terms to compute; see Eq. 17 in methods_notes.pdf. Copy comp3.
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
            if (isnan(term1)) {
                cerr << "term1 isnan" << endl;
                cerr << "thesum is " << thesum << " fa[hi=" << hi << "] = " << fa[hi] << endl;
                exit (21);
            }

            // Term 1.1 is F/N-1 a div(ahat)
            Flt term1_1 = this->FOverNm1 * fa[hi] * this->div_ahat[hi];
            if (isnan(term1_1)) {
                cerr << "term1_1 isnan" << endl;
                exit (21);
            }

            // Term 1.2 is F/N-1 grad(ahat) . grad(a)
            Flt term1_2 = this->FOverNm1 * (this->grad_ahat[0][hi] * this->grad_a[i][0][hi]
                                            + this->grad_ahat[1][hi] * this->grad_a[i][1][hi]);
            if (isnan(term1_2)) {
                cerr << "term1_2 isnan" << endl;
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

            this->divJ[i][hi] = term1 - term1_1 - term1_2 - term2 - term3;
        }
    }

    virtual void integrate_a (void) {

        // 2. Do integration of a (RK in the 1D model). Involves computing axon branching flux.

        // Pre-compute:
        // 1) The intermediate val alpha_c.
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
            // Also compute ahat.
            vector<Flt> eps(this->nhex, 0.0);
            for (unsigned int j=0; j<this->N; ++j) {
                if (j==i) { continue; }
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    this->ahat[h] += this->a[j][h];
                    eps[h] += static_cast<Flt>(pow (this->a[j][h], this->l));
                }
            }

            // Multiply it by epsilon[i]/(N-1). Now it's ready to subtract from the solutions
            Flt eps_over_N = this->epsilon[i]/(this->N-1);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                eps[h] *= eps_over_N;
                // Store a[i][h] * eps[h] for analysis (it's also used 4 times below)
                //this->a_eps[i][h] = this->a[i][h] * eps[h];
            }

            // 1.1 Compute divergence and gradient of ahat
            this->compute_divahat();
            this->spacegrad2D (this->ahat, this->grad_ahat);

            // Runge-Kutta integration for A
            vector<Flt> qq(this->nhex, 0.0);
            this->compute_divJ (this->a[i], i); // populates divJ[i]

            vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] - this->dc[i][h] - this->a[i][h] * eps[h];
                qq[h] = this->a[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2(this->nhex, 0.0);
            this->compute_divJ (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] - this->dc[i][h] - qq[h] * eps[h];
                qq[h] = this->a[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3(this->nhex, 0.0);
            this->compute_divJ (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] - this->dc[i][h] - qq[h] * eps[h];
                qq[h] = this->a[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4(this->nhex, 0.0);
            this->compute_divJ (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] - this->dc[i][h] - qq[h] * eps[h];
                this->a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
            }

            // Do any necessary computation which involves summing a here
            this->sum_a_computation (i);

            // Now apply the transfer function
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->a[i][h] = this->transfer_a (this->a[i][h], i);
            }
        }
    }

}; // RD_James_norm
