/*
 * 2D Karbowski system with *divisive* normalization of a_i and competition.
 */

#include "rd_james_comp8.h"

template <class Flt> //       RD_James_comp8 base class gives divisive normalization
class RD_James_comp9 : public RD_James_comp8<Flt>
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
     * Simple constructor; no arguments. Just calls base constructor.
     */
    RD_James_comp9 (void)
        : RD_James_comp8<Flt>() {
    }

    virtual void allocate (void) {
        RD_James_comp8<Flt>::allocate();
        this->resize_vector_param (this->epsilon, this->N);
        this->resize_vector_vector (this->a_eps, this->N);
    }

    virtual void init (void) {
        RD_James_comp8<Flt>::init();
    }

    /*!
     * Computation methods
     */
    //@{
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
                //this->a_eps[i][h] = this->a[i][h] * eps[h];
            }

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
    //@}

}; // RD_James_norm
