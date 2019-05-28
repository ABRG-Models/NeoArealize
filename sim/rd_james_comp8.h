/*
 * 2D Karbowski system with divisive normalization of a_i.
 */

#include "rd_james.h"

/*!
 * Additional specialisation to add competition (by modifying the divJ
 * diffusion component)
 */
template <class Flt>
class RD_James_comp8 : public RD_James<Flt>
{
public:
    //! overall 'gain' of the normalization equation
    alignas(Flt) Flt eta = 100.0;
    //! offset of the normalization equation
    alignas(Flt) Flt xi = 1.0;
    //! power value of the normalization equation
    alignas(Flt) Flt q = 2.0;
    //! eta to the power of q
    alignas(Flt) Flt eta_to_q = 1.0;

    /*!
     * Response of a after being passed through the normalization
     * equation.
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > a_res;

    /*!
     * Simple constructor; no arguments. Just calls base constructor
     */
    RD_James_comp8 (void)
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
        this->resize_vector_vector (this->a_res, this->N);

    }
    virtual void init (void) {
        RD_James<Flt>::init();
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

        // Don't really need to do this every step, but it'll take negligible time
        this->eta_to_q = pow (this->eta, this->q);

        // 1. Compute Karb2004 Eq 3. (coupling between connections made by each TC type)
        Flt nsum = 0.0;
        Flt csum = 0.0;
#pragma omp parallel for reduction(+:nsum,csum)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->n[hi] = 0;
            // First, use n[hi] so sum c over all i:
            for (unsigned int i=0; i<this->N; ++i) {
                this->n[hi] += this->c[i][hi];
            }
            // Prevent sum of c being too large:
            this->n[hi] = (this->n[hi] > 1.0) ? 1.0 : this->n[hi];
            csum += this->c[0][hi];
            // Now compute n for real:
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
        // Pre-compute the sum of all a_i across all TC sheets.
        Flt total_sum_a = 0.0;
        for (unsigned int i=0; i<this->N; ++i) {
//#pragma omp parallel for shared(i,k)
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->alpha_c[i][h] = this->alpha[i] * this->c[i][h];
            }

            // Compute the sum of a[i] across the sheet.
            Flt sum_a_to_q = 0.0;
#pragma omp parallel for reduction(+:sum_a_to_q)
            for (unsigned int h=0; h<this->nhex; ++h) {
                sum_a_to_q += pow (this->a[i][h], this->q);
            }

            total_sum_a += sum_a_to_q;
        }
        Flt eta_to_q_plus_sum_a_to_q = eta_to_q + total_sum_a;

        // Runge-Kutta:
        // No OMP here - there are only N(<10) loops, which isn't
        // enough to load the threads up.
        for (unsigned int i=0; i<this->N; ++i) {

            // Runge-Kutta integration for A
            vector<Flt> q(this->nhex, 0.0);
            this->compute_divJ (this->a[i], i); // populates divJ[i]

#if 0
            // Compute the sum of a[i] across the sheet.
            Flt sum_a_to_q = 0.0;
#pragma omp parallel for reduction(+:sum_a_to_q)
            for (unsigned int h=0; h<this->nhex; ++h) {
                sum_a_to_q += pow (this->a[i][h], this->q);
            }
            Flt eta_to_q_plus_sum_a_to_q = eta_to_q + sum_a_to_q;
            if (this->stepCount % 100 == 0) {
                cout << "sum_a_to_q = " << sum_a_to_q << endl;
                cout << "a_i0 = " << this->a[i][0] << endl;
            }
#endif

            vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a[i][h], this->k));
                q[h] = this->a[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k));
                q[h] = this->a[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k));
                q[h] = this->a[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] + this->alpha_c[i][h] - this->beta[i] * this->n[h] * static_cast<Flt>(pow (q[h], this->k));
                this->a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;

                // Divisive normalization step (across all TC types)
                this->a_res[i][h] = this->eta * (pow (this->a[i][h], this->q) / (eta_to_q_plus_sum_a_to_q));
                if (this->stepCount % 100 == 0) {
                    cout << "a_res_i0 = " << this->a_res[i][0] << endl;
                }

                // Prevent a from becoming negative:
                //this->a[i][h] = (this->a[i][h] < 0.0) ? 0.0 : this->a[i][h];
            }
        }

        // 3. Do integration of c
        for (unsigned int i=0; i<this->N; ++i) {

#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                // Note: betaterm used in compute_dci_dt()
                this->betaterm[i][h] = this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a_res[i][h], this->k));
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

}; // RD_James
