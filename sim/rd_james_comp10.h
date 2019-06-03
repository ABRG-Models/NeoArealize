/*
 * 2D Karbowski system with upper bound on \sum_i a_i (using sigmoids
 * possibly) AND competition per element as encoded in comp4.
 */

#include "rd_james.h"

template <class Flt>
class RD_James_comp10 : public RD_James<Flt>
{
public:

    /*!
     * The power to which a_j is raised for the inter-TC axon
     * competition term.
     */
    alignas(Flt) Flt l = 3.0;

    /*!
     * An accumulation parameter.
     */
    alignas(Flt) Flt y = 0.0;

    /*!
     * Parameters for the sigmoid transfer function for a_i
     */
    //@{
    //! height parameter for sigmoid
    alignas(Flt) Flt sig_h = 2.0;
    //! offset
    alignas(Flt) Flt sig_o = 5.0;
    //! sharpness
    alignas(Flt) Flt sig_s = 0.5;
    //@}

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

    //! overall 'gain' of the normalization equation
    alignas(Flt) Flt eta = 100.0;
    //! offset of the normalization equation
    alignas(Flt) Flt xi = 1.0;
    //! power value of the normalization equation
    alignas(Flt) Flt q = 2.0;
    //! eta to the power of q
    alignas(Flt) Flt eta_to_q = 1.0;

    /*!
     * Response of a after being passed through the sigmoid transfer
     * function.
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > a_res;

    /*!
     * To record dci/dt, as this is used in the computation of a
     */
    alignas(alignof(vector<vector<Flt> >))
    vector<vector<Flt> > dc;

    /*!
     * Simple constructor; no arguments. Just calls base constructor
     */
    RD_James_comp10 (void)
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
        this->resize_vector_param (this->epsilon, this->N);
        this->resize_vector_vector (this->a_eps, this->N);
        this->resize_vector_vector (this->a_res, this->N);
        this->resize_vector_vector (this->dc, this->N);

    }
    virtual void init (void) {
        RD_James<Flt>::init();
    }
    //@}

    /*!
     * Computation methods
     */
    //@{

    void integrate_c (void) {
        // 3. Do integration of c
        for (unsigned int i=0; i<this->N; ++i) {

#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                // Note: betaterm used in compute_dci_dt()
                this->betaterm[i][h] = this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a_res[i][h], this->k));
            }

            // Runge-Kutta integration for C (or ci)
            vector<Flt> qq(this->nhex,0.);
            vector<Flt> k1 = this->compute_dci_dt (this->c[i], i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                qq[h] = this->c[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2 = this->compute_dci_dt (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                qq[h] = this->c[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3 = this->compute_dci_dt (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                qq[h] = this->c[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4 = this->compute_dci_dt (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                this->dc[i][h] = (k1[h]+2. * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
                Flt c_cand = this->c[i][h] + this->dc[i][h];
                // Avoid over-saturating c_i and make sure dc is similarly modified.
                this->dc[i][h] = (c_cand > 1.0) ? (1.0 - this->c[i][h]) : this->dc[i][h];
                this->c[i][h] =   (c_cand > 1.0) ? 1.0 : c_cand;
            }
        }
    }

    void integrate_a (void) {

        // 2. Do integration of a (RK in the 1D model). Involves computing axon branching flux.

        // Pre-compute:
        // 1) The intermediate val alpha_c.
        // 2) The sum of all a_i across ALL TC sheets.
        Flt total_sum_a = 0.0;
        for (unsigned int i=0; i<this->N; ++i) {
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
        Flt eta_to_q_plus_sum_a_to_q = this->eta_to_q + total_sum_a;

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
                    eps[h] += static_cast<Flt>(pow (this->a_res[j][h], this->l));
                }
            }

            // Multiply it by epsilon[i]/(N-1). Now it's ready to subtract from the solutions
            Flt eps_over_N = this->epsilon[i]/(this->N-1);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                eps[h] *= eps_over_N;
                // Store a[i][h] * eps[h] for analysis (it's also used 4 times below)
                //this->a_eps[i][h] = this->a_res[i][h] * eps[h];
            }

            // Runge-Kutta integration for A
            vector<Flt> qq(this->nhex, 0.0);
            this->compute_divJ (this->a_res[i], i); // populates divJ[i]

            vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] + this->y * this->a_res[i][h] * this->a_res[i][h] - this->dc[i][h] - this->a_res[i][h] * eps[h];
                qq[h] = this->a_res[i][h] + k1[h] * this->halfdt;
            }

            vector<Flt> k2(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] + this->y * qq[h] * qq[h] - this->dc[i][h] - qq[h] * eps[h];
                qq[h] = this->a[i][h] + k2[h] * this->halfdt;
            }

            vector<Flt> k3(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h]  + this->y * qq[h] * qq[h] - this->dc[i][h] - qq[h] * eps[h];
                qq[h] = this->a[i][h] + k3[h] * this->dt;
            }

            vector<Flt> k4(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h]  + this->y * qq[h] * qq[h] - this->dc[i][h] - qq[h] * eps[h];
                this->a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
                // Sigmoid transfer function
                this->a_res[i][h] =  sig_h / (1 - exp (sig_o - sig_s * this->a[i][h]));
            }
            if (this->stepCount % 100 == 0) {
                cout << "step " << this->stepCount << ": a_i500 = " << this->a[i][500] << ", a_res_i500 = " << this->a_res[i][500] << endl;
            }
        }
    }

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

        // 2. Call Runge Kutta numerical integration code
        this->integrate_a();
        this->integrate_c();
    }


    /*!
     * Override save to additionally save a_res.
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
            // The a_res variable
            path.str("");
            path.clear();
            path << "/a_res_" << i;
            data.add_contained_vals (path.str().c_str(), this->a_res[i]);

        }
        data.add_contained_vals ("/n", this->n);
    }

}; // RD_James
