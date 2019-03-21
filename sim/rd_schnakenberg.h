#include "rd_base.h"

/*!
 * Two component Schnakenberg Reaction Diffusion system
 */
template <class Flt>
class RD_Schnakenberg : public RD_Base
{
public:
    /*!
     * Reactant A
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> A;

    /*!
     * Reactant B
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> B;

    /*!
     * J(x,t) - the "flux current". This is a vector field. May need J_A and J_B.
     */
    alignas(alignof(array<vector<Flt>, 2>))
    array<vector<Flt>, 2> J;

    /*!
     * Holds the divergence of the J_i(x)s
     */
    //alignas(alignof(vector<vector<Flt> >))
    //vector<vector<Flt> > divJ;

    /*!
     * The power to which a_i(x,t) is raised in Eqs 1 and 2 in the
     * paper.
     */
    alignas(Flt) Flt k = 3.0;

    /*!
     * The diffusion parameters.
     */
    //@{
    alignas(Flt) Flt D_A = 0.1;
    alignas(Flt) Flt D_B = 0.1;
    //@}

    /*!
     * k parameters
     */
    alignas(alignof(vector<Flt>))
    vector<Flt> k;

    /*!
     * Simple constructor; no arguments.
     */
    RD_Schnakenberg (void) :
        RD_Base() {
        DBG("RD_Schakenberg constructor")
    }

    /*!
     * Destructor required to free up HexGrid memory
     */
    ~RD_Schnakenbert (void) :
        ~RD_Base() {
        DBG("RD_Schakenberg deconstructor")
    }

    /*!
     * Perform memory allocations, vector resizes and so on.
     */
    void allocate (void) {
        // Always call allocate() from the base class first.
        RD_Base::allocate();
        // Resize and zero-initialise the various containers
        this->resize_vector (this->A);
        this->resize_vector (this->B);
        this->resize_vector_param (this->k);
    }

    /*!
     * Initialise variables and parameters. Carry out one-time
     * computations required of the model.
     */
    void init (void) {
        // Initialise A, B with noise
        this->noiseify_vector (this->A);
        this->noiseify_vector (this->B);
    }


    /*!
     * HDF5 file saving/loading methods.
     */
    //@{

    /*!
     * Save the variables.
     */
    void save (void) {
        stringstream fname;
        fname << this->logpath << "/dat_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        HdfData data(fname.str());
        stringstream path;
        // The A variables
        path << "/A" << i;
        data.add_contained_vals (path.str().c_str(), this->A);
        // The B variable
        path.str("");
        path.clear();
        path << "/B" << i;
        data.add_contained_vals (path.str().c_str(), this->B);

        //data.add_contained_vals ("/n", this->n);
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

        // 2. Do integration of A

        // Runge-Kutta integration for A
        vector<Flt> q(this->nhex, 0.0);
        this->compute_divJ (A, i); // populates divJ[i]

        vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            k1[h] = this->divJ[i][h] + this->alpha_c_beta_nA[h]; // or whatever for Schakenberg
            q[h] = this->A[h] + k1[h] * halfdt;
        }

        vector<Flt> k2(this->nhex, 0.0);
        this->compute_divJ (q, i);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            k2[h] = this->divJ[i][h] + this->alpha_c_beta_nA[h];
            q[h] = this->A[h] + k2[h] * halfdt;
        }

        vector<Flt> k3(this->nhex, 0.0);
        this->compute_divJ (q, i);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            k3[h] = this->divJ[i][h] + this->alpha_c_beta_nA[h];
            q[h] = this->A[h] + k3[h] * dt;
        }

        vector<Flt> k4(this->nhex, 0.0);
        this->compute_divJ (q, i);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            k4[h] = this->divJ[i][h] + this->alpha_c_beta_nA[h];
            A[h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * sixthdt;
        }

        // 3. Do integration of B
        // Runge-Kutta integration for B
        vector<Flt> q(nhex,0.);
        vector<Flt> k1 = compute_dci_dt (B, i);
#pragma omp parallel for
        for (unsigned int h=0; h<nhex; h++) {
            q[h] = B[h] + k1[h] * halfdt;
        }

        vector<Flt> k2 = compute_dci_dt (q, i);
#pragma omp parallel for
        for (unsigned int h=0; h<nhex; h++) {
            q[h] = B[h] + k2[h] * halfdt;
        }

        vector<Flt> k3 = compute_dci_dt (q, i);
#pragma omp parallel for
        for (unsigned int h=0; h<nhex; h++) {
            q[h] = B[h] + k3[h] * dt;
        }

        vector<Flt> k4 = compute_dci_dt (q, i);
#pragma omp parallel for
        for (unsigned int h=0; h<nhex; h++) {
            B[h] += (k1[h]+2. * (k2[h] + k3[h]) + k4[h]) * sixthdt;
        }
    }

}; // RD_Schnakenberg
