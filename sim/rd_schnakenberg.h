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
     * Schnakenberg
     * F = k1 - k2 A + k3 A^2 B
     * G = k4        - k3 A^2 B
     */
    alignas(Flt) Flt k1 = 1.0;
    alignas(Flt) Flt k2 = 1.0;
    alignas(Flt) Flt k3 = 1.0;
    alignas(Flt) Flt k4 = 1.0;

    /*!
     * The diffusion parameters.
     */
    //@{
    alignas(Flt) Flt D_A = 0.1;
    alignas(Flt) Flt D_B = 0.1;
    //@}

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
        {
            // Runge-Kutta integration for A. This time, I'm taking
            // ownership of this code and properly understanding it.

            // Amp: "A at a midpoint"
            vector<Flt> Amp(this->nhex, 0.0);
            // The function F from Schnakenberg
            vector<Flt> F(this->nhex, 0.0);

            // f(A,B) = F(A,B) + lapA
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                F[h] = this->k1 - (this->k2 * this->A[h]) + (this->k3 * this->A[h] * this->A[h] * this->B[h]);
            }
            this->compute_laplace (A, lapA); // From base class

            vector<Flt> K1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                K1[h] = (F[h] + this->lapA[h]) * dt;
                // Amp allows us to compute new Laplacian for subsequent Ks
                Amp[h] = this->A[h] + K1[h] * 0.5 ;// * halfdt;
            }

            // f(Amp,B) = F(Amp,B) + lapAmp
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                F[h] = this->k1 - (this->k2 * Amp[h]) + (this->k3 * Amp[h] * Amp[h] * this->B[h]);
            }
            this->compute_laplace (Amp, lapA);

            vector<Flt> K2(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                K2[h] = SOMETHINGUNKNOWN * dt;
                q[h] = this->A[h] + K1[h] * halfdt;
            }

            vector<Flt> K2(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                K2[h] = this->lapA[h] + F[h];
                q[h] = this->A[h] + K2[h] * dt;
            }

            vector<Flt> K3(this->nhex, 0.0);
            this->compute_divJ (q, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                K3[h] = this->lapA[h] + this->alpha_c_beta_nA[h];
                A[h] += (K0[h] + 2.0 * (K1[h] + K2[h]) + K3[h]) * sixthdt;
            }
        }

        // 3. Do integration of B
        {
            // Addme
        }
    }

}; // RD_Schnakenberg
