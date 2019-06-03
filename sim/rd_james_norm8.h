/*
 * 2D Karbowski system with *divisive* normalization of a_i, deriving
 * from RD_James_norm base class.
 */

#include "rd_james_norm.h"

template <class Flt>
class RD_James_norm8 : public RD_James_norm<Flt>
{
public:

    /*!
     * Members used in divisive normalization
     */
    //@{
    //! overall 'gain' of the normalization equation
    alignas(Flt) Flt eta = 100.0;
    //! offset of the normalization equation
    alignas(Flt) Flt xi = 1.0;
    //! power value of the normalization equation
    alignas(Flt) Flt q = 2.0;
    //! eta to the power of q
    alignas(Flt) Flt eta_to_q = 1.0;
    //! eta to the power of q plus the sum of (a to the power of q)
    alignas(vector<Flt>) vector<Flt> eta_to_q_plus_sum_a_to_q;
    //@}

    /*!
     * Holds the total sum of all a_i^q.
     */
    //alignas(Flt) Flt total_sum_a_to_q;

    /*!
     * An N element vector holding the sum of a_i^q for each TC type.
     */
    alignas(vector<Flt>) vector<Flt> sum_a_to_q;

    /*!
     * Simple constructor; no arguments. Just calls base constructor.
     */
    RD_James_norm8 (void)
        : RD_James_norm<Flt>() {
    }

    virtual void allocate (void) {
        RD_James_norm<Flt>::allocate();
        this->resize_vector_param (this->eta_to_q_plus_sum_a_to_q, this->N);
        this->resize_vector_param (this->sum_a_to_q, this->N);
    }

    virtual void init (void) {
        RD_James_norm<Flt>::init();
        // Don't really need to do this every step, but it'll take negligible time
        this->eta_to_q = pow (this->eta, this->q);
    }

    /*!
     * Computation methods
     */
    //@{

    /*!
     * A possibly normalization-function specific task to carry out
     * once after the sum of a has been computed.
     */
    virtual void sum_a_computation (void) {

        // 2) The sum of all a_i across ALL TC sheets.
        //this->total_sum_a_to_q = 0.0;
        for (unsigned int i=0; i<this->N; ++i) {
            // Compute the sum of a[i] across the sheet.
            this->sum_a_to_q[i] = 0.0;
#pragma omp parallel for //reduction(+:sum_a_to_q)
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->sum_a_to_q[i] += pow (this->a[i][h], this->q);
            }
            //this->total_sum_a_to_q += this->sum_a_to_q[i];
            this->eta_to_q_plus_sum_a_to_q[i] = this->eta_to_q + this->sum_a_to_q[i];
        }
    }

    /*!
     * The normalization/transfer function.
     */
    virtual Flt transfer_a (const Flt& _a, const unsigned int _i) {
        Flt a_rtn = 0.0;
        // Prevent a from becoming negative, necessary only when competition is implemented:
        a_rtn = (_a < 0.0) ? 0.0 : _a;
        // Divisive normalization step (across all TC types)
        a_rtn = this->eta * (pow (a_rtn, this->q) / (this->eta_to_q_plus_sum_a_to_q[_i]));
        return a_rtn;
    }

}; // RD_James_norm
