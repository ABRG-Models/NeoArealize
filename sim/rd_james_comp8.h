/*
 * 2D Karbowski system with *divisive* normalization of a_i, deriving
 * from RD_James_norm base class.
 */

#include "rd_james_norm.h"

template <class Flt>
class RD_James_comp8 : public RD_James_norm<Flt>
{
public:

    /*!
     * An N element vector holding the sum of a_i for each TC type.
     */
    alignas(vector<Flt>) vector<Flt> sum_a;

    /*!
     * Simple constructor; no arguments. Just calls base constructor.
     */
    RD_James_comp8 (void)
        : RD_James_norm<Flt>() {
    }

    virtual void allocate (void) {
        RD_James_norm<Flt>::allocate();
        this->resize_vector_param (this->sum_a, this->N);
    }

    virtual void init (void) {
        RD_James_norm<Flt>::init();
    }

    /*!
     * Computation methods
     */
    //@{

    /*!
     * A possibly normalization-function specific task to carry out
     * once after the sum of a has been computed.
     */
    virtual void sum_a_computation (const unsigned int _i) {
        // Compute the sum of a[i] across the sheet.
        this->sum_a[_i] = 0.0;
        Flt sum_tmp = 0.0;
#pragma omp parallel for reduction(+:sum_tmp)
        for (unsigned int h=0; h<this->nhex; ++h) {
            sum_tmp += this->a[_i][h];
        }
        this->sum_a[_i] = sum_tmp;
    }

    /*!
     * The normalization/transfer function.
     */
    virtual Flt transfer_a (const Flt& _a, const unsigned int _i) {
        // Divisive normalization step
        Flt a_rtn = this->nhex * _a / this->sum_a[_i];
        // Prevent a from becoming negative, necessary only when competition is implemented:
        return (a_rtn < 0.0) ? 0.0 : a_rtn;
    }

}; // RD_James_norm
