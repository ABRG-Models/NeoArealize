/*
 * 2D Karbowski system with *sigmoidal* normalization of a_i, deriving
 * from RD_James_norm base class.
 */

#include "rd_james_norm.h"

template <class Flt>
class RD_James_comp10 : public RD_James_norm<Flt>
{
public:
    //! Sigmoid parameters
    //@{
    //! offset
    alignas(Flt) Flt o = 5.0;
    //! sharpness
    alignas(Flt) Flt s = 0.5;
    //@}

    //! Constructor boilerplate
    RD_James_comp10 (void)
        : RD_James_norm<Flt>() {
    }

    //! The normalization/transfer function. Maybe unroll this one?
    virtual inline Flt transfer_a (const Flt& _a, const unsigned int _i) {
        //Flt a_after = (1.0 / (1.0 + exp(this->o - this->s * _a)));
        //cout << "before: " << _a << ", after: " << a_after << endl;
        Flt a_after = (_a < 2.0 ? _a : 2.0);
        a_after = (a_after >= 0.0 ? _a : 0.0);
        return a_after;
    }

}; // RD_James_norm
