#ifndef _HEX_H_
#define _HEX_H_

#include <string>
#include <list>
#include <utility>
#include <cmath>
#include "morph/BezCoord.h"

using std::string;
using std::to_string;
using std::list;
using std::abs;
using std::sqrt;
using std::pair;

#define DEBUG_WITH_COUT 1
#ifdef DEBUG_WITH_COUT
#include <iostream>
using std::cout;
using std::endl;
#endif

namespace morph {

    const float SQRT_OF_3_F = 1.73205081;
    /*!
     * Describes a regular hexagon arranged with vertices pointing
     * vertically and two flat sides perpendicular to the horizontal
     * axis:
     *
     *            *
     *         *     *
     *         *     *
     *            *
     *
     * The centre of the hex in a Cartesian right hand coordinate
     * system is represented with x, y and z:
     *
     *  y
     *  ^
     *  |
     *  |
     *  0-----> x     z out of screen/page
     *
     * Directions are "r" "g" and "b" and their negatives:
     *
     *         b  * g
     * -r <--  *     * ---> r
     *         *     *
     *         -g * -b
     *
     */
    class Hex
    {
    public:
        /*!
         * Constructor taking index, dimension and integer position
         * indices. Computes Cartesian location from these.
         */
        Hex (const unsigned int& idx, const float& d_,
             const int& r_, const int& g_) {
            this->vi = idx;
            this->d = d_;
            this->ri = r_;
            this->gi = g_;
            this->computeCartesian();
        }

        /*!
         * Produce a string containing information about this hex,
         * showing grid location in dimensionless r,g (but not b)
         * units. Also show nearest neighbours.
         */
        string output (void) const {
            string s("Hex ");
            s += to_string(this->vi).substr(0,2) + " (";
            s += to_string(this->ri).substr(0,4) + ",";
            s += to_string(this->gi).substr(0,4) + "). ";

            if (this->has_ne) {
                s += "E: (" + to_string(this->ne->ri).substr(0,4) + "," + to_string(this->ne->gi).substr(0,4) + ") ";
            }
            if (this->has_nse) {
                s += "SE: (" + to_string(this->nse->ri).substr(0,4) + "," + to_string(this->nse->gi).substr(0,4) + ") ";
            }
            if (this->has_nsw) {
                s += "SW: (" + to_string(this->nsw->ri).substr(0,4) + "," + to_string(this->nsw->gi).substr(0,4) + ") ";
            }
            if (this->has_nw) {
                s += "W: (" + to_string(this->nw->ri).substr(0,4) + "," + to_string(this->nw->gi).substr(0,4) + ") ";
            }
            if (this->has_nnw) {
                s += "NW: (" + to_string(this->nnw->ri).substr(0,4) + "," + to_string(this->nnw->gi).substr(0,4) + ") ";
            }
            if (this->has_nne) {
                s += "NE: (" + to_string(this->nne->ri).substr(0,4) + "," + to_string(this->nne->gi).substr(0,4) + ") ";
            }

            return s;
        }

        /*!
         * Produce a string containing information about this hex,
         * focussing on Cartesian position information.
         */
        string outputCart (void) const {
            string s("Hex ");
            s += to_string(this->vi).substr(0,2) + " (";
            s += to_string(this->ri).substr(0,4) + ",";
            s += to_string(this->gi).substr(0,4) + ") is at (x,y) = ("
                + to_string(this->x).substr(0,4) +"," + to_string(this->y).substr(0,4) + ")";
            return s;
        }

        /*!
         * Convert ri, gi and bi indices into x and y coordinates
         * based on the hex-to-hex distance d.
         */
        void computeCartesian (void) {
            this->x = this->d*this->ri + (d/2.0f)*this->gi - (d/2.0f)*this->bi;
            float dv = (this->d*morph::SQRT_OF_3_F)/2.0f;
            this->y = dv*this->gi + dv*this->bi;
        }

        float distanceFrom (const pair<float, float> cartesianPoint) const {
            float dx = abs(cartesianPoint.first - x);
            float dy = abs(cartesianPoint.second - y);
            float d = sqrt (dx*dx + dy*dy);
            //cout << "distance: " << d << endl;
            return d;
        }

        float distanceFrom (const BezCoord& cartesianPoint) const {
            float dx = abs(cartesianPoint.x() - x);
            float dy = abs(cartesianPoint.y() - y);
            float d = sqrt (dx*dx + dy*dy);
            return d;
        }

        /*!
         * Vector index. This is the index into those data vectors
         * which hold the relevant data pertaining to this hex. This
         * is a scheme which allows me to keep the data in separate
         * vectors and all the hex position information in this class.
         * What happens when I delete some hex elements?  Simple - I
         * can re-set the vi indices after creating a grid of Hex
         * elements and then pruning down.
         */
        unsigned int vi;

        /*!
         * Cartesian coordinates of the centre of the Hex.
         */
        //@{
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        //@}

        /*!
         * The centre-to-centre distance from one Hex to an
         * immediately adjacent Hex.
         */
        float d = 1.0f;

        /*!
         * The distance from the centre of the Hex to any of the
         * vertices.
         */
        float getRv (void) {
            float rv = this->d/morph::SQRT_OF_3_F;
            return rv;
        }

        /*!
         * The vertical distance between hex centres on adjacent rows.
         */
        float getDv (void) {
            float dv = (this->d*morph::SQRT_OF_3_F)/2.0f;
            return dv;
        }

        /*!
         * Indices in hex directions. These lie in the x-y
         * plane. They index in positive and negative directions,
         * starting from the Hex at (0,0,z)
         */
        //@{
        /*!
         * Index in r direction - positive "East", that is in the +x
         * direction.
         */
        int ri = 0;
        /*!
         * Index in g direction - positive "NorthEast". In a direction
         * 30 degrees East of North or 60 degrees North of East.
         */
        int gi = 0;
        /*!
         * Index in b direction - positive "NorthEast". In a direction
         * 30 degrees East of North or 60 degrees North of East.
         */
        int bi = 0;
        //@}

        /*!
         * Set to true if this Hex has been marked as being on a
         * boundary. It is expected that client code will then re-set
         * the neighbour relations so that onBoundary() would return
         * true.
         */
        bool boundaryHex = false;

        /*!
         * Set true if this Hex is known to be inside the boundary.
         */
        bool insideBoundary = false;

        /*!
         * Return true if this is a boundary hex - one on the outside
         * edge of a hex grid.
         */
        bool onBoundary() {
            if (this->has_ne == false
                || this->has_nne == false
                || this->has_nnw == false
                || this->has_nw == false
                || this->has_nsw == false
                || this->has_nse == false) {
                return true;
            }
            return false;
        }

        /*!
         * Setters for neighbour iterators
         */
        //@{
        void set_ne (list<Hex>::iterator it) {
            this->ne = it;
            this->has_ne = true;
        }
        void set_nne (list<Hex>::iterator it) {
            this->nne = it;
            this->has_nne = true;
        }
        void set_nnw (list<Hex>::iterator it) {
            this->nnw = it;
            this->has_nnw = true;
        }
        void set_nw (list<Hex>::iterator it) {
            this->nw = it;
            this->has_nw = true;
        }
        void set_nsw (list<Hex>::iterator it) {
            this->nsw = it;
            this->has_nsw = true;
        }
        void set_nse (list<Hex>::iterator it) {
            this->nse = it;
            this->has_nse = true;
        }
        //@}

        /*!
         * Un-set neighbour iterators
         */
        //@{
        void unset_ne (void) {
            this->has_ne = false;
        }
        void unset_nne (void) {
            this->has_nne = false;
        }
        void unset_nnw (void) {
            this->has_nnw = false;
        }
        void unset_nw (void) {
            this->has_nw = false;
        }
        void unset_nsw (void) {
            this->has_nsw = false;
        }
        void unset_nse (void) {
            this->has_nse = false;
        }
        //@}

        //private:??
        /*!
         * Nearest neighbours
         */
        //@{
        /*!
         * Nearest neighbour to the East; in the plus r direction.
         */
        list<Hex>::iterator ne;
        /*!
         * Set true when ne has been set. Use of iterators rather than
         * pointers means we can't do any kind of check to see if the
         * iterator is valid, so we have to keep a separate boolean
         * value.
         *
         * FIXME: Use std::bitset for these boolean switches?
         */
        bool has_ne = false;

        /*!
         * Nearest neighbour to the NorthEast; in the plus g
         * direction.
         */
        //@{
        list<Hex>::iterator nne;
        bool has_nne = false;
        //@}

        /*!
         * Nearest neighbour to the NorthWest; in the plus b
         * direction.
         */
        //@{
        list<Hex>::iterator nnw;
        bool has_nnw = false;
        //@}

        /*!
         * Nearest neighbour to the West; in the minus r direction.
         */
        //@{
        list<Hex>::iterator nw;
        bool has_nw = false;
        //@}

        /*!
         * Nearest neighbour to the SouthWest; in the minus g
         * direction.
         */
        //@{
        list<Hex>::iterator nsw;
        bool has_nsw = false;
        //@}

        /*!
         * Nearest neighbour to the SouthEast; in the minus b
         * direction.
         */
        //@{
        list<Hex>::iterator nse;
        bool has_nse = false;
        //@}

        //@}
    };

} // namespace morph

#endif // _HEX_H_
