/*!
 * Implementation of HexGrid
 *
 * Author: Seb James
 *
 * Date: 2018/07
 */

#include "HexGrid.h"
#include <cmath>
#include <float.h>
#include <limits>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <morph/BezCurvePath.h>
#include <morph/BezCoord.h>

#define DBGSTREAM std::cout
#define DEBUG 1
//#define DEBUG2 1
#include "MorphDbg.h"

using std::ceil;
using std::abs;
using std::endl;
using std::stringstream;
using std::vector;
using std::set;
using std::runtime_error;
using std::numeric_limits;

using morph::BezCurvePath;
using morph::BezCoord;
using morph2::Hex;

morph2::HexGrid::HexGrid ()
    : d(1.0f)
    , x_span(1.0f)
    , z(0.0f)
{
}

morph2::HexGrid::HexGrid (float d_, float x_span_, float z_)
{
    this->d = d_;
    this->x_span = x_span_;
    this->z = z_;

    this->init();
}

pair<float, float>
morph2::HexGrid::computeCentroid (const list<Hex>& pHexes)
{
    pair<float, float> centroid;
    centroid.first = 0;
    centroid.second = 0;
    for (auto h : pHexes) {
        centroid.first += h.x;
        centroid.second += h.y;
    }
    centroid.first /= pHexes.size();
    centroid.second /= pHexes.size();
    return centroid;
}

void
morph2::HexGrid::setBoundary (const list<Hex>& pHexes)
{
    this->boundaryCentroid = this->computeCentroid (pHexes);

    list<Hex>::iterator bpoint = this->hexen.begin();
    list<Hex>::iterator bpi = this->hexen.begin();
    while (bpi != this->hexen.end()) {
        list<Hex>::const_iterator ppi = pHexes.begin();
        while (ppi != pHexes.end()) {
            // NB: The assumption right now is that the pHexes are
            // from the same dimension hex grid as this->hexen.
            if (bpi->ri == ppi->ri && bpi->gi == ppi->gi) {
                // Set h as boundary hex.
                bpi->boundaryHex = true;
                bpoint = bpi;
                break;
            }
            ++ppi;
        }
        ++bpi;
    }

    // Check that the boundary is contiguous.
    set<unsigned int> seen;
    list<Hex>::iterator hi = bpoint;
    if (this->boundaryContiguous (bpoint, hi, seen) == false) {
        stringstream ee;
        ee << "The boundary is not a contiguous sequence of hexes.";
        throw runtime_error (ee.str());
    }

#if 0 // Possibly reset the domain here.
    // Boundary IS contiguous, discard hexes outside the boundary.
    this->discardOutside();
#endif
}

#ifdef UNTESTED_UNUSED
void
morph2::HexGrid::offsetCentroid (void)
{
    for (auto h : this->hexen) {
        cout << " * : " << h.x << "," << h.y << endl;
    }
    for (auto h : this->hexen) {
        h.subtractLocation (this->boundaryCentroid);
        cout << "***: " << h.x << "," << h.y << endl;
    }
    this->boundaryCentroid = make_pair (0.0, 0.0);
}
#endif

//#define DONT_OFFSET_CENTROID_BEFORE_SETTING_BOUNDARY 1
void
morph2::HexGrid::setBoundary (const BezCurvePath& p)
{
    this->boundary = p;

    if (!this->boundary.isNull()) {
        DBG ("Applying boundary...");

        // Compute the points on the boundary using half of the hex to
        // hex spacing as the step size.
        vector<BezCoord> bpoints = this->boundary.getPoints (this->d/2.0f, true); // true to invert y axis

        this->boundaryCentroid = BezCurvePath::getCentroid (bpoints);
        DBG ("Boundary centroid: " << boundaryCentroid.first << "," << boundaryCentroid.second);

#ifdef DONT_OFFSET_CENTROID_BEFORE_SETTING_BOUNDARY
        list<Hex>::iterator nearbyBoundaryPoint = this->findHexNearest (this->boundaryCentroid);
        DBG ("Hex near boundary centroid at x,y: " << nearbyBoundaryPoint->x << "," << nearbyBoundaryPoint->y);
        auto bpi = bpoints.begin();
#else // Offset BezCoords of the boundary BezCurvePath by its centroid, to make the centroid 0,0.
        auto bpi = bpoints.begin();
        while (bpi != bpoints.end()) {
            bpi->subtract (this->boundaryCentroid);
            ++bpi;
        }
        this->boundaryCentroid = make_pair (0.0, 0.0);
        list<Hex>::iterator nearbyBoundaryPoint = this->hexen.begin(); // i.e the Hex at 0,0
        bpi = bpoints.begin();
#endif
        while (bpi != bpoints.end()) {
            nearbyBoundaryPoint = this->setBoundary (*bpi++, nearbyBoundaryPoint);
            DBG2 ("Added boundary point " << nearbyBoundaryPoint->ri << "," << nearbyBoundaryPoint->gi);
        }

        // Check that the boundary is contiguous.
        set<unsigned int> seen;
        list<Hex>::iterator hi = nearbyBoundaryPoint;
        if (this->boundaryContiguous (nearbyBoundaryPoint, hi, seen) == false) {
            stringstream ee;
            ee << "The boundary which was constructed from "
               << p.name << " is not a contiguous sequence of hexes.";
            throw runtime_error (ee.str());
        }

        // Given that the boundary IS contiguous, can now set a domain
        // of hexes (rectangular region, such that computations can be
        // efficient) and discard hexes outside the domain.
        // setDomain() will define a regular domain, then discard
        // those hexes outside the regular domain and populate all the
        // d_ vectors.
        this->setDomain();
    }
}

list<Hex>::iterator
morph2::HexGrid::setBoundary (const BezCoord& point, list<Hex>::iterator startFrom)
{
    // Searching from "startFrom", search out, via neighbours until
    // the hex closest to the boundary point is located. How to know
    // if it's closest? When all neighbours are further from the
    // currently closest point?

    bool neighbourNearer = true;

    list<Hex>::iterator h = startFrom;
    float d = h->distanceFrom (point);
    float d_ = 0.0f;

    while (neighbourNearer == true) {

        neighbourNearer = false;
        if (h->has_ne && (d_ = h->ne->distanceFrom (point)) < d) {
            d = d_;
            h = h->ne;
            neighbourNearer = true;

        } else if (h->has_nne && (d_ = h->nne->distanceFrom (point)) < d) {
            d = d_;
            h = h->nne;
            neighbourNearer = true;

        } else if (h->has_nnw && (d_ = h->nnw->distanceFrom (point)) < d) {
            d = d_;
            h = h->nnw;
            neighbourNearer = true;

        } else if (h->has_nw && (d_ = h->nw->distanceFrom (point)) < d) {
            d = d_;
            h = h->nw;
            neighbourNearer = true;

        } else if (h->has_nsw && (d_ = h->nsw->distanceFrom (point)) < d) {
            d = d_;
            h = h->nsw;
            neighbourNearer = true;

        } else if (h->has_nse && (d_ = h->nse->distanceFrom (point)) < d) {
            d = d_;
            h = h->nse;
            neighbourNearer = true;
        }
    }

    DBG2 ("Nearest hex to point (" << point.x() << "," << point.y() << ") is at (" << h->ri << "," << h->gi << ")");

    // Mark it for being on the boundary
    h->boundaryHex = true;

    return h;
}

bool
morph2::HexGrid::findBoundaryHex (list<Hex>::const_iterator& hi) const
{
    DBG ("Testing Hex ri,gi = " << hi->ri << "," << hi->gi << " x,y = " << hi->x << "," << hi->y);
    if (hi->boundaryHex == true) {
        // No need to change the Hex iterator
        return true;
    }

    if (hi->has_ne) {
        list<Hex>::const_iterator ci(hi->ne);
        if (this->findBoundaryHex (ci) == true) {
            hi = ci;
            return true;
        }
    }
    if (hi->has_nne) {
        list<Hex>::const_iterator ci(hi->nne);
        if (this->findBoundaryHex (ci) == true) {
            hi = ci;
            return true;
        }
    }
    if (hi->has_nnw) {
        list<Hex>::const_iterator ci(hi->nnw);
        if (this->findBoundaryHex (ci) == true) {
            hi = ci;
            return true;
        }
    }
    if (hi->has_nw) {
        list<Hex>::const_iterator ci(hi->nw);
        if (this->findBoundaryHex (ci) == true) {
            hi = ci;
            return true;
        }
    }
    if (hi->has_nsw) {
        list<Hex>::const_iterator ci(hi->nsw);
        if (this->findBoundaryHex (ci) == true) {
            hi = ci;
            return true;
        }
    }
    if (hi->has_nse) {
        list<Hex>::const_iterator ci(hi->nse);
        if (this->findBoundaryHex (ci) == true) {
            hi = ci;
            return true;
        }
    }

    return false;
}

bool
morph2::HexGrid::boundaryContiguous (void) const
{
    list<Hex>::const_iterator bhi = this->hexen.begin();
    if (this->findBoundaryHex (bhi) == false) {
        // Found no boundary hex
        return false;
    }
    set<unsigned int> seen;
    list<Hex>::const_iterator hi = bhi;
    return this->boundaryContiguous (bhi, hi, seen);
}

bool
morph2::HexGrid::boundaryContiguous (list<Hex>::const_iterator bhi, list<Hex>::const_iterator hi, set<unsigned int>& seen) const
{
    DBG2 ("Called for hi=" << hi->vi);
    bool rtn = false;
    list<Hex>::const_iterator hi_next;

    DBG2 ("Inserting " << hi->vi << " into seen which is Hex ("<< hi->ri << "," << hi->gi<<")");
    seen.insert (hi->vi);

    DBG2 (hi->output());

    if (rtn == false && hi->has_ne && hi->ne->boundaryHex == true && seen.find(hi->ne->vi) == seen.end()) {
        hi_next = hi->ne;
        rtn = (this->boundaryContiguous (bhi, hi_next, seen));
    }
    if (rtn == false && hi->has_nne && hi->nne->boundaryHex == true && seen.find(hi->nne->vi) == seen.end()) {
        hi_next = hi->nne;
        rtn = this->boundaryContiguous (bhi, hi_next, seen);
    }
    if (rtn == false && hi->has_nnw && hi->nnw->boundaryHex == true && seen.find(hi->nnw->vi) == seen.end()) {
        hi_next = hi->nnw;
        rtn =  (this->boundaryContiguous (bhi, hi_next, seen));
    }
    if (rtn == false && hi->has_nw && hi->nw->boundaryHex == true && seen.find(hi->nw->vi) == seen.end()) {
        hi_next = hi->nw;
        rtn =  (this->boundaryContiguous (bhi, hi_next, seen));
    }
    if (rtn == false && hi->has_nsw && hi->nsw->boundaryHex == true && seen.find(hi->nsw->vi) == seen.end()) {
        hi_next = hi->nsw;
        rtn =  (this->boundaryContiguous (bhi, hi_next, seen));
    }
    if (rtn == false && hi->has_nse && hi->nse->boundaryHex == true && seen.find(hi->nse->vi) == seen.end()) {
        hi_next = hi->nse;
        rtn =  (this->boundaryContiguous (bhi, hi_next, seen));
    }

    if (rtn == false) {
        // Checked all neighbours
        if (hi == bhi) {
            DBG2 ("Back at start, nowhere left to go! return true.");
            rtn = true;
        } else {
            DBG2 ("Back at hi=(" << hi->ri << "," << hi->gi << "), while bhi=(" <<  bhi->ri << "," << bhi->gi << "), return false");
            rtn = false;
        }
    }

    DBG2 ("Boundary " << (rtn ? "IS" : "isn't") << " contiguous so far...");

    return rtn;
}

void
morph2::HexGrid::markHexesInside (list<Hex>::iterator hi)
{
    if (hi->boundaryHex == true) {
        hi->insideBoundary = true;
        return;

    } else if (hi->insideBoundary == true) {
        return;

    } else {

        hi->insideBoundary = true;

        if (hi->has_ne) {
            this->markHexesInside (hi->ne);
        }
        if (hi->has_nne) {
            this->markHexesInside (hi->nne);
        }
        if (hi->has_nnw) {
            this->markHexesInside (hi->nnw);
        }
        if (hi->has_nw) {
            this->markHexesInside (hi->nw);
        }
        if (hi->has_nsw) {
            this->markHexesInside (hi->nsw);
        }
        if (hi->has_nse) {
            this->markHexesInside (hi->nse);
        }
    }
}

void
morph2::HexGrid::markHexesInsideDomain (const array<int, 4>& extnts)
{
    // Check ri,gi,bi and reduce to equivalent ri,gi,bi=0.
    // Use gi to determine whether outside top/bottom region
    // Add gi contribution to ri to determine whether outside left/right region
    auto hi = this->hexen.begin();
    while (hi != this->hexen.end()) {
        float hz = hi->ri + 0.5*(hi->gi) + (hi->gi%2 ? 0.5f : 0.0f);
        if (hz < extnts[0]) {
            // outside
            DBG2 ("Outside. Horz idx: " << hz << " < extnts[0]: " << extnts[0]);
        } else if (hz > extnts[1]) {
            // outside
            DBG2 ("Outside. Horz idx: " << hz << " > extnts[1]: " << extnts[1]);
        } else if (hi->gi < extnts[2]) {
            // outside
            DBG2 ("Outside. Vert idx: " << hi->gi << " < extnts[2]: " << extnts[2]);
        } else if (hi->gi > extnts[3]) {
            // outside
            DBG2 ("Outside. Vert idx: " << hi->gi << " > extnts[3]: " << extnts[3]);
        } else {
            // inside
            DBG2 ("INSIDE. Horz,vert index: " << hz << "," << hi->gi << ". hi->bi should be 0: " << h->bi);
            hi->insideDomain = true;
        }
        ++hi;
    }
}

void
morph2::HexGrid::computeDistanceToBoundary (void)
{
    list<Hex>::iterator h = this->hexen.begin();
    while (h != this->hexen.end()) {
        if (h->boundaryHex == true) {
            h->distToBoundary = 0.0f;
        } else {
            // Not a boundary hex
            list<Hex>::iterator bh = this->hexen.begin();
            while (bh != this->hexen.end()) {
                if (bh->boundaryHex == true) {
                    float delta = h->distanceFrom (*bh);
                    if (delta < h->distToBoundary || h->distToBoundary < 0.0f) {
                        h->distToBoundary = delta;
                    }
                }
                ++bh;
            }
        }
        DBG2 ("Hex: " << h->vi <<"  d to bndry: " << h->distToBoundary
              << " on bndry? " << (h->boundaryHex?"Y":"N"));
        ++h;
    }
}

array<int, 4>
morph2::HexGrid::findBoundaryExtents (void)
{
    // Return object contains {ri-left, ri-right, gi-bottom, gi-top}
    // i.e. {xmin, xmax, ymin, ymax}
    array<int, 4> rtn = {{0,0,0,0}};

    // Find the furthest left and right hexes and the further up and down hexes.
    array<float, 4> limits = {{0,0,0,0}};
    bool first = true;
    for (auto h : this->hexen) {
        if (h.boundaryHex == true) {
            if (first) {
                limits = {{h.x, h.x, h.y, h.y}};
                first = false;
            }
            if (h.x < limits[0]) {
                limits[0] = h.x;
            }
            if (h.x > limits[1]) {
                limits[1] = h.x;
            }
            if (h.y < limits[2]) {
                limits[2] = h.y;
            }
            if (h.y > limits[3]) {
                limits[3] = h.y;
            }
        }
    }

    // Now compute the ri and gi values that these xmax/xmin/ymax/ymin
    // correspond to. THIS, if nothing else, should auto-vectorise!
    // d_ri is the distance moved in ri direction per x, d_gi is distance
    float d_ri = this->hexen.front().getD();
    float d_gi = this->hexen.front().getV();
    rtn[0] = (int)(limits[0] / d_ri);
    rtn[1] = (int)(limits[1] / d_ri);
    rtn[2] = (int)(limits[2] / d_gi);
    rtn[3] = (int)(limits[3] / d_gi);

    DBG ("ll,lr,lb,lt:     {" << limits[0] << ","  << limits[1] << ","  << limits[2] << ","  << limits[3] << "}");
    DBG ("d_ri: " << d_ri << ", d_gi: " << d_gi);
    DBG ("ril,rir,gib,git: {" << rtn[0] << ","  << rtn[1] << ","  << rtn[2] << ","  << rtn[3] << "}");

    // Add 'growth buffer'
    rtn[0] -= this->d_growthbuffer_horz;
    rtn[1] += this->d_growthbuffer_horz;
    rtn[2] -= this->d_growthbuffer_vert;
    rtn[3] += this->d_growthbuffer_vert;

    return rtn;
}

void
morph2::HexGrid::d_clear (void)
{
    this->d_x.clear();
    this->d_y.clear();
    this->d_ri.clear();
    this->d_gi.clear();
    this->d_bi.clear();
    this->d_flags.clear();
}

void
morph2::HexGrid::d_push_back (list<Hex>::iterator hi)
{
    d_x.push_back (hi->x);
    d_y.push_back (hi->y);
    d_ri.push_back (hi->ri);
    d_gi.push_back (hi->gi);
    d_bi.push_back (hi->bi);
    d_flags.push_back (hi->getFlags());
}

void
morph2::HexGrid::setDomain (void)
{
    // 1. Find extent of boundary, both left/right and up/down, with
    // 'buffer region' already added.
    array<int, 4> extnts = this->findBoundaryExtents();

    // 1.5 set rowlen and numrows
    this->d_rowlen = extnts[1]-extnts[0]+1;
    this->d_numrows = extnts[3]-extnts[2]+1;
    this->d_size = this->d_rowlen * this->d_numrows;

    // 2. Mark Hexes inside and outside the domain.
    // Mark those hexes inside the boundary
    this->markHexesInsideDomain (extnts);

    // 3. Discard hexes outside domain
    this->discardOutsideDomain();

    // 4. Populate d_ vectors
    // Use neighbour relations to go from bottom left to top right.
    // Find hex on bottom row.
    list<Hex>::iterator hi = this->hexen.begin(); // FIXME: better to reverse traverse?
    while (hi != this->hexen.end()) {
        if (hi->gi == extnts[2]) {
            // We're on the bottom row
            break;
        }
        ++hi;
    }
    DBG ("hi is on bottom row, posn xy:" << hi->x << "," << hi->y << " or rg:" << hi->ri << "," << hi->gi);
    while (hi->has_nw == true) {
        hi = hi->nw;
    }
    DBG ("hi is at bottom left posn xy:" << hi->x << "," << hi->y << " or rg:" << hi->ri << "," << hi->gi);

    // hi should now be the bottom left hex.
    list<Hex>::iterator blh = hi;

    // Sanity check
    if (blh->has_nne == false || blh->has_ne == false || blh->has_nnw == true) {
        throw runtime_error ("We expect the bottom left hex to have an east and a "
                             "north east neighbour, but no north west neighbour.");
    }

    // Now raster through the hexes, building the d_ vectors
    this->d_clear();

    bool next_row_ne = true;

    this->d_push_back (hi);

    do {
        hi = hi->ne;

        this->d_push_back (hi);

        DBG2 ("Pushed back flags: " << hi->getFlags() << " for r/g: " << hi->ri << "," << hi->gi);

        if (hi->has_ne == false) {
            if (hi->gi == extnts[3]) {
                // last (i.e. top) row and no neighbour east, so finished.
                DBG ("Fin. (top row)");
                break;
            } else {

                if (next_row_ne == true) {
                    hi = blh->nne;
                    next_row_ne = false;
                    blh = hi;
                } else {
                    hi = blh->nnw;
                    next_row_ne = true;
                    blh = hi;
                }

                this->d_push_back (hi);
            }
        }
    } while (hi->has_ne == true);

    DBG ("Size of d_x: " << this->d_x.size() << " and d_size=" << this->d_size);
}

void
morph2::HexGrid::discardOutsideDomain (void)
{
    // Run through and discard those hexes outside the boundary:
    auto hi = this->hexen.begin();
    while (hi != this->hexen.end()) {
        if (hi->insideDomain == false) {
            // When erasing a Hex, I need to update the neighbours of
            // its neighbours.
            hi->disconnectNeighbours();
            // Having disconnected the neighbours, erase the Hex.
            hi = this->hexen.erase (hi);
        } else {
            ++hi;
        }
    }
    DBG("Number of hexes in this->hexen is now: " << this->hexen.size());

    // The Hex::vi indices need to be re-numbered.
    this->renumberVectorIndices();

    // Finally, do something about the hexagonal grid vertices; set
    // this to true to mark that the iterators to the outermost
    // vertices are no longer valid and shouldn't be used.
    this->gridReduced = true;
}

void
morph2::HexGrid::discardOutsideBoundary (void)
{
    // Mark those hexes inside the boundary
    list<Hex>::iterator centroidHex = this->findHexNearest (this->boundaryCentroid);
    this->markHexesInside (centroidHex);

#ifdef DEBUG
    // Do a little count of them:
    unsigned int numInside = 0;
    unsigned int numOutside = 0;
    for (auto hi : this->hexen) {
        if (hi.insideBoundary == true) {
            ++numInside;
        } else {
            ++numOutside;
        }
    }
    DBG("Num inside: " << numInside << "; num outside: " << numOutside);
#endif

    // Run through and discard those hexes outside the boundary:
    auto hi = this->hexen.begin();
    while (hi != this->hexen.end()) {
        if (hi->insideBoundary == false) {
            // Here's the problem I think. When erasing a Hex, I need
            // to update the neighbours of its neighbours.
            hi->disconnectNeighbours();
            // Having disconnected the neighbours, erase the Hex.
            hi = this->hexen.erase (hi);
        } else {
            ++hi;
        }
    }
    DBG("Number of hexes in this->hexen is now: " << this->hexen.size());

    // The Hex::vi indices need to be re-numbered.
    this->renumberVectorIndices();

    // Finally, do something about the hexagonal grid vertices; set
    // this to true to mark that the iterators to the outermost
    // vertices are no longer valid and shouldn't be used.
    this->gridReduced = true;
}

list<Hex>::iterator
morph2::HexGrid::findHexNearest (const pair<float, float>& pos)
{
    list<Hex>::iterator nearest = this->hexen.end();
    list<Hex>::iterator hi = this->hexen.begin();
    float dist = FLT_MAX;
    while (hi != this->hexen.end()) {
        float dx = pos.first - hi->x;
        float dy = pos.second - hi->y;
        float dl = sqrt (dx*dx + dy*dy);
        if (dl < dist) {
            dist = dl;
            nearest = hi;
        }
        ++hi;
    }
    DBG("Nearest Hex to " << pos.first << "," << pos.second << " is (r,g):" << nearest->ri << "," << nearest->gi << " (x,y):" << nearest->x << "," << nearest->y);
    return nearest;
}

void
morph2::HexGrid::renumberVectorIndices (void)
{
    unsigned int vi = 0;
    this->vhexen.clear();
#if 0
    this->vi_self.clear();
#endif
    auto hi = this->hexen.begin();
    while (hi != this->hexen.end()) {
        hi->vi = vi++;
        this->vhexen.push_back (&(*hi));
#if 0
        this->vi_self.push_back (hi->vi);
#endif
        ++hi;
    }
#if 0
    // Now that all the vector indices for selves are set, iterate
    // through and set neighbours
    this->vi_ne.clear();
    this->vi_nne.clear();
    this->vi_nnw.clear();
    this->vi_nw.clear();
    this->vi_nsw.clear();
    this->vi_nse.clear();
    hi = this->hexen.begin();
    while (hi != this->hexen.end()) {
        if (hi->has_ne) {
            this->vi_ne.push_back (hi->ne->vi);
        }
        if (hi->has_nne) {
            this->vi_nne.push_back (hi->nne->vi);
        }
        if (hi->has_nnw) {
            this->vi_nnw.push_back (hi->nnw->vi);
        }
        if (hi->has_nw) {
            this->vi_nw.push_back (hi->nw->vi);
        }
        if (hi->has_nsw) {
            this->vi_nsw.push_back (hi->nsw->vi);
        }
        if (hi->has_nse) {
            this->vi_nse.push_back (hi->nse->vi);
        }
        ++hi;
    }
#endif
}

unsigned int
morph2::HexGrid::num (void) const
{
    return this->hexen.size();
}

unsigned int
morph2::HexGrid::lastVectorIndex (void) const
{
    return this->hexen.rbegin()->vi;
}

string
morph2::HexGrid::output (void) const
{
    stringstream ss;
    ss << "Hex grid with " << this->hexen.size() << " hexes." << endl;
    auto i = this->hexen.begin();
    float lasty = this->hexen.front().y;
    unsigned int rownum = 0;
    ss << endl << "Row/Ring " << rownum++ << ":" << endl;
    while (i != this->hexen.end()) {
        if (i->y > lasty) {
            ss << endl << "Row/Ring " << rownum++ << ":" << endl;
            lasty = i->y;
        }
        ss << i->output() << endl;
        ++i;
    }
    return ss.str();
}

string
morph2::HexGrid::extent (void) const
{
    stringstream ss;
    if (gridReduced == false) {
        ss << "Grid vertices: \n"
           << "           NW: (" << this->vertexNW->x << "," << this->vertexNW->y << ") "
           << "      NE: (" << this->vertexNE->x << "," << this->vertexNE->y << ")\n"
           << "     W: (" << this->vertexW->x << "," << this->vertexW->y << ") "
           << "                              E: (" << this->vertexE->x << "," << this->vertexE->y << ")\n"
           << "           SW: (" << this->vertexSW->x << "," << this->vertexSW->y << ") "
           << "      SE: (" << this->vertexSE->x << "," << this->vertexSE->y << ")";
    } else {
        ss << "Initial grid vertices are no longer valid.";
    }
    return ss.str();
}

float
morph2::HexGrid::getd (void) const
{
    return this->d;
}

float
morph2::HexGrid::getXmin (float phi) const
{
    float xmin = 0.0f;
    float x_ = 0.0f;
    bool first = true;
    for (auto h : this->hexen) {
        x_ = h.x * cos (phi) + h.y * sin (phi);
        if (first) {
            xmin = x_;
            first = false;
        }
        if (x_ < xmin) {
            xmin = x_;
        }
    }
    return xmin;
}

float
morph2::HexGrid::getXmax (float phi) const
{
    float xmax = 0.0f;
    float x_ = 0.0f;
    bool first = true;
    for (auto h : this->hexen) {
        x_ = h.x * cos (phi) + h.y * sin (phi);
        if (first) {
            xmax = x_;
            first = false;
        }
        if (x_ > xmax) {
            xmax = x_;
        }
    }
    return xmax;
}

void
morph2::HexGrid::init (float d_, float x_span_, float z_)
{
    this->d = d_;
    this->x_span = x_span_;
    this->z = z_;
    this->init();
}

void
morph2::HexGrid::init (void)
{
    // Use span_x to determine how many rings out to traverse.
    float halfX = this->x_span/2.0f;
    DBG ("halfX:" << halfX);
    DBG ("d:" << d);
    unsigned int maxRing = abs(ceil(halfX/this->d));
    DBG ("ceil(halfX/d):" << ceil(halfX/d));

    DBG ("Creating hexagonal hex grid with maxRing: " << maxRing);

    // The "vector iterator" - this is an identity iterator that is
    // added to each Hex in the grid.
    unsigned int vi = 0;

    // Vectors of list-iterators to hexes in this->hexen. Used to keep a
    // track of nearest neighbours. I'm using vector, rather than a
    // list as this allows fast random access of elements and I'll not
    // be inserting or erasing in the middle of the arrays.
    vector<list<Hex>::iterator> prevRingEven;
    vector<list<Hex>::iterator> prevRingOdd;

    // Swap pointers between rings.
    vector<list<Hex>::iterator>* prevRing = &prevRingEven;
    vector<list<Hex>::iterator>* nextPrevRing = &prevRingOdd;

    // Direction iterators used in the loop for creating hexes
    int ri = 0;
    int gi = 0;

    // Create central "ring" first (the single hex)
    this->hexen.emplace_back (vi++, this->d, ri, gi);

    // Put central ring in the prevRing vector:
    {
        list<Hex>::iterator h = this->hexen.end(); --h;
        prevRing->push_back (h);
    }

    // Now build up the rings around it, setting neighbours as we
    // go. Each ring has 6 more hexes than the previous one (except
    // for ring 1, which has 6 instead of 1 in the centre).
    unsigned int numInRing = 6;

    // How many hops in the same direction before turning a corner?
    // Increases for each ring. Increases by 1 in each ring.
    unsigned int ringSideLen = 1;

    // These are used to iterate along the six sides of the hexagonal
    // ring that's inside, but adjacent to the hexagonal ring that's
    // under construction.
    int walkstart = 0;
    int walkinc = 0;
    int walkmin = walkstart-1;
    int walkmax = 1;

    for (unsigned int ring = 1; ring <= maxRing; ++ring) {

        DBG2 ("\n\n************** numInRing: " << numInRing << " ******************");

        // Set start ri, gi. This moves up a hex and left a hex onto
        // the start hex of the new ring.
        --ri; ++gi;

        nextPrevRing->clear();

        // Now walk around the ring, in 6 walks, that will bring us
        // round to just before we started. walkstart has the starting
        // iterator number for the vertices of the hexagon.
        DBG2 ("Before r; walkinc: " << walkinc << ", walkmin: " << walkmin << ", walkmax: " << walkmax);

        // Walk in the r direction first:
        DBG2 ("============ r walk =================");
        for (unsigned int i = 0; i<ringSideLen; ++i) {

            DBG2 ("Adding hex at " << ri << "," << gi);
            this->hexen.emplace_back (vi++, this->d, ri++, gi);
            auto hi = this->hexen.end(); hi--;
            auto lasthi = hi;
            --lasthi;

            // Set vertex
            if (i==0) { vertexNW = hi; }

            // 1. Set my W neighbour to be the previous hex in THIS ring, if possible
            if (i > 0) {
                hi->set_nw (lasthi);
                DBG2 (" r walk: Set me (" << hi->ri << "," << hi->gi << ") as E neighbour for hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as E neighbour to previous hex in the ring:
                lasthi->set_ne (hi);
            } else {
                // i must be 0 in this case, we would set the SW
                // neighbour now, but as this won't have been added to
                // the ring, we have to leave it.
                DBG2 (" r walk: I am (" << hi->ri << "," << hi->gi << "). Omitting SW neighbour of first hex in ring.");
            }

            // 2. SW neighbour
            int j = walkstart + (int)i - 1;
            DBG2 ("i is " << i << ", j is " << j << ", walk min/max: " << walkmin << " " << walkmax);
            if (j>walkmin && j<walkmax) {
                // Set my SW neighbour:
                hi->set_nsw ((*prevRing)[j]);
                // Set me as NE neighbour to those in prevRing:
                DBG2 (" r walk: Set me (" << hi->ri << "," << hi->gi << ") as NE neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nne (hi);
            }
            ++j;
            DBG2 ("i is " << i << ", j is " << j);

            // 3. Set my SE neighbour:
            if (j<=walkmax) {
                hi->set_nse ((*prevRing)[j]);
                // Set me as NW neighbour:
                DBG2 (" r walk: Set me (" << hi->ri << "," << hi->gi << ") as NW neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nnw (hi);
            }

            // Put in me nextPrevRing:
            nextPrevRing->push_back (hi);
        }
        walkstart += walkinc;
        walkmin   += walkinc;
        walkmax   += walkinc;

        // Walk in -b direction
        DBG2 ("Before -b; walkinc: " << walkinc << ", walkmin: " << walkmin << ", walkmax: " << walkmax);
        DBG2 ("=========== -b walk =================");
        for (unsigned int i = 0; i<ringSideLen; ++i) {
            DBG2 ("Adding hex at " << ri << "," << gi);
            this->hexen.emplace_back (vi++, this->d, ri++, gi--);
            auto hi = this->hexen.end(); hi--;
            auto lasthi = hi;
            --lasthi;

            // Set vertex
            if (i==0) { vertexNE = hi; }

            // 1. Set my NW neighbour to be the previous hex in THIS ring, if possible
            if (i > 0) {
                hi->set_nnw (lasthi);
                DBG2 ("-b walk: Set me (" << hi->ri << "," << hi->gi << ") as SE neighbour for hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as SE neighbour to previous hex in the ring:
                lasthi->set_nse (hi);
            } else {
                // Set my W neighbour for the first hex in the row.
                hi->set_nw (lasthi);
                DBG2 ("-b walk: Set me (" << hi->ri << "," << hi->gi << ") as E neighbour for last walk's hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as E neighbour to previous hex in the ring:
                lasthi->set_ne (hi);
            }

            // 2. W neighbour
            int j = walkstart + (int)i - 1;
            DBG2 ("i is " << i << ", j is " << j << " prevRing->size(): " <<prevRing->size() );
            if (j>walkmin && j<walkmax) {
                // Set my W neighbour:
                hi->set_nw ((*prevRing)[j]);
                // Set me as E neighbour to those in prevRing:
                DBG2 ("-b walk: Set me (" << hi->ri << "," << hi->gi << ") as E neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_ne (hi);
            }
            ++j;
            DBG2 ("i is " << i << ", j is " << j);

            // 3. Set my SW neighbour:
            DBG2 ("i is " << i << ", j is " << j);
            if (j<=walkmax) {
                hi->set_nsw ((*prevRing)[j]);
                // Set me as NE neighbour:
                DBG2 ("-b walk: Set me (" << hi->ri << "," << hi->gi << ") as NE neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nne (hi);
            }

            nextPrevRing->push_back (hi);
        }
        walkstart += walkinc;
        walkmin += walkinc;
        walkmax += walkinc;
        DBG2 ("walkinc: " << walkinc << ", walkmin: " << walkmin << ", walkmax: " << walkmax);

        // Walk in -g direction
        DBG2 ("=========== -g walk =================");
        for (unsigned int i = 0; i<ringSideLen; ++i) {

            DBG2 ("Adding hex at " << ri << "," << gi);
            this->hexen.emplace_back (vi++, this->d, ri, gi--);
            auto hi = this->hexen.end(); hi--;
            auto lasthi = hi;
            --lasthi;

            // Set vertex
            if (i==0) { vertexE = hi; }

            // 1. Set my NE neighbour to be the previous hex in THIS ring, if possible
            if (i > 0) {
                hi->set_nne (lasthi);
                DBG2 ("-g walk: Set me (" << hi->ri << "," << hi->gi << ") as SW neighbour for hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as SW neighbour to previous hex in the ring:
                lasthi->set_nsw (hi);
            } else {
                // Set my NW neighbour for the first hex in the row.
                hi->set_nnw (lasthi);
                DBG2 ("-g walk: Set me (" << hi->ri << "," << hi->gi << ") as SE neighbour for last walk's hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as SE neighbour to previous hex in the ring:
                lasthi->set_nse (hi);
            }

            // 2. NW neighbour
            int j = walkstart + (int)i - 1;
            DBG2 ("i is " << i << ", j is " << j);
            if (j>walkmin && j<walkmax) {
                // Set my NW neighbour:
                hi->set_nnw ((*prevRing)[j]);
                // Set me as SE neighbour to those in prevRing:
                DBG2 ("-g walk: Set me (" << hi->ri << "," << hi->gi << ") as SE neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nse (hi);
            }
            ++j;
            DBG2 ("i is " << i << ", j is " << j);

            // 3. Set my W neighbour:
            if (j<=walkmax) {
                hi->set_nw ((*prevRing)[j]);
                // Set me as E neighbour:
                DBG2 ("-g walk: Set me (" << hi->ri << "," << hi->gi << ") as E neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_ne (hi);
            }

            // Put in me nextPrevRing:
            nextPrevRing->push_back (hi);
        }
        walkstart += walkinc;
        walkmin += walkinc;
        walkmax += walkinc;
        DBG2 ("walkinc: " << walkinc << ", walkmin: " << walkmin << ", walkmax: " << walkmax);

        // Walk in -r direction
        DBG2 ("=========== -r walk =================");
        for (unsigned int i = 0; i<ringSideLen; ++i) {

            DBG2 ("Adding hex at " << ri << "," << gi);
            this->hexen.emplace_back (vi++, this->d, ri--, gi);
            auto hi = this->hexen.end(); hi--;
            auto lasthi = hi;
            --lasthi;

            // Set vertex
            if (i==0) { vertexSE = hi; }

            // 1. Set my E neighbour to be the previous hex in THIS ring, if possible
            if (i > 0) {
                hi->set_ne (lasthi);
                DBG2 ("-r walk: Set me (" << hi->ri << "," << hi->gi << ") as W neighbour for hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as W neighbour to previous hex in the ring:
                lasthi->set_nw (hi);
            } else {
                // Set my NE neighbour for the first hex in the row.
                hi->set_nne (lasthi);
                DBG2 ("-r walk: Set me (" << hi->ri << "," << hi->gi << ") as SW neighbour for last walk's hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as SW neighbour to previous hex in the ring:
                lasthi->set_nsw (hi);
            }

            // 2. NE neighbour:
            int j = walkstart + (int)i - 1;
            DBG2 ("i is " << i << ", j is " << j);
            if (j>walkmin && j<walkmax) {
                // Set my NE neighbour:
                hi->set_nne ((*prevRing)[j]);
                // Set me as SW neighbour to those in prevRing:
                DBG2 ("-r walk: Set me (" << hi->ri << "," << hi->gi << ") as SW neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nsw (hi);
            }
            ++j;
            DBG2 ("i is " << i << ", j is " << j);

            // 3. Set my NW neighbour:
            if (j<=walkmax) {
                hi->set_nnw ((*prevRing)[j]);
                // Set me as SE neighbour:
                DBG2 ("-r walk: Set me (" << hi->ri << "," << hi->gi << ") as SE neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nse (hi);
            }

            nextPrevRing->push_back (hi);
        }
        walkstart += walkinc;
        walkmin += walkinc;
        walkmax += walkinc;
        DBG2 ("walkinc: " << walkinc << ", walkmin: " << walkmin << ", walkmax: " << walkmax);

        // Walk in b direction
        DBG2 ("============ b walk =================");
        for (unsigned int i = 0; i<ringSideLen; ++i) {
            DBG2 ("Adding hex at " << ri << "," << gi);
            this->hexen.emplace_back (vi++, this->d, ri--, gi++);
            auto hi = this->hexen.end(); hi--;
            auto lasthi = hi;
            --lasthi;

            // Set vertex
            if (i==0) { vertexSW = hi; }

            // 1. Set my SE neighbour to be the previous hex in THIS ring, if possible
            if (i > 0) {
                hi->set_nse (lasthi);
                DBG2 (" b in-ring: Set me (" << hi->ri << "," << hi->gi << ") as NW neighbour for hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as NW neighbour to previous hex in the ring:
                lasthi->set_nnw (hi);
            } else { // i == 0
                // Set my E neighbour for the first hex in the row.
                hi->set_ne (lasthi);
                DBG2 (" b in-ring: Set me (" << hi->ri << "," << hi->gi << ") as W neighbour for last walk's hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as W neighbour to previous hex in the ring:
                lasthi->set_nw (hi);
            }

            // 2. E neighbour:
            int j = walkstart + (int)i - 1;
            DBG2 ("i is " << i << ", j is " << j);
            if (j>walkmin && j<walkmax) {
                // Set my E neighbour:
                hi->set_ne ((*prevRing)[j]);
                // Set me as W neighbour to those in prevRing:
                DBG2 (" b walk: Set me (" << hi->ri << "," << hi->gi << ") as W neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nw (hi);
            }
            ++j;
            DBG2 ("i is " << i << ", j is " << j);

            // 3. Set my NE neighbour:
            if (j<=walkmax) {
                hi->set_nne ((*prevRing)[j]);
                // Set me as SW neighbour:
                DBG2 (" b walk: Set me (" << hi->ri << "," << hi->gi << ") as SW neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nsw (hi);
            }

            nextPrevRing->push_back (hi);
        }
        walkstart += walkinc;
        walkmin += walkinc;
        walkmax += walkinc;
        DBG2 ("walkinc: " << walkinc << ", walkmin: " << walkmin << ", walkmax: " << walkmax);

        // Walk in g direction up to almost the last hex
        DBG2 ("============ g walk =================");
        for (unsigned int i = 0; i<ringSideLen; ++i) {

            DBG2 ("Adding hex at " << ri << "," << gi);
            this->hexen.emplace_back (vi++, this->d, ri, gi++);
            auto hi = this->hexen.end(); hi--;
            auto lasthi = hi;
            --lasthi;

            // Set vertex
            if (i==0) { vertexW = hi; }

            // 1. Set my SW neighbour to be the previous hex in THIS ring, if possible
            DBG2(" g walk: i is " << i << " and ringSideLen-1 is " << (ringSideLen-1));
            if (i == (ringSideLen-1)) {
                // Special case at end; on last g walk hex, set the NE neighbour
                // Set my NE neighbour for the first hex in the row.
                hi->set_nne ((*nextPrevRing)[0]); // (*nextPrevRing)[0] is an iterator to the first hex

                DBG2 (" g in-ring: Set me (" << hi->ri << "," << hi->gi << ") as SW neighbour for this ring's first hex at (" << (*nextPrevRing)[0]->ri << "," << (*nextPrevRing)[0]->gi << ")");
                // Set me as NW neighbour to previous hex in the ring:
                (*nextPrevRing)[0]->set_nsw (hi);

            }
            if (i > 0) {
                hi->set_nsw (lasthi);
                DBG2 (" g in-ring: Set me (" << hi->ri << "," << hi->gi << ") as NE neighbour for hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as NE neighbour to previous hex in the ring:
                lasthi->set_nne (hi);
            } else {
                // Set my SE neighbour for the first hex in the row.
                hi->set_nse (lasthi);
                DBG2 (" g in-ring: Set me (" << hi->ri << "," << hi->gi << ") as NW neighbour for last walk's hex at (" << lasthi->ri << "," << lasthi->gi << ")");
                // Set me as NW neighbour to previous hex in the ring:
                lasthi->set_nnw (hi);
            }

            // 2. E neighbour:
            int j = walkstart + (int)i - 1;
            DBG2 ("i is " << i << ", j is " << j);
            if (j>walkmin && j<walkmax) {
                // Set my SE neighbour:
                hi->set_nse ((*prevRing)[j]);
                // Set me as NW neighbour to those in prevRing:
                DBG2 (" g walk: Set me (" << hi->ri << "," << hi->gi << ") as NW neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nnw (hi);
            }
            ++j;
            DBG2 ("i is " << i << ", j is " << j);

            // 3. Set my E neighbour:
            if (j==walkmax) { // We're on the last square and need to
                              // set the East neighbour of the first
                              // hex in the last ring.
                hi->set_ne ((*prevRing)[0]);
                // Set me as W neighbour:
                DBG2 (" g walk: Set me (" << hi->ri << "," << hi->gi << ") as W neighbour for hex at (" << (*prevRing)[0]->ri << "," << (*prevRing)[0]->gi << ")");
                (*prevRing)[0]->set_nw (hi);

            } else if (j<walkmax) {
                hi->set_ne ((*prevRing)[j]);
                // Set me as W neighbour:
                DBG2 (" g walk: Set me (" << hi->ri << "," << hi->gi << ") as W neighbour for hex at (" << (*prevRing)[j]->ri << "," << (*prevRing)[j]->gi << ")");
                (*prevRing)[j]->set_nw (hi);
            }

            // Put in me nextPrevRing:
            nextPrevRing->push_back (hi);
        }
        // Should now be on the last hex.

        // Update the walking increments for finding the vertices of
        // the hexagonal ring. These are for walking around the ring
        // *inside* the ring of hexes being created and hence note
        // that I set walkinc to numInRing/6 BEFORE incrementing
        // numInRing by 6, below.
        walkstart = 0;
        walkinc = numInRing / 6;
        walkmin = walkstart - 1;
        walkmax = walkmin + 1 + walkinc;

        // Always 6 additional hexes in the next ring out
        numInRing += 6;

        // And ring side length goes up by 1
        ringSideLen++;

        // Swap prevRing and nextPrevRing.
        vector<list<Hex>::iterator>* tmp = prevRing;
        prevRing = nextPrevRing;
        DBG2 ("New prevRing contains " << prevRing->size() << " elements");
        nextPrevRing = tmp;
    }

    DBG ("Finished creating " << this->hexen.size() << " hexes in " << maxRing << " rings.");
}
