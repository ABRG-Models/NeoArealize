/*
 * Utility functions
 */

#ifndef _TOOLS_H_
#define _TOOLS_H_

#include <vector>
#include <string>

using std::vector;
using std::string;

/*!
 * Character sets useful when calling util::sanitize().
 *
 * These are ordered so that the most common chars appear earliest.
 */
//@{
#define CHARS_NUMERIC            "0123456789"
#define CHARS_ALPHA              "etaoinshrdlcumwfgypbvkjxqzETAOINSHRDLCUMWFGYPBVKJXQZ"
#define CHARS_ALPHALOWER         "etaoinshrdlcumwfgypbvkjxqz"
#define CHARS_ALPHAUPPER         "ETAOINSHRDLCUMWFGYPBVKJXQZ"
#define CHARS_NUMERIC_ALPHA      "etaoinshrdlcumwfgypbvkjxqz0123456789ETAOINSHRDLCUMWFGYPBVKJXQZ"
#define CHARS_NUMERIC_ALPHALOWER "etaoinshrdlcumwfgypbvkjxqz0123456789"
#define CHARS_NUMERIC_ALPHAUPPER "0123456789ETAOINSHRDLCUMWFGYPBVKJXQZ"
//@}

/*!
 * These are the chars which are acceptable for use in both unix, mac
 * AND windows file names. This doesn guarantee a safe Windows
 * filename, it imposes some extra conditions (no . at end of name,
 * some files such as NUL.txt AUX.txt disallowed).
 */
#define COMMON_FILE_SAFE_CHARS        CHARS_NUMERIC_ALPHA"_-.{}^[]`=,;"

/*!
 * Chars which are safe for IP domainnames
 */
#define IP_DOMAINNAME_SAFE_CHARS      CHARS_NUMERIC_ALPHA"-."

/*!
 * Chars which are safe for IP addresses
 */
#define IP_ADDRESS_SAFE_CHARS         CHARS_NUMERIC"."

namespace morph
{
    /*!
     * Allows the use of transform and tolower() on strings with
     * GNU compiler
     */
    class to_lower
    {
    public:
        /*!
         * Apply lower case operation to the char c.
         */
        char operator() (const char c) const {
            return tolower(c);
        }
    };

    /*!
     * Allows the use of transform and toupper() on strings with
     * GNU compiler
     */
    class to_upper
    {
    public:
        /*!
         * Apply upper case operation to the char c.
         */
        char operator() (const char c) const {
            return toupper(c);
        }
    };

    class Tools
    {
    public:
        /*!
         * Return a random double precision number in the range [0,1], sampled
         * from a uniform distribution.
         */
        static double randDouble (void);

        /*!
         * Return a random single precision number in the range [0,1],
         * sampled from a uniform distribution.
         */
        static float randSingle (void);

        /*!
         * Create the directory and any parent directories
         * which need to be created.
         *
         * Makes use of mkdir() and acts like the system
         * command mkdir -p path.
         *
         * If uid/gid is set to >-1, then chown each
         * directory. This means that ownership is set for the
         * directories in the path even if the directories do
         * not need to be created.
         *
         * \param path The path (relative or absolute) to the
         * directory which should be created.
         *
         * \param mode the permissions mode which should be
         * set on the directory. This is applied even if the
         * directory was not created.
         *
         * \param uid The user id to apply to the
         * directory. This is applied even if the directory
         * was not created. This is NOT applied if it is set
         * to -1.
         *
         * \param gid The group id to apply to the
         * directory. This is applied even if the directory
         * was not created. This is NOT applied if it is set
         * to -1.
         */
        static void createDir (const string& path,
                               const mode_t mode = 0775,
                               const int uid = -1, const int gid = -1);

        /*!
         * Return true if input contains only space, tab, newline
         * chars.
         */
        static bool containsOnlyWhitespace (string& input);

        /*!
         * Do a search and replace, search for searchTerm,
         * replacing with replaceTerm. if replaceAll is true,
         * replace all occurrences of searchTerm, otherwise
         * just replace the first occurrence of searchTerm
         * with replaceTerm.
         *
         * \return the number of terms replaced.
         */
        static int searchReplace (const string& searchTerm,
                                  const string& replaceTerm,
                                  string& data,
                                  const bool replaceAll = true);
    };


} // namespace

#endif // _TOOLS_H_
