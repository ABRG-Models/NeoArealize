#include "tools.h"
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include <stdio.h>
#ifdef __WIN__
# include <direct.h>
# define GetCurrentDir _getcwd
#else
# include <unistd.h>
# define GetCurrentDir getcwd
#endif

extern "C" {
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <dirent.h>
}

using std::stringstream;
using std::string;
using std::vector;
using std::runtime_error;

double
morph::Tools::randDouble (void)
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

float
morph::Tools::randSingle (void)
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void
morph::Tools::createDir (const string& path,
                         const mode_t mode,
                         const int uid, const int gid)
{
    if (path.empty()) {
        // Create no directory. Just return.
        return;
    }

    // Set to true if we are provided with an absolute filepath
    bool pathIsAbsolute(false);

    // Set umask to 0000 to stop it interfering with mode
    int oldUmask = umask (0000);
    string::size_type pos, lastPos = path.size()-1;
    vector<string> dirs;
    if ((pos = path.find_last_of ('/', lastPos)) == string::npos) {
        // Path is single directory.
        dirs.push_back (path);
    } else {
        // Definitely DO have a '/' in the path somewhere:
        if (path[0] == '/') {
            pathIsAbsolute = true;
            while ((pos = path.find_last_of ('/', lastPos)) != 0) {
                dirs.push_back (path.substr(pos+1, lastPos-pos));
                lastPos = pos-1;
            }
            dirs.push_back (path.substr(1, lastPos));
        } else {
            // Non absolute...
            while ((pos = path.find_last_of ('/', lastPos)) != 0) {
                dirs.push_back (path.substr(pos+1, lastPos-pos));
                lastPos = pos-1;
                if (pos == string::npos) {
                    break;
                }
            }
        }
    }

    vector<string>::reverse_iterator i = dirs.rbegin();
    string prePath("");
    bool first(true);
    while (i != dirs.rend()) {
        if (first && !pathIsAbsolute) {
            prePath += "./" + *i;
            first = false;
        } else {
            prePath += "/" + *i;
        }
        int rtn = mkdir (prePath.c_str(), mode);
        if (rtn) {
            int e = errno;
            stringstream emsg;
            emsg << "createDir(): mkdir() set error: ";
            switch (e) {
            case EACCES:
                emsg << "Permission is denied";
                break;
            case EEXIST:
                // Path exists, though maybe not as a directory.
                // Set mode/ownership before moving on:
                if (uid>-1 && gid>-1) {
                    chown (prePath.c_str(), static_cast<uid_t>(uid), static_cast<gid_t>(gid));
                    chmod (prePath.c_str(), mode);
                }
                i++;
                continue;
                break;
            case EFAULT:
                emsg << "Bad address";
                break;
            case ELOOP:
                emsg << "Too many symlinks in " << prePath;
                break;
            case ENAMETOOLONG:
                emsg << "File name (" << prePath << ") too long";
                break;
            case ENOENT:
                emsg << "Path '" << prePath << "' invalid (part or all of it)";
                break;
            case ENOMEM:
                emsg << "Out of kernel memory";
                break;
            case ENOSPC:
                emsg << "Out of storage space/quota exceeded.";
                break;
            case ENOTDIR:
                emsg << "component of the path '" << prePath << "' is not a directory";
                break;
            case EPERM:
                emsg << "file system doesn't support directory creation";
                break;
            case EROFS:
                emsg << "path '" << prePath << "' refers to location on read only filesystem";
                break;
            default:
                emsg << "unknown error";
                break;
            }
            throw runtime_error (emsg.str());
        }
        if (uid>-1 && gid>-1) {
            chown (prePath.c_str(), static_cast<uid_t>(uid), static_cast<gid_t>(gid));
        }
        i++;
    }

    // Reset umask
    umask (oldUmask);
}

bool
morph::Tools::containsOnlyWhitespace (string& input)
{
    bool rtn = true;
    for (string::size_type i = 0; i < input.size(); ++i) {
        if (input[i] == ' ' || input[i] == '\t' || input[i] == '\n' || input[i] == '\r') {
            // continue.
        } else {
            rtn = false;
            break;
        }
    }
    return rtn;
}

int
morph::Tools::searchReplace (const string& searchTerm,
                             const string& replaceTerm,
                             string& data,
                             const bool replaceAll)
{
    int count = 0;
    string::size_type pos = 0;
    string::size_type ptr = string::npos;
    string::size_type stl = searchTerm.size();
    if (replaceAll) {
        pos = data.size();
        while ((ptr = data.rfind (searchTerm, pos)) != string::npos) {
            data.erase (ptr, stl);
            data.insert (ptr, replaceTerm);
            count++;
            if (ptr >= stl) {
                // This is a move backwards along the
                // string far enough that we don't
                // match a substring of the last
                // replaceTerm in the next search.
                pos = ptr - stl;
            } else {
                break;
            }
        }
    } else {
        // Replace first only
        if ((ptr = data.find (searchTerm, pos)) != string::npos) {
            data.erase (ptr, stl);
            data.insert (ptr, replaceTerm);
            count++;
        }
    }

    return count;
}
