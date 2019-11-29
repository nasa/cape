#include <Python.h>
#include <stdio.h>
#include <math.h>

// Local includes
#include "capec_BaseFile.h"
#include "capec_CSVFile.h"


// Count data lines in open file
int
capec_CSVFileCountLines(FILE *fp)
{
    // Line counts
    long nline = 0;
    // File position
    long pos;
    // Strings
    char c;
    
    // Remember current location
    pos = ftell(fp);
    
    // Read lines
    while (!feof(fp)) {
        // Read spaces
        capcec_FileAdvanceWhiteSpace(fp);
        // Get next character
        c = getc(fp);
        // Check it
        if (c == '\n') {
            // Empty line
            continue;
        } else if (feof(fp)) {
            // Last line
            break;
        }
        // Read to end of line (comment or not)
        capec_FileAdvanceEOL(fp);
        // Check if it was a comment
        if (c != '#') {
            // Increase line count
            nline += 1;
        }
    }
    
    // Return to original location
    fseek(fp, pos, SEEK_SET);
    
    // Output
    return nline;
}
