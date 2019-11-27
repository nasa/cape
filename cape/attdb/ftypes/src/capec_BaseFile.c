#include <stdio.h>

// Local includes


// Read to end of line
void
capec_FileAdvanceEOL(FILE *fp)
{
    // Buffer
    char buff[80];
    
    // Read white space (80 chars at a time)
    while (fscanf(fp, "%80[^\n]", buff) == 1)
    {
        
    }
    // Read newline character
    getc(fp);
}

// Advance past any whitespace (not newline) characters
void
capcec_FileAdvanceWhiteSpace(FILE *fp)
{
    // Buffer
    char buff[80];
    
    // Read white space (80 chars at a time)
    while (fscanf(fp, "%80[ \t\r]", buff) == 1)
    {
        
    }
}

