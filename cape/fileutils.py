
# Pure Python to tail a file
def tail(fname: str, n=1):
    # Number of lines read
    m = 0
    # Open file binary to enable relative search
    with open(fname, 'rb') as fb:
        # Go to end of file
        pos = fb.seek(0, 2)
        # Special case for empty file
        if pos < 2:
            # Return whole file if 0 or 1 chars
            fb.seek(0)
            return fb.read().decode("utf-8")
        # Loop backwards through file until *n* newline chars
        while m < n:
            # Go back two chars so we can read previous one
            # Note special case:
            #    We don't actually check the final char!
            #    This avoids checking if file ends with \n
            pos = fb.seek(-2, 1)
            # Check for beginning of file
            if pos == 0:
                break
            # Read that character
            c = fb.read(1)
            # Check for newline
            if c == b"\n":
                m += 1
        # File is no after last \n; read to EOF
        return fb.read().decode("utf-8")

