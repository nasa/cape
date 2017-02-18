

Report Definitions
------------------
The above sample contains two sample report definitions, but several options
were left out.  The example below contains a more complete set of options.

    .. code-block:: javascript
    
        "Report": {
            // List of report names
            "Reports": ["case", "mach"],
            // 
            // Full definition for "case" report
            "case": {
                "Title": "OVERFLOW Iterative Force \\& Moment Histories",
                "Subtitle": "SLS-10008 VAC1 Ascent Analysis",
                "Author": "Ames SLS CFD Team",
                "Affiliation": "NASA/ARC TNA",
                "Restriction": "SBU - ITAR",
                "Logo": "NASA_logo.pdf",
                "Frontispiece": "NASA_logo.pdf",
                "Figures": ["CaseSummary", "IterFM"],
                "ZeroFigures": ["ZeroSummary", "SurfGrid"],
                "ErrorFigures": ["ZeroSummary", "SurfGrid"]
            },
            // Full definition of "mach" report
            "mach": {
                "Title": "OVERFLOW Force \\& Moment Mach Sweeps",
                "Subtitle": "SLS-10008 VAC1 Ascent Analysis",
                "Author": "Ames SLS CFD Team",
                "Affiliation": "NASA/ARC TNA",
                "Restriction": "SBU - ITAR",
                "Logo": "NASA_logo.pdf",
                "Frontispiece": "NASA_logo.pdf",
                "Sweeps": ["mach"]
            }
        }
        
As these examples show, there are several options for the title page, which do
not need much description.  However, notice the somewhat unwieldy text ``"\\&"``
in the titles.  Text from the ``"Report"`` section as a whole is first
interpreted as a Python string and then inserted into a LaTeX document; the
first backslash is to get Python to interpret ``"\\"`` as a backslash, and
``"\&"`` is interpreted by PDFLaTeX as an ampersand character (since ``&`` is a
functional character in LaTeX code.  This also means that complex LaTeX such as 
equations (between ``"$"`` characters) and formatted text (e.g.
``"\\texttt{RSRB\\_FwdBSM}"``) can be used.

The ``"Logo"`` is placed at the bottom left of each page, and the
``"Frontispiece"`` is placed on the title page in a larger version.  Both of
these images (which do not have to be the same file as they are in this example)
are given relative to the ``report/`` folder, which is where the PDFLaTeX
document is compiled.

The ``"Restriction"`` tag is text that is placed on the bottom center of each
page in bold text.  Finally, the ``"ZeroFigures"`` and ``"ErrorFigures"`` allow
the user to use a different set of figures (and thus subfigures) when either
there are zero iterations or an error status.  This can be informative when
plotting the iterative history of zero iterations is otherwise not too useful.

Sweep Definitions
-----------------
Sweeps are defined by a list of constraints, which can be either exact or to
within a certain tolerance.  The user may also specify global constraints that
restrict the results to a subset of the run matrix/database.  Finally, the user
may subdivide the sweep into smaller sweeps using ``"Carpet"`` constraints.  In
the following example, all cases with zero sideslip are grouped into one
"sweep", but it is split into separate curves for each angle of attack.

    .. code-block:: javascript
    
        "Sweeps": {
            // Mach/alpha carpet plot
            "mach-alpha": {
                // For two cases in the same set, these vars must be equal
                "EqCons": ["beta"],
                // Split these cases into subgroups with same alpha
                "CarpetEqCons": ["alpha"],
                // Only consider supersonic cases
                "GlobalCons": ["mach > 1"],
                // Plot result against Mach number
                "XAxis": "mach"
            }
        }

For some run matrices, especially when the data or conditions comes from a wind
tunnel test, finding cases with the exact same angle of attack may be
inadequate.  For example, the angle of attack may fluctuate slightly between
otherwise similar cases at different Mach numbers.  Using a tolerance can often
solve this problem.

    .. code-block:: javascript
    
        "Sweeps": {
            // Mach/alpha carpet plot
            "mach-alpha": {
                // All cases in set must have *beta* within 0.05 of first case
                "TolCons": {"beta": 0.05},
                // Do the same thing with *alpha*
                "CarpetTolCons": {"alpha": 0.1},
                // Plot results against Mach number
                "XAxis": "mach"
            }
        }