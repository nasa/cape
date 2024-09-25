r"""
:mod:`cape.pycart.archivist`: Archiving actions specific to Cart3D
===================================================================

This module provides customizations of the generic
:class:`cape.cfdx.archivist.CaseArchivist` that assist in reducing the
file count while running Cart3D, especially in adaptive mode.
"""

# Local imports
from ..cfdx import archivist


# Custom class
class CaseArchivist(archivist.CaseArchivist):
    r"""Custom case archiving class for Cart3D

    :Call:
        >>> a = CaseArchivist(opts, where=None, casename=None)
    :Inputs:
        *opts*: :class:`ArchiveOpts`
            Case archiving options
        *where*: {``None``} | :class:`str`
            Root of CFD case folder (default: CWD)
        *casename*: {``None``} | :class:`str`
            Name of CFD case folder (default: last two levels of CWD)
    """
    def tar_adapt(self, test: bool = False):
        r"""Tar ``adaptXX`` folders except most recent

        No action will be taken if the *TarAdapt* setting is ``False``.

        For example, if there are 3 adaptations in the current folder,
        the following substitutions will be made.

            * ``adapt00/`` --> :file:`adapt00.tar`
            * ``adapt01/`` --> :file:`adapt01.tar`
            * ``adapt02/`` --> :file:`adapt02.tar`
            * ``adapt03/``

        The most recent adaptation is left untouched, and the old
        folders are deleted.

        :Call:
            >>> a.tar_adapt(test=False)
        :Inputs:
            *a*: :class:`CaseArchivist`
                Archive controller for one case
            *test*: ``True`` | {``False``}
                If ``True``, log which files would be deleted, but don't
                actually delete, copy, or tar anything
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0 (``cape.pycart.manage``)
            * 2015-01-10 ``@ddalle``: v1.1; format as an option
            * 2015-12-02 ``@ddalle``: v1.2; Options
            * 2024-06-24 ``@ddalle``: v1.3; never tar ``adapt00``
            * 2024-09-24 ``@ddalle``: v2.0; CaseArchivist
        """
        # Get *TarAdapt* option
        v = self.opts.get_opt("TarAdapt", vdef=True)
        # Exit if not set
        if (not v) or (v == "none"):
            return
        # Initialize settings
        self.begin("restart", test)
        # Find the adapt folders
        matchdict = self.search("adapt[0-9][0-9]")
        # Get matches
        matches = matchdict['']
        # Sort by number in case un-tarring messed up modtimes
        matches.sort(key=lambda x: int(x[-2:]))
        # Loop through matches
        for adaptdir in matches[:-1]:
            # Skip ``adapt00/``
            if adaptdir == "adapt00":
                continue
            # Otherwise tar it
            self.tar_local(adaptdir, {adaptdir: 0})
            # Remove folder
            self.delete_dir(adaptdir)

    def tar_viz(self, test: bool = False):
        r"""Move visualization surface and cut plane files to tar balls

        This reduces file count by tarring ``Components.i.*.plt`` and
        ``cutPlanes.*.plt``.

        :Call:
            >>> a.tar_viz(test=False)
        :Inputs:
            *a*: :class:`CaseArchivist`
                Archive controller for one case
            *test*: ``True`` | {``False``}
                If ``True``, log which files would be deleted, but don't
                actually delete, copy, or tar anything
        :Versions:
            * 2014-12-18 ``@ddalle``: v1.0 (``cape.pycart.manage``)
            * 2015-01-10 ``@ddalle``: v1.1; add format option
            * 2024-09-24 ``@ddalle``: v2.0; CaseArchivist
        """
        # Get *TarViz* option
        v = self.opts.get_opt("TarViz", vdef=True)
        # Exit if not set
        if (not v) or (v == "none"):
            return
        # Initialize settings
        self.begin("restart", test)
        # Use regexs for now
        search_method = self.opts.get_SearchMethod()
        self.opts.set_SearchMethod("glob")
        # Globs and tarballs
        viz_globs = (
            'Components.i.[0-9]*.stats',
            'Components.i.[0-9]*',
            'cutPlanes.[0-9]*',
            'pointSensors.[0-9]*',
            'lineSensors.[0-9]*',
        )
        viz_tarnames = (
            'Components.i.stats',
            'Components.i',
            'cutPlanes',
            'pointSensors',
            'lineSensors',
        )
        # Loop through globs
        for fglob, tarname in zip(viz_globs, viz_tarnames):
            # Create tar ball
            self.tar_local(tarname, {fglob: 0})
            # Remove files
            matchdict = self.search(fglob)
            # Delete the files, except most recent
            self.delete_files(matchdict, 1)
        # Reset search method
        self.opts.set_SearchMethod(search_method)

    # Clear check files created during start/stop running
    def clean_checkfiles(self, istart: int = 0, test: bool = False):
        r"""Clear check files that were created since iteration *istart*

        :Call:
            >>> a.clean_checkfiles(istart=0)
        :Inputs:
            *a*: :class:`CaseArchivist`
                Archive controller for one case
            *istart*: :class:`int`
                Do not delete check files prior to iteration *istart*
            *test*: ``True`` | {``False``}
                If ``True``, log which files would be deleted, but don't
                actually delete, copy, or tar anything
        :Versions:
            * 2016-03-04 ``@ddalle``: v1.0 (``cape.pycart.manage``)
            * 2024-09-24 ``@ddalle``: v2.0; CaseArchivist
        """
        # Get number of files to keep
        nkeep = self.opts.get_opt("NCheckPoint")
        # Exit if non-positive
        if nkeep is None or nkeep < 0:
            return
        # Initialize settings
        self.begin("restart", test)
        # Use regexs for now
        search_method = self.opts.get_SearchMethod()
        self.opts.set_SearchMethod("regex")
        # Get the check.*  files
        filenames = self.search("check.[0-9]+")['']
        filenames.sort()
        # Loop through the glob except for the last *nkeep* files
        for fc in filenames[:-nkeep]:
            # Iteration number
            i = int(fc.split('.')[1])
            # Check if it is a recent enough iteration
            if i <= istart:
                continue
            # Delete the file
            self.delete_file(fc)
        # Get the check.*  files
        filenames = self.search("check.[0-9]+.td")['']
        filenames.sort()
        # Loop through the glob except for the last *nkeep* files
        for fc in filenames[:-nkeep]:
            # Iteration number
            i = int(fc.split('.')[1])
            # Check if it is a recent enough iteration
            if i <= istart:
                continue
            # Delete the file
            self.delete_file(fc)
        # Reset search method
        self.opts.set_SearchMethod(search_method)
