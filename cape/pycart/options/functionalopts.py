"""
:mod:`cape.pycart.options.Functional`: Objective Function Options
===================================================================

This module provides an interface for defining Cart3D's output
functional for output-based mesh adaptation.  The options read from
this file are written to the ``$__Design_Info`` section of
``input.cntl`1``. Each output function is a linear combination of terms
where each term can be a component of a force, a
component of a moment, or a point/line sensor.

The following is a representative example of a complex output function.

    .. code-block:: javascript

        "Functional": {
            // Term 1: normal force coefficient on "wings"
            "CN": {
                "type": "optForce",
                "force": 2,
                "frame": 0,
                "weight": 1.0,
                "compID": "wings",
                "J": 0,
                "N": 1,
                "target": 0.0
            },
            // Term 2: 0.5 times side force on "entire"
            "CY": {
                "type": "optForce",
                "force": 1,
                "frame": 0,
                "weight": 0.5,
                "compID": "entire"
            },
            // Term 3: 0.001 times point sensor called "p1"
            "p1": {
                "type": "optSensor",
                "weight": 0.001
            }
        }

See the :ref:`JSON "Functional" section <json-Functional>` for a
description of all available options.
"""


# Import options-specific utilities
from ...optdict import OptionsDict


# Class for output functional settings
class FunctionalOpts(OptionsDict):
    """Dictionary-based interface for Cart3D output functionals"""

    # Function to return all the optForce dicts found
    def get_optForces(self):
        """Return a list of output forces to be used in functional
        
        An output force has the following parameters:
        
            *type*: {``"optForce"``}
                Output type
            *compID*: :class:`str` | :class:`int`
                Name of component from which to calculate force/moment
            *force*: {``0``} | ``1`` | ``2``
                Axis number of force to use (0-based)
            *frame*: {``0``} | ``1``
                Force frame; ``0`` for body axes and ``1`` for stability axes
            *weight*: {``1.0``} | :class:`float`
                Weight multiplier for force's contribution to total
            *J*: {``0``} | ``1``
                Modifier for the force; not normally used
            *N*: {``1``} | :class:`int`
                Exponent on force coefficient
            *target*: {``0.0``} | :class:`float`
                Target value; functional is ``weight*(F-target)**N``
        
        :Call:
            >>> optForces = opts.get_optForces()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *optForces*: :class:`list` (:class:`dict`)
                List of output force dictionaries
        :Versions:
            * 2014-11-19 ``@ddalle``: First version
        """
        # Initialize output
        optForces = {}
        # Loop through keys
        for k in self.keys():
            # Get the key value.
            v = self[k]
            # Check if it's a dict.
            if type(v).__name__ != "dict": continue
            # Check if it's a force
            if v.get('type', 'optForce') == 'optForce':
                # Append the key.
                optForces[k] = v
        # Output
        return optForces
        
    # Function to return all the optSensor dicts found
    def get_optSensors(self):
        """Return a list of output sensors to be used in functional
        
        An output sensor has the following parameters.  The name of the output
        sensor defines which point/line sensor is used; therefore it must match
        exactly a point/line sensor as defined in ``input.cntl``.
        
            *type*: {``"optSensor"``}
                Output type
            *weight*: {``1.0``} | :class:`float`
                Weight multiplier for force's contribution to total
            *J*: {``0``} | ``1``
                Modifier for the force; not normally used
            *N*: {``1``} | :class:`int`
                Exponent on force coefficient
            *target*: {``0.0``} | :class:`float`
                Target value; functional is ``weight*(F-target)**N``
        
        :Call:
            >>> optSensors = opts.get_optSensors()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *optSensors*: :class:`list` (:class:`dict`)
                List of output sensor dictionaries
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Initialize output
        optSensors = {}
        # Loop through keys.
        for k in self.keys():
            # Get the key value.
            v = self[k]
            # Check if it's a dict.
            if type(v).__name__ != "dict": continue
            # Check if it's a sensor.
            if v.get('type', 'optForce') == 'optSensor':
                # Append the key.
                optSensors[k] = v
        # Output
        return optSensors
        
    # Function to return all the optMoment_Point dicts found
    def get_optMoments(self):
        """Return a list of moment coefficients to be used in functional
        
        An output force has the following parameters:
        
            *type*: {``"optMoment""``} | ``"optMoment_point"``
                Output type
            *compID*: :class:`str` | :class:`int`
                Name of component from which to calculate force/moment
            *force*: {``0``} | ``1`` | ``2``
                Axis number of force to use (0-based)
            *frame*: {``0``} | ``1``
                Force frame; ``0`` for body axes and ``1`` for stability axes
            *weight*: {``1.0``} | :class:`float`
                Weight multiplier for force's contribution to total
            *J*: {``0``} | ``1``
                Modifier for the force; not normally used
            *N*: {``1``} | :class:`int`
                Exponent on force coefficient
            *target*: {``0.0``} | :class:`float`
                Target value; functional is ``weight*(F-target)**N``
        
        :Call:
            >>> optMoments = opts.get_optMoments()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *optMoments*: :class:`list` (:class:`dict`)
                List of output moment coefficient dictionaries
        :Versions:
            * 2015-05-14 ``@ddalle``: First version
        """
        # Initialize output
        optMoments = {}
        # Loop through keys.
        for k in self.keys():
            # Get the key value.
            v = self[k]
            # Check if it's a dict.
            if type(v).__name__ != "dict": continue
            # Check if it's a sensor.
            if v.get('type', 'optForce') in ['optMoment', 'optMoment_Point']:
                # Append the key.
                optMoments[k] = v
        # Output
        return optMoments
    
