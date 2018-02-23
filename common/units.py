"""
NEURON unit interopability using 'Pint' package.

@see    http://pint.readthedocs.io
"""

import re
import pint
ureg = pint.UnitRegistry()
Q_ = Quantity = ureg.Quantity

from neuron import h

# Define NEURON units that are not in pint's default units
# TODO: check/add full list of NMODL/modlunit units
ureg.define('Ohm = ohm')
ureg.define('mho = 1/ohm')
ureg.define('cm2 = cm^2')


def get_nrn_units(nrn_obj, attr, hoc_classname=None):
    """
    Get units of NEURON variable as a pint.Quantity object.

    @param  nrn_obj : nrn.HocObject
            NEURON object

    @param  attr : str
            Variable name of NEURON object

    @param  hoc_classname : str
            if 'attr' is not a mechanism, Section, or global Hoc variable name,
            specify the Hoc classname here. E.g. to set 'Exp2Syn.tau1',
            use attr='tau1' and hoc_classname='Exp2Syn'

    @return q : pint.Quantity
            Units of given NEURON variable

    @throws err : ValueError
            Error thrown if variable name is not found by Hoc.
    """
    # TODO: see h.units() documentation: extract classname from nrn_obj
    #       so we can pass it to h.units('classname.attr'), e.g. if nrn_obj
    #       is h.Exp2Syn
    if hoc_classname is None:
        full_attr = attr
    else:
        full_attr = '{}.{}'.format(hoc_classname, attr)
    try:
        nrn_units = h.units(full_attr) # can return '' -> treated as dimensionless by Pint
    except RuntimeError as e:
        if e.args[0] == 'hoc error':
            raise ValueError('Attribute {} not recognized by hoc'.format(attr))

    units_nodash = nrn_units.replace('-', '*')
    units_exponents = re.sub(r'([a-zA-Z]+)(\d)',r'\1^\2', units_nodash)
    target_units = ureg(units_exponents)
    return target_units


def to_nrn_units(nrn_obj, attr, quantity, hoc_classname=None):
    """
    Convert quantity to same units as NEURON variable.

    @see    get_nrn_units() for description of arguments

    @param  quantity : pint.Quantity
            Quantity object consisting of value and units

    @throws err : pint.errors.DimensionalityError
            Error thrown in case of dimensionality (not units) mismatch.

    @return q : pint.Quantity
            Original quantity converted to units of the NEURON object.

    EXAMPLE
    -------
    
        > import neuron, units
        > quantity = units.Quantity(value, param_spec['units'])
        > converted_quantity = units.to_nrn_units(neuron.h, 'gnabar_hh', quantity)
        > value = converted_quantity.magnitude
    """
    target_units = get_nrn_units(nrn_obj, attr, hoc_classname)
    return quantity.to(target_units)


def compatible_units(nrn_obj, attr, quantity, hoc_classname=None):
    """
    Check if units of given quantity are compatible with those of NEURON variable.

    @see    get_nrn_units() for description of arguments

    @param  quantity : pint.Quantity
            Quantity object consisting of value and units

    @return compatible : bool
            True if units are compatible.
    """
    try:
        to_nrn_units(nrn_obj, attr, quantity, hoc_classname)
    except pint.errors.DimensionalityError:
        return True
    else:
        return False


def same_units(nrn_obj, attr, quantity, hoc_classname=None):
    """
    Check if units of given quantity are the same as those of NEURON variable.

    @see    get_nrn_units() for description of arguments

    @param  quantity : pint.Quantity
            Quantity object consisting of value and units

    @return same : bool
            True if units are the same
    """
    nrn_quantity = get_nrn_units(nrn_obj, attr, hoc_classname)
    return nrn_quantity.units == quantity.units


def set_nrn_quantity(nrn_obj, attr, quantity, hoc_classname=None):
    """
    Set attribute of NEURON object using a pint.Quantity object that includes
    a value and units.

    @see    get_nrn_units() for description of arguments

    @param  quantity : pint.Quantity
            Quantity object consisting of value and units

    @throws err : pint.errors.DimensionalityError
            Error thrown in case of dimensionality (not units) mismatch.
    """
    val_converted = to_nrn_units(nrn_obj, attr, quantity, hoc_classname)
    setattr(nrn_obj, attr, val_converted.magnitude)

