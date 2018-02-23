"""
Cell mechanism
"""

from __future__ import division
import re
import sympy
import numpy as np

import common.units as pint_units
import sympy.physics.units as sympy_units
from sympy.printing.str import StrPrinter
# NOTE:
# Pint quantity has attributes dimensionality, units, magnitude
# Sympy quantity has attributes dimensionality, scale_factor

# See http://docs.sympy.org/latest/modules/numeric-computation.html
# and backends with examples in https://github.com/sympy/sympy/blob/master/sympy/printing
# from sympy.utilities.lambdify import lambdify # sympy.lambdify
from sympy.utilities.autowrap import ufuncify
# from sympy.printing.theanocode import theano_function
# from sympy.printing.llvmjitcode import llvm_callable


# Symbolic computation with Sympy
exp = sympy.exp
log = sympy.log
# pow is built-in function

# Shorthands for units
ureg = pint_units.ureg # so we can use variable name 'units'
mV = ureg('mV')
ms = ureg('ms')

# Symbolic variable for membrane voltage
v = sympy.symbols('v')

# TODO: find good way to combine symbolic expressions and units.
#       You cannot expr.subs(quantity), but you can multiply
#       a quantity with a symbolic expression and retrieve it via
#       '.magnitude' property. However, we want to ensure that when we
#       substitute quantities into a symbolic expression, all the units
#       match up.
# SEE:  http://docs.sympy.org/latest/modules/physics/units/index.html
#       https://github.com/sympy/sympy/tree/master/sympy/physics/units/tests
#       Possibly use sympy built-in unit system.

# METHOD 1
#       Make wrapper object that keeps for each symbolic variable, the
#       Symbol and units object. Can do this by subclassing Symbol/Quantity,
#       depending on which behaviour you want to keep. Or by creating true
#       wrapper class ~Neo.spiketrain. Then to check units you can do one of
#       the following: lambdify the expression, pass quantities and see
#       that units match (no error) and final units are correct. You can
#       lambdify by finding intersection with expr.free_symbols and providing
#       that as arguments list, then give params in that order.
#       This method allows you to use sympy or external units/quantities module.

# METHOD 2
#       Use sympy.physics.units.Quantity object vor all variables.
#       See http://docs.sympy.org/latest/modules/physics/units/examples.html
#       E.g. first you can subs() q.dimensions and see that final dimensionality
#       is correct, then subs() q.scale_factor and check that final scale is correct.

# def SymbolicQuantity(x, units=None):
#     """
#     Wrapper for sympy.Symbol
#     """
#     return Quantity(sympy.symbols(x), units)

class SymbolicQuantity(sympy.Symbol):
    """
    Wrapper for sympy.Symbol that also stores units.
    """
    def __new__(*args, **kwargs):
        """
        Symbol name is passed via __new__ so need to override.
        """
        cls = args[0]
        return super(SymbolicQuantity, cls).__new__(*args, **kwargs)

    def __init__(self, name, default_value=1.0, units=None):

        super(SymbolicQuantity, self).__init__()
        self._default_value = default_value

        # Create Pint units
        unit_str = units
        self._pint_units = pint_units.Quantity(default_value, unit_str)
        base_units = self._pint_units.to_base_units()
        dims_exponents = base_units.dimensionality.items()
        
        # Create SymPy units
        if len(dims_exponents) == 0:
            sp_dims = 1 # '1' is dimensionless in sympy
        else:
            dim0, exp0 = dims_exponents[0]
            sp_dims = getattr(sympy_units.dimensions, dim0.strip('[]')) ** exp0
        for dim_str, dim_exp in dims_exponents[1:]:
            sp_dims *= getattr(sympy_units.dimensions, dim_str.strip('[]')) ** dim_exp
        
        sp_scale = base_units.magnitude
        self._sympy_units = sympy_units.Quantity(name, sp_dims, sp_scale)


class QuantitativeExpr(sympy.Expr):
    """
    Wrapper around sympy expressions.
    """

    def __new__(*args, **kwargs):
        """
        Symbol name is passed via __new__ so need to override.
        """
        cls = args[0]
        return super(QuantitativeExpr, cls).__new__(*args, **kwargs)


    def __init__(self, expr):
        print("I got:\n" + str(expr))
        raw_expr = expr.as_expr() # getattr(expr, 'expr', expr)
        self.expr = raw_expr # Original wrapped expression
        
        return super(QuantitativeExpr, self).__init__() # just calles object.__init__

    def as_expr(self):
        """
        Return as sympy Expression

        @note   A SymPy expression e is e.func(*e.args)

        @see    http://docs.sympy.org/latest/tutorial/manipulation.html#recursing-through-an-expression-tree
        """
        expr = self.expr
        return expr.func(*[a.as_expr() for a in expr.args])


    # def __repr__(self):
    #     return repr(self.as_expr())
    #     # in sympy.core.basic.py :
    #     # from sympy.printing import sstr
    #     # return sstr(self.expr, order=None)

    # def __str__(self):
    #     return str(self.as_expr())

    ############################################################################
    # All mathematical operators should be done against self.expr so that
    # result is of same type as other operand
    ############################################################################

    # def __abs__(self):
    #     return abs(self.expr)

    # def __neg__(self):
    #     return -self.expr

    # def __div__(self, x):
    #     return self.expr / getattr(x, 'expr', x)

    # def __rdiv__(self, x):
    #     return getattr(x, 'expr', x) / self.expr

    # def __truediv__(self, x):
    #     return self.expr / getattr(x, 'expr', x)

    # def __rtruediv__(self, x):
    #     return getattr(x, 'expr', x) / self.expr

    # def __mul__(self, x):
    #     return self.expr * getattr(x, 'expr', x)

    # def __rmul__(self, x):
    #     return getattr(x, 'expr', x) * self.expr
    
    # def __add__(self, x):
    #     return self.expr + getattr(x, 'expr', x)

    # def __radd__(self, x):
    #     return getattr(x, 'expr', x) + self.expr

    # def __sub__(self, x):
    #     return self.expr - getattr(x, 'expr', x)

    # def __rsub__(self, x):
    #     return getattr(x, 'expr', x) - self.expr

    # def __pow__(self, x):
    #     return self.expr ** getattr(x, 'expr', x)

    # def __or__(self, x):
    #     return self.expr or getattr(x, 'expr', x)

    # def __eq__(self, x):
    #     return self.expr == getattr(x, 'expr', x)

    # def __ne__(self, x):
    #     return self.expr != getattr(x, 'expr', x)

    # def __gt__(self, x):
    #     return self.expr > getattr(x, 'expr', x)

    # def __ge__(self, x):
    #     return self.expr >= getattr(x, 'expr', x)

    # def __lt__(self, x):
    #     return self.expr < getattr(x, 'expr', x)

    # def __le__(self, x):
    #     return self.expr <= getattr(x, 'expr', x)


class MechanismType(type):
    """
    Metaclass for cell mechanism.

    @see    https://docs.python.org/3/reference/datamodel.html#metaclasses

    @note   metaclass returns a new type for classes that use it
    """

    def __new__(mcls, name, bases, namespace, **kwds):
        """
        Create a class object for a class definition that has MechanismType
        as its metaclass.

        @param      mcls : type
                    the metaclass object (prototype describing type MechanismType)

        @return     cls : MechanismType
                    a new Class object (an instance of type MechanismType)

        @note       This method is called once every time a class is defined with 
                    MechanismType as its metaclass. Whereas hat class' __new__ 
                    method is called every time an instance is created.
        """
        # Store definitions of states etc. in structured form
        newattrs = {}
        newattrs['_MECH_PARAMS'] = {}
        newattrs['_MECH_STATE_VARS'] = {}
        newattrs['_MECH_STATE_DERIV'] = {} # expression for derivative of state variable
        newattrs['_MECH_STATE_INF'] = {} # (optional) expression for steady state value of state
        newattrs['_MECH_STATE_TAU'] = {} # (optional) expression for time constant of state
        newattrs['_MECH_MEMB_CURRENTS'] = {} # (optional) expression for time constant of state

        for attr_name, attr_val in namespace.iteritems():
            # Process special attributes created in class scope
            attr_type = getattr(attr_val, '_mech_attr_type', None)
            if attr_type == 'parameter':
                newattrs['_MECH_PARAMS'][attr_val._param_name] = attr_val
            
            elif attr_type == 'statevar':
                newattrs['_MECH_STATE_VARS'][attr_val._state_name] = attr_val
            
            elif attr_type == 'statevar_steadystate':
                newattrs['_MECH_STATE_INF'][attr_val._state_name] = attr_val
            
            elif attr_type == 'statevar_timeconst':
                newattrs['_MECH_STATE_TAU'][attr_val._state_name] = attr_val
            
            elif attr_type == 'statevar_derivative':
                newattrs['_MECH_STATE_DERIV'][attr_val._state_name] = attr_val

            elif attr_type == 'current':
                ion_currents = newattrs['_MECH_MEMB_CURRENTS'].setdefault(attr_val._ion_species, {})
                if attr_name in ion_currents.keys():
                    raise ValueError('Duplicate current name \'{}\' for ion \'{}\'').format(
                        attr_name, attr_val._ion_species)
            
            # TODO: is it wise to also keep them available in class namespace? Maybe remove this after debugging
            newattrs[attr_name] = attr_val
        
        return super(MechanismType, mcls).__new__(mcls, name, bases, newattrs)


    def __init__(cls, name, bases, namespace):
        """
        @param  cls : type
                the class object returned by __new__
        """
        return super(MechanismType, cls).__init__(name, bases, namespace)


class MechanismBase:
    """
    Base class for Cell Mechanism
    """
    __metaclass__ = MechanismType


#    def __new__(cls, *args, **kwargs):
#        """
#        @param      cls : type
#                    the class object created by metaclass
#
#        @return     self : MechanismBase
#                    new instance of type cls
#        """
#        return super(MechanismBase, cls).__new__(cls, *args, **kwargs)


    def __init__(*args, **kwargs):
        self = args[0]
        super(MechanismBase, self).__init__()
        # Copy class parameters so they can be modified
        self._IMECH_PARAMS = {
            p : MechanismBase.define_parameter(
                                v._param_name,
                                v._default_value,
                                str(v._pint_units.units)) 
                for p,v in self._MECH_PARAMS.iteritems()
        }


    @staticmethod
    def define_parameter(name, default_value, units):
        param = SymbolicQuantity(name, default_value=default_value, units=units)
        param._param_name = name
        param._mech_attr_type = 'parameter'
        return param


    @staticmethod
    def define_state(name, power):
        state = SymbolicQuantity(name)
        state._mech_attr_type = 'statevar'
        state._state_name = name
        state.power = power
        return state


    @staticmethod
    def state_steadystate(expr, state):
        expr = QuantitativeExpr(expr)
        expr._mech_attr_type = 'statevar_steadystate'
        expr._state_name = state
        return expr


    @staticmethod
    def state_timeconst(expr, state):
        expr = QuantitativeExpr(expr)
        expr._mech_attr_type = 'statevar_timeconst'
        expr._state_name = state
        return expr


    @staticmethod
    def state_derivative(expr, state):
        expr = QuantitativeExpr(expr)
        expr._mech_attr_type = 'statevar_derivative'
        expr._state_name = state
        return expr


    @staticmethod
    def define_current(expr, ion):
        expr = QuantitativeExpr(expr)
        expr._mech_attr_type = 'current'
        expr._ion_species = ion
        return expr


    def plot_steadystate_gating(self):
        """
        Plot steady state values and time constants of gating vars
        """
        # TODO: collect gating functions and plot
        import matplotlib.pyplot as plt
        for state_name in self._MECH_STATE_VARS.keys():

            # Get steady state and time constant
            s_inf = self._MECH_STATE_INF.get(state_name, None)
            s_tau = self._MECH_STATE_TAU.get(state_name, None)
            if s_inf is None and s_tau is None:
                continue

            # Convert quantity to expression
            fn_inf = self.make_func(s_inf)
            fn_tau = self.make_func(s_tau)
            v_range = np.linspace(-100, 100, 400)

            plt.figure()
            plt.suptitle('State variable {}'.format(state_name))
            plt.plot(v_range, fn_inf(v_range), label=r'$x_{\inf}$')
            plt.plot(v_range, fn_tau(v_range), label=r'$\tau_x$')
            plt.xlabel('V (mV)')
            plt.legend()

            # fig, axes = plt.subplots(2,1)
            # axes[0].plot(v_range, fn_inf(v_range))
            # axes[0].text(.5, .5, '${}$'.format(sympy.latex(s_inf.expr)),
            #                             horizontalalignment='right',
            #                             verticalalignment='top')
            # axes[1].plot(v_range, fn_tau(v_range))
            # axes[1].text(.5, .5, '${}$'.format(sympy.latex(s_tau.expr)),
            #                             horizontalalignment='right',
            #                             verticalalignment='top')

        plt.show(block=False)


    def make_func(self, expr):
        """
        Make ufunc by substituting parameters
        """
        func = sympy.lambdify(v, expr.as_expr(), 'numpy')
        # func = ufuncify(v, raw_expr)
        return func


    def compile_mod():
        # TODO: compile Python representation of channel to .mod file
        pass


    def compile_nrn_c():
        # TODO: compile Python representation directly to C file. See mod2c
        pass


    def compile_nrn_lib():
        # TODO: Compile to library that can be linked into libnrnmech.so, see nrnivmodl script
        #       This will require making an adapted nrnivmodl script.
        pass


    def register_nonvint():
        """
        Register with NEURON's nonvint_block_supervisor

        @see    example at
                https://github.com/nrnhines/nrntest/blob/master/nrniv/rxd/kchanarray.py
                or see RxD module
        """
        # TODO: base on kchanarray example, except pass compiled C-functions as callbacks
        #       rather than slow Python functions
        pass


# Alias static methods
mech = MechanismBase
Parameter = mech.define_parameter
State = mech.define_state
Current = mech.define_current
state_steadystate = mech.state_steadystate
state_timeconst = mech.state_timeconst
state_derivative = mech.state_derivative


class HHNaChannel(MechanismBase):
    """
    Hodgkin-Huxley Na channel mechanism.
    """

    # TODO: use __new__ or metaclass to process attributes, see example https://github.com/NeuralEnsemble/python-neo/blob/master/neo/core/spiketrain.py

    # PARAMETER block
    gnabar = Parameter('gnabar', 0.12, 'S/cm^2')
    ENa = Parameter('gnabar', 50.0, 'mV')

    # STATE block
    # TODO: make sympy symbol + register power
    m = State('m', power=3)
    h = State('h', power=1)

    # RHS_EXPRESSIONS
    ## m gate
    alpha_m = 0.1 * -(v+40) / (exp(-(v+40)/(10)) - 1)
    beta_m = 4.0 * exp(-(v+65)/(18))
    minf = state_steadystate(alpha_m / (alpha_m + beta_m), state='m')
    tau_m = state_timeconst(1.0 / (alpha_m + beta_m), state='m')

    ## h gate
    alpha_h = 0.07 * exp(-(v+65)/(20))
    beta_h = 1.0 / (exp(-(v+35)/(10)) + 1)
    hinf = state_steadystate(alpha_h / (alpha_h + beta_h), state='h')
    tau_h = state_timeconst(1.0 / (alpha_h + beta_h), state='h')

    # DERIVATIVE block
    dm = state_derivative((minf - m) / tau_m, state='m')
    dh = state_derivative((hinf - h) / tau_h, state='h')

    # BREAKPOINT block
    ina = Current(gnabar * m**m.power * h**h.power * (v-ENa), ion='na')

if __name__ == '__main__':
    # Create channel
    chan = HHNaChannel()
    chan.plot_steadystate_gating()