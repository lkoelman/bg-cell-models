"""
Common functionality for folding algorithms.

@author     Lucas Koelman

@date       04-01-2018
"""

from enum import Enum, unique

@unique
class ReductionMethod(Enum):
    Rall = 0
    Stratford = 1           # Stratford, K., Mason, A., Larkman, A., Major, G., and Jack, J. J. B. (1989) - The modelling of pyramidal neurones in the visual cortex
    BushSejnowski = 2       # Bush, P. C. & Sejnowski, T. J. Reduced compartmental models of neocortical pyramidal cells. Journal of Neuroscience Methods 46, 159-166 (1993).
    Marasco = 3             # Marasco, A., Limongiello, A. & Migliore, M. Fast and accurate low-dimensional reduction of biophysically detailed neuron models. Scientific Reports 2, (2012).


class FoldingAlgorithm(object):
    """
    Abstract base class for folding algorithms.
    """

    def preprocess_impl(reduction):
        """
        Preprocess cell for folding reduction.

        Calculates properties needed during reduction and saves them on reduction
        object and individual SectionRef instances.

        @param  reduction   FoldReduction object

        @effect             - assign identifiers to each sections
                            - compute electrotonic properties in each segment of cell
                            - determine interpolation path for channel distributions and save them
        """
        raise NotImplementedError(
                "Virtual method of abstract base class FoldingAlgorithm not implemented.")

    def prepare_folds_impl(reduction):
        """
        Prepare next collapse operation: assign topology information
        to each Section.

        (Implementation of interface declared in reduce_cell.CollapseReduction)
        """
        raise NotImplementedError(
                "Virtual method of abstract base class FoldingAlgorithm not implemented.")


    def calc_folds_impl(reduction, i_pass, Y_criterion):
        """
        Collapse branches at branch points identified by given criterion.
        """
        raise NotImplementedError(
                "Virtual method of abstract base class FoldingAlgorithm not implemented.")


    def make_folds_impl(reduction):
        """
        Make equivalent Sections for branches that have been folded.
        """
        raise NotImplementedError(
                "Virtual method of abstract base class FoldingAlgorithm not implemented.")


    def postprocess_impl(reduction):
        """
        Post-process cell for Marasco reduction.

        (interface declared in reduce_cell.CollapseReduction)

        @param  reduction       reduce_cell.CollapseReduction object
        """
        raise NotImplementedError(
                "Virtual method of abstract base class FoldingAlgorithm not implemented.")
