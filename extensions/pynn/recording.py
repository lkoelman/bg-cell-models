"""
Utilities for recording NEURON traces using PyNN.

@author     Lucas Koelman

@date       20/03/2018
"""

from pyNN import errors
from pyNN import recording
from pyNN.neuron.recording import Recorder, recordable_pattern
import neuron
h = neuron.h

from common import nrnutil

logger = recording.logger


class TraceSpecRecorder(Recorder):
    """
    Extension of pyNN Recorder class for NEURON simulator that understands
    trace specifications in the format used by NetPyne

    @see        Based on NetPyne's Cell.RecordTraces in file:
                https://github.com/Neurosim-lab/netpyne/blob/master/netpyne/cell.py

    USAGE
    -----

        >>> from pyNN.neuron.populations import Population
        >>> from extensions.pynn.recording import TraceSpecRecorder
        >>> Population._recorder_class = TraceSpecRecorder
        >>> ... (recording setup code)
    """

    def record(self, variables, ids, sampling_interval=None):
        """
        Override the default record() method to accept trace specifications
        in the format used by NetPyne.

        @override   pyNN.recording.Recorder.record()

        @param      variables : iterable(tuple(str, <dict or str>))

                    Any iterable collection where the first element of each item
                    is the trace name, and the second element the trace
                    specification. The trace specification can be either a
                    string like the default PyNN variable names, or a dict
                    that specifies the trace in NetPyne format.
        
        USAGE
        -----

            >>> trace_specs = {
            >>>     'GP_cai':{'sec':'soma[0]','loc':0.5,'var':'cai'},
            >>>     'GP_ainf':{'sec':'soma[0]','loc':0.5,'mech':'gpRT','var':'a_inf'}, 
            >>>     'GP_r':{'sec':'soma[0]','loc':0.5,'mech':'gpRT','var':'r'},
            >>> }
            >>> pop.record(trace_specs.items())
        
        """
        logger.debug('Recorder.record(<%d cells>)' % len(ids))
        if sampling_interval is not None:
            if sampling_interval != self.sampling_interval and len(self.recorded) > 0:
                raise ValueError("All neurons in a population must be recorded with the same sampling interval.")

        ids = set([id for id in ids if id.local])

        for variable in variables:
            trace_name, trace_spec = variable
            # if not self.population.can_record(trace_spec):
            #     raise errors.RecordingError(trace_spec, self.population.celltype)
            
            # Get cells that aren't recording this trace yet
            new_ids = ids.difference(self.recorded[trace_name])

            self.recorded[trace_name] = self.recorded[trace_name].union(ids)
            self._record(variable, new_ids, sampling_interval)


    def _record_state_variable(self, cell, variable):
        """
        Record the variable specified by the object 'variable'.

        @override   pyNN.neuron.recording.Recorder._record_state_variable()

        @param      cell : ID._cell
                    Instantiated cell model (CellType.model(**params))

        @param      variable : tuple(str, <str or dict>)
                    A trace specifier consisting of the trace name as first
                    element and full trace specification as second element.
        """
        trace_name, trace_spec = variable
        recorded = False

        # First try to interpret spec as PyNN format (string)
        if hasattr(cell, 'recordable') and trace_spec in cell.recordable:
            hoc_var = cell.recordable[trace_spec]
        elif trace_spec == 'v':
            hoc_var = cell.source_section(0.5)._ref_v  # or use "seg.v"?
        elif trace_spec == 'gsyn_exc':
            hoc_var = cell.esyn._ref_g
        elif trace_spec == 'gsyn_inh':
            hoc_var = cell.isyn._ref_g
        elif isinstance(trace_spec, str):
            source, var_name = self._resolve_variable(cell, trace_spec)
            hoc_var = getattr(source, "_ref_%s" % var_name)
        else:
            # spec is in NetPyne format
            hoc_obj = cell.resolve_section(trace_spec['sec'])
            vec, marker = self._record_trace(hoc_obj, trace_spec, self.sampling_interval)
            recorded = True
            if marker is not None:
                _pp_markers = getattr(cell, '_pp_markers', [])
                _pp_markers.append(marker)
                cell._pp_markers = _pp_markers
        
        # Make NEURON Vector and record into it
        if not recorded:
            vec = h.Vector()
            if self.sampling_interval == self._simulator.state.dt:
                vec.record(hoc_var)
            else:
                vec.record(hoc_var, self.sampling_interval)

        cell.traces[trace_name] = vec
        
        # Record global time variable 't' if not recorded already
        if not cell.recording_time:
            cell.record_times = h.Vector()
            if self.sampling_interval == self._simulator.state.dt:
                cell.record_times.record(h._ref_t)
            else:
                cell.record_times.record(h._ref_t, self.sampling_interval)
            
            cell.recording_time += 1


    def _record_trace(self, hoc_obj, spec, rec_dt, duration=None):
        """
        Record the given traces from section

        WARNING: For multithreaded execution, section _must_ have POINT_PROCESS
                 associated with it to identify the tread. Hence, a PointProcessMark
                 (see ppmark.mod) will be inserted if no PP present.
        """
        pp_marker = None
        hoc_ptr = None  # pointer to Hoc variable that will be recorded
        pp = None       # Hoc POINT_PROCESS instance
        vec = None      # Hoc.Vector that will be recorded into

        # Get Section and segment
        if isinstance(hoc_obj, neuron.nrn.Segment):
            seg = hoc_obj
            sec = seg.sec
            if 'loc' in spec:
                seg = sec(spec['loc'])
        else:
            sec = hoc_obj
            seg = sec(spec['loc'])


        if 'loc' in spec: # hoc_obj is Section

            # Get pointer/reference to variable to record
            if 'mech' in spec:  # eg. soma(0.5).hh._ref_gna
                mech_instance = getattr(seg, spec['mech'])
                hoc_ptr = getattr(mech_instance, '_ref_'+spec['var'])
            else:
                # No mechanism. E.g. soma(0.5)._ref_v
                hoc_ptr = getattr(seg, '_ref_'+spec['var'])
            
            # find a POINT_PROCESS in segment to improve efficiency
            seg_pps = seg.point_processes()
            if any(seg_pps):
                pp = seg_pps[0]
            else:
                pp = h.PointProcessMark(seg)
                pp_marker = pp
        

        elif 'pointp' in spec: # hoc_obj is POINT_PROCESS
            # Look for the point process in Section
            seg_pps = seg.point_processes()
            for hobj in seg_pps:
                modname = nrnutil.get_mod_name(hobj)
                if spec['pointp'] == modname:
                    pp = hoc_obj
                    break
            
            if pp is None:
                raise ValueError("Could not find point process '{}' "
                    "in segment {}".format(spec['pointp'], seg))
            
            hoc_ptr = getattr(pp, '_ref_'+spec['var'])


        else: # global vars, e.g. h.t
            hoc_ptr = getattr(h, '_ref_'+spec['var'])


        # Record from pointer into Vector
        if duration is not None:
            vec = h.Vector(duration/rec_dt+1).resize(0)
        else:
            vec = h.Vector()

        if pp is not None:
            vec.record(pp, hoc_ptr, rec_dt)
        else:
            vec.record(hoc_ptr, rec_dt)

        return vec, pp_marker