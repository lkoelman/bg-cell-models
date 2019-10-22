COMMENT
A VecStim is an artificial spiking cell that generates
events at times that are specified in a Vector.

A maintained version can be found at 
github.com/neuronsimulator/nrn/master/share/examples/nrniv/netcon/vecevent.mod

HOC Example:
-----------

// assumes spt is a Vector whose elements are all > 0
// and are sorted in monotonically increasing order
objref vs
vs = new VecStim()
vs.play(spt)
// now launch a simulation, and vs will produce spike events
// at the times contained in spt

Python Example:
--------------

from neuron import h
spt = h.Vector(10).indgen(1, 0.2)
vs = h.VecStim()
vs.play(spt)

def pr():
  print (h.t)

nc = h.NetCon(vs, None)
nc.record(pr)

cvode = h.CVode()
h.finitialize()
cvode.solve(20)

ENDCOMMENT

NEURON {
	THREADSAFE
	ARTIFICIAL_CELL VecStim
	POINTER ptr
}

ASSIGNED {
	index
	etime (ms)
	ptr
}


INITIAL {
	index = 0
	element()
	if (index > 0) {
		net_send(etime - t, 1)
	}
}

NET_RECEIVE (w) {
	if (flag == 1) {
		net_event(t)
		element()
		if (index > 0) {
			net_send(etime - t, 1)
		}
	}
}

DESTRUCTOR {
VERBATIM
	void* vv = (void*)(_p_ptr);  
        if (vv) {
		hoc_obj_unref(*vector_pobj(vv));
	}
ENDVERBATIM
}

PROCEDURE element() {
VERBATIM	
  { void* vv; int i, size; double* px;
	i = (int)index;
	if (i >= 0) {
		vv = (void*)(_p_ptr);
		if (vv) {
			size = vector_capacity(vv);
			px = vector_vec(vv);
			if (i < size) {
				etime = px[i];
				index += 1.;
			}else{
				index = -1.;
			}
		}else{
			index = -1.;
		}
	}
  }
ENDVERBATIM
}

PROCEDURE play() {
VERBATIM
	void** pv;
	void* ptmp = NULL;
	if (ifarg(1)) {
		ptmp = vector_arg(1);
		hoc_obj_ref(*vector_pobj(ptmp));
	}
	pv = (void**)(&_p_ptr);
	if (*pv) {
		hoc_obj_unref(*vector_pobj(*pv));
	}
	*pv = ptmp;
ENDVERBATIM
}
