TITLE Activity-dependent weight adjuster using Van Rossum et al. (2000)

COMMENT

Homeostatic plasticity (activity-dependent weight scaling) according to 
equation (3) in article Van Rossum et al. (2000) - Stable hebbian Learning 
from Spike Timing-Dependent Plasticity.


Example
-------

TODO: Python example usage

>>> # make HPWA in section
>>> # make netcon from post_cell to HPWA (set threshold for biophysical cell)
>>> # add weight refs

ENDCOMMENT


VERBATIM
// Definitions for synaptic scaling procs
void raise_activity_sensor(double time);
void decay_activity_sensor(double time);
void update_scale_factor(double time);
double get_avg_activity();


// Linked list node for storing refs to observed hoc variables
typedef struct node {
    double id;          // user-specified identifier
    double* hoc_ref;    // hoc reference to observed variable
    struct node* next;  // next node in linked list
} WeightRef;

#define WREFS_INH (*((WeightRef**)&(_p_inh_weight_refs)))
#define WREFS_EXC (*((WeightRef**)&(_p_exc_weight_refs)))

ENDVERBATIM


NEURON {
    POINT_PROCESS VanRossumHPWA
    POINTER temp_wref        : temporary variable for passing weight reference

    GLOBAL scaleinhib   : Set to TRUE (1) for I-cell scaling in addition to E-cell scaling. Default is off (0).
    GLOBAL activitytau  : Activity time constant (ms^-1)  
    GLOBAL activitybeta : Scaling strength constant (s^-1 Hz^-1)
    GLOBAL activitygamma: Scaling update constant (s^-2 Hz^-1)
}


PARAMETER {
    goal_activity = 10: target firing rate
    enabled = 1
    scaleinhib = 0      : Whether or not we should scale I cells as well as E cells

    activitytau = 100.0e3   : Activity sensor time constant (ms^-1) (van Rossum et al., 2000)
    activityoneovertau      : Store 1 / tau for faster calculation
    activitybeta = 4.0e-8   : was e-5 Scaling strength constant (s^-1 Hz^-1) (van Rossum et al., 2000)
    activitygamma = 1.0e-10 : was e-7 Scaling update constant (s^-2 Hz^-1) (van Rossum et al., 2000)
}

ASSIGNED {
    temp_wref : temporary variable for passing weight reference
    exc_weight_refs : refs to excitatory weights
    inh_weight_refs : refs to inhibitory weights

    max_err     : max error value
    max_scale   : max scaling factor
    scalefactor : default scaling factor for this cell's excitatory synapses
    lastupdate  : time of last activity sensor decay / spike udpate
    activity    : activity sensor value ('a' in Van Rossum et al. (2000))
    activity_integral_err : intergral of cell's activity divergence from target
    spkcnt      : spike count of post-synaptic cell
    
}

: Constructor, called only once
CONSTRUCTOR {
VERBATIM {

    WREFS_EXC = emalloc(sizeof(WeightRef))
    WREFS_EXC->id = 0.0
    WREFS_EXC->hoc_ref = NULL;
    WREFS_EXC->next = NULL;

    WREFS_INH = emalloc(sizeof(WeightRef))
    WREFS_EXC->id = 0.0
    WREFS_INH->hoc_ref = NULL;
    WREFS_INH->next = NULL;

}
ENDVERBATIM
}


DESTRUCTOR {
VERBATIM {
    // Free linked lists
    WeightRef* current = WREFS_EXC
    WeightRef* next_node;
    while (current != NULL) {
        next_node = current->next;
        free(current);
        current = next_node;
    }

    WeightRef* current = WREFS_INH
    WeightRef* next_node;
    while (current != NULL) {
        next_node = current->next;
        free(current);
        current = next_node;
    }

}
ENDVERBATIM
}


: TODO: is there any reason to put this in CONSTRUCTOR instead?
INITIAL {

    activity = 0    : Sensor for this cell's recent activity (default 0MHz i.e. cycles per ms)
    max_err = 0     : Max error value
    max_scale = 4   : Max scaling factor
    lastupdate = 0  : Time of last activity sensor decay / spike update
    scalefactor = 1.0 : Default scaling factor for this cell's AMPA synapses
    : goal_activity = -1 : Cell's target activity (MHz i.e. cycles per ms)
    activity_integral_err = 0.0 : Integral of cell's activity divergence from target activity
    activityoneovertau = 1.0 / activitytau
}


: Receive spikes from post-synaptic cell or self-scheduled updates.
: 
: @param    w : float
:           NetCon weight, w > 0 indicates spikes from post-synaptic cell.
NET_RECEIVE(w) {
: TODO schedule updates in case cell doesn't fire
VERBATIM
    decay_activity_sensor(t); // Allow activity sensor to decay on every update

    if (enabled) {
        if (goal_activity < 0) {
            // If scaling has just been turned on, set goal activity to historical average firing rate This is only meaningful if sensor has had a chance to measure correct activity over a relatively long period of time, so don't call setscaling(1) until at least ~800s.
            //ip->goal_activity = get_avg_activity();
            goal_activity = activity; // Take current activity sensor value
            //ip->max_err = ip->goal_activity * 0.5; // Error value saturates at +- 50% of goal activity rate
        }
        // Only update if cell is not inhib OR we are scaling all I+E cells
        update_scale_factor(t); // Run synaptic scaling procedure to find scalefactor
    }
    lastupdate = t; // Store time of last update
ENDVERBATIM

    if (w > 0) {
VERBATIM
        raise_activity_sensor(t); // Update activity sensor
ENDVERBATIM
    }

VERBATIM
    if (enabled) {
        // TODO: all excitatory wref *= scalefactor
        // TODO: all inhibitory wref *= 1 / scalefactor
    }
ENDVERBATIM
}


FUNCTION add_wref(excitatory, identifier) {
VERBATIM
    uint32_t group_id = (uint32_t) _lgrp_id;
    assert(group_id < MAX_GROUPS);
    
    GETGROUPS; // set local var TargetGroup** grps
    TargetGroup* groups = *grps;
    TargetGroup  target = groups[group_id];

    // Look for end of linked list and append controlled variable
    WeightRef* current;
    if (_lexcitatory) {
        current = WREFS_EXC;
    } else {
        current = WREFS_INH;
    }
    while (current->next != NULL) {
        current = current->next;
    }

    current->next = emalloc(sizeof(WeightRef));
    current->next->id = _lidentifier;
    current->next->hoc_ref = _p_temp_wref;
    current->next->next = NULL;

    // fprintf(stderr, "Added ref to group %d\n", group_id);
ENDVERBATIM
}


VERBATIM
/**
 * Get average spike rate of the cell since t = 0.
 */
double get_avg_activity () {
  return spkcnt / t;
}
ENDVERBATIM

VERBATIM 
void raise_activity_sensor (double time) {
  // Update the cell's activity sensor value, assuming this function has been called at the same
  // time as a spike at time t
  // REQUIRES: time of current spike in ms
  // ENSURES: returns activity value in MHz (due to ms timing)
  
  // Raise the activity by (-a + 1) / tau
  activity = activity + (-activity + 1.0) / activitytau;
}
ENDVERBATIM

VERBATIM
void decay_activity_sensor (double time) {
  // Decay the cell's activity sensor value according to the time since last decay update.
  // In algorithm described in van Rossum et al. (2000), this is called every discrete timestep t
  // But this procedure is only called on NET_RECEIVE events, so we need to decay
  // taking into account the time since the last decay operation.

  // a_t = a_t0 * e(-(1/tau * t-t0))
  activity = activity * exp(-activityoneovertau * (time - lastupdate));
}
ENDVERBATIM


VERBATIM
/**
 * Implements weight scaling according to van Rossum et al. (2000).
 * Calculates dw/dt and updates w according to equation (3), p. 2.
 */
void update_scale_factor (double time) {
  // Get difference between goal and current activity
  double err = goal_activity - activity;

  // Set scalefactor
  scalefactor += (activitybeta * scalefactor * err + activitygamma * scalefactor * activity_integral_err);

  // Bound scalefactor to max_scale to prevent Inf values
  if (scalefactor > max_scale) {
    scalefactor = max_scale;
  }

  // Calculate integral error term between sensor and target activity for next time (t')
  double timecorrection = time - lastupdate;
  // e.g. If last update was 1ms ago, then the time correction = 1
  // If last update was 0.1ms ago correction = 0.1, so the accumulated error will be much smaller
  // If it's been a long time since the last update, the error will be correspondingly much larger

  activity_integral_err += (err * timecorrection);
}
ENDVERBATIM