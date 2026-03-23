You can add LTC transit priority as a thin “Transit Priority Module” that plugs into your existing SURTRAC-style / multi-agent controller and talks to detectors plus City of London’s TIMMS/TMC stack.

Below is the markdown file you asked for.

Transit Signal Priority (TSP) Architecture for DeusNegotiatio at Oxford–Hyde Park
1. Goals and Compliance Context
Provide priority to LTC buses using cameras, GPS/transponders, and lane‑specific detectors while respecting overall corridor performance and safety.

Align with City of London’s Transportation Intelligent Mobility Management System (TIMMS) goals: reduced delays, improved transit reliability, and sensor/video‑based signal upgrades.
​

Ensure compatibility with:

GPS‑based transit signal priority used by the City.
​

Black transit signal heads and “TRANSIT SIGNAL” indications for bus‑only lanes, distinct from yellow signals for general traffic.

2. High‑Level System Overview
The TSP feature is a dedicated module that sits beside your existing intersection RL agent and MultiAgentCoordinator.

Existing components (unchanged):

DeusNegotiatio RL agent per intersection (phase decision and timing).

MultiAgentCoordinator for neighbor messaging and corridor coordination.

Detector abstraction layer (loops, LiDAR, cameras, etc.).
​

New components (TSP layer):

Transit Detection & Eligibility Service (T-DES).

TSP Request Manager (TSP-RM).

Priority Strategy Engine (PSE).

TSP Policy Interface to the RL agent.

Compliance Rules & Safety Guardrail module.

3. Transit Detection & Eligibility Service (T‑DES)
3.1 Inputs

On‑board GPS / AVL from LTC buses

Corridor‑level GPS‑based TSP as used in TIMMS; bus position, heading, route ID, schedule adherence (early/on‑time/late).
​

Roadside detection

Cameras / video analytics (recognize LTC buses, estimate distance/speed).

RF transponders or DSRC/ITS‑G5 if you simulate connected buses.

Traditional loops / magnetometers in bus lanes (if modeled).

3.2 Core Functions

Map each detected bus to:

Approach & lane (mixed traffic vs bus‑only lane).

ETA to stop line given speed and distance.

Schedule status: late, on‑time, early (priority only if late/on‑time, configurable).

Output per intersection per approach:

has_transit_approach: bool

eta_seconds: float

priority_level: {NONE, NORMAL, HIGH}

4. TSP Request Manager (TSP‑RM)
4.1 Responsibilities

Convert T‑DES outputs into formal TSP requests for each intersection agent.

Track request lifecycle:

PENDING → GRANTED → SERVED / EXPIRED.

Enforce minimum spacing between TSP events to protect cross‑street LOS.

4.2 Request Data Structure (Conceptual)

python
class TransitPriorityRequest:
    def __init__(self, intersection_id, approach_id, eta, priority_level,
                 requested_actions, expiry_time):
        self.intersection_id = intersection_id
        self.approach_id = approach_id
        self.eta = eta
        self.priority_level = priority_level
        self.requested_actions = requested_actions  # e.g. ['green_extend', 'early_green']
        self.expiry_time = expiry_time
        self.status = "PENDING"
5. Priority Strategy Engine (PSE)
5.1 Supported Strategies

The PSE decides what “knob” to turn when a valid TSP request exists:

Green extension

Keep current green for bus approach until bus clears (within max extension).

Early green / red truncation

Shorten current conflicting green to start bus green earlier, within min green and safety constraints.

Phase insertion / re‑order (phase rotation)

Insert a bus phase (e.g., black bus signal or protected turn from red bus lane) or re‑order phases to serve bus first, then restore plan.

Dwell recovery

After a strong TSP action, slightly compensate cross‑street green to keep average fairness over time.

5.2 Selection Logic (Simplified)

If eta is very short and approach is already green → green extension.

If eta falls into next cycle boundary → early green for bus movement.

If bus uses dedicated curb or centre‑running bus lane with black signal → insert or advance dedicated transit phase.

6. TSP Policy Interface to RL Agent
6.1 Observation Augmentation

Extend the RL state so the agent “knows” about transit:

For each approach (or phase):

transit_eta (capped, normalized).

transit_priority_flag (0/1).

transit_phase_id if a special black signal phase exists.

This is similar to how you already add neighbor context through MultiAgentCoordinator features.
​

6.2 Action Shaping / Constraints

Wrap the RL agent’s output in a TSP‑aware policy layer:

Agent proposes a phase / duration as usual.

TSP Policy Interface checks:

Active / pending TSP requests.

PSE strategy recommendations.

Safety & compliance rules (min green, clearance intervals, pedestrian phases, etc.).

If required, modify or override the suggested action:

Extend current green.

Start bus phase early.

Defer conflicting turn phases temporarily.

The agent still learns within this constrained action space; TSP appears as part of the environment.

7. Compliance & Safety Guardrails
7.1 City of London and Provincial Context

TIMMS / Intelligent Signals

Prioritize transit while still managing general traffic and incidents.
​

Use upgraded sensors and video to support future connected/automated vehicles.
​

Rapid Transit Signals

Use black transit signals and “TRANSIT SIGNAL” signage to clearly separate bus indications from general yellow traffic signals.

Ontario signal practice (OTM Book 12)

Respect standard safety principles: minimum greens, clearance intervals, pedestrian protection, and conflict‑free signal groups.
​

7.2 Guardrail Rules in the Architecture

Never reduce any phase below minimum green or clearance intervals defined by standards.
​

Transit phase must:

Not conflict with pedestrian crossings or opposing vehicle flows.

Follow the defined signal group and conflict matrix for black transit heads vs yellow vehicle heads.

Maximum cumulative TSP per unit time per intersection (e.g., max number of strong TSP actions per 10–15 minutes).

Corridor‑level constraints: coordination with TMC (simulated TIMMS) to avoid cascading TSP that destabilizes a corridor.
​

8. Multi‑Agent / Corridor Coordination
MultiAgentCoordinator gains transit‑aware features:

Share predicted downstream LTC arrival times so neighboring agents can prepare green bands for buses (moving green wave for LTC).

Corridor‑level policy (could be rule‑based at first):

If a bus is moving along the corridor, upstream intersections gently bias their phase choices to create a soft transit progression without fully sacrificing cross‑streets.

9. Simulation and Training Hooks
9.1 SUMO / CARLA Integration

Represent LTC buses as a dedicated vehicle class with:

Known routes, dwell times, and schedule adherence.

Dedicated curb or centre‑running bus lanes where applicable, with separate signal groups for transit phases.

Implement detectors:

Virtual GPS feed from buses to T‑DES.

Camera / LiDAR sensors already in your multi‑sensor design for rich state (optional).
​

9.2 Reward Shaping

Extend RL reward:

Large penalty for bus delay (especially when late).

Moderate penalty for general vehicle delay and stops.

Safety hard penalties for red‑light conflicts or pedestrian violations (should be prevented by guardrails anyway).

10. Module Diagram (Textual)
Sensors & Data

GPS/AVL (LTC) → T‑DES

Cameras / LiDAR / loops → T‑DES

TSP Logic Layer

T‑DES → TSP‑RM → PSE → TSP Policy Interface

Control Core

RL Agent (DeusNegotiatio) ↔ MultiAgentCoordinator

TSP Policy Interface wraps RL outputs and enforces guardrails

External / Supervisory

Corridor / TIMMS layer (simulated TMC) for corridor‑scale constraints and monitoring.
​

11. Implementation Checklist
Add T‑DES class with GPS + detector fusion and eligibility logic.

Add TSP‑RM with request queue, expiry, and per‑intersection TSP limits.

Add PSE with strategy selection (extend, early green, phase insertion/rotation).

Extend RL observation space with transit features and retrain agent.

Implement TSP Policy Interface that:

Intercepts actions.

Applies PSE decisions.

Enforces compliance guardrails.

Model black transit signals and bus‑only phases in your SUMO network to match London’s rapid transit signal behavior.

If you want, next step I can draft concrete Python class skeletons (T‑DES, TSP‑RM, PSE, and the policy wrapper) that slot directly into your current DeusNegotiatio codebase.