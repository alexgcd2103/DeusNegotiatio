# Oxford–Hyde Park SUMO Training Guide

**Project**: Oxford Street West @ Hyde Park Road Intersection  
**Location**: London, Ontario  
**Design Year**: 2046  
**Design Standard**: LOS C Modified  

This guide provides all essential information to build and train the SUMO traffic simulation and RL agent for the Oxford–Hyde Park intersection redesign project.

---

## 1. Network Geometry Specification

### 1.1 Intersection Layout

**Configuration**: Signalized 4-leg intersection (Oxford E–W, Hyde Park N–S)

**Approach Extent**: 115 m from stop line on all legs

**Lane Configuration** (all approaches):
- Right turn lane: 3.0 m width
- Through lane 1: 3.3 m width  
- Through lane 2: 3.3 m width
- Left turn lane: 3.0 m width

**Turn Lane Storage**: 100 m length (sized for 95 m worst-case queue)

**Median**: 2.0 m raised median with signal mounting

**Pedestrian Infrastructure**:
- Multi-use path: 2.0 m width
- Boulevard buffer: 1.0 m between curb and path

### 1.2 SUMO Network File (`*.net.xml`) Parameters

```xml
<!-- Example edge definition for Northbound approach -->
<edge id="NB_approach" from="south_node" to="center" priority="2">
    <lane id="NB_0" index="0" speed="16.67" length="115.0" width="3.0"/>  <!-- Right -->
    <lane id="NB_1" index="1" speed="16.67" length="115.0" width="3.3"/>  <!-- Through -->
    <lane id="NB_2" index="2" speed="16.67" length="115.0" width="3.3"/>  <!-- Through -->
    <lane id="NB_3" index="3" speed="16.67" length="115.0" width="3.0"/>  <!-- Left -->
</edge>
```

**Key Parameters**:
- Speed limit: 60 km/h (16.67 m/s)
- All approaches: 4 lanes each
- Departure edges mirror approach configuration
- Use `netedit` or `netconvert` to create junction with traffic lights

---

## 2. Traffic Demand (2046 PM Peak Hour)

### 2.1 Vehicle Flows – Design Hour Volume (DHV)

**Passenger Car Volumes** (vehicles/hour):

| Approach   | Left | Through | Right | Total |
|------------|------|---------|-------|-------|
| Northbound | 74   | 642     | 68    | 784   |
| Eastbound  | 529  | 897     | 53    | 1,478 |
| Southbound | 468  | 767     | 731   | 1,966 |
| Westbound  | 199  | 829     | 441   | 1,469 |

**Truck Volumes** (vehicles/hour by approach):
- Northbound: 11
- Eastbound: 6
- Southbound: 15
- Westbound: 14

### 2.2 SUMO Route File (`*.rou.xml`) Implementation

**Vehicle Types**:
```xml
<vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="16.67"/>
<vType id="truck" accel="1.3" decel="4.0" sigma="0.5" length="12.0" maxSpeed="13.89"/>
```

**Flow Definition Template**:
```xml
<!-- Northbound Left Turn: 74 veh/h -->
<flow id="NB_L" type="car" begin="0" end="3600" number="74" 
      from="NB_approach" to="WB_departure" departLane="3"/>

<!-- Northbound Through: 642 veh/h -->
<flow id="NB_T" type="car" begin="0" end="3600" number="642" 
      from="NB_approach" to="NB_departure" departLane="best"/>

<!-- Northbound Right: 68 veh/h -->
<flow id="NB_R" type="car" begin="0" end="3600" number="68" 
      from="NB_approach" to="EB_departure" departLane="0"/>
```

**Truck Distribution Strategy**:
- Distribute trucks proportionally across through movements
- Or assign all trucks to through lanes for conservative estimate

### 2.3 Pedestrian Flows

**Pedestrian Volumes** (crossings/hour):
- Northbound crosswalk: 186
- Eastbound crosswalk: 94
- Southbound: 0 (negligible)
- Westbound: 0 (negligible)

**SUMO Person Flow**:
```xml
<personFlow id="ped_NB" begin="0" end="3600" number="186">
    <walk from="NB_sidewalk_south" to="NB_sidewalk_north"/>
</personFlow>

<personFlow id="ped_EB" begin="0" end="3600" number="94">
    <walk from="EB_sidewalk_west" to="EB_sidewalk_east"/>
</personFlow>
```

### 2.4 Cyclists (Optional)

Negligible volumes (0-3/hour). Can add low-volume bicycle flows for realism:
```xml
<vType id="bicycle" vClass="bicycle" maxSpeed="5.56"/>
<flow id="bike_mixed" type="bicycle" begin="0" end="3600" number="3" from="..." to="..."/>
```

---

## 3. Signal Control System

### 3.1 Design Objectives

**Target Level of Service**: LOS C Modified
- Most movements: LOS B–C
- Critical movements (SB/WB left): LOS D–E acceptable
- Queue lengths: Must not exceed 100 m storage

### 3.2 Phase Structure

**Recommended Phase Plan**:

| Phase | Movements                    | Notes                          |
|-------|------------------------------|--------------------------------|
| 1     | NB/SB through + perm. right | Main N-S arterial             |
| 2     | EB/WB through + perm. right | Main E-W arterial             |
| 3     | NB/SB protected left        | High SB left demand           |
| 4     | EB/WB protected left        | High EB left demand           |

**Phase Timing Constraints**:
- Minimum green: 5-10 seconds per phase
- Pedestrian clearance: 15-20 seconds for crosswalk phases
- Yellow interval: 4 seconds
- All-red clearance: 2 seconds

### 3.3 Traffic Light Program (TLS) File

**Static Timing Example** (starting point for RL):
```xml
<tlLogic id="center" type="static" programID="0" offset="0">
    <!-- Phase 1: NB/SB through -->
    <phase duration="40" state="GGrrGGrrrrrrrrr"/>
    <phase duration="4"  state="yyrryyrrrrrrrr"/>  <!-- Yellow -->
    
    <!-- Phase 2: EB/WB through -->
    <phase duration="40" state="rrGGrrrrGGrrrr"/>
    <phase duration="4"  state="rryyrrrryyrrrr"/>
    
    <!-- Phase 3: NB/SB left -->
    <phase duration="20" state="rrrGrrrrrrrGrr"/>
    <phase duration="4"  state="rrryrrrrrrryrr"/>
    
    <!-- Phase 4: EB/WB left -->
    <phase duration="20" state="rrrrrrrGrrrrrG"/>
    <phase duration="4"  state="rrrrrrryrrrrrry"/>
</tlLogic>
```

---

## 4. Performance Metrics & Target Values

### 4.1 Engineering Design Targets (from Portfolio)

**Queue Length Benchmarks** (2046 PM):

| Movement      | Capacity | Target LOS | Max Queue (m) |
|---------------|----------|------------|---------------|
| NB Left       | 308      | C          | 4             |
| NB Through    | 1,392    | C          | 45            |
| SB Left       | 354      | E          | 84            |
| SB Through    | 1,403    | C          | 69            |
| SB Right      | 1,200    | C          | 64            |
| EB Left       | 327      | C          | 30            |
| EB Through    | 1,662    | B          | 65            |
| EB Right      | 809      | B          | 67            |
| WB Left       | 378      | F          | 95            |
| WB Through    | 1,679    | B          | 71            |
| WB Right      | 781      | B          | 6             |

**Critical Movements**:
- Southbound Left: 84 m queue, LOS E
- Westbound Left: 95 m queue, LOS F (worst case)

### 4.2 SUMO Output Metrics

**Lane-Level Metrics** (from SUMO lane outputs):
- Mean queue length (m)
- Mean waiting time (s)
- Mean speed (m/s)
- Number of vehicles departed
- Number of vehicles arrived

**Intersection-Level Aggregation**:
- Total system waiting time
- Average delay per vehicle
- Throughput (vehicles/hour served)
- Queue spillback events (queue > 100 m)

---

## 5. RL Agent Configuration

### 5.1 State Space

**Observation Vector** (per timestep):
- Queue length per lane (12 values: 4 approaches × 3 movements)
- Waiting time per lane (12 values)
- Current phase (one-hot encoded, 4 values)
- Time since phase change (1 value)
- Pedestrian waiting count (2 values: NB, EB crosswalks)

**Total State Dimension**: ~30 features

### 5.2 Action Space

**Discrete Actions**:
- Action 0: Switch to Phase 1 (NB/SB through)
- Action 1: Switch to Phase 2 (EB/WB through)
- Action 2: Switch to Phase 3 (NB/SB left)
- Action 3: Switch to Phase 4 (EB/WB left)
- Action 4: Extend current phase

**Constraints**:
- Minimum phase duration: 5 seconds
- Yellow/all-red transitions handled automatically by environment

### 5.3 Reward Function

**Reward Components** (per timestep):

```python
reward = (
    - α * total_waiting_time          # Total vehicle wait (all lanes)
    - β * total_queue_length           # Total queue length (all lanes)
    - γ * queue_spillback_penalty      # Heavy penalty if queue > 100m
    - δ * pedestrian_waiting_time      # Pedestrian delay
    + ε * throughput                   # Vehicles departed this step
)
```

**Recommended Weights** (starting values):
- α = 0.01 (waiting time weight)
- β = 0.05 (queue length weight)
- γ = 10.0 (spillback penalty, triggered if queue > 100 m)
- δ = 0.02 (pedestrian weight)
- ε = 0.5 (throughput reward)

**Tuning Strategy**:
- Increase γ if queues regularly exceed storage
- Increase δ if pedestrians are starved
- Adjust β to match SB/WB left emphasis from engineering analysis

### 5.4 Training Hyperparameters

**RL Algorithm**: PPO (Proximal Policy Optimization) or DQN

**PPO Configuration**:
```python
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
}
```

**Training Schedule**:
- Episode length: 3600 seconds (1 hour simulation)
- Total episodes: 5,000–10,000
- Evaluation frequency: Every 100 episodes
- Early stopping: If mean reward plateaus for 500 episodes

---

## 6. File Structure & Workflow

### 6.1 Required SUMO Files

```
project/
├── network/
│   ├── oxford_hydepark.net.xml          # Network geometry
│   ├── oxford_hydepark.rou.xml          # Traffic demand
│   ├── oxford_hydepark.sumocfg          # SUMO configuration
│   └── tls.add.xml                      # Traffic light definitions
├── training/
│   ├── train_agent.py                   # RL training script
│   ├── env_wrapper.py                   # Gym environment wrapper
│   └── config.yaml                      # Training configuration
├── evaluation/
│   ├── evaluate_agent.py                # Test trained policy
│   └── metrics_analysis.py              # Performance visualization
└── docs/
    └── SUMO-Training-Guide.md           # This file
```

### 6.2 SUMO Configuration File (`*.sumocfg`)

```xml
<configuration>
    <input>
        <net-file value="network/oxford_hydepark.net.xml"/>
        <route-files value="network/oxford_hydepark.rou.xml"/>
        <additional-files value="network/tls.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    <output>
        <queue-output value="output/queue.xml"/>
        <summary-output value="output/summary.xml"/>
    </output>
</configuration>
```

### 6.3 Training Workflow

**Step 1: Build Network**
```bash
# Use netedit GUI or command-line netconvert
netedit oxford_hydepark.net.xml
# Configure 4-leg junction, 4 lanes per approach, TLS
```

**Step 2: Generate Demand**
```bash
# Create rou.xml with flows matching DHV tables above
# Test demand independently:
sumo -c oxford_hydepark.sumocfg --no-step-log
```

**Step 3: Train RL Agent**
```bash
python training/train_agent.py --config training/config.yaml
# Monitor TensorBoard for reward curves
tensorboard --logdir training/logs/
```

**Step 4: Evaluate Performance**
```bash
python evaluation/evaluate_agent.py --model training/best_model.zip
# Generate metrics report comparing to engineering targets
```

**Step 5: Validation Against Design**
- Compare SUMO queue lengths to Appendix G targets
- Verify no systematic spillback beyond 100 m
- Confirm LOS approximation (delay-based or queue-based)
- Check pedestrian service levels

---

## 7. Validation Checklist

Before deploying or reporting the trained agent, verify:

**Geometry**:
- [ ] 4 lanes per approach (R, T, T, L)
- [ ] Lane widths: 3.0–3.5 m
- [ ] Turn lane storage: 100 m
- [ ] Approach length: ≥115 m

**Demand**:
- [ ] DHV flows match Table 2.1 (2046 PM)
- [ ] Truck flows included per approach
- [ ] Pedestrian flows for NB/EB crosswalks

**Performance**:
- [ ] SB left queue ≤ 100 m (target 84 m)
- [ ] WB left queue ≤ 100 m (target 95 m)
- [ ] Most movements achieve LOS C or better
- [ ] Throughput matches DHV (no gridlock)

**RL Agent**:
- [ ] Reward function emphasizes queue control
- [ ] Training converges (stable mean reward)
- [ ] Policy generalizes to demand variations (±10%)
- [ ] Pedestrian phases not starved

---

## 8. Quick Reference – Key Numbers

| Parameter | Value | Source |
|-----------|-------|--------|
| Design Year | 2046 | Portfolio |
| Peak Hour | PM | Portfolio |
| Total Intersection Volume | 5,697 veh/h | Sum of DHV |
| Worst Queue (Design) | 95 m (WB Left) | Appendix G |
| Lane Storage | 100 m | Design |
| Approach Length | 115 m | Portfolio |
| Lane Widths | 3.0–3.5 m | Design |
| Speed Limit | 60 km/h (16.67 m/s) | Municipal |
| Simulation Duration | 3600 s (1 hour) | Standard |
| Target LOS | C (Modified) | Portfolio |

---

## 9. Troubleshooting

**Issue**: Queues exceed 100 m frequently  
**Solution**: Increase protected left phase duration; adjust reward γ weight

**Issue**: Pedestrians never get green  
**Solution**: Increase pedestrian weight δ; enforce minimum ped phase

**Issue**: Throughput below DHV  
**Solution**: Check for gridlock; verify departure edges clear properly; increase cycle efficiency

**Issue**: RL training doesn't converge  
**Solution**: Normalize state features; reduce action space; tune learning rate

**Issue**: SUMO crashes or routes invalid  
**Solution**: Validate `.net.xml` connections in `netedit`; check lane permissions

---

## 10. References

- **Portfolio Document**: `Resubmission-DP.docx-2.md` (file:73)
- **Appendices**: `Appendices.docx.md` (file:72)
  - Appendix A: Traffic volumes (Table 3.5)
  - Appendix D: Elevations and grading
  - Appendix E: Drainage design
  - Appendix F: Asphalt design
  - Appendix G: Capacity, LOS, queue analysis
  - Appendix H: Cost estimate
  - Appendix I: Design drawings
- **City of London Design Standards**: Municipal roadway design manual
- **Provincial Standards (Ontario)**: MTO Geometric Design Standards

---

## Contact & Support

For questions about this SUMO training setup or the Oxford–Hyde Park intersection design:

**Project Team**: InterseXtion Solutions  
**Design Year**: 2046  
**Last Updated**: January 13, 2026  

**Note**: This training guide is derived directly from the engineering design portfolio and appendices. All traffic volumes, geometric parameters, and performance targets are based on professional traffic engineering analysis for the year 2046 design horizon.

---

**End of SUMO Training Guide**
