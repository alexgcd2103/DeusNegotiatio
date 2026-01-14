# Training DeusNegotiatio on the Oxford-Hyde Park Redesigned Intersection
## SUMO Network Creation, Traffic Demand Calibration, and Agent Training Pipeline

---

## Executive Summary

This document provides a complete, step-by-step methodology for **recreating the redesigned Oxford-Hyde Park intersection** in the SUMO traffic simulator and **training the DeusNegotiatio reinforcement learning agent** on massive volumes of simulated traffic data. The process encompasses accurate geometric modeling of your proposed LOS C Modified design, realistic traffic demand generation calibrated to 2026-2046 projections, pedestrian and cyclist integration, and a comprehensive training pipeline that generates thousands of simulation episodes to produce a robust, adaptive traffic control policy.

By the end of this implementation, you will have a fully trained agent capable of handling the complex, high-variance traffic conditions at the redesigned intersection with multiple turning lanes, simultaneous pedestrian phases, and multi-modal transportation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [SUMO Network Creation from Design Specifications](#2-sumo-network-creation-from-design-specifications)
3. [Traffic Demand Generation and Calibration](#3-traffic-demand-generation-and-calibration)
4. [Pedestrian and Cyclist Integration](#4-pedestrian-and-cyclist-integration)
5. [Environment Setup: SUMO-RL Interface](#5-environment-setup-sumo-rl-interface)
6. [Training Data Generation Pipeline](#6-training-data-generation-pipeline)
7. [Agent Training Architecture](#7-agent-training-architecture)
8. [Validation and Performance Metrics](#8-validation-and-performance-metrics)
9. [Appendix: Code Templates and Configuration Files](#9-appendix-code-templates-and-configuration-files)

---

## 1. Project Overview

### 1.1 Intersection Design Specifications (From Your Portfolio)

Based on your InterseXion Solutions design portfolio, the redesigned Oxford-Hyde Park intersection features:

**Geometric Configuration (LOS C Modified Case)**:
- **Oxford Street (East-West Arterial)**:
  - Eastbound: 2 through lanes + 1 left-turn lane + 1 right-turn lane (4 lanes total)
  - Westbound: 2 through lanes + 1 left-turn lane + 1 right-turn lane (4 lanes total)
  
- **Hyde Park Road (North-South Arterial)**:
  - Northbound: 2 through lanes + 1 left-turn lane + 1 right-turn lane (4 lanes total)
  - Southbound: 2 through lanes + 1 left-turn lane + 1 right-turn lane (4 lanes total)

**Lane Widths** (City of London Standards):
- Through lanes (center): 3.3-3.5 m
- Turning lanes: 3.0 m minimum
- Curb lanes: 3.5 m

**Multi-Modal Infrastructure**:
- 2-meter-wide asphalt multi-use path (pedestrian + cyclist)
- 1-meter buffer from curb for snow storage
- Pedestrian crossing infrastructure at all four approaches
- Projected 2046 peak hour: 186 pedestrians, 3 cyclists

**Traffic Signal Configuration**:
- NEMA-standard phase sequencing
- Protected left-turn phases
- Pedestrian walk/don't-walk signals
- All-red clearance intervals

### 1.2 Training Objectives

The goal is to train DeusNegotiatio to:

1. **Optimize multi-objective performance**: Vehicle throughput, pedestrian safety, emissions reduction, delay minimization
2. **Handle temporal variability**: Morning rush (6-9 AM), midday (12-1 PM), evening rush (5-7 PM), off-peak
3. **Adapt to demand uncertainty**: Special events (university activities), weather impacts, seasonal patterns
4. **Coordinate turning movements**: Complex left-turn conflicts, simultaneous pedestrian phases
5. **Learn robust policies**: Transfer learned behavior across different traffic scenarios

### 1.3 Data Requirements

To train a robust agent, we need **massive simulation volumes**:

- **Minimum Episodes**: 10,000-50,000 episodes (each 1-2 hours simulated time)
- **Scenario Diversity**: 20+ traffic demand profiles covering all time periods and special events
- **Total Simulation Time**: 20,000-100,000 simulated hours
- **State-Action Pairs**: 5-10 million transitions for experience replay buffer

---

## 2. SUMO Network Creation from Design Specifications

### 2.1 Installation and Environment Setup

#### Prerequisites

```bash
# Ubuntu/Linux
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# Set environment variables
export SUMO_HOME="/usr/share/sumo"
export PATH="$PATH:$SUMO_HOME/bin"

# Verify installation
sumo --version  # Should show SUMO Version 1.18+ or higher

# Python dependencies
pip install sumolib traci numpy pandas matplotlib
pip install torch stable-baselines3  # For RL training
pip install sumo-rl  # SUMO-RL wrapper (optional but recommended)
```

### 2.2 Network Geometry Creation

#### Method 1: Manual XML Definition (Recommended for Precision)

Create the network from scratch to match your exact design specifications.

**File: `oxford_hydepark.nod.xml`** (Node definitions)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    
    <!-- Central Intersection Node -->
    <node id="center" x="0.0" y="0.0" type="traffic_light" 
          tl="oxford_hydepark_tl" radius="20"/>
    
    <!-- Oxford Street East-West Approach Nodes -->
    <node id="oxford_east_approach" x="150.0" y="0.0" type="priority"/>
    <node id="oxford_west_approach" x="-150.0" y="0.0" type="priority"/>
    
    <!-- Hyde Park Road North-South Approach Nodes -->
    <node id="hydepark_north_approach" x="0.0" y="150.0" type="priority"/>
    <node id="hydepark_south_approach" x="0.0" y="-150.0" type="priority"/>
    
    <!-- Extended Network Boundaries (100m+ approaches as per design) -->
    <node id="oxford_east_boundary" x="250.0" y="0.0" type="priority"/>
    <node id="oxford_west_boundary" x="-250.0" y="0.0" type="priority"/>
    <node id="hydepark_north_boundary" x="0.0" y="250.0" type="priority"/>
    <node id="hydepark_south_boundary" x="0.0" y="-250.0" type="priority"/>
    
</nodes>
```

**File: `oxford_hydepark.edg.xml`** (Edge/Lane definitions)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    
    <!-- Oxford Street Eastbound (Approaching Intersection) -->
    <edge id="oxford_eb_approach" from="oxford_west_approach" to="center" 
          priority="2" numLanes="4" speed="16.67">
        <!-- 16.67 m/s = 60 km/h, arterial speed -->
        <lane index="0" allow="all" width="3.0"/>  <!-- Right-turn lane -->
        <lane index="1" allow="all" width="3.4"/>  <!-- Through lane 1 -->
        <lane index="2" allow="all" width="3.4"/>  <!-- Through lane 2 -->
        <lane index="3" allow="all" width="3.0"/>  <!-- Left-turn lane -->
    </edge>
    
    <!-- Oxford Street Westbound (Approaching Intersection) -->
    <edge id="oxford_wb_approach" from="oxford_east_approach" to="center" 
          priority="2" numLanes="4" speed="16.67">
        <lane index="0" allow="all" width="3.0"/>  <!-- Right-turn lane -->
        <lane index="1" allow="all" width="3.4"/>  <!-- Through lane 1 -->
        <lane index="2" allow="all" width="3.4"/>  <!-- Through lane 2 -->
        <lane index="3" allow="all" width="3.0"/>  <!-- Left-turn lane -->
    </edge>
    
    <!-- Hyde Park Road Northbound (Approaching Intersection) -->
    <edge id="hydepark_nb_approach" from="hydepark_south_approach" to="center" 
          priority="2" numLanes="4" speed="16.67">
        <lane index="0" allow="all" width="3.0"/>  <!-- Right-turn lane -->
        <lane index="1" allow="all" width="3.4"/>  <!-- Through lane 1 -->
        <lane index="2" allow="all" width="3.4"/>  <!-- Through lane 2 -->
        <lane index="3" allow="all" width="3.0"/>  <!-- Left-turn lane -->
    </edge>
    
    <!-- Hyde Park Road Southbound (Approaching Intersection) -->
    <edge id="hydepark_sb_approach" from="hydepark_north_approach" to="center" 
          priority="2" numLanes="4" speed="16.67">
        <lane index="0" allow="all" width="3.0"/>  <!-- Right-turn lane -->
        <lane index="1" allow="all" width="3.4"/>  <!-- Through lane 1 -->
        <lane index="2" allow="all" width="3.4"/>  <!-- Through lane 2 -->
        <lane index="3" allow="all" width="3.0"/>  <!-- Left-turn lane -->
    </edge>
    
    <!-- Departing/Exiting Edges (post-intersection) -->
    <edge id="oxford_eb_exit" from="center" to="oxford_east_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="oxford_wb_exit" from="center" to="oxford_west_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="hydepark_nb_exit" from="center" to="hydepark_north_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="hydepark_sb_exit" from="center" to="hydepark_south_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    
    <!-- Extended Approach Segments (100m queue storage as designed) -->
    <edge id="oxford_eb_extended" from="oxford_west_boundary" to="oxford_west_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="oxford_wb_extended" from="oxford_east_boundary" to="oxford_east_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="hydepark_nb_extended" from="hydepark_south_boundary" to="hydepark_south_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="hydepark_sb_extended" from="hydepark_north_boundary" to="hydepark_north_approach" 
          priority="2" numLanes="2" speed="16.67"/>
    
    <!-- Exit Extensions -->
    <edge id="oxford_eb_boundary_exit" from="oxford_east_approach" to="oxford_east_boundary" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="oxford_wb_boundary_exit" from="oxford_west_approach" to="oxford_west_boundary" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="hydepark_nb_boundary_exit" from="hydepark_north_approach" to="hydepark_north_boundary" 
          priority="2" numLanes="2" speed="16.67"/>
    <edge id="hydepark_sb_boundary_exit" from="hydepark_south_approach" to="hydepark_south_boundary" 
          priority="2" numLanes="2" speed="16.67"/>
    
</edges>
```

**File: `oxford_hydepark.con.xml`** (Connection/Turn Movement definitions)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
             xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">
    
    <!-- Oxford Eastbound Connections -->
    <!-- Right turn: Lane 0 → Hyde Park SB -->
    <connection from="oxford_eb_approach" to="hydepark_sb_exit" fromLane="0" toLane="0"/>
    
    <!-- Through: Lanes 1,2 → Oxford EB Exit -->
    <connection from="oxford_eb_approach" to="oxford_eb_exit" fromLane="1" toLane="0"/>
    <connection from="oxford_eb_approach" to="oxford_eb_exit" fromLane="2" toLane="1"/>
    
    <!-- Left turn: Lane 3 → Hyde Park NB -->
    <connection from="oxford_eb_approach" to="hydepark_nb_exit" fromLane="3" toLane="1"/>
    
    <!-- Oxford Westbound Connections -->
    <connection from="oxford_wb_approach" to="hydepark_nb_exit" fromLane="0" toLane="0"/>
    <connection from="oxford_wb_approach" to="oxford_wb_exit" fromLane="1" toLane="0"/>
    <connection from="oxford_wb_approach" to="oxford_wb_exit" fromLane="2" toLane="1"/>
    <connection from="oxford_wb_approach" to="hydepark_sb_exit" fromLane="3" toLane="1"/>
    
    <!-- Hyde Park Northbound Connections -->
    <connection from="hydepark_nb_approach" to="oxford_wb_exit" fromLane="0" toLane="0"/>
    <connection from="hydepark_nb_approach" to="hydepark_nb_exit" fromLane="1" toLane="0"/>
    <connection from="hydepark_nb_approach" to="hydepark_nb_exit" fromLane="2" toLane="1"/>
    <connection from="hydepark_nb_approach" to="oxford_eb_exit" fromLane="3" toLane="1"/>
    
    <!-- Hyde Park Southbound Connections -->
    <connection from="hydepark_sb_approach" to="oxford_eb_exit" fromLane="0" toLane="0"/>
    <connection from="hydepark_sb_approach" to="hydepark_sb_exit" fromLane="1" toLane="0"/>
    <connection from="hydepark_sb_approach" to="hydepark_sb_exit" fromLane="2" toLane="1"/>
    <connection from="hydepark_sb_approach" to="oxford_wb_exit" fromLane="3" toLane="1"/>
    
</connections>
```

**File: `oxford_hydepark.tll.xml`** (Traffic Light Logic - NEMA Standard 8-Phase)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<tlLogics xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
          xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/tllogic_file.xsd">
    
    <tlLogic id="oxford_hydepark_tl" type="static" programID="0" offset="0">
        
        <!-- Phase 1: Oxford EB/WB Through + Right (Main Street Green) -->
        <phase duration="30" state="GGGgrrrGGGgrrr"/>
        
        <!-- Phase 2: Yellow Transition -->
        <phase duration="3" state="yyygrrryyygrrr"/>
        
        <!-- Phase 3: All Red Clearance -->
        <phase duration="2" state="rrrgrrrrrrgrrr"/>
        
        <!-- Phase 4: Oxford EB/WB Left Turn (Protected) -->
        <phase duration="15" state="rrrGrrrrrrGrrr"/>
        
        <!-- Phase 5: Yellow Transition -->
        <phase duration="3" state="rrryrrrrrryrrr"/>
        
        <!-- Phase 6: All Red Clearance -->
        <phase duration="2" state="rrrrrrrrrrrrrr"/>
        
        <!-- Phase 7: Hyde Park NB/SB Through + Right -->
        <phase duration="25" state="rrrrrGGGgrrrGGg"/>
        
        <!-- Phase 8: Yellow Transition -->
        <phase duration="3" state="rrrrryyygrrryyg"/>
        
        <!-- Phase 9: All Red Clearance -->
        <phase duration="2" state="rrrrrrrrrrrrrr"/>
        
        <!-- Phase 10: Hyde Park NB/SB Left Turn (Protected) -->
        <phase duration="12" state="rrrrrrrGrrrrrG"/>
        
        <!-- Phase 11: Yellow Transition -->
        <phase duration="3" state="rrrrrrryrrrrry"/>
        
        <!-- Phase 12: All Red Clearance -->
        <phase duration="2" state="rrrrrrrrrrrrrr"/>
        
    </tlLogic>
    
</tlLogics>
```

**State String Encoding**:
- Each character represents one connection (turn movement)
- G = Green (go), g = Green (caution), y = Yellow, r = Red
- Order: Oxford EB (R,T,T,L), Hyde Park NB (R,T,T,L), Oxford WB (R,T,T,L), Hyde Park SB (R,T,T,L)

#### Network Compilation

```bash
# Compile the network from XML files
netconvert \
  --node-files=oxford_hydepark.nod.xml \
  --edge-files=oxford_hydepark.edg.xml \
  --connection-files=oxford_hydepark.con.xml \
  --tllogic-files=oxford_hydepark.tll.xml \
  --output-file=oxford_hydepark.net.xml \
  --junctions.join \
  --geometry.remove \
  --ramps.guess \
  --junctions.corner-detail=5 \
  --rectangular-lane-cut

# Visualize the network
sumo-gui -n oxford_hydepark.net.xml
```

#### Method 2: Import from OpenStreetMap (Alternative)

For real-world geometry reference:

```bash
# Download OSM data for London, ON area containing intersection
# Use https://www.openstreetmap.org/export
# Bounding box: Lat/Lon around 43.0055, -81.3186

# Convert OSM to SUMO network
netconvert \
  --osm-files=london_oxford_hydepark.osm \
  --output-file=oxford_hydepark_osm.net.xml \
  --geometry.remove \
  --ramps.guess \
  --junctions.join \
  --tls.guess-signals \
  --tls.default-type=actuated

# Then manually edit lanes/connections to match your design specs
# Use NETEDIT GUI for visual editing
netedit -s oxford_hydepark_osm.net.xml
```

### 2.3 Adding Detectors and Sensors

**File: `detectors.add.xml`** (Induction loops for vehicle detection)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    
    <!-- Oxford Eastbound Detectors (50m before intersection) -->
    <inductionLoop id="det_oxford_eb_r" lane="oxford_eb_approach_0" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_oxford_eb_t1" lane="oxford_eb_approach_1" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_oxford_eb_t2" lane="oxford_eb_approach_2" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_oxford_eb_l" lane="oxford_eb_approach_3" pos="100" freq="60" file="detector_output.xml"/>
    
    <!-- Repeat for other approaches: WB, NB, SB -->
    <inductionLoop id="det_oxford_wb_r" lane="oxford_wb_approach_0" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_oxford_wb_t1" lane="oxford_wb_approach_1" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_oxford_wb_t2" lane="oxford_wb_approach_2" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_oxford_wb_l" lane="oxford_wb_approach_3" pos="100" freq="60" file="detector_output.xml"/>
    
    <inductionLoop id="det_hydepark_nb_r" lane="hydepark_nb_approach_0" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_hydepark_nb_t1" lane="hydepark_nb_approach_1" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_hydepark_nb_t2" lane="hydepark_nb_approach_2" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_hydepark_nb_l" lane="hydepark_nb_approach_3" pos="100" freq="60" file="detector_output.xml"/>
    
    <inductionLoop id="det_hydepark_sb_r" lane="hydepark_sb_approach_0" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_hydepark_sb_t1" lane="hydepark_sb_approach_1" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_hydepark_sb_t2" lane="hydepark_sb_approach_2" pos="100" freq="60" file="detector_output.xml"/>
    <inductionLoop id="det_hydepark_sb_l" lane="hydepark_sb_approach_3" pos="100" freq="60" file="detector_output.xml"/>
    
    <!-- Multi-Entry-Exit (E3) Detectors for comprehensive data -->
    <e3Detector id="e3_oxford_eb" freq="60" file="e3_output.xml">
        <detEntry lane="oxford_eb_approach_0" pos="120"/>
        <detEntry lane="oxford_eb_approach_1" pos="120"/>
        <detEntry lane="oxford_eb_approach_2" pos="120"/>
        <detEntry lane="oxford_eb_approach_3" pos="120"/>
        <detExit lane="oxford_eb_exit_0" pos="10"/>
        <detExit lane="oxford_eb_exit_1" pos="10"/>
        <detExit lane="hydepark_nb_exit_1" pos="10"/>
        <detExit lane="hydepark_sb_exit_0" pos="10"/>
    </e3Detector>
    
    <!-- Repeat E3 detectors for other approaches -->
    
</additional>
```

---

## 3. Traffic Demand Generation and Calibration

### 3.1 Traffic Projections from Your Portfolio

Your design portfolio provides these projections:

| Horizon Year | AADT Growth | Peak Hour Volume (PHV) |
|-------------|-------------|------------------------|
| 2026 (Construction) | Baseline | Tables 2.1 in Appendix A |
| 2031 (5-year) | +15% | Tables 2.2 |
| 2036 (10-year) | +28% | Tables 2.3 |
| 2046 (20-year) | +52% | Tables 2.4 |

**Key Traffic Characteristics**:
- **Directional split**: Heavy eastbound AM, heavy westbound PM
- **Turn percentages**: Variable by approach (need Synchro outputs from Appendix G)
- **Peak periods**: 6-9 AM (inbound), 5-7 PM (outbound)
- **Special events**: University class changes (8-9 AM, 12-1 PM)

### 3.2 Creating Realistic Traffic Demand Files

#### Step 1: Define Origin-Destination (OD) Matrix

**File: `od_matrix_2046_am_peak.txt`** (TAZ-based OD for morning rush)

```
$OR;D2
* From-Time  To-Time
7.00 9.00
* Factor
1.00
* some
* additional
* comments
         1          2          3          4
    1  0.00     350.00     280.00     120.00
    2  400.00     0.00     180.00     210.00
    3  320.00     200.00     0.00     150.00
    4  140.00     220.00     160.00     0.00
```

Where TAZ (Traffic Analysis Zones):
- TAZ 1 = Oxford East boundary
- TAZ 2 = Oxford West boundary
- TAZ 3 = Hyde Park North boundary
- TAZ 4 = Hyde Park South boundary

#### Step 2: Convert OD Matrix to SUMO Trips

```bash
# Define TAZ districts
# File: districts.taz.xml
cat > districts.taz.xml << 'EOF'
<tazs>
    <taz id="1" edges="oxford_eb_extended oxford_eb_boundary_exit"/>
    <taz id="2" edges="oxford_wb_extended oxford_wb_boundary_exit"/>
    <taz id="3" edges="hydepark_nb_extended hydepark_nb_boundary_exit"/>
    <taz id="4" edges="hydepark_sb_extended hydepark_sb_boundary_exit"/>
</tazs>
EOF

# Convert OD matrix to trips
od2trips \
  -d od_matrix_2046_am_peak.txt \
  -n districts.taz.xml \
  -o trips_2046_am_peak.trips.xml \
  --timeline.day-in-hours
```

#### Step 3: Route Assignment

```bash
# Use DUAROUTER to assign routes based on shortest path
duarouter \
  -n oxford_hydepark.net.xml \
  -t trips_2046_am_peak.trips.xml \
  -o routes_2046_am_peak.rou.xml \
  --ignore-errors \
  --repair \
  --routing-algorithm dijkstra \
  --max-alternatives 3 \
  --weights.random-factor 1.2
```

### 3.3 Traffic Demand Calibration Using Real-World Turn Percentages

**Python Script: `calibrate_traffic_demand.py`**

```python
import xml.etree.ElementTree as ET
import random

# Turn percentages from your Synchro analysis (Appendix G Tables 8.1-8.4)
# Example for Oxford EB approach during AM peak
turn_percentages = {
    'oxford_eb': {'right': 0.15, 'through': 0.70, 'left': 0.15},
    'oxford_wb': {'right': 0.12, 'through': 0.75, 'left': 0.13},
    'hydepark_nb': {'right': 0.18, 'through': 0.65, 'left': 0.17},
    'hydepark_sb': {'right': 0.20, 'through': 0.62, 'left': 0.18}
}

def assign_turn_movements(route_file, output_file, turn_probs):
    """
    Assign specific turn movements based on observed percentages
    """
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    for vehicle in root.findall('vehicle'):
        route = vehicle.find('route')
        edges = route.get('edges').split()
        origin_edge = edges[0]
        
        # Determine approach
        if 'oxford_eb' in origin_edge:
            approach = 'oxford_eb'
        elif 'oxford_wb' in origin_edge:
            approach = 'oxford_wb'
        elif 'hydepark_nb' in origin_edge:
            approach = 'hydepark_nb'
        elif 'hydepark_sb' in origin_edge:
            approach = 'hydepark_sb'
        else:
            continue
        
        # Probabilistically assign turn
        turn_prob = random.random()
        probs = turn_probs[approach]
        
        if turn_prob < probs['right']:
            turn_type = 'right'
        elif turn_prob < probs['right'] + probs['through']:
            turn_type = 'through'
        else:
            turn_type = 'left'
        
        # Assign appropriate exit edge based on turn
        # (Logic depends on your network structure)
        new_route = generate_route_for_turn(origin_edge, turn_type)
        route.set('edges', new_route)
    
    tree.write(output_file)

def generate_route_for_turn(origin, turn):
    """Map origin + turn to appropriate route"""
    route_map = {
        ('oxford_eb_extended', 'right'): 'oxford_eb_extended oxford_eb_approach hydepark_sb_exit hydepark_sb_boundary_exit',
        ('oxford_eb_extended', 'through'): 'oxford_eb_extended oxford_eb_approach oxford_eb_exit oxford_eb_boundary_exit',
        ('oxford_eb_extended', 'left'): 'oxford_eb_extended oxford_eb_approach hydepark_nb_exit hydepark_nb_boundary_exit',
        # ... complete for all origin-turn combinations
    }
    return route_map.get((origin, turn), '')

# Run calibration
assign_turn_movements(
    'routes_2046_am_peak.rou.xml',
    'routes_2046_am_peak_calibrated.rou.xml',
    turn_percentages
)
```

### 3.4 Multi-Scenario Demand Generation

Generate **20+ traffic scenarios** for comprehensive training:

| Scenario ID | Time Period | Demand Level | Description |
|------------|-------------|--------------|-------------|
| S01 | 6-9 AM | High | Morning rush, EB heavy |
| S02 | 9-11 AM | Medium | Mid-morning |
| S03 | 12-1 PM | Medium-High | Lunch rush + pedestrians |
| S04 | 2-5 PM | Medium | Afternoon |
| S05 | 5-7 PM | High | Evening rush, WB heavy |
| S06 | 7-11 PM | Low-Medium | Evening |
| S07 | 11 PM-6 AM | Very Low | Night |
| S08 | Special Event | Very High | University event surge |
| S09 | Accident | Asymmetric | Lane blockage |
| S10 | Weather | Reduced Speed | Rain/snow conditions |
| S11-S20 | Variations | Mixed | Randomized perturbations |

**Automated Generation Script**:

```python
import subprocess

scenarios = [
    {'name': 'am_rush', 'base_flow': 1800, 'duration': 10800, 'eb_factor': 1.4, 'wb_factor': 0.7},
    {'name': 'pm_rush', 'base_flow': 1900, 'duration': 7200, 'eb_factor': 0.6, 'wb_factor': 1.5},
    {'name': 'midday', 'base_flow': 1200, 'duration': 3600, 'eb_factor': 1.0, 'wb_factor': 1.0},
    # ... define all 20+ scenarios
]

for scenario in scenarios:
    # Generate randomTrips with appropriate parameters
    subprocess.run([
        'python', f'{os.environ["SUMO_HOME"]}/tools/randomTrips.py',
        '-n', 'oxford_hydepark.net.xml',
        '-o', f'trips_{scenario["name"]}.trips.xml',
        '-e', str(scenario['duration']),
        '--period', str(3600 / scenario['base_flow']),  # Inter-vehicle time
        '--binomial', '3',  # Randomness
        '--validate'
    ])
    
    # Route generation
    subprocess.run([
        'duarouter',
        '-n', 'oxford_hydepark.net.xml',
        '-t', f'trips_{scenario["name"]}.trips.xml',
        '-o', f'routes_{scenario["name"]}.rou.xml',
        '--ignore-errors'
    ])
```

---

## 4. Pedestrian and Cyclist Integration

### 4.1 Pedestrian Crossing Infrastructure

**File: `pedestrian_crossings.add.xml`**

```xml
<additional>
    <!-- Oxford EB Crossing -->
    <crossing id="cross_oxford_eb" 
              edges="oxford_eb_approach oxford_eb_exit" 
              node="center" 
              width="3.0" 
              priority="true"
              linkIndex="14"/>
    
    <!-- Repeat for all 4 crossings: oxford_wb, hydepark_nb, hydepark_sb -->
    
    <!-- Pedestrian sidewalks -->
    <poly id="sidewalk_oxford_north" type="sidewalk" 
          color="0.5,0.5,0.5" fill="1" layer="0.5"
          shape="..."/>  <!-- Define shape coordinates -->
    
</additional>
```

### 4.2 Pedestrian Demand Generation

Based on your projection: **186 pedestrians during peak hour**

**Generate pedestrian trips**:

```bash
# Random pedestrian trips
python $SUMO_HOME/tools/randomTrips.py \
  -n oxford_hydepark.net.xml \
  --pedestrians \
  -o pedestrians_peak.rou.xml \
  -e 3600 \
  -p 19.35  # 3600s / 186 peds = 19.35s inter-arrival
```

**More realistic: crossing-focused pedestrian movements**

```python
# pedestrian_demand.py
import xml.etree.ElementTree as ET

# Define crossing patterns
crossing_demand = {
    'oxford_north_crossing': 52,  # peds/hour
    'oxford_south_crossing': 48,
    'hydepark_east_crossing': 44,
    'hydepark_west_crossing': 42
}

root = ET.Element('routes')

ped_id = 0
for crossing, count in crossing_demand.items():
    inter_arrival = 3600 / count
    
    for i in range(count):
        depart_time = i * inter_arrival + random.uniform(-5, 5)
        
        person = ET.SubElement(root, 'person', {
            'id': f'ped_{ped_id}',
            'depart': f'{depart_time:.2f}',
            'type': 'pedestrian'
        })
        
        # Define walk across crossing
        walk = ET.SubElement(person, 'walk', {
            'edges': get_crossing_edges(crossing),
            'speed': '1.3'  # m/s, realistic pedestrian speed
        })
        
        ped_id += 1

tree = ET.ElementTree(root)
tree.write('pedestrians_realistic.rou.xml')
```

### 4.3 Cyclist Integration

**3 cyclists during peak hour** → Very low volume, integrate into vehicle flow:

```bash
# Generate cyclist trips
python $SUMO_HOME/tools/randomTrips.py \
  -n oxford_hydepark.net.xml \
  --vehicle-class bicycle \
  -o cyclists_peak.rou.xml \
  -e 3600 \
  -p 1200  # 3 cyclists in 1 hour
```

**Define bicycle vehicle type**:

```xml
<vType id="bicycle" vClass="bicycle" 
       maxSpeed="6.0" speedDev="1.0" 
       length="1.8" width="0.65" 
       accel="0.8" decel="1.5"/>
```

---

## 5. Environment Setup: SUMO-RL Interface

### 5.1 Install SUMO-RL Framework

```bash
pip install sumo-rl gym pettingzoo
```

### 5.2 Custom Environment for Oxford-Hyde Park

**File: `oxford_hydepark_env.py`**

```python
import sumo_rl
import gym
from gym import spaces
import numpy as np
import traci

class OxfordHydeParkEnv(gym.Env):
    """
    Custom Gym environment for Oxford-Hyde Park intersection
    """
    
    def __init__(self, 
                 net_file='oxford_hydepark.net.xml',
                 route_file='routes_2046_am_peak_calibrated.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5,
                 yellow_time=3,
                 min_green=10,
                 max_green=50):
        
        super().__init__()
        
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        
        # Traffic light ID from network
        self.ts_id = 'oxford_hydepark_tl'
        
        # Define observation space (state representation)
        # 16 lanes × (queue_length + avg_speed + occupancy) + phase info
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(16*3 + 12,),  # 60-dimensional state
            dtype=np.float32
        )
        
        # Define action space (phase selection)
        # 8 phases (4 main + 4 protected left turns)
        self.action_space = spaces.Discrete(8)
        
        self.sumo_running = False
        self.simulation_step = 0
        
    def reset(self):
        """Reset simulation to initial state"""
        if self.sumo_running:
            traci.close()
        
        sumo_cmd = [
            'sumo-gui' if self.use_gui else 'sumo',
            '-n', self.net_file,
            '-r', self.route_file,
            '--additional-files', 'detectors.add.xml,pedestrian_crossings.add.xml',
            '--no-warnings',
            '--no-step-log',
            '--time-to-teleport', '-1',
            '--waiting-time-memory', '1000',
            '--max-depart-delay', '0'
        ]
        
        traci.start(sumo_cmd)
        self.sumo_running = True
        self.simulation_step = 0
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        
        # Apply action (set traffic light phase)
        current_phase = traci.trafficlight.getPhase(self.ts_id)
        
        if action != self._phase_to_action(current_phase):
            # Phase change requested
            traci.trafficlight.setPhase(self.ts_id, self._action_to_phase(action))
        
        # Simulate for delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.simulation_step += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if episode is done
        done = self.simulation_step >= self.num_seconds
        
        info = {
            'total_wait_time': self._get_total_wait_time(),
            'queue_length': self._get_total_queue_length(),
            'throughput': self._get_throughput(),
            'pedestrian_wait': self._get_pedestrian_wait_time()
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Construct state vector from SUMO"""
        state = []
        
        # Lane-level features for all 16 approach lanes
        lanes = [
            'oxford_eb_approach_0', 'oxford_eb_approach_1', 
            'oxford_eb_approach_2', 'oxford_eb_approach_3',
            'oxford_wb_approach_0', 'oxford_wb_approach_1',
            'oxford_wb_approach_2', 'oxford_wb_approach_3',
            'hydepark_nb_approach_0', 'hydepark_nb_approach_1',
            'hydepark_nb_approach_2', 'hydepark_nb_approach_3',
            'hydepark_sb_approach_0', 'hydepark_sb_approach_1',
            'hydepark_sb_approach_2', 'hydepark_sb_approach_3'
        ]
        
        for lane in lanes:
            # Queue length (number of halting vehicles)
            queue = traci.lane.getLastStepHaltingNumber(lane)
            
            # Average speed
            avg_speed = traci.lane.getLastStepMeanSpeed(lane)
            
            # Occupancy (percentage of lane occupied)
            occupancy = traci.lane.getLastStepOccupancy(lane)
            
            state.extend([queue, avg_speed, occupancy])
        
        # Current phase information (one-hot encoding of 12 phases)
        current_phase = traci.trafficlight.getPhase(self.ts_id)
        phase_encoding = np.zeros(12)
        phase_encoding[current_phase] = 1.0
        state.extend(phase_encoding)
        
        return np.array(state, dtype=np.float32)
    
    def _compute_reward(self):
        """Multi-objective reward function"""
        
        # Vehicle delay (primary metric)
        total_wait_time = self._get_total_wait_time()
        
        # Queue lengths (secondary metric)
        total_queue = self._get_total_queue_length()
        
        # Throughput (positive reinforcement)
        throughput = self._get_throughput()
        
        # Pedestrian wait time
        ped_wait = self._get_pedestrian_wait_time()
        
        # Composite reward (weights from DeusNegotiatio architecture)
        reward = (
            -0.4 * total_wait_time / 100.0 +  # Normalize
            -0.2 * total_queue / 50.0 +
            0.3 * throughput / 20.0 +
            -0.1 * ped_wait / 30.0
        )
        
        return reward
    
    def _get_total_wait_time(self):
        """Sum of waiting times for all vehicles"""
        wait_time = 0
        for veh_id in traci.vehicle.getIDList():
            wait_time += traci.vehicle.getWaitingTime(veh_id)
        return wait_time
    
    def _get_total_queue_length(self):
        """Sum of halting vehicles across all lanes"""
        queue = 0
        for lane in traci.lane.getIDList():
            if 'approach' in lane:
                queue += traci.lane.getLastStepHaltingNumber(lane)
        return queue
    
    def _get_throughput(self):
        """Number of vehicles that departed in last delta_time"""
        return traci.simulation.getDepartedNumber()
    
    def _get_pedestrian_wait_time(self):
        """Average waiting time for pedestrians"""
        ped_ids = traci.person.getIDList()
        if len(ped_ids) == 0:
            return 0
        
        total_wait = sum(traci.person.getWaitingTime(pid) for pid in ped_ids)
        return total_wait / len(ped_ids)
    
    def _action_to_phase(self, action):
        """Map discrete action to SUMO phase index"""
        # Define action-to-phase mapping based on traffic light logic
        action_phase_map = {
            0: 0,   # Oxford EW through + right
            1: 3,   # Oxford EW left turn
            2: 6,   # Hyde Park NS through + right
            3: 9,   # Hyde Park NS left turn
            # Add more mappings if using more complex phase structure
        }
        return action_phase_map.get(action, 0)
    
    def _phase_to_action(self, phase):
        """Reverse mapping"""
        phase_action_map = {0: 0, 3: 1, 6: 2, 9: 3}
        return phase_action_map.get(phase, 0)
    
    def close(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False
```

---

## 6. Training Data Generation Pipeline

### 6.1 Parallel Simulation Execution

To generate **10,000-50,000 episodes**, parallelize simulations:

**File: `generate_training_data.py`**

```python
import multiprocessing as mp
from oxford_hydepark_env import OxfordHydeParkEnv
import numpy as np
import pickle
from tqdm import tqdm

def run_single_episode(episode_config):
    """Run one simulation episode and collect data"""
    episode_id, scenario, seed = episode_config
    
    np.random.seed(seed)
    
    env = OxfordHydeParkEnv(
        route_file=f'routes_{scenario}.rou.xml',
        num_seconds=3600,  # 1 hour episodes
        use_gui=False
    )
    
    state = env.reset()
    episode_data = []
    total_reward = 0
    
    done = False
    while not done:
        # Random policy for data collection (exploratory)
        action = env.action_space.sample()
        
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        episode_data.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info
        })
        
        state = next_state
        total_reward += reward
    
    env.close()
    
    return {
        'episode_id': episode_id,
        'scenario': scenario,
        'transitions': episode_data,
        'total_reward': total_reward,
        'metrics': {
            'avg_wait_time': np.mean([t['info']['total_wait_time'] for t in episode_data]),
            'avg_queue': np.mean([t['info']['queue_length'] for t in episode_data]),
            'throughput': sum([t['info']['throughput'] for t in episode_data])
        }
    }

def generate_massive_dataset(num_episodes=50000, num_workers=16):
    """Generate large-scale training dataset using parallel workers"""
    
    scenarios = [
        'am_rush', 'pm_rush', 'midday', 'evening', 'night',
        'special_event', 'accident', 'weather'
    ] * (num_episodes // 8)  # Repeat scenarios to reach target episodes
    
    # Create episode configurations
    episode_configs = [
        (i, scenarios[i % len(scenarios)], np.random.randint(0, 1000000))
        for i in range(num_episodes)
    ]
    
    # Parallel execution
    print(f"Generating {num_episodes} episodes using {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(run_single_episode, episode_configs),
            total=num_episodes
        ))
    
    # Save dataset in chunks (avoid memory overflow)
    chunk_size = 5000
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i+chunk_size]
        with open(f'training_data/dataset_chunk_{i//chunk_size:04d}.pkl', 'wb') as f:
            pickle.dump(chunk, f)
    
    print(f"Dataset generation complete. Saved {len(results)} episodes.")
    
    return results

if __name__ == '__main__':
    # Generate 50,000 episodes
    generate_massive_dataset(num_episodes=50000, num_workers=16)
```

**Execution**:
```bash
# Create output directory
mkdir -p training_data

# Run data generation (this will take hours/days depending on hardware)
python generate_training_data.py
```

### 6.2 Data Storage and Management

Organize generated data:

```
training_data/
├── dataset_chunk_0000.pkl  (episodes 0-4999)
├── dataset_chunk_0001.pkl  (episodes 5000-9999)
├── ...
├── dataset_chunk_0009.pkl  (episodes 45000-49999)
├── metadata.json           (statistics, scenario distributions)
└── validation_set/
    └── dataset_chunk_val.pkl  (10% held out for validation)
```

---

## 7. Agent Training Architecture

### 7.1 Deep Q-Network (DQN) Training Pipeline

**File: `train_deusnegotiatio.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pickle
from tqdm import tqdm

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class PrioritizedReplayBuffer:
    """Experience replay with prioritized sampling"""
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

def load_dataset_chunks(data_dir='training_data'):
    """Load all dataset chunks"""
    import glob
    
    all_transitions = []
    chunk_files = sorted(glob.glob(f'{data_dir}/dataset_chunk_*.pkl'))
    
    for chunk_file in tqdm(chunk_files, desc="Loading dataset"):
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
        
        for episode in chunk:
            all_transitions.extend(episode['transitions'])
    
    return all_transitions

def train_deusnegotiatio(
    num_training_steps=1000000,
    batch_size=64,
    gamma=0.99,
    lr=0.0001,
    target_update_freq=1000,
    buffer_capacity=100000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Main training loop for DeusNegotiatio agent"""
    
    print(f"Training on device: {device}")
    
    # Initialize networks
    state_dim = 60  # From environment
    action_dim = 8   # 8 phase options
    
    q_network = DuelingDQN(state_dim, action_dim).to(device)
    target_network = DuelingDQN(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    
    # Initialize replay buffer
    replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
    
    # Load pre-generated dataset
    print("Loading training data...")
    transitions = load_dataset_chunks()
    
    # Populate replay buffer
    print("Populating replay buffer...")
    for trans in tqdm(transitions[:buffer_capacity]):
        replay_buffer.add(
            trans['state'],
            trans['action'],
            trans['reward'],
            trans['next_state'],
            trans['done']
        )
    
    # Training loop
    print("Starting training...")
    losses = []
    
    for step in tqdm(range(num_training_steps)):
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = \
            replay_buffer.sample(batch_size, beta=0.4 + step * 0.0001)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Compute Q-values
        q_values = q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            next_q_values = q_network(next_states)
            next_actions = next_q_values.max(1)[1]
            
            target_q_values = target_network(next_states)
            target_q_value = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target = rewards + gamma * target_q_value * (1 - dones)
        
        # Compute loss
        td_errors = target - q_value
        loss = (weights * td_errors.pow(2)).mean()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
        optimizer.step()
        
        # Update priorities
        priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        replay_buffer.update_priorities(indices, priorities)
        
        losses.append(loss.item())
        
        # Update target network
        if step % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Logging
        if step % 1000 == 0:
            avg_loss = np.mean(losses[-1000:])
            print(f"Step {step}, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if step % 10000 == 0:
            torch.save({
                'step': step,
                'model_state_dict': q_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, f'checkpoints/deusnegotiatio_step_{step}.pt')
    
    print("Training complete!")
    return q_network

if __name__ == '__main__':
    # Train the agent
    trained_model = train_deusnegotiatio(num_training_steps=1000000)
    
    # Save final model
    torch.save(trained_model.state_dict(), 'models/deusnegotiatio_final.pt')
```

### 7.2 Training Execution

```bash
# Create directories
mkdir -p checkpoints models logs

# Start training (GPU recommended)
python train_deusnegotiatio.py

# Monitor training (if using tensorboard)
tensorboard --logdir=logs
```

**Expected Training Time**:
- **Dataset Generation**: 48-72 hours (50k episodes, 16 cores)
- **Model Training**: 12-24 hours (1M steps, GPU)
- **Total Pipeline**: 3-5 days

---

## 8. Validation and Performance Metrics

### 8.1 Evaluation Script

**File: `evaluate_agent.py`**

```python
from oxford_hydepark_env import OxfordHydeParkEnv
import torch
import numpy as np
from train_deusnegotiatio import DuelingDQN

def evaluate_agent(model_path, num_episodes=100, scenarios=None):
    """Evaluate trained agent on test scenarios"""
    
    if scenarios is None:
        scenarios = ['am_rush', 'pm_rush', 'midday', 'evening']
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DuelingDQN(state_dim=60, action_dim=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    results = {scenario: [] for scenario in scenarios}
    
    for scenario in scenarios:
        print(f"\nEvaluating on scenario: {scenario}")
        
        env = OxfordHydeParkEnv(
            route_file=f'routes_{scenario}.rou.xml',
            num_seconds=3600,
            use_gui=False
        )
        
        for episode in range(num_episodes // len(scenarios)):
            state = env.reset()
            episode_reward = 0
            episode_metrics = {
                'wait_times': [],
                'queue_lengths': [],
                'throughput': 0
            }
            
            done = False
            with torch.no_grad():
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
                    
                    next_state, reward, done, info = env.step(action)
                    
                    episode_reward += reward
                    episode_metrics['wait_times'].append(info['total_wait_time'])
                    episode_metrics['queue_lengths'].append(info['queue_length'])
                    episode_metrics['throughput'] += info['throughput']
                    
                    state = next_state
            
            results[scenario].append({
                'reward': episode_reward,
                'avg_wait_time': np.mean(episode_metrics['wait_times']),
                'avg_queue': np.mean(episode_metrics['queue_lengths']),
                'total_throughput': episode_metrics['throughput']
            })
        
        env.close()
    
    # Print summary statistics
    print("\n=== Evaluation Results ===")
    for scenario, episodes in results.items():
        avg_reward = np.mean([e['reward'] for e in episodes])
        avg_wait = np.mean([e['avg_wait_time'] for e in episodes])
        avg_queue = np.mean([e['avg_queue'] for e in episodes])
        avg_throughput = np.mean([e['total_throughput'] for e in episodes])
        
        print(f"\n{scenario}:")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Wait Time: {avg_wait:.2f} s")
        print(f"  Avg Queue Length: {avg_queue:.2f} veh")
        print(f"  Avg Throughput: {avg_throughput:.0f} veh/hr")
    
    return results

if __name__ == '__main__':
    results = evaluate_agent('models/deusnegotiatio_final.pt', num_episodes=100)
```

### 8.2 Comparison with Baseline (SURTRAC-style Fixed Timing)

```python
def compare_with_baseline():
    """Compare DeusNegotiatio with fixed-time control"""
    
    print("Evaluating DeusNegotiatio...")
    rl_results = evaluate_agent('models/deusnegotiatio_final.pt', num_episodes=100)
    
    print("\nEvaluating Fixed-Time Baseline...")
    # Run simulation with default traffic light program
    baseline_results = evaluate_baseline()
    
    # Compute improvements
    for scenario in rl_results.keys():
        rl_wait = np.mean([e['avg_wait_time'] for e in rl_results[scenario]])
        baseline_wait = np.mean([e['avg_wait_time'] for e in baseline_results[scenario]])
        
        improvement = (baseline_wait - rl_wait) / baseline_wait * 100
        print(f"\n{scenario}: {improvement:.1f}% reduction in wait time")
```

---

## 9. Appendix: Code Templates and Configuration Files

### 9.1 SUMO Configuration File

**File: `simulation.sumocfg`**

```xml
<configuration>
    <input>
        <net-file value="oxford_hydepark.net.xml"/>
        <route-files value="routes_2046_am_peak_calibrated.rou.xml,pedestrians_realistic.rou.xml,cyclists_peak.rou.xml"/>
        <additional-files value="detectors.add.xml,pedestrian_crossings.add.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    
    <processing>
        <time-to-teleport value="-1"/>
        <max-depart-delay value="900"/>
        <routing-algorithm value="dijkstra"/>
    </processing>
    
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
    
    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
    </gui_only>
</configuration>
```

### 9.2 Complete Training Pipeline Script

**File: `run_full_pipeline.sh`**

```bash
#!/bin/bash
set -e

echo "=== DeusNegotiatio Training Pipeline ==="
echo "Oxford-Hyde Park Intersection"

# Step 1: Network creation
echo "\n[1/5] Creating SUMO network..."
netconvert \
  --node-files=oxford_hydepark.nod.xml \
  --edge-files=oxford_hydepark.edg.xml \
  --connection-files=oxford_hydepark.con.xml \
  --tllogic-files=oxford_hydepark.tll.xml \
  --output-file=oxford_hydepark.net.xml

# Step 2: Traffic demand generation
echo "\n[2/5] Generating traffic demand for all scenarios..."
python generate_traffic_scenarios.py

# Step 3: Generate training dataset
echo "\n[3/5] Generating training dataset (50,000 episodes)..."
echo "This will take 48-72 hours..."
python generate_training_data.py

# Step 4: Train agent
echo "\n[4/5] Training DeusNegotiatio agent..."
echo "This will take 12-24 hours on GPU..."
python train_deusnegotiatio.py

# Step 5: Evaluate
echo "\n[5/5] Evaluating trained agent..."
python evaluate_agent.py

echo "\n=== Pipeline Complete ==="
echo "Trained model saved to: models/deusnegotiatio_final.pt"
```

---

## Conclusion

This comprehensive methodology provides everything needed to:

1. **Recreate your exact intersection design** in SUMO with proper lane configurations, turning movements, and multi-modal infrastructure
2. **Generate realistic, calibrated traffic demand** matching your 2026-2046 projections from Synchro analysis
3. **Create massive training datasets** with 50,000+ diverse simulation episodes
4. **Train a robust DeusNegotiatio agent** using deep reinforcement learning with prioritized experience replay
5. **Validate performance** against baseline fixed-time control and SURTRAC-style reactive systems

The resulting trained agent will be capable of handling the complex, high-variance traffic conditions at Oxford-Hyde Park with adaptive, learned policies that continuously improve over time.

**Next Steps**:
1. Populate exact traffic volumes from your Appendix A Tables 2.1-2.4
2. Extract turn percentages from your Appendix G Tables 8.1-8.4
3. Run the full pipeline on a GPU-equipped machine
4. Fine-tune reward function weights based on City of London priorities
5. Deploy in simulation testing phase before real-world pilot

Your trained agent will represent a significant advancement over traditional traffic control, embodying the vision of DeusNegotiatio as the "deity of traffic" for this critical London, Ontario intersection.