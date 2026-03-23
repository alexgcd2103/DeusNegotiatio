import sys
import os
import traci
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ensure_sumo_home
ensure_sumo_home()

from envs.oxford_hydepark_env import OxfordHydeParkEnv

def verify_emergency():
    print("Starting Emergency Priority/Pull-over Verification...")
    env = OxfordHydeParkEnv(use_gui=False)
    env.reset()
    
    found_ev = False
    
    for i in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        is_preempted = info.get('emergency_preemption', False)
        vehs = traci.vehicle.getIDList()
        evs = [v for v in vehs if traci.vehicle.getTypeID(v) == "emergency"]
        
        if evs:
            found_ev = True
            ev_id = evs[0]
            x, y = traci.vehicle.getPosition(ev_id)
            ev_road = traci.vehicle.getRoadID(ev_id)
            dist = np.sqrt((x - 500.0)**2 + (y - 500.0)**2)
            phase = traci.trafficlight.getPhase('C')
            
            # Check surrounding vehicles on the same road
            surrounding = [v for v in vehs if traci.vehicle.getRoadID(v) == ev_road and v != ev_id]
            lat_offsets = []
            for sv in surrounding[:5]:
                try:
                    lat_offsets.append(traci.vehicle.getLateralLanePosition(sv))
                except traci.exceptions.TraCIException:
                    lat_offsets.append(0.0)
            
            print(f"Step {i:3d} | EV {ev_id} | Dist: {dist:6.1f}m | Phase: {phase} | Preempted: {is_preempted} | LatOffsets: {lat_offsets}")
                
        if terminated:
            break
            
    env.close()

if __name__ == "__main__":
    verify_emergency()
