import numpy as np

class InfraredSimulator:
    """
    Simulates a thermal camera sensor.
    Generates heat intensity values based on vehicle activity.
    """
    
    def __init__(self,
                 ambient_temp=20.0,
                 engine_temp_base=90.0,
                 brake_temp_add=30.0,
                 sensor_range=50.0):
        self.ambient_temp = ambient_temp
        self.engine_temp_base = engine_temp_base
        self.brake_temp_add = brake_temp_add
        self.sensor_range = sensor_range
        
    def capture(self, vehicles):
        """
        Generate thermal features from vehicle states.
        
        Returns:
            dict: {
                'quadrant_heat': [q1, q2, q3, q4], # Normalized heat per intersection quadrant
                'emergency_heat': bool # High heat signature detection
            }
        """
        # 4 Quadrants (NE, NW, SW, SE)
        # 0: NE (x>0, y>0), 1: NW (x<0, y>0), 2: SW (x<0, y<0), 3: SE (x>0, y<0)
        quadrant_heat = [0.0] * 4
        max_heat_signature = 0.0
        
        for veh in vehicles:
            # Calculate heat signature
            # Base heat from engine (moving fast = hotter airflow/work)
            # Actually, engine block is hot but airflow cools it. 
            # Thermal cameras see TIRES and EXHAUST.
            # Tires get hotter with speed. Brakes get HOT.
            
            speed = veh.get('speed', 0.0)
            accel = veh.get('acceleration', 0.0)
            
            # Heat model
            heat = self.engine_temp_base
            
            # Speed factor (tires warm up)
            heat += speed * 2.0
            
            # Braking factor (huge heat spike)
            if accel < -1.0: # Decelerating significantly
                heat += self.brake_temp_add * abs(accel)
                
            max_heat_signature = max(max_heat_signature, heat)
            
            # Locality
            x, y = veh['x'], veh['y']
            dist = np.sqrt(x**2 + y**2)
            
            if dist > self.sensor_range:
                continue
                
            # Inverse square law for sensor intensity
            measured_intensity = heat / (dist**0.5 + 1.0)
            
            # Assign to quadrant
            q_idx = 0
            if x >= 0 and y >= 0: q_idx = 0      # NE
            elif x < 0 and y >= 0: q_idx = 1     # NW
            elif x < 0 and y < 0: q_idx = 2      # SW
            else: q_idx = 3                      # SE
            
            quadrant_heat[q_idx] += measured_intensity

        # Normalize heat
        # Assume a 'max' heat saturation level of 500 units per quadrant
        quadrant_heat = np.clip(np.array(quadrant_heat) / 500.0, 0, 1.0)
        
        # Emergency vehicle detection (very high heat or special signature)
        # For simulation, we check metadata or assume very high speed + heat
        is_emergency = max_heat_signature > 150.0 
        
        return {
            'quadrant_heat': quadrant_heat.tolist(),
            'emergency_heat': 1.0 if is_emergency else 0.0
        }
