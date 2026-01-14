class MultiAgentCoordinator:
    """
    Handles coordination between intersection agents.
    """
    def __init__(self, agent_id, neighbor_map):
        self.agent_id = agent_id
        self.neighbors = neighbor_map # dict of direction -> agent_id
        self.incoming_messages = {}
        
    def receive_message(self, sender_id, message):
        """
        Store received outflow predictions or requests.
        """
        self.incoming_messages[sender_id] = message
        
    def get_neighbor_states(self):
        """
        Integrate neighbor info into local state context.
        """
        # Parse incoming messages to extract relevant features
        # e.g. "North neighbor predicts 20 cars coming in 30s"
        context_features = []
        for n_id, msg in self.incoming_messages.items():
            # Dummy feature extraction
            obs = msg.get('outflow_prediction', {}).get('predicted_vehicles', 0)
            context_features.append(obs)
            
        return context_features

    def consensus_check(self, proposed_action):
        """
        Check if proposed action conflicts with strong neighbor requests.
        """
        # Placeholder for consensus voting logic
        return True # Approved
