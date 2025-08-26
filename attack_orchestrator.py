import numpy as np
from .agents import QuantumVulnerabilityAgent, NeuralArchitectureSearchAgent, MetaLearningAttackAgent

class AdvancedAttackOrchestrator:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or self._default_config()
        
        # Initialize all agents with corrected constructor calls
        self.quantum_agent = QuantumVulnerabilityAgent(
            model, 
            epsilon=self.config['epsilon'],
            quantum_depth=self.config['quantum_depth']
        )
        
        self.nas_agent = NeuralArchitectureSearchAgent(
            model,
            epsilon=self.config['epsilon'],
            population_size=self.config['nas_population_size']
        )
        
        self.meta_agent = MetaLearningAttackAgent(
            model,
            epsilon=self.config['epsilon'],
            meta_lr=self.config['meta_learning_rate']
        )
    
    def _default_config(self):
        return {
            'epsilon': 0.1,
            'quantum_depth': 6,
            'nas_population_size': 12,
            'meta_learning_rate': 0.01
        }
    
    def execute_multi_agent_attack(self, input_data, attack_types=['quantum', 'nas', 'meta']):
        """Execute coordinated multi-agent attack"""
        results = {}
        
        if 'quantum' in attack_types:
            print("Executing Quantum Vulnerability Attack...")
            results['quantum'] = self.quantum_agent.execute_quantum_superposition_attacks(input_data)
        
        if 'nas' in attack_types:
            print("Executing Neural Architecture Search Attack...")
            results['nas'] = self.nas_agent.evolve_attack_architectures(input_data)
        
        if 'meta' in attack_types:
            print("Executing Meta-Learning Attack...")
            # For meta-learning, we need support and query examples
            support_examples = [input_data] * 5 # Simplified for demo
            query_examples = [input_data]
            results['meta'] = self.meta_agent.few_shot_attack_adaptation(
                support_examples, query_examples
            )
        
        return self._combine_attack_results(results)
    
    def _combine_attack_results(self, results):
        """Combine results from multiple agents"""
        combined = {
            'best_attack': None,
            'best_score': 0,
            'agent_results': results,
            'ensemble_perturbation': None
        }
        
        # Find best individual attack
        for agent_name, result in results.items():
            score = self._extract_score(result)
            if score > combined['best_score']:
                combined['best_score'] = score
                combined['best_attack'] = {
                    'agent': agent_name,
                    'result': result
                }
        
        # Create ensemble perturbation
        perturbations = []
        for result in results.values():
            perturbation = self._extract_perturbation(result)
            if perturbation is not None:
                perturbations.append(perturbation)
        
        if perturbations:
            combined['ensemble_perturbation'] = np.mean(perturbations, axis=0)
        
        return combined
    
    def _extract_score(self, result):
        """Extract score from agent result - FIXED METHOD"""
        if isinstance(result, dict):
            # Try different score keys depending on agent type
            score_keys = ['vulnerability_score', 'final_fitness', 'average_effectiveness']
            for key in score_keys:
                if key in result:
                    return result[key]
        return 0.0
    
    def _extract_perturbation(self, result):
        """Extract perturbation from agent result - FIXED METHOD"""
        if isinstance(result, dict):
            # Try different perturbation keys
            perturbation_keys = ['best_attack_vector', 'perturbation', 'attack_perturbation']
            for key in perturbation_keys:
                if key in result and result[key] is not None:
                    perturbation = result[key]
                    if hasattr(perturbation, 'shape'):
                        return perturbation
                    elif isinstance(perturbation, (list, tuple)):
                        return np.array(perturbation)
        return None
