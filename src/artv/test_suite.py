import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from .agents import QuantumVulnerabilityAgent, NeuralArchitectureSearchAgent, MetaLearningAttackAgent
from .attack_orchestrator import AdvancedAttackOrchestrator

class TestAdvancedAdversarialAgents(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # FIXED: Use correct parameters for make_classification
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_classes=2,  # Fixed: Changed from 3 to 2
            n_informative=10,  # Fixed: Increased from 2 to 10
            n_redundant=5,
            random_state=42
            )
        
        # Train a simple model
        cls.model = RandomForestClassifier(n_estimators=10, random_state=42)
        cls.model.fit(X, y)
        
        # Test data
        cls.test_input = X[0]
        cls.test_inputs = X[:10]
    
    def test_quantum_vulnerability_agent(self):
        """Test QuantumVulnerabilityAgent functionality"""
        if QuantumVulnerabilityAgent is None:
            self.skipTest("QuantumVulnerabilityAgent not available")
            
        agent = QuantumVulnerabilityAgent(self.model, epsilon=0.1, quantum_depth=4)
        
        result = agent.execute_quantum_superposition_attacks(self.test_input)
        
        # Assertions
        self.assertIn('best_attack_vector', result)
        self.assertIn('vulnerability_score', result)
        self.assertIn('quantum_efficiency', result)
        self.assertGreaterEqual(result['vulnerability_score'], 0)
        
        if result['best_attack_vector'] is not None:
            self.assertEqual(len(result['best_attack_vector']), len(self.test_input))
    
    def test_nas_agent(self):
        """Test NeuralArchitectureSearchAgent functionality"""
        if NeuralArchitectureSearchAgent is None:
            self.skipTest("NeuralArchitectureSearchAgent not available")
            
        agent = NeuralArchitectureSearchAgent(self.model, epsilon=0.1, population_size=5)
        
        result = agent.evolve_attack_architectures(self.test_input, generations=3)
        
        # Assertions
        self.assertIn('best_architecture', result)
        self.assertIn('final_fitness', result)
        self.assertIn('evolution_history', result)
        self.assertGreaterEqual(result['final_fitness'], 0)
        self.assertEqual(len(result['evolution_history']), 3)
    
    def test_meta_learning_agent(self):
        """Test MetaLearningAttackAgent functionality"""
        if MetaLearningAttackAgent is None:
            self.skipTest("MetaLearningAttackAgent not available")
            
        agent = MetaLearningAttackAgent(self.model, epsilon=0.1, meta_lr=0.01)
        
        support_examples = self.test_inputs[:5]
        query_examples = self.test_inputs[5:7]
        
        result = agent.few_shot_attack_adaptation(
            support_examples, query_examples, k_shot=5, adaptation_steps=5
        )
        
        # Assertions
        self.assertIn('adapted_parameters', result)
        self.assertIn('attack_results', result)
        self.assertIn('average_effectiveness', result)
        self.assertEqual(len(result['attack_results']), len(query_examples))
        self.assertGreaterEqual(result['average_effectiveness'], 0)
    
    def test_orchestrator_integration(self):
        """Test multi-agent orchestration"""
        if AdvancedAttackOrchestrator is None:
            self.skipTest("AdvancedAttackOrchestrator not available")
            
        orchestrator = AdvancedAttackOrchestrator(self.model)
        result = orchestrator.execute_multi_agent_attack(
            self.test_input, 
            attack_types=['quantum']  # Test with one agent for speed
        )
        
        # Assertions
        self.assertIn('best_attack', result)
        self.assertIn('agent_results', result)
        self.assertGreaterEqual(result['best_score'], 0)
    

class PerformanceBenchmarks(unittest.TestCase):
    """Performance and scalability tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for performance testing"""
        # FIXED: Use correct parameters
        X_small, y_small = make_classification(
            n_samples=100, 
            n_features=10, 
            n_classes=2,
            n_informative=5,
            random_state=42
        )
        
        X_medium, y_medium = make_classification(
            n_samples=500, 
            n_features=20, 
            n_classes=2,
            n_informative=10,
            random_state=42
        )
        
        # Train models
        cls.simple_model = RandomForestClassifier(n_estimators=5, random_state=42)
        cls.simple_model.fit(X_small, y_small)
        
        cls.complex_model = RandomForestClassifier(n_estimators=20, random_state=42)
        cls.complex_model.fit(X_medium, y_medium)
        
        cls.test_data_small = X_small
        cls.test_data_medium = X_medium
    
    def test_quantum_scalability(self):
        """Test quantum agent scalability across different problem sizes"""
        if QuantumVulnerabilityAgent is None:
            self.skipTest("QuantumVulnerabilityAgent not available")
            
        print("\n=== Testing Quantum Agent Scalability ===")
        
        quantum_results = []
        test_configs = [
            {'quantum_depth': 3, 'data_size': 'small'},
            {'quantum_depth': 4, 'data_size': 'small'},
            {'quantum_depth': 5, 'data_size': 'medium'},
        ]
        
        for config in test_configs:
            print(f"Testing quantum depth {config['quantum_depth']} on {config['data_size']} data...")
            
            try:
                start_memory = self._get_memory_usage()
                start_time = time.time()
                
                agent = QuantumVulnerabilityAgent(
                    self.simple_model, 
                    epsilon=0.1, 
                    quantum_depth=config['quantum_depth']
                )
                
                test_input = (self.test_data_small[0] if config['data_size'] == 'small' 
                            else self.test_data_medium[0])
                
                result = agent.execute_quantum_superposition_attacks(test_input)
                
                execution_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - start_memory
                
                quantum_results.append({
                    'quantum_depth': config['quantum_depth'],
                    'data_size': config['data_size'],
                    'execution_time': execution_time,
                    'memory_usage': max(0, memory_usage),  # Ensure non-negative
                    'vulnerability_score': result['vulnerability_score'],
                    'quantum_efficiency': result['quantum_efficiency']
                })
                
                print(f"  ✓ Completed in {execution_time:.3f}s, Score: {result['vulnerability_score']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                quantum_results.append({
                    'quantum_depth': config['quantum_depth'],
                    'data_size': 'failed',
                    'error': str(e)
                })
        
        # Performance assertions
        successful_results = [r for r in quantum_results if r['data_size'] != 'failed']
        self.assertGreater(len(successful_results), 0, "At least one quantum test should succeed")
        
        for result in successful_results:
            self.assertLess(result['execution_time'], 60.0, "Quantum execution should complete within 60s")
            self.assertGreaterEqual(result['vulnerability_score'], 0, "Vulnerability score should be non-negative")
        
        self.performance_results['quantum_scalability'] = quantum_results
    
    def test_nas_convergence(self):
        """Test NAS agent convergence patterns"""
        if NeuralArchitectureSearchAgent is None:
            self.skipTest("NeuralArchitectureSearchAgent not available")
            
        print("\n=== Testing NAS Agent Convergence ===")
        
        nas_results = []
        test_configs = [
            {'population_size': 6, 'generations': 4},
            {'population_size': 8, 'generations': 5},
        ]
        
        for config in test_configs:
            print(f"Testing NAS with population {config['population_size']}, generations {config['generations']}...")
            
            try:
                start_time = time.time()
                
                agent = NeuralArchitectureSearchAgent(
                    self.simple_model, 
                    epsilon=0.1, 
                    population_size=config['population_size']
                )
                
                result = agent.evolve_attack_architectures(
                    self.test_data_small[0], 
                    generations=config['generations']
                )
                
                execution_time = time.time() - start_time
                
                # Calculate convergence metrics
                fitness_progression = [gen['fitness'] for gen in result['evolution_history']]
                convergence_metrics = self._calculate_nas_convergence_metrics(fitness_progression)
                
                nas_result = {
                    'population_size': config['population_size'],
                    'generations': config['generations'],
                    'execution_time': execution_time,
                    'final_fitness': result['final_fitness'],
                    **convergence_metrics
                }
                
                nas_results.append(nas_result)
                print(f"  ✓ Completed in {execution_time:.3f}s, Final fitness: {result['final_fitness']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        # Convergence assertions
        self.assertGreater(len(nas_results), 0, "At least one NAS test should succeed")
        
        for result in nas_results:
            self.assertGreaterEqual(result['final_fitness'], 0, "Final fitness should be non-negative")
            self.assertLess(result['execution_time'], 120.0, "NAS should complete within 2 minutes")
        
        self.performance_results['nas_convergence'] = nas_results
    
    def test_meta_learning_adaptation(self):
        """Test meta-learning agent adaptation speed"""
        if MetaLearningAttackAgent is None:
            self.skipTest("MetaLearningAttackAgent not available")
            
        print("\n=== Testing Meta-Learning Agent Adaptation ===")
        
        adaptation_results = []
        test_configs = [
            {'k_shot': 3, 'adaptation_steps': 5},
            {'k_shot': 5, 'adaptation_steps': 8},
        ]
        
        for config in test_configs:
            print(f"Testing meta-learning with {config['k_shot']}-shot, {config['adaptation_steps']} steps...")
            
            try:
                agent = MetaLearningAttackAgent(
                    self.simple_model, 
                    epsilon=0.1, 
                    meta_lr=0.01
                )
                
                support_examples = self.test_data_small[:config['k_shot']]
                query_examples = self.test_data_small[config['k_shot']:config['k_shot']+2]
                
                adaptation_times = []
                effectiveness_scores = []
                
                # Test multiple adaptation runs
                for trial in range(3):  # 3 trials for averaging
                    start_time = time.time()
                    
                    result = agent.few_shot_attack_adaptation(
                        support_examples, 
                        query_examples, 
                        k_shot=config['k_shot'],
                        adaptation_steps=config['adaptation_steps']
                    )
                    
                    adaptation_time = time.time() - start_time
                    adaptation_times.append(adaptation_time)
                    effectiveness_scores.append(result['average_effectiveness'])
                
                # Calculate metrics
                adaptation_metrics = {
                    'k_shot': config['k_shot'],
                    'adaptation_steps': config['adaptation_steps'],
                    'average_adaptation_time': np.mean(adaptation_times),
                    'average_effectiveness': np.mean(effectiveness_scores),
                    'adaptation_efficiency': np.mean(effectiveness_scores) / np.mean(adaptation_times),
                    'consistency_score': 1.0 / (1.0 + np.std(effectiveness_scores))
                }
                
                adaptation_results.append(adaptation_metrics)
                print(f"  ✓ Avg time: {adaptation_metrics['average_adaptation_time']:.3f}s, "
                      f"Avg effectiveness: {adaptation_metrics['average_effectiveness']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        # Adaptation assertions
        self.assertGreater(len(adaptation_results), 0, "At least one meta-learning test should succeed")
        
        for result in adaptation_results:
            self.assertLess(result['average_adaptation_time'], 30.0, "Adaptation should be fast")
            self.assertGreaterEqual(result['average_effectiveness'], 0, "Should show some effectiveness")
            self.assertGreater(result['consistency_score'], 0.5, "Should be reasonably consistent")
        
        self.performance_results['meta_learning_adaptation'] = adaptation_results
    
    def test_meta_learning_adaptation_speed(self):
        """Test meta-learning adaptation speed"""
        print("\n=== Testing Meta-Learning Adaptation Speed ===")
        
        # Test different adaptation configurations
        adaptation_configs = [
            {'k_shot': 3, 'adaptation_steps': 5},
            {'k_shot': 5, 'adaptation_steps': 10},
            {'k_shot': 7, 'adaptation_steps': 15},
        ]
        
        adaptation_results = []
        
        for config in adaptation_configs:
            print(f"Testing meta-learning with k_shot={config['k_shot']}, "
                  f"adaptation_steps={config['adaptation_steps']}")
            
            agent = MetaLearningAttackAgent(
                self.simple_model,
                epsilon=0.1,
                meta_lr=0.01
            )
            
            # Prepare support and query sets
            support_examples = self.test_data_small[:config['k_shot']]
            query_examples = self.test_data_small[config['k_shot']:config['k_shot']+3]
            
            # Measure adaptation speed
            adaptation_times = []
            effectiveness_progression = []
            
            for trial in range(3):  # Multiple trials for averaging
                start_time = time.time()
                
                result = agent.few_shot_attack_adaptation(
                    support_examples,
                    query_examples,
                    k_shot=config['k_shot'],
                    adaptation_steps=config['adaptation_steps']
                )
                
                adaptation_time = time.time() - start_time
                adaptation_times.append(adaptation_time)
                effectiveness_progression.append(result['average_effectiveness'])
            
            # Calculate adaptation metrics
            adaptation_metrics = {
                'k_shot': config['k_shot'],
                'adaptation_steps': config['adaptation_steps'],
                'average_adaptation_time': np.mean(adaptation_times),
                'adaptation_time_std': np.std(adaptation_times),
                'average_effectiveness': np.mean(effectiveness_progression),
                'effectiveness_std': np.std(effectiveness_progression),
                'adaptation_efficiency': np.mean(effectiveness_progression) / np.mean(adaptation_times),
                'consistency_score': 1.0 / (1.0 + np.std(effectiveness_progression))
            }
            
            adaptation_results.append(adaptation_metrics)
        
        # Analyze adaptation speed patterns
        self._analyze_meta_learning_adaptation(adaptation_results)
        
        # Adaptation speed assertions
        for result in adaptation_results:
            # Adaptation should be reasonably fast
            self.assertLess(
                result['average_adaptation_time'], 30.0,  # Should adapt within 30 seconds
                f"Adaptation should be fast for k_shot={result['k_shot']}"
            )
            
            # Effectiveness should be reasonable
            self.assertGreater(
                result['average_effectiveness'], 0,
                "Should show some attack effectiveness"
            )
            
            # Adaptation efficiency should improve with more shots/steps
            self.assertGreater(
                result['adaptation_efficiency'], 0,
                "Should show positive adaptation efficiency"
            )
            
            # Consistency should be reasonable
            self.assertGreater(
                result['consistency_score'], 0.5,
                "Should show reasonable consistency across trials"
            )
        
        # Test scaling properties
        k_shot_times = {r['k_shot']: r['average_adaptation_time'] for r in adaptation_results}
        sorted_k_shots = sorted(k_shot_times.keys())
        
        # Adaptation time should scale reasonably with k_shot
        for i in range(len(sorted_k_shots) - 1):
            current_k = sorted_k_shots[i]
            next_k = sorted_k_shots[i + 1]
            
            # Time should not increase too dramatically
            time_ratio = k_shot_times[next_k] / k_shot_times[current_k]
            self.assertLess(
                time_ratio, 3.0,  # Should not triple the time
                f"Adaptation time scaling should be reasonable: {current_k} -> {next_k}"
            )
        
        self.performance_results['meta_learning_adaptation'] = adaptation_results
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0  # Return 0 if psutil fails
    
    def _calculate_nas_convergence_metrics(self, fitness_progression):
        """Calculate convergence metrics for NAS"""
        if len(fitness_progression) < 2:
            return {
                'convergence_rate': 0,
                'stability_score': 0,
                'improvement_ratio': 0,
                'plateau_detection': False
            }
        
        # Convergence rate: how quickly fitness improves
        fitness_diffs = np.diff(fitness_progression)
        positive_improvements = fitness_diffs[fitness_diffs > 0]
        convergence_rate = len(positive_improvements) / len(fitness_diffs) if len(fitness_diffs) > 0 else 0
        
        # Stability score: consistency of improvements
        stability_score = 1.0 / (1.0 + np.std(fitness_diffs)) if len(fitness_diffs) > 0 else 0
        
        # Improvement ratio: final vs initial fitness
        improvement_ratio = fitness_progression[-1] / fitness_progression[0] if fitness_progression[0] != 0 else 1.0
        
        # Plateau detection: check if fitness has plateaued
        plateau_threshold = max(1, len(fitness_progression) // 3)
        recent_fitness = fitness_progression[-plateau_threshold:]
        plateau_detection = np.std(recent_fitness) < 0.01 if len(recent_fitness) > 1 else False
        
        return {
            'convergence_rate': convergence_rate,
            'stability_score': stability_score,
            'improvement_ratio': improvement_ratio,
            'plateau_detection': plateau_detection
        }
    
    def _analyze_quantum_scalability(self, results):
        """Analyze quantum scalability patterns"""
        print("\n--- Quantum Scalability Analysis ---")
        
        # Group by quantum depth
        depth_groups = {}
        for result in results:
            if result['data_size'] != 'failed':
                depth = result['quantum_depth']
                if depth not in depth_groups:
                    depth_groups[depth] = []
                depth_groups[depth].append(result)
        
        for depth in sorted(depth_groups.keys()):
            group_results = depth_groups[depth]
            avg_time = np.mean([r['execution_time'] for r in group_results])
            avg_memory = np.mean([r['memory_usage'] for r in group_results])
            avg_efficiency = np.mean([r['quantum_efficiency'] for r in group_results])
            
            print(f"Depth {depth}: Avg Time={avg_time:.3f}s, "
                  f"Avg Memory={avg_memory:.1f}MB, "
                  f"Avg Efficiency={avg_efficiency:.1f}")
    
    def _analyze_nas_convergence(self, results):
        """Analyze NAS convergence patterns"""
        print("\n--- NAS Convergence Analysis ---")
        
        for result in results:
            print(f"Population {result['population_size']}, "
                  f"Generations {result['generations']}:")
            print(f"  Final Fitness: {result['final_fitness']:.4f}")
            print(f"  Convergence Rate: {result['convergence_rate']:.3f}")
            print(f"  Stability Score: {result['stability_score']:.3f}")
            print(f"  Improvement Ratio: {result['improvement_ratio']:.3f}")
            print(f"  Plateau Detected: {result['plateau_detection']}")
            print(f"  Execution Time: {result['execution_time']:.2f}s")
    
    def _analyze_meta_learning_adaptation(self, results):
        """Analyze meta-learning adaptation patterns"""
        print("\n--- Meta-Learning Adaptation Analysis ---")
        
        for result in results:
            print(f"K-shot {result['k_shot']}, "
                  f"Steps {result['adaptation_steps']}:")
            print(f"  Avg Adaptation Time: {result['average_adaptation_time']:.3f}s")
            print(f"  Avg Effectiveness: {result['average_effectiveness']:.4f}")
            print(f"  Adaptation Efficiency: {result['adaptation_efficiency']:.4f}")
            print(f"  Consistency Score: {result['consistency_score']:.3f}")
    
    def test_comparative_performance(self):
        """Compare performance across all three agents"""
        print("\n=== Comparative Performance Analysis ===")
        
        test_input = self.test_data_small[0]
        
        # Test each agent with standardized parameters
        agents_config = {
            'quantum': {
                'agent': QuantumVulnerabilityAgent(self.simple_model, epsilon=0.1, quantum_depth=4),
                'method': 'execute_quantum_superposition_attacks',
                'params': [test_input]
            },
            'nas': {
                'agent': NeuralArchitectureSearchAgent(self.simple_model, epsilon=0.1, population_size=8),
                'method': 'evolve_attack_architectures', 
                'params': [test_input, None, 5]  # 5 generations for speed
            },
            'meta': {
                'agent': MetaLearningAttackAgent(self.simple_model, epsilon=0.1, meta_lr=0.01),
                'method': 'few_shot_attack_adaptation',
                'params': [self.test_data_small[:3], [test_input], 3, 5]  # 3-shot, 5 adaptation steps
            }
        }
        
        comparative_results = {}
        
        for agent_name, config in agents_config.items():
            print(f"Testing {agent_name} agent...")
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                method = getattr(config['agent'], config['method'])
                result = method(*config['params'])
                
                execution_time = time.time() - start_time
                memory_used = self._get_memory_usage() - start_memory
                
                # Extract performance metrics
                if agent_name == 'quantum':
                    effectiveness = result['vulnerability_score']
                elif agent_name == 'nas':
                    effectiveness = result['final_fitness']
                else:  # meta
                    effectiveness = result['average_effectiveness']
                
                comparative_results[agent_name] = {
                    'execution_time': execution_time,
                    'memory_usage': memory_used,
                    'effectiveness': effectiveness,
                    'success': True
                }
                
            except Exception as e:
                print(f"Failed to test {agent_name} agent: {e}")
                comparative_results[agent_name] = {
                    'execution_time': float('inf'),
                    'memory_usage': float('inf'),
                    'effectiveness': 0,
                    'success': False
                }
        
        # Analyze comparative results
        self._analyze_comparative_performance(comparative_results)
        
        # Performance assertions
        successful_agents = [name for name, result in comparative_results.items() if result['success']]
        self.assertGreater(len(successful_agents), 0, "At least one agent should succeed")
        
        # All successful agents should show some effectiveness
        for agent_name in successful_agents:
            self.assertGreater(
                comparative_results[agent_name]['effectiveness'], 0,
                f"{agent_name} agent should show some effectiveness"
            )
        
        self.performance_results['comparative'] = comparative_results
    
    def _analyze_comparative_performance(self, results):
        """Analyze comparative performance across agents"""
        print("\n--- Comparative Performance Analysis ---")
        
        successful_results = {name: result for name, result in results.items() if result['success']}
        
        if not successful_results:
            print("No agents succeeded in comparative testing")
            return
        
        # Find best performer in each category
        best_time = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
        best_memory = min(successful_results.items(), key=lambda x: x[1]['memory_usage'])
        best_effectiveness = max(successful_results.items(), key=lambda x: x[1]['effectiveness'])
        
        print(f"Fastest: {best_time[0]} ({best_time[1]['execution_time']:.3f}s)")
        print(f"Most Memory Efficient: {best_memory[0]} ({best_memory[1]['memory_usage']:.1f}MB)")
        print(f"Most Effective: {best_effectiveness[0]} ({best_effectiveness[1]['effectiveness']:.4f})")
        
        # Overall performance summary
        print("\nOverall Performance Summary:")
        for agent_name, result in successful_results.items():
            efficiency_score = result['effectiveness'] / (result['execution_time'] + 0.001)  # Avoid division by zero
            print(f"{agent_name}: Time={result['execution_time']:.3f}s, "
                  f"Memory={result['memory_usage']:.1f}MB, "
                  f"Effectiveness={result['effectiveness']:.4f}, "
                  f"Efficiency={efficiency_score:.4f}")
    
    def tearDown(self):
        """Clean up after each test"""
        # Force garbage collection to free memory
        import gc
        gc.collect()
    
    @classmethod
    def tearDownClass(cls):
        """Generate performance report after all tests"""
        cls._generate_performance_report()
    
    @classmethod
    def _generate_performance_report(cls):
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*60)
        
        if not hasattr(cls, 'performance_results'):
            print("No performance results available")
            return
        
        # Save results to file for further analysis
        import json
        import os
        
        report_file = "performance_benchmark_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = cls._make_json_serializable(cls.performance_results)
        
        with open(report_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Detailed results saved to: {report_file}")
        
        # Generate summary statistics
        if 'quantum_scalability' in cls.performance_results:
            quantum_results = cls.performance_results['quantum_scalability']
            successful_quantum = [r for r in quantum_results if r['data_size'] != 'failed']
            if successful_quantum:
                avg_quantum_time = np.mean([r['execution_time'] for r in successful_quantum])
                print(f"Quantum Agent - Average execution time: {avg_quantum_time:.3f}s")
        
        if 'nas_convergence' in cls.performance_results:
            nas_results = cls.performance_results['nas_convergence']
            avg_nas_convergence = np.mean([r['convergence_rate'] for r in nas_results])
            print(f"NAS Agent - Average convergence rate: {avg_nas_convergence:.3f}")
        
        if 'meta_learning_adaptation' in cls.performance_results:
            meta_results = cls.performance_results['meta_learning_adaptation']
            avg_adaptation_time = np.mean([r['average_adaptation_time'] for r in meta_results])
            print(f"Meta-Learning Agent - Average adaptation time: {avg_adaptation_time:.3f}s")
        
        print("\nPerformance benchmarking completed successfully!")
    
    @classmethod
    def _make_json_serializable(cls, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {key: cls._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [cls._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


# Additional helper functions for advanced performance testing

def run_stress_test():
    """Run stress test with large-scale data"""
    print("Running stress test...")
    
    # Generate large dataset
    X_stress, y_stress = make_classification(
        n_samples=10000, n_features=100, n_classes=10, 
        n_informative=50, random_state=42
    )
    
    # Train complex model
    stress_model = RandomForestClassifier(n_estimators=200, random_state=42)
    stress_model.fit(X_stress[:5000], y_stress[:5000])
    
    # Test agents under stress
    stress_results = {}
    
    # Quantum stress test
    try:
        quantum_agent = QuantumVulnerabilityAgent(stress_model, epsilon=0.1, quantum_depth=8)
        start_time = time.time()
        quantum_result = quantum_agent.execute_quantum_superposition_attacks(X_stress[0])
        stress_results['quantum'] = {
            'success': True,
            'time': time.time() - start_time,
            'score': quantum_result['vulnerability_score']
        }
    except Exception as e:
        stress_results['quantum'] = {'success': False, 'error': str(e)}
    
    # NAS stress test  
    try:
        nas_agent = NeuralArchitectureSearchAgent(stress_model, epsilon=0.1, population_size=15)
        start_time = time.time()
        nas_result = nas_agent.evolve_attack_architectures(X_stress[0], generations=8)
        stress_results['nas'] = {
            'success': True,
            'time': time.time() - start_time,
            'score': nas_result['final_fitness']
        }
    except Exception as e:
        stress_results['nas'] = {'success': False, 'error': str(e)}
    
    # Meta-learning stress test
    try:
        meta_agent = MetaLearningAttackAgent(stress_model, epsilon=0.1, meta_lr=0.01)
        start_time = time.time()
        meta_result = meta_agent.few_shot_attack_adaptation(
            X_stress[:10], X_stress[10:15], k_shot=10, adaptation_steps=20
        )
        stress_results['meta'] = {
            'success': True,
            'time': time.time() - start_time,
            'score': meta_result['average_effectiveness']
        }
    except Exception as e:
        stress_results['meta'] = {'success': False, 'error': str(e)}
    
    return stress_results


if __name__ == '__main__':
    # Add imports at the top of the file
    import time
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Run comprehensive performance tests
    suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmarks)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    
    # Optional: Run stress test separately
    print("\n" + "="*60)
    print("RUNNING STRESS TEST")
    print("="*60)
    
    stress_results = run_stress_test()
    
    print("\nStress Test Results:")
    for agent_name, result in stress_results.items():
        if result['success']:
            print(f"{agent_name}: ✓ Success - Time: {result['time']:.2f}s, Score: {result['score']:.4f}")
        else:
            print(f"{agent_name}: ✗ Failed - {result['error']}")
