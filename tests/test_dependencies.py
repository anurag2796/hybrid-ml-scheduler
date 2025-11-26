
import unittest
import os
from src.workload_generator import WorkloadGenerator, Task

class TestTaskDependencies(unittest.TestCase):
    def test_dependency_generation(self):
        """Test that tasks are generated with dependencies"""
        wg = WorkloadGenerator(seed=42)
        tasks = wg.generate_workload(
            num_tasks=100,
            dependency_prob=0.5, # High probability for testing
            max_dependencies=3
        )
        
        # Check that some tasks have dependencies
        tasks_with_deps = [t for t in tasks if t.dependencies]
        self.assertGreater(len(tasks_with_deps), 0, "No dependencies generated")
        
        # Verify dependency validity
        for task in tasks:
            for dep_id in task.dependencies:
                # Dependency ID must be less than Task ID (no cycles, only backward refs)
                self.assertLess(dep_id, task.task_id, f"Task {task.task_id} depends on future task {dep_id}")
                self.assertGreaterEqual(dep_id, 0, "Dependency ID cannot be negative")

    def test_serialization(self):
        """Test that dependencies are saved and loaded correctly"""
        wg = WorkloadGenerator(seed=123)
        tasks = wg.generate_workload(num_tasks=50, dependency_prob=0.8)
        
        filename = "test_deps.csv"
        wg.save_workload(filename)
        
        try:
            wg_loaded = WorkloadGenerator.load_workload(filename)
            
            self.assertEqual(len(wg.tasks), len(wg_loaded.tasks))
            
            for t1, t2 in zip(wg.tasks, wg_loaded.tasks):
                self.assertEqual(t1.task_id, t2.task_id)
                self.assertEqual(t1.dependencies, t2.dependencies)
                
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == '__main__':
    unittest.main()
