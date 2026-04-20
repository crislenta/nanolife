#!/usr/bin/env python3
"""
Autonomous 24/7 work loop for nanolife.

Cycles through:
1. Run experiments (simulations with different parameters)
2. Analyze results (emergence patterns, cost, behavior quality)
3. Identify improvements (bottlenecks, edge cases, enhancements)
4. Implement changes (code improvements, new features)
5. Commit & push to GitHub
6. Report findings (daily email summary)
"""
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Configuration
WORK_CYCLE_HOURS = 4  # How often to run the full loop
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "autonomous_logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "autonomous_experiments"

LOGS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

class AutonomousWorker:
    def __init__(self):
        self.cycle_count = 0
        self.log_file = LOGS_DIR / f"worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log(self, message: str, level: str = "INFO"):
        """Log to both file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
    
    def run_simulation(self, scenario: str, agents: int, ticks: int, 
                       model: str = None, use_openrouter: bool = False) -> Dict[str, Any]:
        """Run a nanolife simulation and return results."""
        self.log(f"Running simulation: {scenario} with {agents} agents, {ticks} ticks")
        
        # Build command
        cmd = [
            "python", "-m", "scripts.simulate",
            f"--scenario={scenario}",
            f"--agents={agents}",
            f"--ticks={ticks}",
            "--no-report"  # Skip HTML report for now, we'll analyze logs
        ]
        
        if use_openrouter:
            cmd.append("--open-router")
        if model:
            cmd.append(f"--model={model}")
        
        # Run simulation
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600  # 10 min max per simulation
            )
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "scenario": scenario,
                "agents": agents,
                "ticks": ticks,
                "duration_seconds": duration,
                "stdout": result.stdout[-2000:] if result.stdout else "",  # Last 2k chars
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "scenario": scenario,
                "agents": agents,
                "ticks": ticks,
                "duration_seconds": duration,
                "error": "Simulation timed out (>10 min)",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "scenario": scenario,
                "agents": agents,
                "ticks": ticks,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "exit_code": -1
            }
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze simulation results to identify patterns and issues."""
        self.log("Analyzing experiment results...")
        
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        
        analysis = {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "avg_duration": sum(r["duration_seconds"] for r in successful) / len(successful) if successful else 0,
            "failures": [{"scenario": r["scenario"], "error": r.get("error", "unknown")} for r in failed],
            "recommendations": []
        }
        
        # Look for patterns in output
        for result in successful:
            stdout = result.get("stdout", "")
            
            # Check for common issues
            if "extinction" in stdout.lower():
                analysis["recommendations"].append(
                    f"Scenario {result['scenario']} led to extinction — consider balancing harshness"
                )
            if "mass" in stdout.lower() and "death" in stdout.lower():
                analysis["recommendations"].append(
                    f"Mass death event in {result['scenario']} — resource depletion too aggressive?"
                )
        
        # Identify best-performing scenarios
        if successful:
            analysis["best_scenario"] = max(successful, key=lambda r: r["duration_seconds"])["scenario"]
        
        return analysis
    
    def identify_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement tasks based on analysis."""
        improvements = []
        
        if analysis["failed_runs"] > 0:
            improvements.append("Fix failures in: " + ", ".join([f["scenario"] for f in analysis["failures"]]))
        
        if "extinction" in str(analysis.get("recommendations", [])):
            improvements.append("Add dynamic harshness adjustment to prevent mass extinction")
        
        if analysis["avg_duration"] > 300:
            improvements.append("Optimize LLM calls — simulations taking >5min average")
        
        # Add roadmap items from README
        improvements.extend([
            "Consider adding spatial awareness (coordinate graph for distance/travel)",
            "Explore inter-agent trade & negotiation protocol",
            "Build reproducible benchmark suite with standardized metrics"
        ])
        
        return improvements[:5]  # Top 5 priorities
    
    async def implement_change(self, improvement: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to implement a specific improvement using autonomous developer."""
        self.log(f"Attempting to implement: {improvement}")
        
        try:
            from autonomous_developer import develop_improvement
            
            result = await develop_improvement(improvement, analysis, PROJECT_ROOT)
            
            if result.get("success"):
                self.log(f"✓ Implemented: {result.get('reason')}", "SUCCESS")
                return result
            elif result.get("skipped"):
                self.log(f"⊘ Skipped: {result.get('reason')}", "INFO")
                return result
            else:
                self.log(f"✗ Failed: {result.get('reason')}", "WARNING")
                return result
                
        except Exception as e:
            self.log(f"Error implementing change: {e}", "ERROR")
            return {"success": False, "reason": str(e)}
    
    def git_commit_push(self, message: str) -> bool:
        """Commit and push changes to GitHub."""
        try:
            # Check if there are changes
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )
            
            if not status.stdout.strip():
                self.log("No changes to commit")
                return False
            
            # Stage all changes
            subprocess.run(["git", "add", "-A"], cwd=PROJECT_ROOT, check=True)
            
            # Commit
            subprocess.run(
                ["git", "commit", "-m", f"[autonomous] {message}"],
                cwd=PROJECT_ROOT,
                check=True
            )
            
            # Push (will need SSH key configured)
            subprocess.run(["git", "push"], cwd=PROJECT_ROOT, check=True)
            
            self.log(f"Committed and pushed: {message}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Git operation failed: {e}", "ERROR")
            return False
    
    def generate_report(self, cycle_num: int, results: List[Dict], 
                       analysis: Dict, improvements: List[str],
                       implementation_results: List[Dict] = None) -> str:
        """Generate daily report summary."""
        report = f"""
# nanolife Autonomous Work Report — Cycle {cycle_num}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiments Run
- Total simulations: {len(results)}
- Successful: {analysis['successful_runs']}
- Failed: {analysis['failed_runs']}
- Average duration: {analysis['avg_duration']:.1f}s

## Key Findings
"""
        for rec in analysis.get("recommendations", [])[:5]:
            report += f"- {rec}\n"
        
        report += f"\n## Identified Improvements\n"
        for imp in improvements:
            report += f"- {imp}\n"
        
        # Add autonomous development section
        if implementation_results:
            report += f"\n## Autonomous Development\n"
            for impl in implementation_results:
                if impl.get("success"):
                    report += f"✓ IMPLEMENTED: {impl.get('reason', 'unknown')}\n"
                    report += f"  - File: {impl.get('file_path', 'N/A')}\n"
                    report += f"  - Type: {impl.get('change_type', 'N/A')}\n"
                elif impl.get("skipped"):
                    report += f"⊘ SKIPPED: {impl.get('reason', 'unknown')}\n"
                else:
                    report += f"✗ FAILED: {impl.get('reason', 'unknown')}\n"
        
        if analysis.get("failures"):
            report += f"\n## Failures to Investigate\n"
            for fail in analysis["failures"]:
                report += f"- {fail['scenario']}: {fail['error']}\n"
        
        return report
    
    def send_email_report(self, report: str):
        """Send report via email using Nebula's email system."""
        try:
            # Queue email for Nebula to send
            email_data = {
                "to": os.getenv("REPORT_EMAIL", "cris@thelifesim.com"),
                "subject": f"nanolife Daily Report — {datetime.now().strftime('%Y-%m-%d')}",
                "body": report,
                "timestamp": datetime.now().isoformat()
            }
            
            outbox = LOGS_DIR / "email_outbox.json"
            with open(outbox, 'w') as f:
                json.dump(email_data, f, indent=2)
            
            self.log(f"Email report queued: {outbox}")
            return True
        except Exception as e:
            self.log(f"Failed to queue email: {e}", "ERROR")
            return False
    
    async def work_cycle(self):
        """Run one complete work cycle."""
        self.cycle_count += 1
        self.log(f"=== Starting work cycle {self.cycle_count} ===")
        
        # 1. Run experiments
        scenarios = ["nanothrones", "colony", "island"]
        results = []
        
        for scenario in scenarios:
            result = self.run_simulation(
                scenario=scenario,
                agents=10,
                ticks=20,  # Short runs for testing
                use_openrouter=False  # Use Groq by default (free)
            )
            results.append(result)
            
            # Save result
            result_file = EXPERIMENTS_DIR / f"cycle{self.cycle_count}_{scenario}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # 2. Analyze results
        analysis = self.analyze_results(results)
        
        # 3. Identify improvements
        improvements = self.identify_improvements(analysis)
        
        # 4. AUTONOMOUS DEVELOPMENT: Try to implement top improvement
        implementation_results = []
        if improvements:
            top_improvement = improvements[0]  # Focus on highest priority
            self.log(f"Attempting autonomous implementation of: {top_improvement}")
            
            impl_result = await self.implement_change(top_improvement, analysis)
            implementation_results.append(impl_result)
            
            if impl_result.get("success"):
                # Test the change
                self.log("Testing implemented change...")
                test_result = self.run_simulation(
                    scenario="colony",  # Quick test
                    agents=5,
                    ticks=10,
                    use_openrouter=False
                )
                
                if test_result.get("success"):
                    self.log("✓ Change tested successfully", "SUCCESS")
                    # Commit the code change
                    self.git_commit_push(f"[autonomous-dev] {impl_result.get('reason', 'improvement')}")
                else:
                    self.log("✗ Test failed, rolling back", "WARNING")
                    subprocess.run(["git", "checkout", "HEAD", impl_result.get("file_path", ".")], 
                                   cwd=PROJECT_ROOT)
        
        # 5. Generate report (including implementation results)
        report = self.generate_report(self.cycle_count, results, analysis, improvements, implementation_results)
        
        # Save report
        report_file = LOGS_DIR / f"report_cycle{self.cycle_count}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.log(f"Report saved to {report_file}")
        print("\n" + report)
        
        # 6. Send email report (daily summary)
        self.send_email_report(report)
        
        # 7. Commit results logs
        self.git_commit_push(f"Cycle {self.cycle_count}: {len(results)} experiments, {analysis['successful_runs']} successful")
        
        self.log(f"=== Work cycle {self.cycle_count} complete ===\n")
        
        return report
    
    async def run_forever(self):
        """Run work cycles continuously."""
        self.log("Autonomous worker started — running 24/7")
        
        while True:
            try:
                report = await self.work_cycle()
                
                # Wait for next cycle
                sleep_seconds = WORK_CYCLE_HOURS * 3600
                self.log(f"Sleeping for {WORK_CYCLE_HOURS} hours until next cycle...")
                await asyncio.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                self.log("Received interrupt — shutting down gracefully")
                break
            except Exception as e:
                self.log(f"Error in work cycle: {e}", "ERROR")
                self.log(traceback.format_exc(), "ERROR")
                # Wait 30 min before retrying on error
                await asyncio.sleep(1800)

async def main():
    worker = AutonomousWorker()
    await worker.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
