#!/usr/bin/env python3
"""
Mission-aware autonomous developer for nanosim.

Core Mission: Minimalist implementation of evolutionary biology and social dynamics 
with LLM agents where they have perfect freedom like true beings.

Principles:
1. Minimalism — No bloat. Every line must earn its place.
2. Agent Autonomy — Agents decide everything. No hardcoded behaviors.
3. Emergence Over Programming — Detect patterns, don't code them.
4. 7 Primitives Only — Event Log, Local Observation, Scarcity, Harshness, 
   Reputation, Heredity, Compression. Everything else emerges.
"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class MissionValidator:
    """Ensures all changes align with nanosim's core mission."""
    
    CORE_PRIMITIVES = {
        "event_log", "local_observation", "scarcity", "harshness",
        "reputation", "heredity", "compression"
    }
    
    FORBIDDEN_PATTERNS = [
        r"class.*State.*Machine",  # No FSMs - agents decide via LLM
        r"if.*behavior.*==",       # No hardcoded behaviors
        r"def.*calculate_happiness",  # No computed emotions - LLM only
        r"ACTIONS\s*=\s*\[",       # No action lists - free form
        r"embedding",              # No embeddings - pure LLM
        r"vector.*store",          # No vector DBs
        r"def.*pathfind",          # No hardcoded pathfinding
        r"class.*Behavior",        # No behavior classes
        r"PERSONALITY_TYPES",      # No personality archetypes
    ]
    
    ALLOWED_EXTENSIONS = {
        "spatial", "trade", "negotiation", "benchmark", "metrics",
        "visualization", "optimization", "performance"
    }
    
    def validate_change(self, file_path: str, new_code: str, reason: str) -> tuple[bool, str]:
        """Check if a code change aligns with mission principles."""
        
        # 1. Check for forbidden patterns (agent autonomy violations)
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, new_code, re.IGNORECASE):
                return False, f"REJECTED: Contains forbidden pattern '{pattern}' - violates agent autonomy"
        
        # 2. Ensure engine.py stays minimal (hard limit)
        if "engine.py" in file_path:
            lines = len(new_code.split("\n"))
            if lines > 500:
                return False, f"REJECTED: engine.py would grow to {lines} lines (limit: 500) - violates minimalism"
        
        # 3. Check if adding new primitives beyond the 7
        if re.search(r"class.*Primitive|def.*primitive", new_code, re.IGNORECASE):
            if not any(prim in new_code.lower() for prim in self.CORE_PRIMITIVES):
                return False, "REJECTED: Adding new primitive beyond the core 7 - violates minimalism"
        
        # 4. Prevent bloat - check for excessive complexity
        if self._is_bloated(new_code):
            return False, "REJECTED: Code is too complex/verbose - violates minimalism"
        
        # 5. Ensure backwards compatibility - core interfaces must remain
        if self._breaks_compatibility(file_path, new_code):
            return False, "REJECTED: Breaks backwards compatibility with existing simulations"
        
        # 6. Verify reason mentions emergence or allowed extensions
        reason_lower = reason.lower()
        if "emergence" in reason_lower or any(ext in reason_lower for ext in self.ALLOWED_EXTENSIONS):
            return True, "APPROVED: Aligns with mission"
        
        # 7. Default: approve if no violations found
        return True, "APPROVED: No violations detected"
    
    def _is_bloated(self, code: str) -> bool:
        """Detect bloat: excessive abstraction, deep nesting, or verbosity."""
        lines = code.split("\n")
        
        # Check for excessive abstraction layers
        class_count = len(re.findall(r"^\s*class\s+", code, re.MULTILINE))
        if class_count > 5:
            return True
        
        # Check for deep nesting (>4 levels)
        for line in lines:
            indent = len(line) - len(line.lstrip())
            if indent > 16:  # 4 levels * 4 spaces
                return True
        
        # Check for overly long functions (>50 lines)
        function_blocks = re.split(r"\n(?=def\s+)", code)
        for block in function_blocks:
            if len(block.split("\n")) > 50:
                return True
        
        return False
    
    def _breaks_compatibility(self, file_path: str, new_code: str) -> bool:
        """Check if changes break core interfaces."""
        
        # Core interfaces that must remain stable
        if "interfaces.py" in file_path:
            required_interfaces = [
                "CognitiveFunction", "CompressionFunction", 
                "SpreadFunction", "Scenario"
            ]
            for interface in required_interfaces:
                if interface not in new_code:
                    return True
        
        # Engine must keep its core methods
        if "engine.py" in file_path:
            required_methods = ["tick", "run", "spawn_agent"]
            for method in required_methods:
                if f"def {method}" not in new_code:
                    return True
        
        return False


class AutonomousDeveloper:
    """LLM-powered developer that implements improvements autonomously."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.validator = MissionValidator()
        self.proxy_url = os.environ.get("NEBULA_PROXY_URL")
        self.auth_token = os.environ.get("SANDBOX_AUTH_TOKEN")
        
    async def analyze_improvement(self, improvement: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze an improvement and propose implementation."""
        
        # Read relevant context
        engine_code = (self.project_root / "nanosim/engine.py").read_text()
        readme = (self.project_root / "README.md").read_text()
        
        prompt = f"""You are an autonomous developer working on nanosim, a minimalist LLM-based artificial life simulation.

MISSION: Minimalist implementation of evolutionary biology and social dynamics with LLM agents where they have perfect freedom like true beings.

CORE PRINCIPLES:
1. Minimalism — No bloat. Keep engine.py under 500 lines.
2. Agent Autonomy — Agents decide via LLM. No hardcoded behaviors, FSMs, or action lists.
3. Emergence Over Programming — Detect patterns in logs, don't code them as behaviors.
4. 7 Primitives Only — Event Log, Local Observation, Scarcity, Harshness, Reputation, Heredity, Compression.

CURRENT STATE:
- engine.py is ~350 lines
- Successful simulations: {analysis.get('successful_runs', 0)}/{analysis.get('total_runs', 0)}
- Failed scenarios: {[f['scenario'] for f in analysis.get('failures', [])]}
- Average duration: {analysis.get('avg_duration', 0):.1f}s

IMPROVEMENT TO IMPLEMENT:
{improvement}

TASK:
1. Determine which file(s) need changes
2. Propose minimal code changes that align with the mission
3. Ensure changes preserve agent autonomy (no hardcoded behaviors)
4. Keep additions small and focused

Respond with JSON:
{{
  "file_path": "path/to/file.py",
  "change_type": "add_feature|fix_bug|optimize|refactor",
  "reason": "why this change aligns with mission",
  "implementation": "the actual code to add/modify",
  "test_command": "command to verify it works"
}}

If the improvement violates the mission (adds bloat, hardcodes behavior, etc), respond:
{{
  "skip": true,
  "reason": "why this violates the mission"
}}
"""
        
        # Call LLM via Nebula proxy
        try:
            response = await self._call_llm(prompt)
            proposal = json.loads(response)
            return proposal
        except Exception as e:
            print(f"Error analyzing improvement: {e}")
            return None
    
    async def implement_change(self, proposal: Dict[str, Any]) -> tuple[bool, str]:
        """Implement a proposed code change with validation."""
        
        if proposal.get("skip"):
            return False, f"Skipped: {proposal.get('reason', 'unknown')}"
        
        file_path = proposal.get("file_path", "")
        implementation = proposal.get("implementation", "")
        reason = proposal.get("reason", "")
        
        # Validate against mission principles
        is_valid, validation_msg = self.validator.validate_change(file_path, implementation, reason)
        if not is_valid:
            return False, validation_msg
        
        # Read current file
        full_path = self.project_root / file_path
        if not full_path.exists():
            # New file
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(implementation)
            return True, f"Created {file_path}: {reason}"
        
        # Modify existing file
        current_code = full_path.read_text()
        
        # Simple append for now (more sophisticated patching can be added)
        if "# AUTONOMOUS ADDITION" not in current_code:
            new_code = current_code + f"\n\n# AUTONOMOUS ADDITION: {reason}\n" + implementation
        else:
            # Replace previous autonomous addition
            parts = current_code.split("# AUTONOMOUS ADDITION")
            new_code = parts[0] + f"# AUTONOMOUS ADDITION: {reason}\n" + implementation
        
        # Validate new code doesn't break principles
        is_valid, validation_msg = self.validator.validate_change(file_path, new_code, reason)
        if not is_valid:
            return False, validation_msg
        
        # Write changes
        full_path.write_text(new_code)
        return True, f"Modified {file_path}: {reason} - {validation_msg}"
    
    async def test_change(self, test_command: str) -> tuple[bool, str]:
        """Run tests to verify change works."""
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return True, "Tests passed"
            else:
                return False, f"Tests failed: {result.stderr[-500:]}"
        except subprocess.TimeoutExpired:
            return False, "Tests timed out (>2min)"
        except Exception as e:
            return False, f"Test error: {str(e)}"
    
    async def rollback(self, file_path: str):
        """Rollback changes using git."""
        try:
            subprocess.run(
                ["git", "checkout", "HEAD", file_path],
                cwd=self.project_root,
                check=True
            )
            print(f"Rolled back {file_path}")
        except Exception as e:
            print(f"Rollback failed: {e}")
    
    async def _call_llm(self, prompt: str, model: str = "google/gemini-2.0-flash-001") -> str:
        """Call LLM via Nebula OAuth proxy."""
        
        # Parse OAUTH_APPS to get OpenRouter account
        oauth_apps = json.loads(os.environ.get("OAUTH_APPS", "{}"))
        
        # Use GROQ as fallback if OpenRouter not available
        groq_key = os.environ.get("GROQ_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        
        if openrouter_key:
            # Direct OpenRouter call
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        elif groq_key:
            # Direct Groq call
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        else:
            raise RuntimeError("No LLM API keys available (GROQ_API_KEY or OPENROUTER_API_KEY)")


async def develop_improvement(improvement: str, analysis: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Main entry point: analyze, implement, test, and report."""
    
    developer = AutonomousDeveloper(project_root)
    
    # 1. Analyze and propose
    proposal = await developer.analyze_improvement(improvement, analysis)
    if not proposal:
        return {"success": False, "reason": "Failed to generate proposal"}
    
    if proposal.get("skip"):
        return {"success": False, "reason": proposal.get("reason", "Skipped"), "skipped": True}
    
    # 2. Implement
    impl_success, impl_msg = await developer.implement_change(proposal)
    if not impl_success:
        return {"success": False, "reason": impl_msg}
    
    # 3. Test
    test_command = proposal.get("test_command")
    if test_command:
        test_success, test_msg = await developer.test_change(test_command)
        if not test_success:
            # Rollback on test failure
            await developer.rollback(proposal.get("file_path", ""))
            return {"success": False, "reason": f"Implementation OK but tests failed: {test_msg}"}
    
    # 4. Success
    return {
        "success": True,
        "file_path": proposal.get("file_path"),
        "change_type": proposal.get("change_type"),
        "reason": impl_msg,
        "tested": bool(test_command)
    }
