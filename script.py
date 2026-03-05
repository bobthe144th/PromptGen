#!/usr/bin/env python3
"""
TeichAI Prompt Dataset Generator - Production Version
Supports both local GGUF models and cloud models via OpenRouter
"""

import os
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
from abc import ABC, abstractmethod
import requests


class ModelBackend(ABC):
    """Abstract base class for model backends"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, stop: List[str]) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name for display"""
        pass


class LocalGGUFBackend(ModelBackend):
    """Local GGUF model backend using llama-cpp-python"""
    
    def __init__(self, model_name: str = "unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", 
                 n_ctx: int = 4096, n_gpu_layers: int = -1):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python --break-system-packages")
        
        print(f"\n🔄 Loading local GGUF model: {model_name}")
        print("This may take a few minutes on first run...")
        
        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        
        # Parse model name (format: org/repo/file.gguf)
        parts = model_name.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid model format. Expected: org/repo/file.gguf, got: {model_name}")
        
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=os.path.expanduser("~/.cache/huggingface")
        )
        
        print(f"✅ Model file downloaded/cached at: {model_path}")
        
        # Check GPU support
        print("\n🔍 Checking GPU support...")
        try:
            test_model = Llama(model_path=model_path, n_ctx=128, n_gpu_layers=0, verbose=False)
            del test_model
            print("✅ llama-cpp-python has GPU offload support")
        except Exception as e:
            print(f"⚠️  GPU check failed: {e}")
        
        # Load model
        print("\n📦 Loading model into memory...")
        print(f"   Requested GPU layers: {n_gpu_layers if n_gpu_layers >= 0 else 'all'}")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_threads=os.cpu_count() or 4,
        )
        
        self.model_name = model_name
        
        print("\n✅ Model loaded successfully!")
        print(f"   Context size: {n_ctx}")
        print(f"   GPU layers requested: {n_gpu_layers if n_gpu_layers >= 0 else 'all'}")
    
    def generate(self, prompt: str, max_tokens: int, temperature: float, stop: List[str]) -> str:
        """Generate text using local GGUF model"""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            echo=False,
            stop=stop,
            logprobs=None,
        )
        return output['choices'][0]['text'].strip()
    
    def get_name(self) -> str:
        return f"Local GGUF: {self.model_name}"


class OpenRouterBackend(ModelBackend):
    """Cloud model backend using OpenRouter API"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        print(f"\n✅ OpenRouter backend configured")
        print(f"   Model: {model}")
        
        # Test API key
        print("\n🔍 Testing API connection...")
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            test_payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            response = requests.post(self.base_url, headers=headers, json=test_payload, timeout=10)
            if response.status_code == 200:
                print("✅ API connection successful!")
            else:
                print(f"⚠️  API test returned status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"⚠️  API test failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int, temperature: float, stop: List[str]) -> str:
        """Generate text using OpenRouter API"""
        # Parse ChatML format to messages
        # Extract system and user messages from the prompt
        messages = []
        
        # Simple parsing - look for <|im_start|>role and <|im_end|> markers
        if "<|im_start|>system" in prompt:
            system_start = prompt.find("<|im_start|>system\n") + len("<|im_start|>system\n")
            system_end = prompt.find("<|im_end|>", system_start)
            if system_end > system_start:
                system_content = prompt[system_start:system_end].strip()
                messages.append({"role": "system", "content": system_content})
        
        if "<|im_start|>user" in prompt:
            user_start = prompt.find("<|im_start|>user\n") + len("<|im_start|>user\n")
            user_end = prompt.find("<|im_end|>", user_start)
            if user_end > user_start:
                user_content = prompt[user_start:user_end].strip()
                messages.append({"role": "user", "content": user_content})
        
        # Fallback if parsing failed
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/teichai/prompt-generator",
            "X-Title": "TeichAI Prompt Generator"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  API request failed: {e}")
            return ""
        except (KeyError, IndexError) as e:
            print(f"⚠️  Failed to parse API response: {e}")
            return ""
    
    def get_name(self) -> str:
        return f"OpenRouter: {self.model}"


class PromptDatasetGenerator:
    """Generate diverse training prompts for language models"""
    
    # Available domains
    DOMAINS = [
        "Coding", "Math", "Science", "Web Development", "Data Science",
        "Machine Learning", "Creative Writing", "Logic", "Reasoning", "General Knowledge", "agentic coding",
    ]
    
    def __init__(self, backend: ModelBackend):
        """Initialize with a model backend"""
        self.backend = backend
        self.domains = self.DOMAINS
        
        print("\n" + "=" * 70)
        print("TEICHAI PROMPT DATASET GENERATOR - PRODUCTION VERSION")
        print("=" * 70)
        print(f"Backend: {self.backend.get_name()}")
    
    def validate_percentages(self, percentages: Dict[str, float]) -> bool:
        """Validate that percentages sum to 100"""
        total = sum(percentages.values())
        if abs(total - 100.0) > 0.01:
            print(f"❌ Error: Percentages sum to {total}, not 100")
            return False
        return True
    
    def calculate_domain_counts(self, total_prompts: int, percentages: Dict[str, float]) -> Dict[str, int]:
        """Calculate number of prompts per domain"""
        domain_counts = {}
        remaining = total_prompts
        
        # Sort domains by percentage (descending)
        sorted_domains = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate prompts proportionally
        for i, (domain, percent) in enumerate(sorted_domains):
            if i == len(sorted_domains) - 1:
                # Last domain gets all remaining
                domain_counts[domain] = remaining
            else:
                count = round(total_prompts * percent / 100)
                domain_counts[domain] = count
                remaining -= count
        
        return domain_counts
    
    def generate_prompts_batch(self, domain: str, batch_size: int, debug: bool = False, temperature: float = 0.9) -> List[str]:
        """
        Generate a batch of prompts for a specific domain
        
        CRITICAL: System prompt ALWAYS requests 10 prompts (hardcoded) for perfect caching.
        We generate 10 prompts every time but only extract batch_size prompts from the output.
        
        This ensures:
        - First batch: Full TTFT overhead (~1-2s)  
        - All subsequent batches: Near-zero TTFT (~0.01s) due to 100% cache hit
        
        Args:
            domain: The domain to generate prompts for
            batch_size: How many prompts we actually need (extract this many from output)
            debug: Show raw model output for debugging
            temperature: Temperature for generation (higher = more diverse)
            
        Returns:
            List of exactly batch_size prompts
        """
        # CRITICAL: System prompt is COMPLETELY STATIC - always requests 10 prompts
        # We extract only batch_size prompts from the output
        # This guarantees identical prompts across batches = perfect cache hits
        system_prompt = f"""You are a helpful AI assistant that generates high-quality training prompts for language models.
Generate exactly 10 diverse, specific prompts in the domain of {domain}. /no_think

Requirements:
- Each prompt should be a clear question or instruction
- Prompts should vary in complexity (some simple, some complex)
- Prompts should be diverse and cover different aspects of {domain}
- Each prompt should be on a new line
- Do not number the prompts
- Do not include any explanations or additional text
- Only output the prompts themselves
- Do not output any prompts that would require information not stated in the prompt
- Make the prompts as hard as possible
- All content of the prompt must be on the ONE line

Example format:
Write a Python function to calculate fibonacci numbers
Explain the concept of recursion with examples"""

        # User prompt is also kept simple and consistent
        user_prompt = "Generate prompts:"
        
        # Qwen uses ChatML format
        full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
        
        try:
            print(f"  Generating batch of {batch_size} prompts for {domain}...", end=" ", flush=True)
            
            # ALWAYS allocate tokens for 10 prompts (what we ask for in system prompt)
            max_tokens_for_batch = 1024  # Enough for 10 prompts (~100 tokens each)
            
            # Generate using backend
            generated_text = self.backend.generate(
                full_prompt,
                max_tokens=max_tokens_for_batch,
                temperature=temperature,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            
            # Debug mode: show raw output
            if debug:
                print(f"\n--- RAW OUTPUT ---")
                print(generated_text)
                print(f"--- END RAW OUTPUT ---\n")
            
            # Split by newlines and clean
            lines = generated_text.split('\n')
            
            # Filter and clean prompts
            prompts = []
            filtered_count = 0
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Remove common prefixes that slip through
                prefixes_to_remove = [
                    r'^\d+\.\s*',  # "1. "
                    r'^-\s*',      # "- "
                    r'^\*\s*',     # "* "
                    r'^•\s*',      # "• "
                ]
                
                for prefix_pattern in prefixes_to_remove:
                    line = re.sub(prefix_pattern, '', line)
                
                # Clean up the line
                cleaned = line.strip()
                
                # Skip if it's meta-commentary
                skip_phrases = [
                    "here are", "here's", "let me generate", "i'll generate",
                    "okay", "sure", "of course", "certainly"
                ]
                if any(phrase in cleaned.lower()[:30] for phrase in skip_phrases):
                    filtered_count += 1
                    continue
                
                # Only keep lines that are substantial prompts (at least 15 characters)
                if len(cleaned) >= 15:
                    prompts.append(cleaned)
            
            # Take only the requested number of prompts
            # We always ask for 10 prompts (for caching) but extract only batch_size
            # So if batch_size=5 (last batch), we take the first 5 of the 10 generated
            prompts = prompts[:batch_size]
            
            print(f"✓ ({len(prompts)} prompts)")
            return prompts
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return []
    
    def rate_prompt_quality(self, prompts: List[str], domain: str) -> List[int]:
        """
        Rate the quality of generated prompts on a scale of 1-5
        
        Scale:
        1 = Excellent - Clear, specific, challenging, well-formed
        2 = Good - Clear and specific but not very challenging  
        3 = Acceptable - Clear but generic or simple
        4 = Poor - Vague, unclear, or problematic
        5 = Unacceptable - Nonsensical, malformed, or requires external context
        
        Args:
            prompts: List of prompts to rate
            domain: Domain the prompts belong to
            
        Returns:
            List of ratings (1-5) corresponding to each prompt
        """
        if not prompts:
            return []
        
        # Batch the prompts for rating (rate 20 at a time for efficiency)
        batch_size = 20
        all_ratings = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # Create numbered list of prompts for the LLM to rate
            prompt_list = "\n".join([f"{j+1}. {p}" for j, p in enumerate(batch)])
            
            system_prompt = f"""You are a quality evaluator for training prompts in the domain of {domain}.

Rate each prompt on a scale of 1-5 where:
1 = Excellent - Clear, specific, challenging, well-formed, no external context needed
2 = Good - Clear and specific but not very challenging
3 = Acceptable - Clear but generic or simple
4 = Poor - Vague, unclear, or somewhat problematic
5 = Unacceptable - Nonsensical, malformed, requires information not in the prompt, or has serious issues

CRITICAL: Output ONLY the ratings as numbers, one per line, nothing else.
Do NOT include explanations, reasoning, or the prompt text.
Output format:
1
2
1
3
..."""

            user_prompt = f"""Rate these {len(batch)} prompts:

{prompt_list}

Output only the rating numbers (1-5), one per line:"""
            
            full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
            
            try:
                generated_text = self.backend.generate(
                    full_prompt,
                    max_tokens=200,
                    temperature=0.3,  # Low temperature for consistent ratings
                    stop=["<|im_end|>", "<|im_start|>"]
                )
                
                # Parse ratings from output
                lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
                
                batch_ratings = []
                for line in lines[:len(batch)]:  # Only take as many as we need
                    # Extract first digit from the line
                    digits = [c for c in line if c.isdigit()]
                    if digits:
                        rating = int(digits[0])
                        # Clamp to valid range
                        rating = max(1, min(5, rating))
                        batch_ratings.append(rating)
                    else:
                        # If we can't parse, assume acceptable quality
                        batch_ratings.append(3)
                
                # If we didn't get enough ratings, fill with 3 (acceptable)
                while len(batch_ratings) < len(batch):
                    batch_ratings.append(3)
                
                all_ratings.extend(batch_ratings)
                
            except Exception as e:
                print(f"⚠️  Error rating batch: {e}")
                # On error, assume all prompts are acceptable
                all_ratings.extend([3] * len(batch))
        
        return all_ratings
    
    def filter_by_quality(self, prompts: List[str], domain_counts: Dict[str, int], 
                         quality_threshold: int, batch_size: int = 10, 
                         max_retries: int = 5) -> List[str]:
        """
        Filter prompts by quality and regenerate those that don't meet the threshold
        
        Args:
            prompts: List of prompts to filter
            domain_counts: Original domain distribution (for regenerating)
            quality_threshold: Maximum acceptable rating (1-5, prompts > threshold are rejected)
            batch_size: Batch size for regeneration
            max_retries: Maximum regeneration attempts
            
        Returns:
            List of prompts that pass quality filter
        """
        print(f"\n🔍 Quality filtering (threshold: ≤{quality_threshold}/5)...")
        print(f"   Max regeneration attempts: {max_retries}")
        
        # We need to track which domain each prompt belongs to for regeneration
        # Build a mapping of prompts to domains
        prompt_to_domain = {}
        idx = 0
        for domain, count in domain_counts.items():
            for _ in range(count):
                if idx < len(prompts):
                    prompt_to_domain[prompts[idx]] = domain
                    idx += 1
        
        passing_prompts = []
        retry_count = 0
        remaining_prompts = prompts.copy()
        
        while remaining_prompts and retry_count < max_retries:
            # Group prompts by domain for efficient rating
            domain_to_prompts = {}
            for prompt in remaining_prompts:
                domain = prompt_to_domain.get(prompt, "General Knowledge")
                if domain not in domain_to_prompts:
                    domain_to_prompts[domain] = []
                domain_to_prompts[domain].append(prompt)
            
            # Rate prompts by domain
            all_failed_prompts = []
            for domain, domain_prompts in domain_to_prompts.items():
                print(f"  Rating {len(domain_prompts)} prompts for {domain}...", end=" ", flush=True)
                ratings = self.rate_prompt_quality(domain_prompts, domain)
                
                # Separate passing and failing prompts
                passed = 0
                failed = 0
                for prompt, rating in zip(domain_prompts, ratings):
                    if rating <= quality_threshold:
                        passing_prompts.append(prompt)
                        passed += 1
                    else:
                        all_failed_prompts.append((prompt, domain, rating))
                        failed += 1
                
                print(f"✓ (Passed: {passed}, Failed: {failed})")
            
            # If no failures, we're done
            if not all_failed_prompts:
                break
            
            # Calculate how many prompts to regenerate per domain
            retry_count += 1
            if retry_count >= max_retries:
                print(f"  ⚠️  Max retries reached. Keeping {len(all_failed_prompts)} lower-quality prompts.")
                # Add the failed prompts anyway to meet the target count
                passing_prompts.extend([p[0] for p in all_failed_prompts])
                break
            
            # Group failed prompts by domain
            failed_by_domain = {}
            for prompt, domain, rating in all_failed_prompts:
                if domain not in failed_by_domain:
                    failed_by_domain[domain] = []
                failed_by_domain[domain].append((prompt, rating))
            
            # Regenerate failed prompts
            print(f"\n  🔄 Regenerating {len(all_failed_prompts)} failed prompts (attempt {retry_count}/{max_retries})...")
            print(f"     📝 Generating in batches of 10, keeping ALL generated prompts as buffer")
            
            regenerated = []
            for domain, failed_list in failed_by_domain.items():
                count_needed = len(failed_list)
                # Round up to nearest batch of 10 to give us a buffer
                batches_needed = (count_needed + 9) // 10
                total_to_generate = batches_needed * 10
                
                print(f"    {domain}: Need {count_needed}, generating {total_to_generate} ({batches_needed} batches of 10)")
                
                # Generate in batches of 10, keep ALL (don't trim)
                domain_regenerated = []
                for batch_num in range(batches_needed):
                    batch_prompts = self.generate_prompts_batch(
                        domain, 
                        10,  # Always 10 for prompt caching consistency
                        temperature=1.0  # Higher temp for more diversity
                    )
                    domain_regenerated.extend(batch_prompts)
                
                print(f"      → Generated {len(domain_regenerated)} total (keeping all as buffer)")
                
                # Update the prompt_to_domain mapping for new prompts
                for prompt in domain_regenerated:
                    prompt_to_domain[prompt] = domain
                
                regenerated.extend(domain_regenerated)
            
            # Set remaining prompts to the newly regenerated ones
            remaining_prompts = regenerated
        
        # Final count - we may have MORE than target due to keeping buffer prompts
        final_count = len(passing_prompts)
        target_count = sum(domain_counts.values())
        
        print(f"✅ Quality filtering complete: {final_count} prompts passed")
        
        if final_count > target_count:
            extra = final_count - target_count
            print(f"   💡 Generated {extra} extra prompts (buffer) - all passed quality filter!")
            print(f"   You can keep all {final_count} or trim to {target_count} in the final output")
        
        return passing_prompts
    
    def generate_all_prompts(self, domain_counts: Dict[str, int], batch_size: int = 10, temperature: float = 0.9) -> List[str]:
        """
        Generate all prompts across all domains
        
        Args:
            domain_counts: Dictionary mapping domains to number of prompts needed
            batch_size: Size of each generation batch (default 10 for quality)
            temperature: Temperature for generation (higher = more diverse)
            
        Returns:
            List of all generated prompts
        
        Note:
            Uses batch_size=10 to maintain quality and uniqueness.
            Prompt caching ensures minimal TTFT overhead - the system prompt
            is cached and reused across all batches for the same domain.
        """
        all_prompts = []
        
        for domain, count in domain_counts.items():
            if count == 0:
                continue
                
            print(f"\nGenerating {count} prompts for {domain}:")
            domain_prompts = []
            
            # Generate in batches of 10 for optimal quality
            # Prompt caching will reuse the tokenized system prompt across batches
            remaining = count
            batch_num = 0
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                batch_prompts = self.generate_prompts_batch(domain, current_batch_size, temperature=temperature)
                domain_prompts.extend(batch_prompts)
                remaining -= len(batch_prompts)
                batch_num += 1
            
            print(f"  Total generated for {domain}: {len(domain_prompts)} ({batch_num} batches)")
            all_prompts.extend(domain_prompts)
        
        return all_prompts
    
    def remove_duplicates(self, prompts: List[str]) -> Tuple[List[str], int]:
        """Remove duplicate prompts (case-insensitive)"""
        seen = set()
        unique = []
        
        for prompt in prompts:
            # Normalize: lowercase and strip whitespace
            normalized = prompt.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(prompt)
        
        duplicates_removed = len(prompts) - len(unique)
        return unique, duplicates_removed
    
    def save_to_markdown(self, prompts: List[str], filename: str):
        """Save prompts to markdown file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Training Prompts Dataset\n\n")
            f.write(f"Total prompts: {len(prompts)}\n\n")
            f.write("---\n\n")
            
            for prompt in prompts:
                f.write(f"{prompt}\n")
        
        print(f"\n✅ Saved {len(prompts)} prompts to {filename}")
    
    def generate_dataset(self, total_prompts: int, percentages: Dict[str, float], 
                        output_file: str = "prompt_dataset.md", batch_size: int = 10,
                        quality_threshold: int = None, max_quality_retries: int = 5):
        """
        Main method to generate the complete dataset
        
        Args:
            total_prompts: Total number of prompts to generate
            percentages: Dictionary mapping domains to percentage (must sum to 100)
            output_file: Output markdown filename
            batch_size: Number of prompts to generate per batch
            quality_threshold: Optional quality filter (1-5, prompts > threshold rejected)
                              None = no quality filtering (default)
                              3 = keep excellent/good/acceptable (reject poor/unacceptable)
                              4 = keep everything except unacceptable
            max_quality_retries: Maximum regeneration attempts for quality filtering (1-10)
        """
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        
        # Validate inputs
        if not self.validate_percentages(percentages):
            return
        
        # Calculate domain distribution
        domain_counts = self.calculate_domain_counts(total_prompts, percentages)
        
        print(f"\n📊 Target Distribution:")
        for domain in self.domains:
            count = domain_counts[domain]
            percent = percentages[domain]
            print(f"  {domain:20s}: {count:4d} prompts ({percent:5.1f}%)")
        print(f"  {'Total':20s}: {sum(domain_counts.values()):4d} prompts")
        
        if quality_threshold is not None:
            print(f"\n🎯 Quality Filtering:")
            print(f"   Threshold: ≤{quality_threshold}/5")
            print(f"   Max retries: {max_quality_retries}")
        
        # Generate initial prompts
        print("\n" + "=" * 70)
        print("GENERATING PROMPTS")
        print("=" * 70)
        print("💡 Using batch_size=10 for optimal quality and uniqueness")
        print("   Prompt caching minimizes TTFT overhead between batches\n")
        all_prompts = self.generate_all_prompts(domain_counts, batch_size)
        
        print(f"\n📈 Initial generation complete: {len(all_prompts)} prompts")
        
        # Quality filtering (optional)
        if quality_threshold is not None:
            print("\n" + "=" * 70)
            print("QUALITY FILTERING")
            print("=" * 70)
            all_prompts = self.filter_by_quality(all_prompts, domain_counts, quality_threshold, batch_size, max_quality_retries)
        
        # Remove duplicates
        print("\n" + "=" * 70)
        print("CHECKING FOR DUPLICATES")
        print("=" * 70)
        unique_prompts, duplicates_removed = self.remove_duplicates(all_prompts)
        print(f"Removed {duplicates_removed} duplicate prompts")
        print(f"Unique prompts: {len(unique_prompts)}")
        
        # Save results
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        self.save_to_markdown(unique_prompts, output_file)
        
        print("\n" + "=" * 70)
        print("✨ GENERATION COMPLETE!")
        print("=" * 70)
        print(f"Final count: {len(unique_prompts)} unique prompts")
        print(f"Target was: {total_prompts} prompts")
        if len(unique_prompts) < total_prompts:
            print(f"⚠️  Note: {total_prompts - len(unique_prompts)} prompts short of target")


def configure_backend() -> ModelBackend:
    """Interactive backend configuration"""
    print("\n" + "=" * 70)
    print("MODEL BACKEND CONFIGURATION")
    print("=" * 70)
    
    print("\nChoose model backend:")
    print("  1. Local GGUF model (llama-cpp-python)")
    print("  2. Cloud model (OpenRouter API)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if choice == "1":
        # Local GGUF backend
        print("\n--- Local GGUF Configuration ---")
        print("\nAvailable models:")
        print("  1. Qwen3-4B-Q4_K_M (default, ~2.3GB)")
        print("  2. Qwen3-8B-Q4_K_M (~4.5GB)")
        print("  3. Custom model")
        
        model_choice = input("\nEnter choice (1-3, default: 1): ").strip() or "1"
        
        if model_choice == "1":
            model_name = "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
        elif model_choice == "2":
            model_name = "unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        else:
            model_name = input("Enter model name (format: org/repo/file.gguf): ").strip()
        
        use_gpu = input("\nUse GPU acceleration? (y/n, default: y): ").strip().lower() or "y"
        n_gpu_layers = -1 if use_gpu == "y" else 0
        
        return LocalGGUFBackend(model_name=model_name, n_gpu_layers=n_gpu_layers)
    
    else:
        # OpenRouter backend
        print("\n--- OpenRouter Configuration ---")
        
        api_key = input("\nEnter OpenRouter API key: ").strip()
        if not api_key:
            print("Error: API key is required")
            return configure_backend()
        
        print("\nPopular models:")
        print("  1. anthropic/claude-3.5-sonnet")
        print("  2. openai/gpt-4-turbo")
        print("  3. google/gemini-pro-1.5")
        print("  4. meta-llama/llama-3.1-70b-instruct")
        print("  5. meta-llama/llama-3.1-405b-instruct")
        print("  6. Custom model")
        
        model_choice = input("\nEnter choice (1-6, default: 1): ").strip() or "1"
        
        models = {
            "1": "anthropic/claude-3.5-sonnet",
            "2": "openai/gpt-4-turbo",
            "3": "google/gemini-pro-1.5",
            "4": "meta-llama/llama-3.1-70b-instruct",
            "5": "meta-llama/llama-3.1-405b-instruct"
        }
        
        if model_choice in models:
            model = models[model_choice]
        else:
            model = input("Enter model name (e.g., anthropic/claude-3.5-sonnet): ").strip()
        
        return OpenRouterBackend(api_key=api_key, model=model)


def main():
    """Main entry point with interactive configuration"""
    
    print("=" * 70)
    print("TEICHAI PROMPT DATASET GENERATOR")
    print("Production Version - Local & Cloud Support")
    print("=" * 70)
    
    # Configure backend
    backend = configure_backend()
    
    # Get dataset parameters
    print("\n" + "=" * 70)
    print("DATASET CONFIGURATION")
    print("=" * 70)
    
    try:
        total_prompts = int(input("\nHow many prompts do you want to generate? "))
        if total_prompts <= 0:
            print("Error: Number of prompts must be positive")
            return
    except ValueError:
        print("Error: Please enter a valid number")
        return
    
    print("\nEnter percentage for each domain (must sum to 100):")
    percentages = {}
    domains = [
        "Coding", "Math", "Science", "Web Development", "Data Science",
        "Machine Learning", "Creative Writing", "Logic", "Reasoning", "General Knowledge", "agentic coding"
    ]
    
    for domain in domains:
        while True:
            try:
                percent = float(input(f"  {domain}: "))
                if percent < 0 or percent > 100:
                    print("    Error: Percentage must be between 0 and 100")
                    continue
                percentages[domain] = percent
                break
            except ValueError:
                print("    Error: Please enter a valid number")
    
    # Verify percentages sum to 100
    total_percent = sum(percentages.values())
    if abs(total_percent - 100.0) > 0.01:
        print(f"\n❌ Error: Percentages sum to {total_percent}, not 100")
        print("Percentages must sum to exactly 100")
        return
    
    output_file = input("\nOutput filename (default: prompt_dataset.md): ").strip()
    if not output_file:
        output_file = "prompt_dataset.md"
    if not output_file.endswith('.md'):
        output_file += '.md'
    
    # Quality filtering configuration
    quality_threshold = None
    max_quality_retries = 5
    use_quality_filter = input("\nUse quality filtering? (y/n, default: n): ").strip().lower()
    if use_quality_filter == 'y':
        print("\nQuality rating scale:")
        print("  1 = Excellent - Clear, specific, challenging, well-formed")
        print("  2 = Good - Clear and specific but not very challenging")
        print("  3 = Acceptable - Clear but generic or simple")
        print("  4 = Poor - Vague, unclear, or problematic")
        print("  5 = Unacceptable - Nonsensical, malformed, or requires external context")
        print("\nQuality threshold: Keep prompts rated ≤ threshold")
        print("  Threshold 2 = Keep only excellent/good (strict, ~85% rejection)")
        print("  Threshold 3 = Keep excellent/good/acceptable (recommended, ~10-15% rejection)")
        print("  Threshold 4 = Keep everything except unacceptable (lenient, ~5% rejection)")
        
        while True:
            try:
                threshold = int(input("\nEnter quality threshold (1-5, default: 3): ").strip() or "3")
                if threshold < 1 or threshold > 5:
                    print("  Error: Threshold must be between 1 and 5")
                    continue
                quality_threshold = threshold
                break
            except ValueError:
                print("  Error: Please enter a valid number")
        
        # Max retries configuration
        while True:
            try:
                retries = int(input("Maximum regeneration attempts (1-10, default: 5): ").strip() or "5")
                if retries < 1 or retries > 10:
                    print("  Error: Must be between 1 and 10")
                    continue
                max_quality_retries = retries
                break
            except ValueError:
                print("  Error: Please enter a valid number")
    
    # Generate dataset
    generator = PromptDatasetGenerator(backend)
    generator.generate_dataset(
        total_prompts, 
        percentages, 
        output_file, 
        quality_threshold=quality_threshold,
        max_quality_retries=max_quality_retries
    )


if __name__ == "__main__":
    main()
