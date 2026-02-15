#!/usr/bin/env python3
"""
LLM Prompt Dataset Generator for Unsloth Container
Generates a dataset of prompts across multiple domains using Qwen 3 4B GGUF
Designed to run inside the Unsloth Docker container with llama.cpp
"""

from llama_cpp import Llama
from typing import List, Dict
import re
import os
from huggingface_hub import hf_hub_download


class PromptDatasetGenerator:
    def __init__(self, 
                 model_repo: str = "unsloth/Qwen3-4B-GGUF",
                 model_file: str = "Qwen3-4B-Q4_K_M.gguf"):
        """
        Initialize the prompt generator with GGUF model
        
        Args:
            model_repo: HuggingFace repository containing the GGUF model
            model_file: Specific GGUF file to use (Q4_K_M is a good balance of quality/speed)
        """
        self.model_repo = model_repo
        self.model_file = model_file
        self.model = None
        self.domains = [
            "Coding",
            "Math",
            "Science",
            "Web Development",
            "Data Science",
            "Machine Learning",
            "Creative Writing",
            "Logic",
            "Reasoning",
            "General Knowledge"
        ]
        
    def load_model(self, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        Load the GGUF model using llama-cpp-python
        
        Args:
            n_ctx: Context window size (default 4096)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all layers)
        """
        print(f"\n🔄 Loading GGUF model: {self.model_repo}/{self.model_file}")
        print("This may take a few minutes on first run...")
        
        # Download the GGUF file from HuggingFace if not already cached
        try:
            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                repo_type="model"
            )
            print(f"✅ Model file downloaded/cached at: {model_path}")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise
        
        # Check if llama-cpp-python has GPU support
        print("\n🔍 Checking GPU support...")
        try:
            from llama_cpp import llama_cpp
            if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
                gpu_support = llama_cpp.llama_supports_gpu_offload()
                if gpu_support:
                    print("✅ llama-cpp-python has GPU offload support")
                else:
                    print("⚠️  llama-cpp-python does NOT have GPU support")
                    print("    You're running on CPU only. To enable GPU:")
                    print("    CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install --force-reinstall llama-cpp-python")
                    n_gpu_layers = 0  # Force CPU mode
            else:
                print("⚠️  Cannot detect GPU support (old llama-cpp-python version)")
                print("    Attempting to use GPU anyway...")
        except Exception as e:
            print(f"⚠️  Could not check GPU support: {e}")
        
        # Initialize llama.cpp with the GGUF file
        # n_gpu_layers=-1 means use all available GPU layers
        # Setting to 0 would use CPU only
        try:
            print(f"\n📦 Loading model into memory...")
            print(f"   Requested GPU layers: {n_gpu_layers if n_gpu_layers >= 0 else 'all'}")
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,  # Context window size
                n_gpu_layers=n_gpu_layers,  # Use GPU if available
                verbose=False,  # Keep output clean during generation
                n_threads=os.cpu_count() or 4,  # Use all CPU threads for CPU portions
            )
            
            print("\n✅ Model loaded successfully!")
            print(f"   Context size: {n_ctx}")
            print(f"   GPU layers requested: {n_gpu_layers if n_gpu_layers >= 0 else 'all'}")
            print("\n💡 GPU verification:")
            print("   - Check the loading output above for 'offloaded X layers to GPU'")
            print("   - If you see that message, GPU is working correctly ✅")
            print("   - Performance metrics disabled for cleaner output during generation")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def validate_percentages(self, percentages: Dict[str, float]) -> bool:
        """Validate that percentages sum to 100 and all domains are covered"""
        if set(percentages.keys()) != set(self.domains):
            print("Error: All domains must be specified")
            return False
        
        total = sum(percentages.values())
        if abs(total - 100.0) > 0.01:
            print(f"Error: Percentages must sum to 100 (current sum: {total})")
            return False
        
        return True
    
    def calculate_domain_counts(self, total_prompts: int, percentages: Dict[str, float]) -> Dict[str, int]:
        """Calculate number of prompts needed per domain"""
        counts = {}
        remaining = total_prompts
        
        # Sort domains by percentage to handle rounding better
        sorted_domains = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        for i, (domain, percent) in enumerate(sorted_domains):
            if i == len(sorted_domains) - 1:
                # Assign all remaining prompts to last domain to ensure exact total
                counts[domain] = remaining
            else:
                count = round(total_prompts * (percent / 100))
                counts[domain] = count
                remaining -= count
        
        return counts
    
    def generate_prompts_batch(self, domain: str, batch_size: int, debug: bool = False, temperature: float = 0.9) -> List[str]:
        """
        Generate a batch of prompts for a specific domain using the GGUF model
        
        Args:
            domain: The domain to generate prompts for
            batch_size: Number of prompts to generate
            
        Returns:
            List of generated prompts
        """
        # Construct the prompt for Qwen 3 using the ChatML format
        # Qwen models expect a specific chat format with system and user messages
        system_prompt = f"""You are a helpful AI assistant that generates high-quality training prompts for language models.
Generate exactly {batch_size} diverse, specific prompts in the domain of {domain}. /no_think

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

        user_prompt = f"{batch_size} prompts for {domain}:"
        
        # Qwen uses ChatML format: <|im_start|>role\ncontent<|im_end|>
        full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
        
        try:
            print(f"  Generating batch of {batch_size} prompts for {domain}...", end=" ", flush=True)
            
            # Generate using llama.cpp
            # The GGUF model through llama-cpp-python is much more memory efficient
            output = self.model(
                full_prompt,
                max_tokens=1024,  # Maximum length of generated text
                temperature=temperature,  # Use provided temperature (higher for more diversity)
                top_p=0.95,  # Nucleus sampling
                echo=False,  # Don't echo the prompt in output
                stop=["<|im_end|>", "<|im_start|>"],  # Stop tokens for Qwen
            )
            
            # Extract the generated text from the response
            generated_text = output['choices'][0]['text'].strip()
            
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
                
                # Skip lines that look like reasoning/commentary
                # These typically start with phrases like "Okay,", "Let me", "Wait,", etc.
                reasoning_indicators = [
                    'okay,', 'let me', 'wait,', 'hmm', 'first,', 'i think',
                    'i need', 'i should', 'the user', 'how about', 'maybe',
                    'alternatively,', 'here are', 'for example:', 'example:',
                    'bad output', 'good output', 'do not', "i'll", 'now,',
                    'so,', 'well,', 'actually,', 'but,', 'and then',
                ]
                
                line_lower = line.lower()
                if any(line_lower.startswith(indicator) for indicator in reasoning_indicators):
                    filtered_count += 1
                    if debug:
                        print(f"  [FILTERED - Reasoning]: {line[:80]}")
                    continue
                
                # Skip very long lines (likely reasoning text, not prompts)
                # Prompts are typically under 200 characters
                if len(line) > 300:
                    filtered_count += 1
                    if debug:
                        print(f"  [FILTERED - Too long]: {line[:80]}")
                    continue
                
                # Skip lines that are questions about generating prompts
                if 'generate' in line_lower and 'prompt' in line_lower:
                    filtered_count += 1
                    if debug:
                        print(f"  [FILTERED - Meta]: {line[:80]}")
                    continue
                
                # Skip metadata or formatting instructions
                if line.startswith('#') or line.startswith('---'):
                    continue
                
                # Remove leading numbers and dots/parentheses/bullets
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
                cleaned = cleaned.strip()
                
                # Only keep lines that are substantial prompts (at least 15 characters)
                if len(cleaned) >= 15:
                    prompts.append(cleaned)
            
            # Take only the requested number of prompts
            # The model might generate more than requested, so we trim to batch_size
            prompts = prompts[:batch_size]
            
            print(f"✓ ({len(prompts)} prompts)")
            return prompts
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return []
    
    def generate_all_prompts(self, domain_counts: Dict[str, int], batch_size: int = 10, temperature: float = 0.9) -> List[str]:
        """
        Generate all prompts across all domains
        
        Args:
            domain_counts: Dictionary mapping domains to number of prompts needed
            batch_size: Size of each generation batch
            temperature: Temperature for generation (higher = more diverse)
            
        Returns:
            List of all generated prompts
        """
        all_prompts = []
        
        for domain, count in domain_counts.items():
            if count == 0:
                continue
                
            print(f"\nGenerating {count} prompts for {domain}:")
            domain_prompts = []
            
            # Generate in batches
            remaining = count
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                batch_prompts = self.generate_prompts_batch(domain, current_batch_size, temperature=temperature)
                domain_prompts.extend(batch_prompts)
                remaining -= len(batch_prompts)
            
            print(f"  Total generated for {domain}: {len(domain_prompts)}")
            all_prompts.extend(domain_prompts)
        
        return all_prompts
    
    def remove_duplicates(self, prompts: List[str]) -> tuple[List[str], int]:
        """
        Remove duplicate prompts (case-insensitive)
        
        Returns:
            Tuple of (unique prompts, number of duplicates removed)
        """
        original_count = len(prompts)
        
        # Use case-insensitive comparison
        seen = set()
        unique_prompts = []
        
        for prompt in prompts:
            prompt_lower = prompt.lower().strip()
            if prompt_lower not in seen:
                seen.add(prompt_lower)
                unique_prompts.append(prompt)
        
        duplicates_removed = original_count - len(unique_prompts)
        return unique_prompts, duplicates_removed
    
    def regenerate_prompts(self, domain_counts: Dict[str, int], needed: int, batch_size: int = 10, temperature: float = 0.9) -> List[str]:
        """
        Regenerate prompts to replace removed duplicates
        Distributes the needed prompts across domains proportionally
        
        Args:
            temperature: Higher temperature for more diverse outputs during regeneration
        """
        print(f"\n🔄 Regenerating {needed} prompts to replace duplicates...")
        if temperature > 0.9:
            print(f"   Using increased temperature ({temperature:.2f}) for more diversity")
        
        # Calculate proportional distribution
        total_original = sum(domain_counts.values())
        
        # Guard against division by zero
        if total_original == 0:
            print("⚠️  No domains have prompts - cannot regenerate")
            return []
        
        regeneration_counts = {}
        allocated = 0
        
        # Sort domains by percentage to ensure larger domains get priority
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        
        # First pass: allocate based on proportion
        for domain, original_count in sorted_domains:
            if original_count == 0:
                regeneration_counts[domain] = 0
                continue
            
            proportion = original_count / total_original
            count = int(needed * proportion)  # Use int() instead of round() for more predictable behavior
            regeneration_counts[domain] = count
            allocated += count
        
        # Second pass: distribute remaining prompts to largest domains
        # This ensures we always generate exactly the needed number
        remaining = needed - allocated
        if remaining > 0:
            # Give remaining prompts to the domains with highest original counts
            for domain, original_count in sorted_domains:
                if remaining <= 0:
                    break
                if original_count > 0:  # Only give to domains that had prompts originally
                    regeneration_counts[domain] += 1
                    remaining -= 1
        
        # Final verification: ensure we're generating at least something
        total_to_generate = sum(regeneration_counts.values())
        if total_to_generate == 0 and needed > 0:
            # Emergency fallback: give all needed prompts to the largest domain
            largest_domain = sorted_domains[0][0]
            regeneration_counts[largest_domain] = needed
            print(f"⚠️  Fallback: Allocating all {needed} prompts to {largest_domain}")
        
        return self.generate_all_prompts(regeneration_counts, batch_size, temperature)
    
    def save_to_markdown(self, prompts: List[str], filename: str):
        """Save prompts to markdown file with one prompt per line"""
        with open(filename, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        print(f"\n✅ Saved {len(prompts)} prompts to {filename}")
    
    def generate_dataset(self, total_prompts: int, percentages: Dict[str, float], 
                        output_file: str = "prompt_dataset.md", batch_size: int = 10):
        """
        Main method to generate the complete dataset
        
        Args:
            total_prompts: Total number of prompts to generate
            percentages: Dictionary mapping domains to percentage (must sum to 100)
            output_file: Output markdown filename
            batch_size: Number of prompts to generate per batch
        """
        print("=" * 70)
        print("LLM PROMPT DATASET GENERATOR (Unsloth GGUF)")
        print("=" * 70)
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
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
        
        # Generate initial prompts
        print("\n" + "=" * 70)
        print("GENERATING PROMPTS")
        print("=" * 70)
        all_prompts = self.generate_all_prompts(domain_counts, batch_size)
        
        print(f"\n📈 Initial generation complete: {len(all_prompts)} prompts")
        
        # Remove duplicates
        print("\n" + "=" * 70)
        print("CHECKING FOR DUPLICATES")
        print("=" * 70)
        unique_prompts, duplicates_removed = self.remove_duplicates(all_prompts)
        print(f"Removed {duplicates_removed} duplicate prompts")
        print(f"Unique prompts: {len(unique_prompts)}")
        
        # Regenerate if needed
        if duplicates_removed > 10:
            prompts_needed = total_prompts - len(unique_prompts)
            max_retries = 10  # Limit regeneration attempts to prevent infinite loops
            retry_count = 0
            current_temperature = 0.9  # Start with default temperature
            previous_unique_count = len(unique_prompts)  # Track progress
            
            while prompts_needed > 0 and retry_count < max_retries:
                # Increase temperature with each retry to promote more diversity
                if retry_count > 0:
                    current_temperature = min(1.2, 0.9 + (retry_count * 0.05))
                
                new_prompts = self.regenerate_prompts(domain_counts, prompts_needed, batch_size, temperature=current_temperature)
                
                # Safety check: if regeneration returned nothing, we're stuck
                if len(new_prompts) == 0:
                    print(f"⚠️  Regeneration produced zero prompts - stopping to prevent infinite loop")
                    break
                
                # Combine and remove duplicates again
                combined = unique_prompts + new_prompts
                unique_prompts, new_duplicates = self.remove_duplicates(combined)
                
                print(f"After regeneration: {len(unique_prompts)} unique prompts ({new_duplicates} new duplicates)")
                
                # Check if we made any progress at all
                progress_made = len(unique_prompts) - previous_unique_count
                if progress_made == 0:
                    print(f"⚠️  No progress made (all {len(new_prompts)} new prompts were duplicates). Stopping.")
                    break
                
                previous_unique_count = len(unique_prompts)
                prompts_needed = total_prompts - len(unique_prompts)
                
                if prompts_needed > 0:
                    retry_count += 1
                    print(f"Still need {prompts_needed} more prompts... (attempt {retry_count}/{max_retries})")
            
            if prompts_needed > 0:
                if retry_count >= max_retries:
                    print(f"\n⚠️  Reached maximum retry limit ({max_retries} attempts)")
                print(f"Generated {len(unique_prompts)} unique prompts (target was {total_prompts})")
                print(f"Success rate: {len(unique_prompts)/total_prompts*100:.1f}%")
                print(f"This is close enough - the model has exhausted unique variations.")
        
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
            print(f"⚠️  Note: {total_prompts - len(unique_prompts)} prompts short of target due to duplicates")


def main():
    """Main entry point with example usage"""
    
    print("=" * 70)
    print("CONFIGURATION")
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
        "Machine Learning", "Creative Writing", "Logic", "Reasoning", "General Knowledge"
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
    
    # Generate dataset
    # The GGUF model is much more efficient than loading full models
    generator = PromptDatasetGenerator()
    generator.generate_dataset(total_prompts, percentages, output_file)


if __name__ == "__main__":
    main()
