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
        
        # Initialize llama.cpp with the GGUF file
        # n_gpu_layers=-1 means use all available GPU layers
        # Setting to 0 would use CPU only
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,  # Context window size
                n_gpu_layers=n_gpu_layers,  # Use GPU if available
                verbose=False,  # Reduce logging noise
                n_threads=os.cpu_count() or 4,  # Use all CPU threads for CPU portions
            )
            print("✅ Model loaded successfully!")
            print(f"   Context size: {n_ctx}")
            print(f"   GPU layers: {n_gpu_layers if n_gpu_layers >= 0 else 'all'}")
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
    
    def generate_prompts_batch(self, domain: str, batch_size: int) -> List[str]:
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

        user_prompt = f"Generate {batch_size} unique prompts for {domain}."
        
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
                temperature=0.9,  # Higher temperature for more diversity
                top_p=0.95,  # Nucleus sampling
                echo=False,  # Don't echo the prompt in output
                stop=["<|im_end|>", "<|im_start|>"],  # Stop tokens for Qwen
            )
            
            # Extract the generated text from the response
            generated_text = output['choices'][0]['text'].strip()
            
            # Split by newlines and clean
            prompts = [
                line.strip() 
                for line in generated_text.split('\n') 
                if line.strip() and not line.strip().startswith('#')
            ]
            
            # Remove any remaining numbering (1., 2., etc.) and bullet points
            cleaned_prompts = []
            for prompt in prompts:
                # Remove leading numbers and dots/parentheses
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', prompt)
                cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
                if cleaned and len(cleaned) > 10:  # Filter out very short non-prompts
                    cleaned_prompts.append(cleaned)
            
            print(f"✓ ({len(cleaned_prompts)} prompts)")
            return cleaned_prompts
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return []
    
    def generate_all_prompts(self, domain_counts: Dict[str, int], batch_size: int = 10) -> List[str]:
        """
        Generate all prompts across all domains
        
        Args:
            domain_counts: Dictionary mapping domains to number of prompts needed
            batch_size: Size of each generation batch
            
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
                batch_prompts = self.generate_prompts_batch(domain, current_batch_size)
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
    
    def regenerate_prompts(self, domain_counts: Dict[str, int], needed: int, batch_size: int = 10) -> List[str]:
        """
        Regenerate prompts to replace removed duplicates
        Distributes the needed prompts across domains proportionally
        """
        print(f"\n🔄 Regenerating {needed} prompts to replace duplicates...")
        
        # Calculate proportional distribution
        total_original = sum(domain_counts.values())
        regeneration_counts = {}
        remaining = needed
        
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (domain, original_count) in enumerate(sorted_domains):
            if original_count == 0:
                regeneration_counts[domain] = 0
                continue
                
            if i == len(sorted_domains) - 1:
                regeneration_counts[domain] = remaining
            else:
                proportion = original_count / total_original
                count = round(needed * proportion)
                regeneration_counts[domain] = count
                remaining -= count
        
        return self.generate_all_prompts(regeneration_counts, batch_size)
    
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
            
            while prompts_needed > 0:
                new_prompts = self.regenerate_prompts(domain_counts, prompts_needed, batch_size)
                
                # Combine and remove duplicates again
                combined = unique_prompts + new_prompts
                unique_prompts, new_duplicates = self.remove_duplicates(combined)
                
                print(f"After regeneration: {len(unique_prompts)} unique prompts ({new_duplicates} new duplicates)")
                prompts_needed = total_prompts - len(unique_prompts)
                
                if prompts_needed > 0:
                    print(f"Still need {prompts_needed} more prompts...")
        
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