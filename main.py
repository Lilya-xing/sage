"""
SAGE Main v2 - New version using 3-layer architecture

Layer 1: State Machine - Hard-coded workflow control
Layer 2: Dynamic Prompt Generator - Code logic for prompt generation
Layer 3: LLM - Executes current task only
"""

import argparse
import os
import json
import time
from typing import Dict, Any, Optional

try:
    import torch  # optional
except Exception:
    torch = None

# Load environment variables
try:
    from dotenv import load_dotenv
    if load_dotenv('sage_config.env'):
        print("Loaded SAGE environment variables from sage_config.env")
except ImportError:
    # If dotenv is not available, manually load environment variables
    env_file = 'sage_config.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("Loaded SAGE environment variables from sage_config.env")
except Exception as e:
    print(f"Warning: Could not load environment variables: {e}")

from core.system import System
from tools.base import Tools
from core.agent import ask_agent
from environment.experiment import ExperimentEnvironment
from utils.common import save_final_results, str2dict
from core.controller import SAGEController

# Import token tracker
try:
    from tools.token_tracker import get_tracker, reset_tracker
    TOKEN_TRACKING_AVAILABLE = True
except ImportError:
    TOKEN_TRACKING_AVAILABLE = False
    def get_tracker():
        return None
    def reset_tracker():
        pass


def get_neuronpedia_config(target_llm: str, sae_path: str, sae_layer: int,
                           neuronpedia_model_id: Optional[str] = None,
                           neuronpedia_source: Optional[str] = None) -> Dict[str, str]:
    """Infer Neuronpedia identifiers for supported target model and SAE pairs."""
    target_llm_lower = target_llm.lower()

    if neuronpedia_model_id:
        model_id = neuronpedia_model_id
    elif 'gpt2' in target_llm_lower:
        model_id = 'gpt2-small'
    elif 'gpt-oss' in target_llm_lower or 'gpt_oss' in target_llm_lower:
        model_id = 'gpt-oss-20b'
    elif 'qwen3' in target_llm_lower:
        model_id = 'qwen3-4b'
    elif 'gemma' in target_llm_lower:
        model_id = 'gemma-2-2b'
    elif 'llama3.1' in target_llm_lower or 'llama-3.1' in target_llm_lower:
        model_id = 'llama3.1-8b-it'
    else:
        raise ValueError(
            f"Cannot infer Neuronpedia model ID from target_llm={target_llm!r}. "
            "Pass --neuronpedia_model_id and --neuronpedia_source explicitly."
        )

    if neuronpedia_source:
        source = neuronpedia_source
    else:
        sae_path_lower = sae_path.lower().replace('_', '-')

        if 'gemma-scope' in sae_path_lower or 'gemmascope' in sae_path_lower:
            width = '8k' if '8k' in sae_path_lower else '16k'
            activation_site = 'mlp' if 'pt-mlp' in sae_path_lower else 'res'
            source = f"{sae_layer}-gemmascope-{activation_site}-{width}"
        elif 'res-jb' in sae_path_lower:
            source = f"{sae_layer}-res-jb"
        elif 'resid-post-aa' in sae_path_lower:
            source = f"{sae_layer}-resid-post-aa"
        elif model_id == 'qwen3-4b':
            source = f"{sae_layer}-transcoder-hp"
        elif model_id == 'gpt-oss-20b':
            source = f"{sae_layer}-resid-post-aa"
        elif 'gemma' in model_id:
            source = f"{sae_layer}-gemmascope-res-16k"
        elif 'gpt2' in model_id:
            source = f"{sae_layer}-res-jb"
        elif 'llama3.1' in model_id:
            source = f"{sae_layer}-resid-post-aa"
        else:
            raise ValueError(
                f"Cannot infer Neuronpedia source for model_id={model_id!r}. "
                "Pass --neuronpedia_source explicitly."
            )

    return {'model_id': model_id, 'source': source}


def call_argparse():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SAGE v2: SAE Automated Interpretability Agent with 3-Layer Architecture')
    parser.add_argument('--agent_llm', type=str, default='gpt-5-nano', help='The LLM agent for reasoning')
    parser.add_argument('--target_llm', type=str, default='google/gemma-2-2b', help='The LLM to interpret')
    parser.add_argument('--model_revision', type=str, default=None, help='Exact Hugging Face revision for local target-model loading')
    parser.add_argument('--sae_path', type=str, default='sae-lens://release=gemma-scope-2b-pt-res-canonical;sae_id=layer_0/width_16k/canonical', help='Path/URI to the pretrained SAE')
    parser.add_argument('--features', type=str2dict, default='layer0=0', help='Features to interpret. Format: "layerX=feat1,feat2"')
    parser.add_argument('--path2save', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--dataset_path', type=str, default='./dataset/corpus.txt', help='Path to dataset corpus')
    parser.add_argument('--dataset_name', type=str, default=None, help='Hugging Face dataset name')
    parser.add_argument('--dataset_config', type=str, default=None, help='Hugging Face dataset config name')
    parser.add_argument('--dataset_split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--text_column', type=str, default='text', help='Column name containing text data')
    parser.add_argument('--device', type=str, default='cpu', help='Compute device')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--max_rounds', type=int, default=14, help='Maximum number of rounds')
    parser.add_argument('--timeout_minutes', type=int, default=30, help='Timeout in minutes')

    # Dataset sampling parameters (match Neuronpedia: n_prompts_total=24576, n_prompts_in_forward_pass=128)
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum corpus samples to evaluate (default: 5000, Neuronpedia: 24576)')
    parser.add_argument('--context_size', type=int, default=64, help='Tokens per prompt (default: 128)')
    parser.add_argument('--batch_size', type=int, default=8, help='Prompts per forward pass (default: 8, Neuronpedia: 128)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of maximally activating corpus examples to retrieve (default: 10)')
    parser.add_argument('--initial_hypotheses', type=int, default=4, help='Number of initial hypotheses (paper main setup: 4)')

    # SAEdashboard options
    parser.add_argument('--use_saedashboard', type=lambda x: x.lower() == 'true', default=True, help='Use SAEdashboard NeuronpediaRunner for activation extraction (default: True). Set to False to use fallback method.')

    # Neuronpedia API options
    parser.add_argument('--use_api_for_activations', type=lambda x: x.lower() == 'true', default=False, help='Use Neuronpedia API for find_maximally_activating_examples and get_activation_trace (default: False)')
    parser.add_argument('--use_api_for_exemplars', type=lambda x: x.lower() == 'true', default=None, help='Fetch top exemplars from Neuronpedia while allowing local custom-text traces')
    parser.add_argument('--neuronpedia_model_id', type=str, default=None, help='Neuronpedia model ID for API calls (e.g., "gpt2-small", "gemma-2-2b", "llama3.1-8b-it"). If not provided, will be inferred from target_llm.')
    parser.add_argument('--neuronpedia_source', type=str, default=None, help='Neuronpedia source/layer identifier for API calls (e.g., "0-gemmascope-mlp-16k", "9-res-jb", "11-resid-post-aa"). Required if use_api_for_activations=True. If not provided, will be inferred from sae_path and layer.')

    args = parser.parse_args()
    return args


def result_has_valid_conclusion(path: str) -> bool:
    """Return whether a saved structured result contains a final SAGE conclusion."""
    try:
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        "[DESCRIPTION]:" in analysis
        and "[EVIDENCE]:" in analysis
        and "[LABEL" in analysis
        for analysis in result.get("analysis_history", [])
    )


def run_single_feature_experiment(args, sae_layer_index: int, feature_index: int) -> Dict[str, Any]:
    """Run experiment for a single feature."""
    print(f"\n{'='*20} Starting Experiment: Layer {sae_layer_index}, Feature {feature_index} {'='*20}")
    
    # Determine save path
    path2save = os.path.join(args.path2save, args.agent_llm, args.target_llm.replace('/', '_'), f"layer_{sae_layer_index}", f"feature_{feature_index}")
    resolved_sae_path = args.sae_path.replace(
        "{layer}", str(sae_layer_index)
    )
    source_template = getattr(args, "neuronpedia_source", None)
    resolved_neuronpedia_source = (
        source_template.replace("{layer}", str(sae_layer_index)) if source_template else None
    )
    
    # Only a structured result with a valid final conclusion is resumable.
    structured_result_path = os.path.join(path2save, "structured_results.json")
    if result_has_valid_conclusion(structured_result_path):
        print("Valid structured results already exist. Skipping.")
        return {"status": "skipped", "reason": "results_exist"}
    
    os.makedirs(path2save, exist_ok=True)
    
    # Reset token tracker (create new statistics for each experiment)
    if TOKEN_TRACKING_AVAILABLE:
        reset_tracker()
    
    try:
        use_api_for_exemplars = getattr(args, "use_api_for_exemplars", None)
        if use_api_for_exemplars is None:
            use_api_for_exemplars = args.use_api_for_activations

        # Get Neuronpedia API configuration (if using API)
        neuronpedia_config = None
        if args.use_api_for_activations or use_api_for_exemplars:
            print("=" * 80)
            print("📡 Neuronpedia exemplar mode enabled")
            print("=" * 80)
            print(f"   - Top exemplars from API: {use_api_for_exemplars}")
            print(f"   - Custom-text traces from API: {args.use_api_for_activations}")
            print("   - use_saedashboard will be automatically disabled")
            print("=" * 80)
            
            # Validate that source is provided or can be inferred
            if not args.neuronpedia_source:
                print(f"⚠️  Warning: --neuronpedia_source not provided. Will attempt to infer from sae_path and layer.")
                print(f"   For better accuracy, please provide --neuronpedia_source explicitly.")
            
            neuronpedia_config = get_neuronpedia_config(
                target_llm=args.target_llm,
                sae_path=resolved_sae_path,
                sae_layer=sae_layer_index,
                neuronpedia_model_id=args.neuronpedia_model_id,
                neuronpedia_source=resolved_neuronpedia_source
            )
            print(f"📡 Neuronpedia API Config: model_id={neuronpedia_config['model_id']}, source={neuronpedia_config['source']}")
            if resolved_neuronpedia_source:
                print(f"   ✅ Source provided explicitly: {resolved_neuronpedia_source}")
            else:
                print(f"   ⚠️  Source inferred from sae_path and layer: {neuronpedia_config['source']}")
                print(f"   💡 Tip: To ensure accuracy, provide --neuronpedia_source explicitly")
            
            # Automatically disable use_saedashboard when using API
            if args.use_saedashboard:
                print(f"   ⚠️  use_saedashboard is enabled but will be ignored (API mode takes precedence)")
            args.use_saedashboard = False
        
        # Initialize system components
        system = System(
            llm_name=args.target_llm,
            model_revision=getattr(args, 'model_revision', None),
            sae_path=resolved_sae_path,
            sae_layer=sae_layer_index,
            feature_index=feature_index,
            device=args.device,
            debug=args.debug,  # Pass debug parameter
            use_api_for_activations=args.use_api_for_activations,
            neuronpedia_model_id=neuronpedia_config['model_id'] if neuronpedia_config else None,
            neuronpedia_source=neuronpedia_config['source'] if neuronpedia_config else None
        )
        
        tools = Tools(
            system=system,
            agent_llm_name=args.agent_llm,
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            use_activations_store=True,
            context_size=args.context_size,
            store_batch_size=args.batch_size,
            default_max_samples=args.max_samples,
            use_saedashboard=args.use_saedashboard,
            use_api_for_activations=use_api_for_exemplars,
            neuronpedia_model_id=neuronpedia_config['model_id'] if neuronpedia_config else None,
            neuronpedia_source=neuronpedia_config['source'] if neuronpedia_config else None,
            default_top_k=args.top_k
        )

        experiment_env = ExperimentEnvironment(tools, debug=args.debug, default_top_k=args.top_k)

        # Initialize log (simplified initialization, all prompts dynamically generated by prompt_generator)
        tools.init_log()
        
        # Create SAGE controller
        controller = SAGEController(
            feature_id=feature_index,
            layer=sae_layer_index,
            llm_client=args.agent_llm,  # Pass string directly, ask_agent function will handle it
            tools=tools,
            experiment_env=experiment_env,
            debug=args.debug,
            max_rounds=args.max_rounds,
            top_k=args.top_k,
            timeout_minutes=args.timeout_minutes,
            initial_hypotheses=getattr(args, 'initial_hypotheses', 4)
        )
        
        # Run experiment
        start_time = time.time()
        results = controller.run()
        end_time = time.time()
        
        # Add execution time
        results["execution_time_seconds"] = end_time - start_time
        results["target_configuration"] = {
            "target_llm": args.target_llm,
            "model_revision": getattr(args, "model_revision", None),
            "sae_path": resolved_sae_path,
            "neuronpedia_source": resolved_neuronpedia_source,
            "activation_backend": (
                "neuronpedia_api" if args.use_api_for_activations
                else "hybrid_api_exemplars_local_trace" if use_api_for_exemplars
                else "local"
            ),
        }
        
        # Save results
        save_final_results(tools.get_log(), path2save)
        
        # Save token statistics (if available)
        if TOKEN_TRACKING_AVAILABLE:
            tracker = get_tracker()
            if tracker:
                token_summary = tracker.get_summary()
                results["token_usage"] = token_summary
                
                # Save detailed token statistics to separate file
                token_stats_path = os.path.join(path2save, 'token_usage.json')
                tracker.save_to_file(token_stats_path)
                
                # Print token statistics summary
                tracker.print_summary()
        
        # Save structured results
        with open(os.path.join(path2save, 'structured_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Check if there is a valid conclusion
        has_conclusion = any(
            "[DESCRIPTION]:" in analysis and "[EVIDENCE]:" in analysis and "[LABEL" in analysis
            for analysis in results.get("analysis_history", [])
        )
        
        if has_conclusion:
            print(f"✅ Experiment for Feature {feature_index} completed successfully with conclusion. Results saved to {path2save}")
            return {"status": "completed", "results": results}
        else:
            print(f"⚠️  Experiment for Feature {feature_index} completed but no valid conclusion generated. Results saved to {path2save}")
            return {"status": "incomplete", "results": results}
        
    except Exception as e:
        print(f"❌ Fatal error during experiment for feature {feature_index}: {e}")
        
        # Save error log
        try:
            save_final_results(tools.get_log(), path2save, filename="error_log.json")
        except:
            pass
        
        return {"status": "error", "error": str(e)}


def main(args) -> int:
    """Run all requested features and return nonzero on any incomplete run."""
    print("🚀 Starting SAGE v2 with 3-Layer Architecture...")
    print(f"📁 Project directory: {os.getcwd()}")
    print(f"🐍 Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not activated')}")
    print(f"📄 Main file: {os.path.abspath(__file__)}")
    print(f"⚙️  Environment config: {os.path.join(os.getcwd(), 'sage_config.env')}")
    print("-" * 50)

    features_to_run = args.features
    total_experiments = sum(len(features) for features in features_to_run.values())
    completed_experiments = 0
    failed_experiments = 0
    print(f"📊 Total experiments to run: {total_experiments}")

    for layer_str, feature_indices in features_to_run.items():
        sae_layer_index = int(layer_str.replace('layer', ''))
        for feature_index in feature_indices:
            try:
                result = run_single_feature_experiment(
                    args, sae_layer_index, feature_index
                )
                completed_experiments += 1
                status = result["status"]
                if status == "completed":
                    print(
                        f"✅ Experiment {completed_experiments}/{total_experiments} "
                        "completed successfully with conclusion"
                    )
                elif status == "skipped":
                    print(
                        f"⏭️  Experiment {completed_experiments}/{total_experiments} "
                        "skipped (already exists)"
                    )
                else:
                    failed_experiments += 1
                    print(
                        f"❌ Experiment {completed_experiments}/{total_experiments} "
                        f"{status}: {result.get('error', 'no valid conclusion')}"
                    )
            except KeyboardInterrupt:
                print("\n🛑 Experiment interrupted by user")
                return 130
            except Exception as e:
                completed_experiments += 1
                failed_experiments += 1
                print(
                    f"❌ Unexpected error in experiment "
                    f"{completed_experiments}/{total_experiments}: {e}"
                )

    print(
        f"\nExperiments processed: {completed_experiments}/{total_experiments}; "
        f"failed or incomplete: {failed_experiments}"
    )
    return 1 if failed_experiments else 0

if __name__ == '__main__':
    raise SystemExit(main(call_argparse()))
