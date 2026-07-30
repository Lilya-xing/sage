import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.controller import SAGEController
from core.state_machine import SAGEState
from scripts.build_formal_manifest import build_matrix, working_diff_sha256
from scripts.evaluate import (
    select_exemplars_for_prediction_evaluation,
    stable_feature_seed,
)
from scripts.evaluate_formal_target import generation_result_valid
from scripts.run_formal_qwen_parallel import (
    build_layer_command,
    parse_devices,
)
from scripts.run_formal_target import build_command, select_entries
from tools.neuronpedia_baseline import build_messages
from tools.output_validator import OutputValidator
from scripts import preflight
import main


class NeuronpediaConfigTests(unittest.TestCase):
    def test_paper_gemma_residual_source_is_inferred(self):
        config = main.get_neuronpedia_config(
            target_llm="google/gemma-2-2b",
            sae_path=(
                "sae-lens://release=gemma-scope-2b-pt-res-canonical;"
                "sae_id=layer_3/width_16k/canonical"
            ),
            sae_layer=3,
        )

        self.assertEqual(config["model_id"], "gemma-2-2b")
        self.assertEqual(config["source"], "3-gemmascope-res-16k")

    def test_paper_qwen_source_is_inferred(self):
        config = main.get_neuronpedia_config(
            target_llm="Qwen/Qwen3-4B",
            sae_path="",
            sae_layer=7,
        )

        self.assertEqual(config["model_id"], "qwen3-4b")
        self.assertEqual(config["source"], "7-transcoder-hp")

    def test_paper_gpt_oss_source_is_inferred(self):
        config = main.get_neuronpedia_config(
            target_llm="openai/gpt-oss-20b",
            sae_path="",
            sae_layer=11,
        )

        self.assertEqual(config["model_id"], "gpt-oss-20b")
        self.assertEqual(config["source"], "11-resid-post-aa")

    def test_explicit_source_wins(self):
        config = main.get_neuronpedia_config(
            target_llm="unknown/model",
            sae_path="",
            sae_layer=23,
            neuronpedia_model_id="custom-model",
            neuronpedia_source="custom-source",
        )

        self.assertEqual(
            config,
            {"model_id": "custom-model", "source": "custom-source"},
        )

    def test_unknown_model_without_explicit_ids_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer Neuronpedia model ID"):
            main.get_neuronpedia_config(
                target_llm="unknown/model",
                sae_path="",
                sae_layer=23,
            )


class CliContractTests(unittest.TestCase):
    def test_max_rounds_reaches_controller(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(
                agent_llm="gpt-5",
                target_llm="google/gemma-2-2b",
                sae_path="unused-in-api-mode",
                path2save=tmp_dir,
                dataset_path="",
                dataset_name=None,
                dataset_config=None,
                dataset_split="train",
                text_column="text",
                device="cpu",
                debug=False,
                max_rounds=7,
                timeout_minutes=3,
                max_samples=10,
                context_size=16,
                batch_size=2,
                top_k=2,
                use_saedashboard=False,
                use_api_for_activations=True,
                neuronpedia_model_id="gemma-2-2b",
                neuronpedia_source="3-gemmascope-res-16k",
            )

            with (
                patch.object(main, "System"),
                patch.object(main, "Tools") as tools_cls,
                patch.object(main, "ExperimentEnvironment"),
                patch.object(main, "SAGEController") as controller_cls,
                patch.object(main, "save_final_results"),
                patch.object(main, "TOKEN_TRACKING_AVAILABLE", False),
            ):
                tools_cls.return_value.get_log.return_value = []
                controller_cls.return_value.run.return_value = {
                    "analysis_history": []
                }

                result = main.run_single_feature_experiment(args, 3, 5125)

            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(controller_cls.call_args.kwargs["max_rounds"], 7)
            self.assertEqual(controller_cls.call_args.kwargs["timeout_minutes"], 3)
            self.assertTrue(
                Path(tmp_dir).joinpath(
                    "gpt-5",
                    "google_gemma-2-2b",
                    "layer_3",
                    "feature_5125",
                    "structured_results.json",
                ).exists()
            )


class ControllerBudgetTests(unittest.TestCase):
    def test_parallel_controller_steps_do_not_consume_paper_round_budget(self):
        controller = SAGEController(
            feature_id=5125,
            layer=11,
            llm_client="gpt-5",
            tools=SimpleNamespace(),
            experiment_env=SimpleNamespace(),
            max_rounds=4,
            timeout_minutes=1,
        )
        iterations = 0

        def execute_parallel_step():
            nonlocal iterations
            iterations += 1
            controller.execution_stats["total_rounds"] += 1
            if iterations == 7:
                controller.state_machine.state = SAGEState.DONE

        with patch.object(
            controller,
            "_execute_round",
            side_effect=execute_parallel_step,
        ):
            result = controller.run()

        self.assertEqual(iterations, 7)
        self.assertEqual(result["final_state"], SAGEState.DONE.value)


class PreflightTests(unittest.TestCase):
    def test_paper_mode_only_requires_api_evaluation_stack(self):
        self.assertEqual(
            preflight.packages_for_mode("paper"),
            ["openai", "requests", "python-dotenv", "numpy", "scipy"],
        )

    def test_local_mode_includes_model_and_sae_stack(self):
        packages = preflight.packages_for_mode("local")
        self.assertIn("torch", packages)
        self.assertIn("datasets", packages)
        self.assertIn("sae-lens", packages)
        self.assertIn("transformer-lens", packages)


class FormalReproductionTests(unittest.TestCase):
    def test_formal_api_runner_uses_paper_spec_and_fixed_matrix(self):
        manifest = {
            "paper_main_spec": {
                "max_rounds": 14,
                "top_k_exemplars": 10,
                "initial_hypotheses": 4,
            },
            "replication_pins": {
                "agent_and_generation_model": "gpt-5-2025-08-07"
            },
            "aggregate_matrix": [
                {
                    "target_llm": "google/gemma-2-2b",
                    "neuronpedia_model_id": "gemma-2-2b",
                    "layer": 7,
                    "source": "7-gemmascope-res-16k",
                    "features": [1, 2],
                    "activation_backend": "neuronpedia_api",
                }
            ],
        }
        entry = select_entries(manifest, "gemma-2-2b", [7])[0]
        command = build_command(manifest, entry, Path("/tmp/formal"))
        rendered = " ".join(command)
        self.assertIn("--agent_llm gpt-5-2025-08-07", rendered)
        self.assertIn("--max_rounds 14", rendered)
        self.assertIn("--top_k 10", rendered)
        self.assertIn("--initial_hypotheses 4", rendered)
        self.assertIn("--features layer7=1,2", rendered)

    def test_working_source_fingerprint_is_a_sha256(self):
        self.assertRegex(working_diff_sha256(), r"^[0-9a-f]{64}$")

    def test_formal_matrix_has_paper_shape_and_valid_feature_ids(self):
        matrix = build_matrix(20260729)
        self.assertEqual(len(matrix), 12)
        self.assertEqual(sum(len(item["features"]) for item in matrix), 120)
        for item in matrix:
            self.assertEqual(len(set(item["features"])), 10)
            self.assertTrue(all(0 <= value < item["d_sae"] for value in item["features"]))

    def test_heldout_selection_is_order_independent_and_repeatable(self):
        exemplars = [
            {
                "text": f"example-{index}",
                "full_text": f"example-{index}",
                "activation": float(100 - index),
                "max_token": str(index),
            }
            for index in range(100)
        ]
        seed = stable_feature_seed(20260729, "gemma-2-2b", "11-gemmascope-res-16k", 5125)
        first = select_exemplars_for_prediction_evaluation(
            exemplars, rng=__import__("random").Random(seed)
        )
        second = select_exemplars_for_prediction_evaluation(
            exemplars, rng=__import__("random").Random(seed)
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first[2]), 10)

    def test_main_prompt_contract_requires_exactly_four_hypotheses(self):
        validator = OutputValidator(initial_hypotheses=4)
        four = """OBSERVATION:
This is a sufficiently detailed observation grounded in several real activation examples.
[HYPOTHESIS LIST]:
Hypothesis_1: one sufficiently detailed and testable hypothesis here
Hypothesis_2: two sufficiently detailed and testable hypothesis here
Hypothesis_3: three sufficiently detailed and testable hypothesis here
Hypothesis_4: four sufficiently detailed and testable hypothesis here
"""
        valid, _ = validator.validate(SAGEState.ANALYZE_EXEMPLARS, four)
        invalid, message = validator.validate(
            SAGEState.ANALYZE_EXEMPLARS,
            four.replace(
                "Hypothesis_4: four sufficiently detailed and testable hypothesis here\n",
                "",
            ),
        )
        self.assertTrue(valid)
        self.assertFalse(invalid)
        self.assertIn("exactly 4", message)

    def test_local_baseline_uses_full_activation_tokens(self):
        messages = build_messages(
            [{
                "tokens": ["short"], "values": [1],
                "full_tokens": ["full", " token"], "full_values": [0, 10],
            }]
        )
        self.assertEqual(len(messages), 8)
        self.assertIn("full\t0", messages[-1]["content"])
        self.assertIn(" token\t10", messages[-1]["content"])


class ParallelAccelerationTests(unittest.TestCase):
    def test_valid_conclusion_is_the_only_resumable_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "structured_results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "analysis_history": [
                            "[DESCRIPTION]: detector\n"
                            "[EVIDENCE]: examples\n"
                            "[LABEL]: lexical"
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(generation_result_valid(result_path))
            self.assertTrue(main.result_has_valid_conclusion(str(result_path)))
            result_path.write_text(
                json.dumps({"analysis_history": ["unfinished"]}),
                encoding="utf-8",
            )
            self.assertFalse(generation_result_valid(result_path))
            self.assertFalse(main.result_has_valid_conclusion(str(result_path)))

    def test_qwen_layer_command_preserves_paper_contract(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "reproduction" / "formal_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        entry = next(
            item for item in manifest["aggregate_matrix"]
            if item["neuronpedia_model_id"] == "qwen3-4b" and item["layer"] == 7
        )
        command = build_layer_command(manifest, entry, entry["features"], Path("/tmp/formal"))
        rendered = " ".join(command)
        self.assertIn("--features layer7=", rendered)
        self.assertNotIn(";layer", rendered)
        self.assertIn("--max_rounds 14", rendered)
        self.assertIn("--top_k 10", rendered)
        self.assertIn("--initial_hypotheses 4", rendered)
        self.assertEqual(parse_devices("4,6,7"), ["4", "6", "7"])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_devices("4,4")


if __name__ == "__main__":
    unittest.main()
