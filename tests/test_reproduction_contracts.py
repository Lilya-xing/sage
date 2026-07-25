import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
