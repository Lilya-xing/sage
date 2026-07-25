import unittest
from unittest.mock import Mock, patch

from core.controller import SAGEController
from core.system import System
from scripts.batch_runner import build_command
from scripts.evaluate import (
    call_neuronpedia_api,
    generative_activation_threshold,
    mean_per_example_pearson,
)
from tools.neuronpedia import NeuronpediaManager


class SafetyTests(unittest.TestCase):
    def test_local_trace_fails_when_model_components_are_missing(self):
        system = object.__new__(System)
        system.use_api_for_activations = False
        system.model = None
        system.tokenizer = None
        system.sae = None
        system.layer = 3

        with self.assertRaisesRegex(RuntimeError, "model, tokenizer, and SAE"):
            system.get_activation_trace("hello")

    @patch("requests.post", side_effect=RuntimeError("network failed"))
    def test_neuronpedia_exemplar_error_is_not_converted_to_empty_data(self, _post):
        system = Mock(feature_index=5125, layer=11)
        manager = NeuronpediaManager(
            system=system,
            use_api_for_activations=True,
            neuronpedia_model_id="gemma-2-2b",
            neuronpedia_source="11-gemmascope-res-16k",
        )

        with self.assertRaisesRegex(RuntimeError, "network failed"):
            manager.get_maximally_activating_examples_from_api()

    @patch("scripts.evaluate.requests.post")
    @patch("scripts.evaluate.requests.get")
    def test_baseline_lookup_is_read_only(self, get_mock, post_mock):
        get_response = Mock(status_code=200)
        get_response.json.return_value = {
            "explanations": [
                {
                    "id": "existing-id",
                    "description": "a test description",
                    "explanationModelName": "gpt-5",
                    "typeName": "oai_token-act-pair",
                }
            ]
        }
        get_response.raise_for_status.return_value = None
        get_mock.return_value = get_response

        result = call_neuronpedia_api(
            model_id="gemma-2-2b",
            layer="11-gemmascope-res-16k",
            feature_index=5125,
            explanation_model_name="gpt-5",
            explanation_type="oai_token-act-pair",
        )

        self.assertEqual(result["source"], "existing")
        post_mock.assert_not_called()

    def test_zero_minute_controller_timeout_is_propagated(self):
        controller = SAGEController(
            feature_id=1,
            layer=3,
            llm_client="gpt-5",
            tools=Mock(),
            experiment_env=Mock(),
            max_rounds=14,
            timeout_minutes=0,
        )

        with self.assertRaises(TimeoutError):
            controller.run()


class MetricTests(unittest.TestCase):
    def test_generative_threshold_is_half_top10_maximum(self):
        exemplars = [{"activation": value} for value in [10.0, 8.0, 6.0]]
        self.assertEqual(generative_activation_threshold(exemplars), 5.0)

    def test_predictive_metric_is_mean_of_example_correlations(self):
        result = mean_per_example_pearson(
            [
                ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
                ([0.0, 1.0, 2.0], [2.0, 1.0, 0.0]),
            ]
        )
        self.assertAlmostEqual(result["correlation"], 0.0, places=7)
        self.assertEqual(result["valid_examples"], 2)


class BatchRunnerTests(unittest.TestCase):
    def test_command_layer_matches_neuronpedia_source(self):
        command = build_command(
            working_dir=Mock(__str__=lambda self: "/tmp/SAGE"),
            cuda=0,
            target_llm="Qwen/Qwen3-4B",
            neuronpedia_model_id="qwen3-4b",
            neuronpedia_source="23-transcoder-hp",
            sae_path="unused",
            feature=42,
            use_api=True,
            agent_llm="gpt-5",
            max_rounds=14,
            timeout_minutes=30,
            top_k=10,
            device="cuda:0",
            path2save="results",
            debug=False,
        )

        self.assertIn("--features layer23=42", command)


if __name__ == "__main__":
    unittest.main()
