"""CPU regression checks; no pretrained weights or dataset required."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from args import parse_args


ROOT = Path(__file__).resolve().parents[1]


def load_definition(path, name, namespace):
    """Load the production definition without importing CUDA/model dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def full_args(*extra):
    with mock.patch.object(sys, 'argv', [
        'prog', '/tmp/data', '--module_rsap_v1', '--text_adjust', 'True',
        '--no_module_visual_guidance', '--loss_sdr', *extra,
    ]):
        return parse_args()


class NoReliabilityConfigTest(unittest.TestCase):
    def test_default_and_opt_in(self):
        self.assertFalse(full_args().no_reliability)
        self.assertTrue(full_args('--no_reliability').no_reliability)

    def test_requires_full_model(self):
        for flags in ([], ['--loss_sdr'], ['--module_rsap_v1', '--text_adjust', 'True']):
            with self.subTest(flags=flags), mock.patch.object(
                sys, 'argv', ['prog', '/tmp/data', '--no_reliability', *flags]
            ):
                with self.assertRaisesRegex(ValueError, 'requires Full'):
                    parse_args()

    def test_result_identity_and_effective_settings(self):
        tag = load_definition('ctta_eval_remote.py', '_result_tag', {})
        config = load_definition('ctta_eval_remote.py', '_stabilization_config', {})
        modules = load_definition('ctta_eval_remote.py', '_tmpa_module_config', {})
        full = full_args()
        ablation = full_args('--no_reliability', '--sdr_min_weight', '0')
        self.assertNotEqual(tag(full), tag(ablation))
        self.assertNotIn('full-wo-reliability', tag(full))
        self.assertIn('full-wo-reliability', tag(ablation))
        self.assertEqual(config(ablation)['sdr']['pixel_weighting'], 'uniform')
        self.assertEqual(config(ablation)['sdr']['effective_min_pixel_weight'], 1.0)
        self.assertFalse(modules(ablation)['reliability_prompt_bank']['reliability_estimation_enabled'])


try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for numerical tests')
class NoReliabilityNumericalTest(unittest.TestCase):
    def setUp(self):
        self.mine = load_definition(
            'model/learnable_shift.py', '_rsap_consensus_visual_mining',
            {'torch': torch, 'F': F},
        )
        self.model = SimpleNamespace(num_classes=3, query_idx=torch.tensor([0, 0, 1, 1, 2, 2]), logit_scale=8.0)
        self.prompts = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
        self.tokens = torch.tensor([[[1., x, 0.] for x in (0., .1, .2, .3, .4, .5)]]).requires_grad_()

    def test_uniform_prototypes_ungated_alpha_and_absent_classes(self):
        prototypes, gates, token_map = self.mine(
            self.model, self.tokens, self.prompts, [0, 0, 1, 1, 2, 2],
            topk=3, use_reliability=False,
        )
        # Six candidates, K=3 => midpoint indices 1, 3, 5, equal weights.
        expected = F.normalize(self.tokens.detach()[0], dim=-1)[[1, 3, 5]].mean(0)
        torch.testing.assert_close(prototypes[0], expected)
        torch.testing.assert_close(prototypes[1], expected)
        torch.testing.assert_close(prototypes[2:], torch.zeros_like(prototypes[2:]))
        torch.testing.assert_close(gates[:, 0], torch.tensor([1., 1., 0., 0., 0., 0.]))
        torch.testing.assert_close(token_map, torch.ones(6))
        self.assertFalse(prototypes.requires_grad)
        self.assertFalse(gates.requires_grad)

    def test_gamma_independence_repeatability_and_default_full(self):
        inputs = (self.model, self.tokens, self.prompts, [0, 0, 1, 1, 2, 2])
        state = torch.random.get_rng_state().clone()
        a = self.mine(*inputs, gamma=0., use_reliability=False)
        b = self.mine(*inputs, gamma=100., use_reliability=False)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        default = self.mine(*inputs)
        explicit = self.mine(*inputs, use_reliability=True)
        for x, y in zip(default, explicit):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertTrue((default[2] < 1).all())

    def test_fewer_candidates_than_k(self):
        prototypes, _, _ = self.mine(
            self.model, self.tokens, self.prompts, [0, 0, 1, 1, 2, 2],
            topk=20, use_reliability=False,
        )
        expected = F.normalize(self.tokens.detach()[0], dim=-1).mean(0)
        torch.testing.assert_close(prototypes[0], expected)

    def test_adaptation_uses_uniform_gsc_without_map_even_with_zero_floor(self):
        namespace = {'torch': torch}
        for name in ('_class_dim', '_normalize_probabilities', 'source_prediction_consistency'):
            load_definition('ctta/modules/consistency.py', name, namespace)
        consistency = namespace['source_prediction_consistency']
        logits = torch.tensor([[[2., 0.]], [[0., 1.]]], requires_grad=True)
        adapted = logits.softmax(0)
        source = torch.tensor([[[.5, .7]], [[.5, .3]]], requires_grad=True)
        expected = consistency(adapted, source)
        namespace.update({
            'needs_source_model': lambda args: True,
            '_forward_source': lambda *args: source.detach(),
            '_forward_adapted': lambda *args: (
                None, adapted, adapted.new_zeros(()), adapted.new_zeros(()), None
            ),
        })
        tune = load_definition('ctta/adaptation.py', 'test_time_tuning_ctta', namespace)
        model = SimpleNamespace(last_reliability_map=None, parameters=lambda: iter([logits]))
        optimizer = torch.optim.SGD([logits], lr=0.)
        scaler = SimpleNamespace(scale=lambda loss: loss, step=lambda opt: opt.step(), update=lambda: None)
        args = full_args('--no_reliability', '--sdr_min_weight', '0')
        reports = tune('dummy', model, torch.zeros(1), (1, 2), optimizer, scaler, args, source_model=object())
        self.assertAlmostEqual(reports[0]['sdr'], expected.item(), places=6)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.)
        self.assertIsNone(source.grad)


if __name__ == '__main__':
    unittest.main()
