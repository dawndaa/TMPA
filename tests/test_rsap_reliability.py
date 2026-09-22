"""CPU checks of RSAP/SDR reliability separation, without pretrained weights."""

import ast
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import parse_args


ROOT = Path(__file__).resolve().parents[1]


def load_definition(path, name, namespace=None):
    """Execute production methods without importing CLIP/CUDA/data dependencies."""
    namespace = {'torch': torch, 'nn': nn, 'F': F} if namespace is None else namespace
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def rsap_args(*extra):
    with mock.patch.object(sys, 'argv', [
        'prog', '/tmp/data', '--module_rsap_v1', '--text_adjust', 'True',
        '--no_module_visual_guidance', *extra,
    ]):
        return parse_args()


class RSAPReliabilityConfigTest(unittest.TestCase):
    def test_default_and_rsap_only_or_full(self):
        self.assertFalse(rsap_args().no_rsap_reliability)
        for flags in ([], ['--loss_sdr']):
            args = rsap_args('--no_rsap_reliability', *flags)
            self.assertTrue(args.no_rsap_reliability)
            self.assertEqual(args.sdr_min_weight, 0.5)

    def test_requires_rsap(self):
        with mock.patch.object(sys, 'argv', ['prog', '/tmp/data', '--loss_sdr', '--no_rsap_reliability']):
            with self.assertRaisesRegex(ValueError, 'requires --module_rsap_v1'):
                parse_args()

    def test_sdr_and_joint_switch_validation(self):
        with mock.patch.object(sys, 'argv', ['prog', '/tmp/data', '--no_sdr_reliability']):
            with self.assertRaisesRegex(ValueError, 'requires --loss_sdr'):
                parse_args()
        with self.assertRaisesRegex(ValueError, 'requires Full'):
            rsap_args('--no_reliability')
        with mock.patch.object(sys, 'argv', ['prog', '/tmp/data', '--loss_sdr', '--no_reliability']):
            with self.assertRaisesRegex(ValueError, 'requires Full'):
                parse_args()

    def test_joint_alias_matches_independent_switches(self):
        tag = load_definition('ctta_eval_remote.py', '_result_tag')
        alias = rsap_args('--loss_sdr', '--no_reliability')
        combined = rsap_args('--loss_sdr', '--no_rsap_reliability', '--no_sdr_reliability')
        self.assertTrue(alias.no_rsap_reliability and alias.no_sdr_reliability)
        self.assertEqual(tag(alias), tag(combined))
        self.assertIn('full-wo-reliability', tag(alias))

    def test_four_configurations_have_distinct_tags_and_metadata(self):
        tag = load_definition('ctta_eval_remote.py', '_result_tag')
        modules = load_definition('ctta_eval_remote.py', '_tmpa_module_config')
        stabilizers = load_definition('ctta_eval_remote.py', '_stabilization_config')
        tags = set()
        for disabled in (False, True):
            for sdr_disabled in (False, True):
                flags = ['--no_rsap_reliability'] if disabled else []
                if sdr_disabled:
                    flags.append('--no_sdr_reliability')
                args = rsap_args('--loss_sdr', *flags)
                tags.add(tag(args))
                self.assertEqual('wo-rsap-reliability' in tag(args), disabled and not sdr_disabled)
                self.assertEqual(modules(args)['rsap_v1']['reliability_enabled'], not disabled)
                self.assertTrue(modules(args)['reliability_prompt_bank']['enabled'])
                self.assertEqual(stabilizers(args)['sdr']['pixel_weighting'], 'uniform' if sdr_disabled else 'reliability')
                self.assertEqual(stabilizers(args)['sdr']['effective_min_pixel_weight'], 1.0 if sdr_disabled else .5)
        self.assertEqual(len(tags), 4)


class RSAPReliabilityNumericalTest(unittest.TestCase):
    def setUp(self):
        self.mine = load_definition('model/learnable_shift.py', '_rsap_consensus_visual_mining')
        self.model = SimpleNamespace(num_classes=3, query_idx=torch.tensor([0, 0, 1, 1, 2, 2]), logit_scale=4.)
        self.prompts = torch.tensor([[1., 0., 0.], [1., .6, 0.], [0., 1., 0.], [.1, 1., 0.], [0., 0., 1.], [0., .1, 1.]])
        self.tokens = torch.tensor([[[1., x, 0.] for x in (0., .1, .2, .3, .4, .5)]]).requires_grad_()

    def inputs(self):
        return self.model, self.tokens, self.prompts, self.model.query_idx.tolist()

    def test_equal_mean_unit_gates_and_original_sdr_map(self):
        full = self.mine(*self.inputs())
        proto, gates, reliability = self.mine(*self.inputs(), use_rsap_reliability=False)
        expected = F.normalize(self.tokens.detach()[0], dim=-1)[[1, 3, 5]].mean(0)
        torch.testing.assert_close(proto[:2], expected.expand(2, -1))
        torch.testing.assert_close(proto[2:], torch.zeros_like(proto[2:]))
        torch.testing.assert_close(gates[:, 0], torch.tensor([1., 1., 0., 0., 0., 0.]))
        torch.testing.assert_close(reliability, full[2], rtol=0, atol=0)
        self.assertGreater(reliability.max().item() - reliability.min().item(), .001)
        self.assertTrue((reliability < 1).all())
        for value in (proto, gates, reliability):
            self.assertFalse(value.requires_grad)

    def test_gamma_affects_sdr_map_but_not_ablated_rsap(self):
        state = torch.random.get_rng_state().clone()
        a = self.mine(*self.inputs(), gamma=0., use_rsap_reliability=False)
        b = self.mine(*self.inputs(), gamma=100., use_rsap_reliability=False)
        for left, right in zip(a[:2], b[:2]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertFalse(torch.allclose(a[2], b[2]))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_fewer_candidates_than_k(self):
        proto, _, _ = self.mine(*self.inputs(), topk=20, use_rsap_reliability=False)
        expected = F.normalize(self.tokens.detach()[0], dim=-1).mean(0)
        torch.testing.assert_close(proto[0], expected)

    def test_forward_and_slide_preserve_sdr_map_and_alpha_gradients(self):
        # Stub only feature extraction and upsampling; execute production mining,
        # fusion, interpolation and sliding-window map assembly on CPU tensors.
        tokens = self.tokens.detach().half()

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = nn.Linear(1, 1).half()

            def encode_image(self, *args):
                return tokens[:, 0], tokens.clone()

        model = self.model
        model.net = Encoder()
        model.parameters = model.net.parameters
        model.device = torch.device('cpu')
        model.text_features = F.normalize(self.prompts.half(), dim=-1)
        model.reliability_query_features = self.prompts.half()
        model.reliability_query_idx_list = model.query_idx.tolist()
        model.alpha = nn.Parameter(torch.full((6, 1), .3, dtype=torch.float16))
        model.patch_size = (1, 1)
        model.ignore_residual = True
        model.feature_up = True
        model.feat_dim = 3
        model.cls_token_lambda = 0.
        model.upsampler = lambda features, image: features
        model.postprocess_result = lambda logits, *args: (None, logits[0].float().softmax(0))
        for name in ('_rsap_consensus_visual_mining', 'forward_feature', 'forward_slide', 'compute_padsize'):
            setattr(model, name, MethodType(load_definition('model/learnable_shift.py', name), model))
        image = torch.ones(1, 3, 2, 3, dtype=torch.float16)
        full_args = rsap_args('--loss_sdr')
        ablated_args = rsap_args('--loss_sdr', '--no_rsap_reliability')
        _, full = model.forward_slide(image, (3, 2), 'dummy', full_args, stride=(2, 3), crop_size=(2, 3))
        full_map = model.last_reliability_map.clone()
        _, sdr_only = model.forward_slide(
            image, (3, 2), 'dummy', rsap_args('--loss_sdr', '--no_sdr_reliability'),
            stride=(2, 3), crop_size=(2, 3),
        )
        torch.testing.assert_close(sdr_only, full, rtol=0, atol=0)
        torch.testing.assert_close(model.last_reliability_map, full_map, rtol=0, atol=0)
        _, ablated = model.forward_slide(image, (3, 2), 'dummy', ablated_args, stride=(2, 3), crop_size=(2, 3))
        torch.testing.assert_close(model.last_reliability_map, full_map, rtol=0, atol=0)
        self.assertFalse(torch.allclose(full, ablated))
        self.assertFalse(full_map.requires_grad)
        ablated[0].sum().backward()
        self.assertTrue(torch.isfinite(model.alpha.grad).all())
        self.assertGreater(model.alpha.grad.abs().sum().item(), 0.)

    def test_sdr_and_joint_ablation_use_uniform_gsc_and_adapted_gradients(self):
        namespace = {'torch': torch}
        for name in (
            '_class_dim', '_normalize_probabilities', 'source_prediction_consistency',
            'reliability_guided_source_consistency',
        ):
            load_definition('ctta/modules/consistency.py', name, namespace)
        for flags in ([], ['--no_sdr_reliability'], ['--no_reliability']):
            with self.subTest(flags=flags):
                args = rsap_args('--loss_sdr', '--sdr_min_weight', '0', *flags)
                logits = torch.tensor([[[2., 0.]], [[0., 1.]]], requires_grad=True)
                adapted = logits.softmax(0)
                source = torch.tensor([[[.5, .7]], [[.5, .3]]], requires_grad=True)
                reliability = torch.tensor([[.1, .8]], requires_grad=True)
                uniform = namespace['source_prediction_consistency'](adapted, source)
                guided = namespace['reliability_guided_source_consistency'](
                    adapted, source, reliability, min_weight=0.,
                )
                self.assertFalse(torch.allclose(uniform, guided))
                namespace.update({
                    'needs_source_model': lambda args: True,
                    '_forward_source': lambda *args: source.detach(),
                    '_forward_adapted': lambda *args: (
                        None, adapted, adapted.new_zeros(()), adapted.new_zeros(()), None
                    ),
                })
                tune = load_definition('ctta/adaptation.py', 'test_time_tuning_ctta', namespace)
                model = SimpleNamespace(
                    last_reliability_map=None if flags else reliability,
                    parameters=lambda: iter([logits]),
                )
                optimizer = torch.optim.SGD([logits], lr=0.)
                scaler = SimpleNamespace(scale=lambda loss: loss, step=lambda opt: opt.step(), update=lambda: None)
                reports = tune('dummy', model, torch.zeros(1), (1, 2), optimizer, scaler, args, source_model=object())
                expected = uniform if flags else guided
                self.assertAlmostEqual(reports[0]['sdr'], expected.item(), places=6)
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertGreater(logits.grad.abs().sum().item(), 0.)
                self.assertIsNone(source.grad)
                self.assertIsNone(reliability.grad)


if __name__ == '__main__':
    unittest.main()
