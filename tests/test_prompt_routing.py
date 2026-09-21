import os
import sys
import tempfile
import unittest
from unittest import mock

from args import parse_args


class PromptRoutingV4Test(unittest.TestCase):
    def _make_prompt_file(self):
        fd, path = tempfile.mkstemp(prefix='rsap_sdr_v4_', suffix='.txt')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('building|urban building|roof structure\n')
            f.write('road|paved road|transportation surface\n')
        return path

    def test_sdr_can_use_single_prompt_prediction_with_auxiliary_reliability_bank(self):
        prompt_path = self._make_prompt_file()
        try:
            argv = [
                'prog',
                '/tmp/data',
                '--name_path', prompt_path,
                '--no_module_cat_prompt',
                '--no_module_visual_guidance',
                '--loss_sdr',
            ]
            with mock.patch.object(sys, 'argv', argv):
                args = parse_args()

            self.assertFalse(args.module_cat_prompt)
            self.assertTrue(args.loss_sdr)
            self.assertEqual(args.cat_prompt_source_path, prompt_path)
            self.assertEqual(args.reliability_prompt_path, prompt_path)
            self.assertNotEqual(args.name_path, prompt_path)

            with open(args.name_path, 'r', encoding='utf-8') as f:
                prediction_prompts = [line.strip() for line in f if line.strip()]
            self.assertEqual(prediction_prompts, ['building', 'road'])
        finally:
            if os.path.exists(prompt_path):
                os.remove(prompt_path)
            if 'args' in locals() and os.path.exists(args.name_path):
                os.remove(args.name_path)

    def test_rsap_still_requires_cat_prompt_prediction(self):
        prompt_path = self._make_prompt_file()
        try:
            argv = [
                'prog',
                '/tmp/data',
                '--name_path', prompt_path,
                '--no_module_cat_prompt',
                '--no_module_visual_guidance',
                '--module_rsap_v1',
                '--text_adjust', 'True',
            ]
            with mock.patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(ValueError, 'requires Cat-Prompt'):
                    parse_args()
        finally:
            if os.path.exists(prompt_path):
                os.remove(prompt_path)


if __name__ == '__main__':
    unittest.main()
