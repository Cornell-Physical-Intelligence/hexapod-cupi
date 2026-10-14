"""Keep checkpoint provenance intact across a certified package migration."""
from copy import deepcopy
import unittest

from locomotion.runs import compat


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_translate_only_known_class_fields_without_mutation(self):
        original = {'class_name': 'locomotion.rate_schedule:CappedRatePPO',
                    'label': 'locomotion.rate_schedule:CappedRatePPO',
                    'nested': [{'class_name': 'locomotion.action_distribution:BoundedMeanGaussian'}]}
        snapshot = deepcopy(original)
        translated = compat.configuration(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(translated['class_name'], 'locomotion.learning.rate_schedule:CappedRatePPO')
        self.assertEqual(translated['label'], original['label'])
        self.assertEqual(translated['nested'][0]['class_name'],
                         'locomotion.learning.action_distribution:BoundedMeanGaussian')

    def test_translation_keeps_configuration_changes_visible(self):
        old = {'class_name': 'locomotion.rate_schedule:CappedRatePPO', 'gamma': .99}
        candidate = {'class_name': 'locomotion.learning.rate_schedule:CappedRatePPO', 'gamma': .98}
        self.assertNotEqual(compat.configuration(old), candidate)

    def test_unknown_class_name_stays_unknown(self):
        self.assertEqual(compat.configuration({'class_name': 'unreviewed:PPO'}),
                         {'class_name': 'unreviewed:PPO'})


if __name__ == '__main__':
    unittest.main()
