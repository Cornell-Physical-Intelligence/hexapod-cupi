"""The discriminator reads its input width from the shared AMP feature contract."""
import unittest

from locomotion import amp_discriminator
from locomotion.amp import feature_contract


class ContractWidthTests(unittest.TestCase):
    def test_input_width_comes_from_the_amp_feature_contract(self):
        self.assertEqual(amp_discriminator.AMP_WIDTH, feature_contract()['width'])
        self.assertEqual(amp_discriminator.AMP_WIDTH, 61)
        discriminator = amp_discriminator.Discriminator([0.]*122, [1.]*122, hidden=())
        self.assertEqual(discriminator.net[0].in_features, 2*feature_contract()['width'])


if __name__ == '__main__':
    unittest.main()
