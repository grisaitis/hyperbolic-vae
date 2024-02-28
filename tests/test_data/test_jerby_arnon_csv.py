from hyperbolic_vae.data.jerby_arnon_csv import JerbyArnonCSVDataset
import unittest
from torch.utils.data import DataLoader


class TestJerbyArnonCSVDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = JerbyArnonCSVDataset()

    def test_len(self):
        self.assertEqual(len(self.dataset), 7872)

    def test_getitem(self):
        data_point, label = self.dataset[0]
        self.assertEqual(data_point, self.data[0][0])
        self.assertEqual(label, self.data[0][1])

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        for i, (data_point, label) in enumerate(dataloader):
            self.assertEqual(data_point, self.data[i][0])
            self.assertEqual(label, self.data[i][1])


if __name__ == '__main__':
    unittest.main()
