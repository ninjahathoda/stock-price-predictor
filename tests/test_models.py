import unittest
from models.numerical.lstm_model import create_lstm_model

class TestLSTMModel(unittest.TestCase):
    def test_model_creation(self):
        model = create_lstm_model((30, 1))
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
