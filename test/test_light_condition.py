from utils.testNetwork_demo_512 import test_network_demo_512
from utils.testNetwork_demo_1024 import test_network_demo_1024
import unittest

class MyTestCase(unittest.TestCase):
    def test_network_512_demo(self):
        try:
            test_network_demo_512(model_folder="trained_model/",
                                  lightFolder="data/example_light_1/",
                                  saveFolder="result3",
                                  img="data/obama.jpg")
        except Exception as error:
            self.fail(f"test_network_demo_512 raised an exception: {error}")

    def test_network_1024_demo(self):
        try:
            test_network_demo_1024(model_folder="trained_model/",
                                  lightFolder="data/example_light",
                                  saveFolder="result3",
                                  img="data/obama.jpg")
        except Exception as error:
            self.fail(f"test_network_demo_1024 raised an exception: {error}")
if __name__ == '__main__':
    unittest.main()
