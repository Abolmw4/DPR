import os

from utils.testNetwork_demo_512 import test_network_demo_512
from utils.testNetwork_demo_1024 import test_network_demo_1024
import random
import unittest



class MyTestCase(unittest.TestCase):
    OUTPUT_PATH = "/home/abolfazl/Documents/DPR/results"
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

    def test_create_different_lighting_condition(self, dir_name: None):
        global __OUTPUT_PATH
        config_file = "/home/abolfazl/Documents/DPR/data/example_light"
        for file in os.listdir(config_file):
            with open(os.path.join(config_file, file), mode='r') as txtFile:
                normal_str = list(map(lambda inpt: format(float(inpt), '.20f') , txtFile.readlines()))
                # ✅ تغییر عدد اول
                if len(normal_str) > 0:
                    normal_str[0] = str(random.uniform(-1, 1))

                # ✅ تغییر از عدد چهارم به بعد (index 3 به بعد)
                for i in range(4, len(normal_str)):
                    normal_str[i] = str(random.uniform(-10, 10))

                if not os.path.exists(os.path.join(MyTestCase.OUTPUT_PATH, dir_name, "example_light")):
                    os.makedirs(os.path.join(MyTestCase.OUTPUT_PATH, dir_name, "example_light"))
                with open(os.path.join(MyTestCase.OUTPUT_PATH, dir_name, "example_light", file), mode='w') as outFile:
                    outFile.write("\n".join(normal_str))

    def test_apply_different_configs_on_imag(self, how_many_test: int = 10):
        i = 1
        while i < how_many_test + 1:
            self.test_create_different_lighting_condition(dir_name=f"test{i}")
            try:
                test_network_demo_512(model_folder="trained_model/",
                                      lightFolder=os.path.join(MyTestCase.OUTPUT_PATH, f"test{i}", "example_light"),
                                      saveFolder=os.path.join(MyTestCase.OUTPUT_PATH, f"test{i}"),
                                      img="data/obama.jpg")
                i += 1
            except Exception as error:
                self.fail(f"test_network_demo_512 raised an exception: {error}")


if __name__ == '__main__':
    unittest.main()
