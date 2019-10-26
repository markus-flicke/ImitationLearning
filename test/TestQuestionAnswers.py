import unittest
import imitations
import os
import pandas as pd
from network import ClassificationNetwork
import numpy as np
DATA_PATH = r"..\dat\teacher"


class TestQuestionAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        implicit test of 1a): load_imitations
        :return:
        """
        cls.observations, cls.actions = imitations.load_imitations(DATA_PATH)

    def test_actions(self):
        """
        Verify data integrity of actions
        * Are all actions mapped to exactly one class?
        :return:
        """
        self.assertTrue(all(np.unique(self.actions) == np.unique(ClassificationNetwork.classes)))

    def test_actions_to_classes(self):
        """
        assignment 1c)
        :return:
        """
        model = ClassificationNetwork()
        df = pd.DataFrame(model.actions_to_classes(self.actions))
        df = df.applymap(int)
        number_of_observation_files = len(os.listdir(DATA_PATH))//2
        number_of_classes_classified = df.sum().sum()

        self.assertEqual(number_of_observation_files, number_of_classes_classified)

    def test_scores_to_action(self):
        """
        assignment 1c)
        :return:
        """
        model = ClassificationNetwork()
        scores = model.actions_to_classes(self.actions)
        action = model.scores_to_action(scores)
        self.assertTrue(action in model.classes)
