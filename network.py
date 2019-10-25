import torch
import numpy as np


class ClassificationNetwork(torch.nn.Module):
    classes = np.array([[0, 0, 0],
                        [-1, 0, 0],
                        [1, 0, 0],
                        [0, 0.5, 0],
                        [0, 0, 0.8],
                        [-1, 0.5, 0],
                        [1, 0.5, 0],
                        [-1, 0, 0.8],
                        [1, 0, 0.8]])

    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        pass

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        # actions = [[-1,0.5,0.8],[0,0,0],...]
        # res = []
        # for action in actions:
        #     l = []
        #     for this_class in torch.Tensor(self.classes):
        #         is_in_class = int(torch.prod(torch.Tensor(action) == this_class))
        #         l.append(is_in_class)
        #     res.append(torch.Tensor(l))
        # return res
        return [torch.Tensor([int(torch.prod(torch.Tensor(action) == this_class )) for this_class in torch.Tensor(self.classes)]) for action in actions]


    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        scores = [torch.Tensor([0.2])]  + [torch.Tensor([0])]* 6+[torch.Tensor([1])]+[torch.Tensor([0])]
        return          (float, float, float)
        """
        _, class_number = torch.max(torch.Tensor(scores), dim=0)
        steer, gas, brake = self.classes[class_number]
        return steer, gas, brake


    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
