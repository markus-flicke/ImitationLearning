import torch
import numpy as np


class ClassificationNetwork(torch.nn.Module):
    classes = [0] * 4
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')

        self.features_2d = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, stride=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 94x94
            torch.nn.Conv2d(2, 4, 3, stride=2),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 46x46
            torch.nn.Conv2d(4, 8, 3, stride=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(negative_slope=0.2),  # 22x22
        ).to(gpu)

        self.features_1d = torch.nn.Sequential(
            torch.nn.Linear(3344, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(negative_slope=0.2)
        ).to(gpu)

        self.scores = torch.nn.Sequential(
            torch.nn.Linear(263, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(16, 4),
            torch.nn.Sigmoid()
        ).to(gpu)

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation. We chose grayscale, because there is no additional information in the colors.
        We should achieve faster convergence because of this choice.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        # Answer to Question 1:
        # batch_size = observation.shape[0]
        # observation = observation[:, :, :, 0] * 0.2989 + observation[:, :, :, 1] * 0.5870 + observation[:, :, :, 2] * 0.1140
        # obs = observation.reshape(batch_size, 1, 96, 96)
        # features_2d = self.features_2d(obs).reshape(batch_size, -1)
        # return self.scores(features_2d)


        batch_size = observation.shape[0] # 64
        # extract sensor values
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, batch_size)

        # deleting the grass
        # gpu = torch.device('cuda')
        # observation = torch.where(observation == 204, torch.tensor(0.).to(gpu), observation)
        # observation = torch.where(observation == 229, torch.tensor(0.).to(gpu), observation)

        # conversion to gray scale
        observation = observation[:, :, :, 0] * 0.2989 + observation[:, :, :, 1] * 0.5870 + observation[:, :, :, 2] * 0.1140
        # crop and reshape observations to 84 x 96
        obs = observation[:, :84, :].reshape(batch_size, 1, 84, 96)

        # get features
        features_2d = self.features_2d(obs).reshape(batch_size, -1)
        features_1d = self.features_1d(features_2d)
        fused_features = torch.cat((
            speed,  # batch size x 1
            abs_sensors,  # batch size x 4
            steering,  # batch size x 1
            gyroscope,  # batch size x 1
            features_1d), 1)  # batch size x 16
        return self.scores(fused_features)

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
        classes = []
        for action in actions:
            c = [0] * 4
            if action[0] < 0:
                c[0] = 1
            if action[0] > 0:
                c[1] = 1
            if action[1] > 0:
                c[2] = 1
            if action[2] > 0:
                c[3] = 1
            classes.append(c)
        return torch.tensor(classes)

    def scores_to_action(self, c):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        scores = [torch.Tensor([1,0,0,0])]
        return          (float, float, float)  // This is the action resulting from the first score -> Therefore, why do we feed a list of scores?
        """
        c=c[0]
        steer = 0
        gas = 0
        brake = 0
        if c[0] == 1:
            steer = -1
        if c[1] == 1:
            steer = 1
        if c[0] == c[1] == 1:
            steer = 0
        if c[2] == 1:
            gas = 0.5
        if c[3] == 1:
            brake = 0.8
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
