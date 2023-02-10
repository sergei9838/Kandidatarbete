import torch

class pytorch_measure:
    def __init__(self,locations,weights):
        """
        Group 1: 
        Group 2: 
        """
        self.locations = torch.nn.parameter.Parameter(locations)#Input must be tensors
        self.weights = torch.nn.parameter.Parameter(weights)

    def total_mass(self):
        """
        Responsibility: Johan
        """
        return sum(self.weights).item()

    def total_variation(self):
        """
        Responsibility: Samuel
        """
        return sum(abs(self.weights)).item()

    def support(self):
        """
        Responsibility: Johan
        """
        pass

    def positive_part(self):
        """
        Responsibility: Samuel
        """
        return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item() > 0]), \
            torch.tensor([weight.item() for weight in self.weights if weight.item() > 0])

    def negative_part(self):
        """
        Responsibility:
        """
        pass

    def put_mass(self) -> bool:
        """
        
        :param:
        :returns: True if all mass could be placed, False otherwise
        Responsibility:
        """
        pass

    def take_mass(self) -> bool:
        """
        Responsibility:
        :param:
        :returns: True if all mass could be taken, False otherwise
        """
        pass

    def sample(self):
        """
        Responsibility:
        """
        pass

    def step(self):
        """
        Responsibility: Hampus
        Takes one optimization step with algoritm
        """
        pass

