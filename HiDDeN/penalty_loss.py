import torch
import torch.nn as nn
import pyiqa

class PenaltyLoss(nn.Module):
    """
    Penalty loss class to enforce training scheme based on sequential penalty methods. 
    The idea is to gradually force deep learning models to produce feasible outputs by imposing increasing 
    penalties on unfeasible solutions. To do it, the penalty_coefficient is increased by increase_factor 
    during training, with a custom defined frequency. 

    Attributes:
        loss_function (function): Loss function used to train the base model.
        penalty_function (function): Penalty function to measure model constraint violation. It must be 
                                        implemented with torch functions to allow for backpropagation.
        penalty_coefficient (float): Initial value of penalty coefficient.
        penalty_increase_factor (float): Multiplicative factor to increase penalty coefficient.
        _min_value (float): Minimum value of penalty_coefficient.
        _max_value (float): Maximum value of penalty_coefficient.
    """

    def __init__(self, constraint_function, penalty_coefficient=0.1, penalty_increase_factor=1.05):
        super(PenaltyLoss, self).__init__()
        self.constraint_function = constraint_function
        self.penalty_coefficient = penalty_coefficient
        self.penalty_increase_factor = penalty_increase_factor
        self._min_value = penalty_coefficient
        self._max_value = 5e2

    def forward(self, predictions, targets):
        if self.penalty_coefficient > 0:
            avg_violation, avg_psnr = self.constraint_function(predictions, targets)
            return self.penalty_coefficient * avg_violation, avg_psnr
        else:
            return torch.tensor(0), torch.tensor(0)

    def increase_penalty(self):
        if self.penalty_coefficient * self.penalty_increase_factor < self._max_value:
            self.penalty_coefficient *= self.penalty_increase_factor
        else:
            self.penalty_coefficient = self._max_value

    def reset_penalty(self):
        self.penalty_coefficient = self._min_value

    def is_max(self):
        return self.penalty_coefficient == self._max_value 
    
class ConstrainedPSNR(nn.Module):
    """
    Computes PSNR on images and measures the average violation of specified threshold.

    Attributes:
        threshold (float): maximum value tolerated for constraint compliance.
        square (bool): if constraint evaluation is linear or squared.
        psnr_function (function): pyiqa implementation of PSNR suitable for backpropagation.
    """
    def __init__(self, threshold, square: bool = False) -> None:
        super(ConstrainedPSNR, self).__init__() 
        self.threshold = threshold
        self.square = square
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.psnr_function = pyiqa.create_metric('psnr', as_loss=True, device=device, loss_reduction='none')
    
    def forward(self, predictions, targets):
        """
        Measure average constraint violation for batch of images. This makes it possible to verify if every single image
        satisfies the constraint.

        Args:
            predictions (tensor): tensor of watermarked images, of size (batch_size, channels, height, width)
            targets (tensor): tensor of original images, of size (batch_size, channels, height, width)
        """
        batch_psnr = self.psnr_function(predictions, targets)
        avg_violation = torch.mean(torch.nn.functional.relu(self.threshold - batch_psnr))
        avg_psnr = torch.mean(batch_psnr).detach()
        if self.square:
            return torch.pow(avg_violation, 2), avg_psnr
        else:
            return avg_violation, avg_psnr
        
