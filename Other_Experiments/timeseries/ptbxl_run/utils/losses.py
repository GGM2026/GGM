
"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(
            divide_no_nan(
                t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)
            )
            * mask
        )


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(
        self,
        insample: t.Tensor,
        freq: int,
        forecast: t.Tensor,
        target: t.Tensor,
        mask: t.Tensor,
    ) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


def id_contrastive_loss(z1, z2, id):
    id = id.cpu().detach().numpy()
    str_pid = [str(i) for i in id]
    str_pid = np.array(str_pid, dtype=object)
    pid1, pid2 = np.meshgrid(str_pid, str_pid)
    pid_matrix = pid1 + "-" + pid2
    pids_of_interest = np.unique(
        str_pid + "-" + str_pid
    )
    bool_matrix_of_interest = np.zeros((len(str_pid), len(str_pid)))
    for pid in pids_of_interest:
        bool_matrix_of_interest += pid_matrix == pid
    rows1, cols1 = np.where(
        np.triu(bool_matrix_of_interest, 1)
    )
    rows2, cols2 = np.where(
        np.tril(bool_matrix_of_interest, -1)
    )

    B, H = z1.size(0), z1.size(1)
    loss = 0
    z1 = t.nn.functional.normalize(z1, dim=1)
    z2 = t.nn.functional.normalize(z2, dim=1)
    view1_array = z1
    view2_array = z2
    norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
    norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
    sim_matrix = t.mm(view1_array, view2_array.transpose(0, 1))
    norm_matrix = t.mm(norm1_vector.transpose(0, 1), norm2_vector)
    temperature = 0.1
    argument = sim_matrix / (norm_matrix * temperature)
    sim_matrix_exp = t.exp(argument)


    triu_sum = t.sum(sim_matrix_exp, 1)
    tril_sum = t.sum(sim_matrix_exp, 0)


    loss_terms = 0

    if len(rows1) > 0:
        triu_elements = sim_matrix_exp[
            rows1, cols1
        ]
        loss_triu = -t.mean(t.log(triu_elements / triu_sum[rows1]))
        loss += loss_triu
        loss_terms += 1

    if len(rows2) > 0:
        tril_elements = sim_matrix_exp[
            rows2, cols2
        ]
        loss_tril = -t.mean(t.log(tril_elements / tril_sum[cols2]))
        loss += loss_tril
        loss_terms += 1

    if loss_terms == 0:
        return 0
    else:
        loss = loss / loss_terms
        return loss
