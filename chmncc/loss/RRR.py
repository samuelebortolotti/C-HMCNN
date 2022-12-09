import torch
import torch.nn as nn
import torch.nn.functional as F
from chmncc.explanations import compute_integrated_gradient


class RRRLoss(nn.Module):
    """
    Right for the Right Reason loss (RRR) as proposed by Ross et. al (2017) with minor changes.
    See https://arxiv.org/abs/1703.03717.
    The RRR loss calculates the Input Gradients as prediction explanation and compares it
    with the (ground-truth) user explanation.
    Taken from https://github.com/ml-research/A-Typology-to-Explore-and-Guide-Explanatory-Interactive-Machine-Learning/blob/main/xil_methods/xil_loss.py
    DOI: https://arxiv.org/abs/2203.03668
    """

    def __init__(
        self,
        net: nn.Module,
        regularizer_rate=100,
        base_criterion=F.cross_entropy,
        weight=None,
        rr_clipping=None,
    ):
        """
        Args:
            net [nn.Module]: trained neural network
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer l
            weight: if specified then weight right reason loss by classes. Tensor
                with shape (c,) c=classes. WARNING !! Currently only working for
                the special case that whole X in fwd has the same class (as is the
                case in isic 2019).
            rr_clipping: clip the RR loss to a maximum per batch.
        """
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.net = net
        self.weight = weight
        self.rr_clipping = rr_clipping

    def forward(self, X, y, expl, logits):
        """
        Returns (loss, right_answer_loss, right_reason_loss)
        Args:
            X: inputs.
            y: ground-truth labels.
            expl: explanation annotations masks (ones penalize regions) [is basically the annotation matrix]
            logits: model output logits.
        """
        # the normal training loss
        right_answer_loss = self.base_criterion(logits, y)

        # get gradients w.r.t. to the input
        log_prob_ys = logits
        log_prob_ys.retain_grad()

        # integrated gradients
        gradXes = None

        self.net.eval()
        # sum each axes contribution
        gradXes = torch.autograd.grad(
            log_prob_ys,
            X,
            torch.ones_like(log_prob_ys),
            create_graph=True,
            allow_unused=True,
        )[0]
        self.net.train()
        gradXes = torch.sum(gradXes, dim=1)
        print(expl.shape, gradXes.shape)
        # when the feature is 0 -> relevant, since if it is 1 we are adding a penality
        A_gradX = torch.mul(expl, gradXes) ** 2

        right_reason_loss = torch.sum(A_gradX)
        right_reason_loss *= self.regularizer_rate

        if self.weight is not None:
            right_reason_loss *= self.weight[y[0]]

        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = (
                    right_reason_loss - right_reason_loss + self.rr_clipping
                )

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss


class IGRRRLoss(RRRLoss):
    """
    Right for the Right Reason loss (RRR) as proposed by Ross et. al (2017) with minor changes.
    See https://arxiv.org/abs/1703.03717.
    The RRR loss calculates the Input Gradients as prediction explanation and compares it
    with the (ground-truth) user explanation.
    Taken from https://github.com/ml-research/A-Typology-to-Explore-and-Guide-Explanatory-Interactive-Machine-Learning/blob/main/xil_methods/xil_loss.py
    DOI: https://arxiv.org/abs/2203.03668

    This one uses integrated gradients as "right reason" part of the loss
    """

    def __init__(
        self,
        net: nn.Module,
        regularizer_rate=100,
        base_criterion=F.cross_entropy,
        weight=None,
        rr_clipping=None,
    ):
        """
        Args:
            net: trained neural network
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer l
            weight: if specified then weight right reason loss by classes. Tensor
                with shape (c,) c=classes. WARNING !! Currently only working for
                the special case that whole X in fwd has the same class (as is the
                case in isic 2019).
            rr_clipping: clip the RR loss to a maximum per batch.
        """
        super().__init__(net, regularizer_rate, base_criterion, weight, rr_clipping)

    def forward(self, X, y, expl, logits):
        """
        This one uses the integrated_gradients
        Returns (loss, right_answer_loss, right_reason_loss)
        Args:
            X: inputs.
            y: ground-truth labels.
            expl: explanation annotations masks (ones penalize regions) [is basically the annotation matrix]
            logits: model output logits.
        """
        # the normal training loss
        right_answer_loss = self.base_criterion(logits, y)

        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()

        # integrated gradients
        gradXes = None

        self.net.eval()
        for i in range(X.shape[0]):
            x_sample = torch.unsqueeze(X[i], 0)
            if gradXes is None:
                gradXes = torch.unsqueeze(
                    compute_integrated_gradient(
                        x_sample, torch.zeros_like(x_sample), self.net
                    ),
                    0,
                )
            else:
                gradXes = torch.cat(
                    (
                        gradXes,
                        torch.unsqueeze(
                            compute_integrated_gradient(
                                x_sample, torch.zeros_like(x_sample), self.net
                            ),
                            0,
                        ),
                    ),
                    0,
                )
        self.net.train()
        # sum each axes contribution
        gradXes = torch.sum(gradXes, dim=1)

        # if expl.shape [n,1,h,w] and gradXes.shape [n,3,h,w] then torch broadcasting
        # is used implicitly
        # when the feature is 0 -> relevant, since if it is 1 we are adding a penality
        A_gradX = torch.mul(expl, gradXes) ** 2

        right_reason_loss = torch.sum(A_gradX)
        right_reason_loss *= self.regularizer_rate

        if self.weight is not None:
            right_reason_loss *= self.weight[y[0]]

        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = (
                    right_reason_loss - right_reason_loss + self.rr_clipping
                )

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss
