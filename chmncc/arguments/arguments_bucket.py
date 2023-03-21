from torch.functional import _unique_consecutive_impl
import torch.nn as nn
import torch
from typing import Dict, List, Tuple
from chmncc.explanations import output_gradients
from chmncc.utils import get_constr_indexes


class ArgumentBucket:
    sample: torch.Tensor
    label_list: List[str]
    input_gradient_dict: Dict[int, Tuple[torch.Tensor, float]]
    label_gradient: Dict[Tuple[int, int], Tuple[torch.Tensor, float]]

    def __init__(
        self,
        sample: torch.Tensor,
        net: nn.Module,
        label_list: List[str],
        to_eval: torch.Tensor,
        R: torch.Tensor,
    ):
        self.sample = sample
        self.sample.requires_grad_(True)
        self.label_list = label_list
        self.input_gradient_dict = {}
        self.label_gradient = {}
        self._compute_input_gradients(net, to_eval)
        self._compute_class_gradients(net, to_eval, R)

    def _compute_input_gradients(self, net: nn.Module, to_eval: torch.Tensor):
        logits = net(self.sample.float())[:, to_eval]
        for i in range(logits.shape[1]):
            grad = output_gradients(self.sample, logits[:, i])
            self.input_gradient_dict[i] = (
                grad,
                torch.linalg.norm(grad.flatten(), dim=0, ord=2) ** 2,
            )
            #  import matplotlib.pyplot as plt
            #
            #  plt.title(
            #      "Correct: Number {} - {} - {}".format(
            #          grad.sum(),
            #          self.label_list[i],
            #          torch.linalg.norm(grad.flatten(), dim=0, ord=2) ** 2,
            #      )
            #  )
            #  plt.imshow(grad.permute(1, 2, 0), cmap="gray")
            #  plt.show()
            #  plt.close()

    def _compute_class_gradients(
        self, net: nn.Module, to_eval: torch.Tensor, R: torch.Tensor
    ):
        # disable constrained layer
        net.constrained_layer = False
        unconstrained_ouput = net(self.sample.float())
        constrained_idx = get_constr_indexes(unconstrained_ouput, R)[0, to_eval]
        print("Const idx", constrained_idx)
        # active constrained layer
        net.constrained_layer = True

        # squeezed R
        sR = R.squeeze(0)[1:, 1:]
        print(self.label_list)
        for parent in range(sR.shape[0]):
            # the outcome considers also the root node, which we excluded
            pred = constrained_idx[parent].item() - 1
            print("For parent", parent, "the const idx", pred)
            for child in range(sR.shape[1]):
                if sR[parent][child] != 1:
                    continue
                print(self.label_list[child], "is subclass of", self.label_list[parent])
                print("Parent prediction is influenced by", self.label_list[pred])
                z = torch.Tensor([0])
                if pred == child:
                    print("It is the child who influence the parent")
                    z = torch.Tensor([1])
                else:
                    print("Parent stands by its own")
                self.label_gradient[(child, parent)] = (
                    z,
                    torch.linalg.norm(z.flatten(), dim=0, ord=2) ** 2,
                )

    def __repr__(self):
        print("Label gradient", self.label_gradient)
        #  print("Input gradient", self.input_gradient_dict)

    def __str__(self):
        return "{}".format(self.label_gradient)
