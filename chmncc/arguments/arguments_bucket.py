"""Module which contains the Argument Bucket class.
The idea is that given a sample, it is possible to populate all the possible explainations or arguments"""
import torch.nn as nn
import torch
from typing import Dict, List, Tuple
from chmncc.explanations import output_gradients
from chmncc.utils import get_constr_indexes, force_prediction_from_batch


class ArgumentBucket:
    """ArgumentBucket class"""

    """current sample"""
    sample: torch.Tensor
    """label list [from integer label to string label]"""
    label_list: List[str]
    """given the class index, it returns the gradient in a torch tensor fashion and the associated score"""
    input_gradient_dict: Dict[int, Tuple[torch.Tensor, float]]
    """label gradient: (class wrt to, parent class): raw gradient and associated score"""
    label_gradient: Dict[Tuple[int, int], Tuple[torch.Tensor, float]]

    def __init__(
        self,
        sample: torch.Tensor,
        net: nn.Module,
        label_list: List[str],
        to_eval: torch.Tensor,
        R: torch.Tensor,
        guess_list: List[int],
        groundtruth_parent: int,
        groundtruth_children: int,
        norm_exponent: int = 2,
        prediction_treshold: float = 0.5,
        force_prediction: bool = False,
        use_softmax: bool = False,
        multiply_by_probability_for_label_gradient: bool = False,
        cincer_approach: bool = False,
    ) -> None:
        """Initialization method
        Args:
            sample: [torch.Tensor]: dataset used
            net [nn.Module]: neural network
            label_list [List[str]]: label list
            to_eval [torch.Tensor]: what to evaluate
            R [torch.Tensor]: adjacency matrix
            guess_list [List[int]]: what the neural network has actually guessed out of sample
            groundtruth_parent [int]: groundtruth parent label
            groundtruth_children [int]: groundtruth children label
            norm_exponent [int] = 2: norm exponent
        """
        self.groundtruth_parent = groundtruth_parent
        self.groundtruth_children = groundtruth_children
        self.sample = sample
        self.sample.requires_grad_(True)
        self.guess_list = guess_list
        self.label_list = label_list
        self.input_gradient_dict = {}
        self.label_gradient = {}
        # compute input gradients
        self._compute_input_gradients(net, to_eval, norm_exponent)
        # compute class gradients
        self._compute_class_gradients(
            net,
            to_eval,
            R,
            norm_exponent,
            multiply_by_probability_for_label_gradient,
            cincer_approach,
        )
        # compute predictions
        prediction = net(self.sample.float())[:, to_eval]
        if force_prediction:
            self.prediction = force_prediction_from_batch(
                prediction.cpu().data, prediction_treshold, use_softmax
            )
        else:
            self.prediction = prediction.cpu().data > prediction_treshold

        self.prediction = self.prediction.squeeze()
        #  print("Prediction", self.prediction)

    def _compute_input_gradients(
        self, net: nn.Module, to_eval: torch.Tensor, norm_exponent: int
    ) -> None:
        """Compute input gradients with respect to each class
        Args:
            net [nn.Module]: neural network
            to_eval [torch.Tensor]: what to evaluate
            norm_exponent [int] = 2: norm exponent for computing the score

        Formula:
            d/dx P_theta(yi|x) foreach i in {1...#classes}
        """
        # get the logits
        logits = net(self.sample.float())[:, to_eval]
        for i in range(logits.shape[1]):
            # input graadients for the ith logit
            grad = output_gradients(self.sample, logits[:, i])
            # add it to the input gradient dictionary
            self.input_gradient_dict[i] = (
                grad,
                torch.linalg.norm(grad.flatten(), dim=0, ord=2) ** norm_exponent,
            )

    def _compute_class_gradients(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        R: torch.Tensor,
        norm_exponent: int,
        multiply_by_probability_for_label_gradient: bool,
        cincer_approach: bool,
    ) -> None:
        """Compute class gradients
        Args:
            net [nn.Module]: neural network
            to_eval [torch.Tensor]: what to evaluate
            R [torch.Tensor]: adjency matrix
            norm_exponent [int] = 2: norm exponent for computing the score

        Formula:
            d/dyi P_theta(yj|x) = d/dyi max{ P_theta(y_j | x), P_theta (y_j, x) }
                                = d/dyi (z P_theta (y_j | x) + (1 - z) P_theta(y_j | x) ) [z in { 0, 1 }]
                                =~ d/dP_theta(yi | x) (z P_theta (y_j | x) + (1 - z) P_theta(y_j | x) ) [z in { 0, 1 }]
                                = z
            foreach i,j in {1...#classes}
        """
        if cincer_approach:
            print("Uso cincer")
        # out
        out = net(self.sample.float())
        # disable constrained layer
        net.constrained_layer = False
        # get unconstrained_ouput
        unconstrained_ouput = net(self.sample.float())
        # get constrained indx: indexes of MCM hierarchy which have influenced the final prediction
        constrained_idx = get_constr_indexes(unconstrained_ouput, R)[0, to_eval]
        # active constrained layer
        net.constrained_layer = True

        # squeezed output
        sout = out.squeeze()
        # squeezed unconstrained_ouput
        s_unconstrained_ouput = unconstrained_ouput.squeeze()
        # squeezed R
        sR = R.squeeze(0)[1:, 1:]
        for parent in range(sR.shape[0]):
            # NB: the outcome considers also the root node, which we excluded
            pred = constrained_idx[parent].item() - 1
            for child in range(sR.shape[1]):
                if sR[parent][child] != 1:  # not influenced
                    continue
                # same element, trivial answer
                if child == parent:
                    continue
                # the child has influenced the parent prediction
                z = torch.Tensor([0])
                if pred == child:
                    # if the predicted child is the one who has influeced the prediction -> gradient = 1
                    value = 1
                    if multiply_by_probability_for_label_gradient:
                        # multiply by probability for label gradient
                        value = 1 * sout[child].item()
                    elif cincer_approach:
                        # difference of probability
                        value = (
                            s_unconstrained_ouput[parent] - s_unconstrained_ouput[child]
                        )
                    z = torch.Tensor([value])
                # register the gradient
                self.label_gradient[(child, parent)] = (
                    z,
                    torch.linalg.norm(z.flatten(), dim=0, ord=2) ** norm_exponent,
                )

    def __repr__(self):
        """Repr"""
        print("Label gradient", self.label_gradient)

    def __str__(self):
        """Str"""
        return "{}".format(self.label_gradient)

    def get_gradients_list_and_names(self) -> Tuple[List[float], List[str]]:
        """Method which returns the list of gradients score and the list of associated gradients' names
        Returns:
            Tuple[List[float], List[str]]: gradients' scores and their names
        """
        titles: List[str] = list()
        gradient_list: List[float] = list()
        for key, value in self.input_gradient_dict.items():
            titles.append(self.label_list[key])
            gradient_list.append(value[1].item())
        for key, value in self.label_gradient.items():
            (parent, child) = key
            titles.append(self.label_list[parent] + "-" + self.label_list[child])
            gradient_list.append(value[1].item())
        return (gradient_list, titles)

    def get_gradents_list_separated(self) -> Tuple[List[float], List[float]]:
        """Method which returns the list of input gradients and label gradients in this exact order
        Returns:
            Tuple[List[float], List[float]]: list of input gradients and label gradients in this exact order
        """
        input_gradient_list = [float(el[1]) for el in self.input_gradient_dict.values()]
        label_gradient_list = [float(el[1]) for el in self.label_gradient.values()]
        return (input_gradient_list, label_gradient_list)

    def get_gradents_list_separated_by_class(
        self, class_lab: int
    ) -> Tuple[List[float], List[float]]:
        """Method which returns the list of input gradients and label gradients in this exact order according to the
        specified class
        Args:
            class_lab [int]: class label
        Returns:
            Tuple[List[float], List[float]]: list of input gradients and label gradients in this exact order
        """
        input_gradient_list = list()
        for key, el in self.input_gradient_dict.items():
            if key == class_lab:
                input_gradient_list.append(float(el[1]))
        label_gradient_list = list()
        for key, el in self.label_gradient.items():
            if key[0] == class_lab:
                label_gradient_list.append(float(el[1]))
        return (input_gradient_list, label_gradient_list)

    def get_gradients_by_names(self) -> Dict[str, List[float]]:
        """Method which returns the dictionary of gradient names and gradient score associated
        Returns:
            Dict[str, List[float]]: dictionary
        """
        gradient_dict = {}
        for key, el in self.input_gradient_dict.items():
            gradient_dict[self.label_list[key]] = float(el[1])
        for key, el in self.label_gradient.items():
            gradient_dict[
                self.label_list[key[0]] + "-" + self.label_list[key[1]]
            ] = float(el[1])
        return gradient_dict

    def get_maximum_ig_score(self) -> float:
        """Returns the maximum score from the list of integrated gradients
        Returns:
            maximum score value: float
        """
        return max([float(score[1]) for score in self.input_gradient_dict.values()])

    def get_maximum_label_score(self) -> float:
        """Returns the maximum score from the list of label gradients
        Returns:
            maximum score value: float
        """
        return max([float(score[1]) for score in self.label_gradient.values()])

    def get_ig_groundtruth(self) -> Tuple[int, Tuple[torch.Tensor, float]]:
        """Returns the Integrated Gradient of the Groundtruth label
        Returns:
            Tuple[int, Tuple[torch.Tensor, float]]: tuple containing class label, gradient and score
        """
        for (key, value) in self.input_gradient_dict.items():
            sample_tensor = self.sample.detach().clone()
            if self.label_list[key] == self.label_list[self.groundtruth_children]:
                return (key, value)
        return (-1, (torch.tensor(float("nan")), 0.0))

    def get_arguments_lists_separated_by_prediction(
        self, parents: List[str]
    ) -> Tuple[
        Dict[int, Tuple[float, torch.Tensor]],
        Dict[Tuple[int, int], Tuple[float, torch.Tensor]],
    ]:
        ig_dict: Dict[int, Tuple[float, torch.Tensor]] = dict()
        label_dict: Dict[Tuple[int, int], Tuple[float, torch.Tensor]] = dict()
        performed_predictions = (self.prediction == True).nonzero().flatten().tolist()
        #  print("Performed Prediction", performed_predictions)
        for pred_el in performed_predictions:
            for key, el in self.input_gradient_dict.items():
                if key == pred_el:
                    ig_dict[key] = float(el[1]), el[0]
            for key, el in self.label_gradient.items():
                if key[0] == pred_el:
                    label_dict[key] = float(el[1]), el[0]
        return (ig_dict, label_dict)
