"""Module which contains the Argument Bucket class.
The idea is that given a sample, it is possible to populate all the possible explainations or arguments"""
import torch.nn as nn
import numpy as np
import torch
from typing import Dict, List, Tuple
from chmncc.explanations import output_gradients
from chmncc.utils import get_constr_indexes, force_prediction_from_batch

from chmncc.probabilistic_circuits.GatingFunction import DenseGatingFunction
from chmncc.probabilistic_circuits.compute_mpe import CircuitMPE

from chmncc.utils import activate_dropout


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
        use_probabilistic_circuits: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
        use_gate_output: bool = False,
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
            prediction_treshold [float] = 0.5: prediction threshold
            force_prediction [bool] = False: whether to force prediction or not
            use_softmax [bool] = False: whether to use softmax or not
            multiply_by_probability_for_label_gradient: bool = False,
            cincer_approach [bool] = False: whether to use cincer approach
            use_probabilistic_circuits [bool] = False: use probabilistic circuits or not
            gate [DenseGatingFunction] = None: gate
            cmpe [CircuitMPE] = None: circuit MPE
            use_gate_output [bool] = False: whether to use the gate output or not
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
        self._compute_input_gradients(
            net,
            to_eval,
            norm_exponent,
            use_probabilistic_circuits,
            use_gate_output,
            gate,
            cmpe,
        )
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
        prediction_not_const = net(self.sample.float())
        prediction = prediction_not_const[:, to_eval]

        if use_probabilistic_circuits:
            # thetas
            thetas = gate(prediction_not_const.float())

            # negative log likelihood and map
            cmpe.set_params(thetas)
            self.prediction = (cmpe.get_mpe_inst(self.sample.shape[0]) > 0).long()
        elif force_prediction:
            self.prediction = force_prediction_from_batch(
                prediction.cpu().data, prediction_treshold, use_softmax
            )
        else:
            self.prediction = prediction.cpu().data > prediction_treshold

        self.prediction = self.prediction.squeeze()

        # dropout thing
        (
            self.label_entropy,
            self.predicted_label_entropy,
        ) = self.get_montecarlo_dropout_entropy_for_labels(net, to_eval, R, label_list)

        #  self.ig_entropy = self.get_montecarlo_dropout_entropy_for_ig(
        #      net,
        #      to_eval,
        #      R,
        #  )

        self.ig_entropy = self.get_joint_entropy_ig_montecarlo_dropout(
            net,
            to_eval,
            R,
        )

    def _compute_input_gradients(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        norm_exponent: int,
        use_probabilistic_circuits: bool = False,
        use_gate_output: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
    ) -> None:
        """Compute input gradients with respect to each class
        Args:
            net [nn.Module]: neural network
            to_eval [torch.Tensor]: what to evaluate
            norm_exponent [int] = 2: norm exponent for computing the score
            use_probabilistic_circuits [bool] = False: whether to use probabilistic circuits or not
            use_gate_output [bool] = False: gate output
            gate [DenseGatingFunction] = None: gate

        Formula:
            d/dx P_theta(yi|x) foreach i in {1...#classes}
        """
        # get the logits
        logits_not_cut = net(self.sample.float())
        logits = logits_not_cut[:, to_eval]

        if use_gate_output:
            logits = gate.get_output(logits_not_cut.float())[:, to_eval]

        if use_probabilistic_circuits:
            thetas = gate(logits_not_cut.float())
            cmpe.set_params(thetas)
            logits = torch.transpose(cmpe.get_marginals_only_positive_part(), 0, 1)[
                :, to_eval
            ]

        for i in range(logits.shape[1]):
            # input gradients for the ith logit
            grad = output_gradients(self.sample, logits[:, i])
            # add it to the input gradient dictionary
            self.input_gradient_dict[i] = (
                grad,
                torch.linalg.norm(grad.flatten(), dim=0, ord=2),  # ** norm_exponent,
            )
            #  if torch.linalg.norm(grad.flatten(), dim=0, ord=2) > 1:
            #      print(torch.linalg.norm(grad.flatten(), dim=0, ord=2))

    def _compute_class_gradients(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        R: torch.Tensor,
        norm_exponent: int,
        multiply_by_probability_for_label_gradient: bool,
        cincer_approach: bool,
        use_probabilistic_circuits: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
    ) -> None:
        """Compute class gradients
        Args:
            net [nn.Module]: neural network
            to_eval [torch.Tensor]: what to evaluate
            R [torch.Tensor]: adjency matrix
            norm_exponent [int] = 2: norm exponent for computing the score
            multiply_by_probability_for_label_gradient [bool]: whether to multiply the outcome label gradient by the probability term
            cincer_approach [bool]: whether to use the cincer approach

        Formula:
            d/dyi P_theta(yj|x) = d/dyi max{ P_theta(y_j | x), P_theta (y_j, x) }
                                = d/dyi (z P_theta (y_j | x) + (1 - z) P_theta(y_j | x) ) [z in { 0, 1 }]
                                =~ d/dP_theta(yi | x) (z P_theta (y_j | x) + (1 - z) P_theta(y_j | x) ) [z in { 0, 1 }]
                                = z
            foreach i,j in {1...#classes}
        """
        # out
        out = net(self.sample.float())
        if use_probabilistic_circuits:
            thetas = gate(out.float())
            cmpe.set_params(thetas)
            out = torch.transpose(cmpe.get_marginals_only_positive_part(), 0, 1)

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
                    torch.linalg.norm(z.flatten(), dim=0, ord=norm_exponent),
                )
                #  print(z, torch.linalg.norm(z.flatten(), dim=0, ord=norm_exponent))

    def marginalize_probability(
        self, prob: torch.Tensor, num_parents: int
    ) -> Dict[int, torch.Tensor]:
        """Method employ to marginalize the probability, still under development

        Args:
            prob [torch.Tensor]: probability
            num_parents [int]: num_parents

        Returns:
            Dict[int, torch.Tensor]: normalized probability
        """
        marginalized_prob = {}
        sum_el = torch.sum(prob)
        for p in range(num_parents):
            marginalized_prob[p] = sum_el - prob[p]
            # TODO compute the gradient
        return marginalized_prob

    def __repr__(self):
        """Repr"""
        print("Label gradient", self.label_gradient)

    def __str__(self) -> str:
        """Str
        Returns:
            str [str]: string representation of the object
        """
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
        """Returns a two dictionaries which contains the arguments wrt the prediction the machine has performed

        Args:
            parents [List[str]]: list of parents

        Returns:
            Dict[int, Tuple[float, torch.Tensor]]: input gradient dictionary wrt prediction
            Dict[Tuple[int, int], Tuple[float, torch.Tensor]]: label dictionary wrt prediction
        """
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

    def find_array_modes(self, arr: np.array):
        from scipy import stats

        mode_result = stats.mode(arr)
        return mode_result.mode.tolist()

    def get_montecarlo_dropout_entropy_for_labels(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        R: torch.Tensor,
        label_list: str,
        num_mc_samples: int = 5,
        use_probabilistic_circuits: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
    ):
        net.eval()

        logits = net(self.sample)
        logits = logits[:, to_eval]

        # TODO make it general
        predicted_child_idx = torch.argmax(logits[:, 4:]) + 4
        predictions = []

        # activate dropout during evaluation
        activate_dropout(net)
        # entropies
        entropies = []
        # number of montecarlo runs
        for _ in range(num_mc_samples):
            logits = net(self.sample)
            mc_outputs = logits[:, to_eval]

            if use_probabilistic_circuits:
                thetas = gate(logits.float())
                cmpe.set_params(thetas)
                mc_outputs = torch.transpose(
                    cmpe.get_marginals_only_positive_part(), 0, 1
                )[:, to_eval]
            entropies.append(mc_outputs.squeeze(0))
            predictions.append(torch.argmax(mc_outputs[:, 4:]) + 4)

        modes = self.find_array_modes(predictions)
        #  if len(modes) != num_mc_samples:
        #      predicted_child_idx = modes[0]

        entropies = torch.stack(entropies)  # Shape: (num_samples, num_classes)

        # Calculate the entropy per node (class) based on the Monte Carlo samples
        entropy_per_node = -torch.sum(entropies * torch.log2(entropies), dim=0)

        return entropy_per_node, entropy_per_node[predicted_child_idx]

    def get_montecarlo_dropout_entropy_for_ig(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        R: torch.Tensor,
        num_mc_samples: int = 5,
        use_probabilistic_circuits: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
    ):
        net.eval()
        # activate dropout during evaluation
        activate_dropout(net)

        def min_max_normalize(vector):
            min_value = torch.min(vector)
            max_value = torch.max(vector)
            normalized_vector = (vector - min_value) / (max_value - min_value)
            return normalized_vector

        # entropies
        entropies = []
        # number of montecarlo runs
        for _ in range(num_mc_samples):
            logits = net(self.sample)
            mc_outputs = logits[:, to_eval]

            if use_probabilistic_circuits:
                thetas = gate(logits.float())
                cmpe.set_params(thetas)
                mc_outputs = torch.transpose(
                    cmpe.get_marginals_only_positive_part(), 0, 1
                )[:, to_eval]

            ig = torch.autograd.grad(
                mc_outputs,
                self.sample,
                grad_outputs=torch.ones_like(mc_outputs),
                create_graph=True,
                retain_graph=True,
            )[0]
            entropies.append(min_max_normalize(ig.squeeze(0)))
        entropies = torch.stack(entropies)  # Shape: (num_samples, num_classes)

        ig_entropy_per_pixel = -torch.sum(
            entropies * torch.log2(entropies + 1e-9), dim=0
        )

        if torch.any(torch.isnan(ig_entropy_per_pixel)):
            print(ig_entropy_per_pixel)
            exit(0)
        return ig_entropy_per_pixel

    def get_joint_entropy_ig_montecarlo_dropout(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        R: torch.Tensor,
        num_mc_samples: int = 5,
        use_probabilistic_circuits: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
    ):
        def compute_joint_entropy(vectors):
            # Take the absolute value of the vectors to ensure non-negative values
            vectors = torch.abs(vectors)

            # Normalize the vectors to make them probability distributions
            vectors_sum = torch.sum(vectors, dim=1, keepdim=True)
            vectors = vectors / vectors_sum

            # Calculate the joint probability distribution
            joint_probs = torch.prod(vectors, dim=0)

            # Add a small epsilon to avoid taking the logarithm of zero
            epsilon = 1e-9
            joint_probs = torch.clamp(joint_probs, min=epsilon)

            # Calculate the joint entropy using the formula
            joint_entropy = -torch.sum(joint_probs * torch.log2(joint_probs))
            return joint_entropy

        net.eval()
        # activate dropout during evaluation
        activate_dropout(net)

        # entropies
        entropies = []
        # number of montecarlo runs
        for _ in range(num_mc_samples):
            logits = net(self.sample)
            mc_outputs = logits[:, to_eval]

            if use_probabilistic_circuits:
                thetas = gate(logits.float())
                cmpe.set_params(thetas)
                mc_outputs = torch.transpose(
                    cmpe.get_marginals_only_positive_part(), 0, 1
                )[:, to_eval]

            ig = torch.autograd.grad(
                mc_outputs,
                self.sample,
                grad_outputs=torch.ones_like(mc_outputs),
                create_graph=True,
                retain_graph=True,
            )[0]
            entropies.append(ig.squeeze(0))

        entropies = torch.stack(entropies)  # Shape: (num_samples, num_classes)

        ig_joint_entropy = compute_joint_entropy(entropies)

        if torch.any(torch.isnan(ig_joint_entropy)):
            print(ig_joint_entropy)
            exit(0)
        #  print(ig_joint_entropy)
        return ig_joint_entropy

    def get_predicted_label_entropy(self):
        return self.predicted_label_entropy.item()

    def get_ig_entropy(self):
        #  return torch.mean(self.ig_entropy).item()
        return self.ig_entropy.item()
