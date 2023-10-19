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

        self.ig_entropy = self.get_montecarlo_dropout_entropy_for_ig(
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

        # activate dropout during evaluation
        activate_dropout(net)
        # entropies positive
        entropies = []
        entropies_positive = []
        entropies_negative = []
        counter_positive = []
        counter_negative = []

        def sum_normalization(vector):
            vector = torch.abs(vector)
            sum_of_elements = torch.sum(vector)
            normalized_vector = vector / sum_of_elements
            return normalized_vector

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
            pred_idx = torch.argmax(mc_outputs[:, 4:]) + 4
            #  new_mc_out = sum_normalization(mc_outputs.squeeze(0))
            new_mc_out = mc_outputs.squeeze(0)
            entropies.append(new_mc_out)

            # include positive and negative
            if len(entropies_positive) == 0:
                entropies_positive = torch.zeros_like(new_mc_out, dtype=torch.float64)
                entropies_negative = torch.zeros_like(new_mc_out, dtype=torch.float64)
                counter_positive = [0 for _ in range(new_mc_out.shape[0])]
                counter_negative = [0 for _ in range(new_mc_out.shape[0])]

            # add things
            for i in range(len(counter_negative)):
                counter_negative[i] += 1
                entropies_negative[i] += new_mc_out[i]

            counter_positive[pred_idx] += 1
            counter_negative[pred_idx] -= 1
            entropies_positive[pred_idx] += new_mc_out[pred_idx]
            entropies_negative[pred_idx] -= new_mc_out[pred_idx]

        # Compute the probability given by mc_dropout
        for i in range(len(counter_negative)):
            if counter_positive[i] != 0:
                entropies_positive[i] /= counter_positive[i]
            if counter_negative != 0:
                entropies_negative[i] /= counter_negative[i]

        # Calculate the entropy per node (class) based on the Monte Carlo samples (original approach)
        entropy_per_node = [torch.tensor(0) for _ in range(len(counter_negative))]
        for i in range(len(counter_negative)):
            if counter_positive[i] > 0 and counter_negative[i] > 0:
                entropy_per_node[i] = -(
                    entropies_positive[i] * torch.log2(entropies_positive[i])
                    + entropies_negative[i] * torch.log2(entropies_negative[i])
                )
            elif counter_positive[i] > 0:
                entropy_per_node[i] = -(
                    entropies_positive[i] * torch.log2(entropies_positive[i])
                )
            else:
                #  print(entropies_negative[i])
                entropy_per_node[i] = -(
                    entropies_negative[i] * torch.log2(entropies_negative[i])
                )
        entropy_per_node = torch.tensor(entropy_per_node, dtype=torch.float64)

        # entropies mean
        return entropy_per_node, entropy_per_node[predicted_child_idx]

    def get_montecarlo_dropout_entropy_for_ig_gaussian(
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
                torch.log(mc_outputs),
                self.sample,
                grad_outputs=torch.ones_like(mc_outputs),
                create_graph=True,
                retain_graph=True,
            )[0]

            magnitude_per_pixels = ig.squeeze(0).view(-1) ** 2
            entropies.append(magnitude_per_pixels)

        # entropies
        entropies = torch.stack(entropies)  # Shape: (num_samples, num_classes)

        def entropy_gaussian(mu, sigma):
            epsilon = 1e-8  # A small positive value to avoid zero or negative sigma
            sigma = max(
                sigma.item(), epsilon
            )  # Ensure sigma is not smaller than epsilon
            entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
            return torch.tensor(entropy)

        # IG entropy
        ig_entropy_per_pixel = []

        for col in range(entropies.shape[1]):
            column_vector = entropies[:, col]
            mu = torch.mean(column_vector)
            sigma = torch.std(column_vector)
            entropy = entropy_gaussian(mu, sigma)
            ig_entropy_per_pixel.append(entropy)

        # ig entropy per pixels
        ig_entropy_per_pixel = torch.tensor(ig_entropy_per_pixel, dtype=torch.float64)
        #  print("entropy per pixel", ig_entropy_per_pixel, ig_entropy_per_pixel.shape)

        return ig_entropy_per_pixel

    def get_montecarlo_dropout_entropy_for_ig(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        R: torch.Tensor,
        num_mc_samples: int = 10,
        use_probabilistic_circuits: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
    ):
        net.eval()
        # activate dropout during evaluation
        activate_dropout(net)

        def sum_normalization(vector):
            vector = torch.abs(vector)
            sum_of_elements = torch.sum(vector)
            normalized_vector = vector / sum_of_elements
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

            # normalization
            normalized_ig = sum_normalization(ig.squeeze(0))
            # compute the entropy per image
            entropy_per_mc_run = -torch.sum(
                normalized_ig * torch.log2(normalized_ig + 1e-9), dim=0
            )

            # entropies
            entropies.append(entropy_per_mc_run)

        # stack the entropies
        entropies = torch.stack(entropies)  # Shape: (num_samples, num_classes)
        # ig entropy per image as the mean per entropies
        ig_entropy_per_image = torch.mean(entropies)
        return ig_entropy_per_image

    def get_montecarlo_dropout_entropy_for_ig_2(
        self,
        net: nn.Module,
        to_eval: torch.Tensor,
        R: torch.Tensor,
        num_mc_samples: int = 10,
        use_probabilistic_circuits: bool = False,
        gate: DenseGatingFunction = None,
        cmpe: CircuitMPE = None,
    ):
        net.eval()
        # activate dropout during evaluation
        activate_dropout(net)

        def sum_normalization(vector):
            vector = torch.abs(vector)
            sum_of_elements = torch.sum(vector)
            normalized_vector = vector / sum_of_elements
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

            # normalization
            normalized_ig = sum_normalization(ig.squeeze(0))

            # entropies
            entropies.append(normalized_ig.flatten())

        # Stack
        entropies = torch.stack(entropies)  # Shape: (num_samples, num_classes)

        # Take the mean across the mc dropout iterations
        entropies = torch.mean(entropies, dim=0)

        # Ig entropy per image as as an entropy per pixels throught multiple runs of mc dropout
        ig_entropy_per_image = -torch.sum(
            entropies * torch.log2(entropies + 1e-9), dim=0
        )

        return ig_entropy_per_image

    def get_predicted_label_entropy(self):
        """Return the predicted label entropy"""
        return self.predicted_label_entropy.item()

    def get_ig_entropy(self):
        """Return the predicted ig entropy"""
        return self.ig_entropy.item()

    def get_random_ranking(self):
        """Returns a random number either 0 or 1"""
        import random

        return random.random()
