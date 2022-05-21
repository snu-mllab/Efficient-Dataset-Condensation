import torch
import torch.nn as nn
import time

from dataset import get_baseline_dataset, get_condense_dataset
from time import time


class Exemplar:
    def __init__(self, prototypes):
        self.prototypes = prototypes


class Condense(Exemplar):
    def __init__(
        self,
        prototypes=None,
        evalset=None,
        ipc=10,
        factor=1,
        strong_aug=False,
    ):
        super().__init__(prototypes=prototypes)
        self.evalset = evalset
        self.ipc = ipc
        self.factor = factor
        self.strong_aug = strong_aug

    def set_feature_model(self, tg_model):
        self.tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        self.num_features = tg_model.fc.in_features

    def compute_prototypes(self, start_iter, iteration, nb_cl_fg, nb_cl, order):

        if nb_cl_fg == 20 and nb_cl == 20:
            condenseset = get_condense_dataset(
                ipc=self.ipc,
                factor=self.factor,
                phase=iteration - start_iter,
            )
        else:
            raise NotImplementedError("Not implemented for other incremental setting")

        if iteration == start_iter:
            for c in range(0, nb_cl_fg):
                self.prototypes[c, :, :, :, :] = condenseset.tensors[
                    [i == c for i in condenseset.targets]
                ].view(
                    self.prototypes.shape[1],
                    self.prototypes.shape[2],
                    self.prototypes.shape[3],
                    self.prototypes.shape[4],
                )
        else:
            for c in range(nb_cl * iteration, nb_cl * iteration + nb_cl):
                self.prototypes[c, :, :, :, :] = condenseset.tensors[
                    [i == (c - (nb_cl * iteration)) for i in condenseset.targets]
                ].view(
                    self.prototypes.shape[1],
                    self.prototypes.shape[2],
                    self.prototypes.shape[3],
                    self.prototypes.shape[4],
                )

    def concat_prototypes(
        self,
        class_index,
        current_cl,
        nb_cl,
        X_protoset_cumul,
        Y_protoset_cumul,
    ):

        index2 = class_index % nb_cl

        new_X_proto = self.prototypes[class_index]
        new_Y_proto = current_cl[index2] * torch.ones(len(new_X_proto))
        X_protoset_cumul = torch.cat([X_protoset_cumul, new_X_proto])
        Y_protoset_cumul = torch.cat([Y_protoset_cumul, new_Y_proto])

        return X_protoset_cumul, Y_protoset_cumul


class Baseline(Exemplar):
    def __init__(self, prototypes=None, evalset=None, type="dm", net="convnet"):
        self.prototypes = prototypes
        self.evalset = evalset
        self.type = type
        self.net = net
        assert self.type in ["dm", "dsa", "herding"]
        if prototypes.shape[1] != 20:
            raise NotImplementedError("Only nb_protos with 20 is implemented")

    def set_feature_model(self, tg_model):
        self.tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        self.num_features = tg_model.fc.in_features

    def compute_prototypes(self, start_iter, iteration, nb_cl_fg, nb_cl, order):
        t1 = time()

        blset = get_baseline_dataset(type=self.type, net=self.net)

        if iteration == start_iter:
            for c in range(0, nb_cl_fg):
                self.prototypes[c, :, :, :, :] = blset.tensors[
                    [i == order[c] for i in blset.targets]
                ].view(
                    self.prototypes.shape[1],
                    self.prototypes.shape[2],
                    self.prototypes.shape[3],
                    self.prototypes.shape[4],
                )
        else:
            for c in range(nb_cl * iteration, nb_cl * iteration + nb_cl):
                self.prototypes[c, :, :, :, :] = blset.tensors[
                    [i == order[c] for i in blset.targets]
                ].view(
                    self.prototypes.shape[1],
                    self.prototypes.shape[2],
                    self.prototypes.shape[3],
                    self.prototypes.shape[4],
                )

        t2 = time()
        print()
        print("#################################################")
        print(f"Elapsed time for loading prototypes is {t2 - t1} seconds")
        print("#################################################")
        print()

    def concat_prototypes(
        self,
        class_index,
        current_cl,
        nb_cl,
        X_protoset_cumul,
        Y_protoset_cumul,
    ):

        index2 = class_index % nb_cl

        new_X_proto = self.prototypes[class_index]
        new_Y_proto = current_cl[index2] * torch.ones(len(new_X_proto))
        X_protoset_cumul = torch.cat([X_protoset_cumul, new_X_proto])
        Y_protoset_cumul = torch.cat([Y_protoset_cumul, new_Y_proto])

        return X_protoset_cumul, Y_protoset_cumul
