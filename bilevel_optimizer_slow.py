# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Iterator

import torch
from torch import Tensor, autograd, nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from archai.common import ml_utils
from archai.common.config import Config
from archai.common.utils import zip_eq
from archai.supergraph.nas.model import Model


def _flatten_concate(xs):
    """
    flatten all tensor from [d1,d2,...dn] to [d]
    and then concat all [d_1] to [d_1+d_2+d_3+...]
    :param xs:
    :return:
    """
    return torch.cat([x.view(-1) for x in xs])

def _get_alphas(model:Model)->Iterator[nn.Parameter]:
    return model.all_owned().param_by_kind('alphas')

def _get_loss(model:Model, lossfn, x, y):
    logits, *_ = model(x) # might also return aux tower logits
    return lossfn(logits, y)

class BilevelOptimizer:
    def __init__(self, conf_alpha_optim:Config, w_momentum: float, w_decay: float,
                 model: Model, lossfn: _Loss) -> None:
        self._w_momentum = w_momentum  # momentum for w
        self._w_weight_decay = w_decay  # weight decay for w
        self._lossfn = lossfn
        self._model = model  # main model with respect to w and alpha

        self._alphas = list(_get_alphas(self._model))

        # this is the optimizer to optimize alphas parameter
        self._alpha_optim = ml_utils.create_optimizer(conf_alpha_optim, self._alphas)

    def state_dict(self)->dict:
        return {
            'alpha_optim': self._alpha_optim.state_dict()
        }

    def load_state_dict(self, state_dict)->None:
        self._alpha_optim.load_state_dict(state_dict['alpha_optim'])

    def _unrolled_model(self, x, y, lr: float, main_optim: Optimizer)->Model:
        # TODO: should this loss be stored for later use?
        loss = _get_loss(self._model, self._lossfn, x, y)
        params = _flatten_concate(self._model.parameters()).detach()

        try:
            moment = _flatten_concate(main_optim.state[v]['momentum_buffer'] for v in self._model.parameters())
            moment.mul_(self._w_momentum)
        except:
            moment = torch.zeros_like(params)

        # flatten all gradients
        grads = _flatten_concate(autograd.grad(loss, self._model.parameters())).data
        # indeed, here we implement a simple SGD with momentum and weight decay
        # theta = theta - eta * (moment + weight decay + dtheta)
        params = params.sub(lr, moment + grads + self._w_weight_decay*params)
        # construct a new model
        return self._params2model(params)

    def _params2model(self, params)->Model:
        """
        construct a new model with initialized weight from params
        it use .state_dict() and load_state_dict() instead of
        .parameters() + fill_()
        :params: flatten weights, need to reshape to original shape
        :return:
        """

        params_d, offset = {}, 0
        for k, v in self._model.named_parameters():
            v_length = v.numel()
            # restore params[] value to original shape
            params_d[k] = params[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(params)

        model_new = copy.deepcopy(self._model)
        model_dict = self._model.state_dict()
        model_dict.update(params_d)
        model_new.load_state_dict(model_dict)

        return model_new.cuda()

    def step(self, x_train: Tensor, y_train: Tensor, x_valid: Tensor, y_valid: Tensor,
             main_optim: Optimizer) -> None:
        # TODO: unlike darts paper, we get lr from optimizer insead of scheduler
        lr = main_optim.param_groups[0]['lr']
        self._alpha_optim.zero_grad()

        # compute the gradient and write it into tensor.grad
        # instead of generated by loss.backward()
        self._backward_bilevel(x_train, y_train, x_valid, y_valid,
                               lr, main_optim)

        # at this point we should have model with updated gradients for w and alpha
        self._alpha_optim.step()

    def _backward_bilevel(self, x_train, y_train, x_valid, y_valid, lr, main_optim):
        """ Compute unrolled loss and backward its gradients """

        # update vmodel with w', but leave alphas as-is
        # w' = w - lr * grad
        unrolled_model = self._unrolled_model(x_train, y_train, lr, main_optim)

        # compute loss on validation set for model with w'
        # wrt alphas. The autograd.grad is used instead of backward()
        # to avoid having to loop through params
        vloss = _get_loss(unrolled_model, self._lossfn, x_valid, y_valid)
        vloss.backward()
        dalpha = [v.grad for v in _get_alphas(unrolled_model)]
        dparams = [v.grad.data for v in unrolled_model.parameters()]

        hessian = self._hessian_vector_product(dparams, x_train, y_train)

        # dalpha we have is from the unrolled model so we need to
        # transfer those grades back to our main model
        # update final gradient = dalpha - xi*hessian
        # TODO: currently alphas lr is same as w lr
        with torch.no_grad():
            for alpha, da, h in zip_eq(self._alphas, dalpha, hessian):
                alpha.grad = da - lr*h
        # now that model has both w and alpha grads,
        # we can run main_optim.step() to update the param values

    def _hessian_vector_product(self, dw, x, y, epsilon_unit=1e-2):
        """
        Implements equation 8

        dw = dw` {L_val(w`, alpha)}
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha {L_trn(w+, alpha)} -dalpha {L_trn(w-, alpha)})/(2*eps)
        eps = 0.01 / ||dw||
        """

        """scale epsilon with grad magnitude. The dw
        is a multiplier on RHS of eq 8. So this scalling is essential
        in making sure that finite differences approximation is not way off
        Below, we flatten each w, concate all and then take norm"""
        # TODO: is cat along dim 0 correct?
        dw_norm = torch.cat([w.view(-1) for w in dw]).norm()
        epsilon = epsilon_unit / dw_norm

        # w+ = w + epsilon * grad(w')
        with torch.no_grad():
            for p, v in zip_eq(self._model.parameters(), dw):
                p += epsilon * v

        # Now that we have model with w+, we need to compute grads wrt alphas
        # This loss needs to be on train set, not validation set
        loss = _get_loss(self._model, self._lossfn, x, y)
        dalpha_plus = autograd.grad(
            loss, self._alphas)  # dalpha{L_trn(w+)}

        # get model with w- and then compute grads wrt alphas
        # w- = w - eps*dw`
        with torch.no_grad():
            for p, v in zip_eq(self._model.parameters(), dw):
                # we had already added dw above so sutracting twice gives w-
                p -= 2. * epsilon * v

        # similarly get dalpha_minus
        loss = _get_loss(self._model, self._lossfn, x, y)
        dalpha_minus = autograd.grad(loss, self._alphas)

        # reset back params to original values by adding dw
        with torch.no_grad():
            for p, v in zip_eq(self._model.parameters(), dw):
                p += epsilon * v

        # apply eq 8, final difference to compute hessian
        h = [(p - m) / (2. * epsilon)
             for p, m in zip_eq(dalpha_plus, dalpha_minus)]
        return h
