import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropyLossWithLogits(nn.Module):
    def __init__(self, pos_weight=None, weight=None, reduction='mean'):
        """Wrapper around binary_cross_entropy_with_logits.

        Args:
            pos_weight (Tensor): weight of positive samples with shape [num_classes]
                (used to handle class imbalance)
            weight (Tensor): per-sample loss weights, same shape as input
            reduction (str): 'none' | 'mean' | 'sum', how the final loss is reduced
        """
        super(BinaryCrossEntropyLossWithLogits, self).__init__()
        self.register_buffer('pos_weight', pos_weight if pos_weight is not None else None)
        self.register_buffer('weight', weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class MeanConstraintLoss(nn.Module):
    def __init__(self, margin=0.5, lambda_reg=0.1, temperature_can=2, temperature_neg=2):
        super().__init__()
        self.margin = margin  # margin of the constraint
        self.lambda_reg = lambda_reg  # strength of the regularization term
        self.temperature_can = temperature_can
        self.temperature_neg = temperature_neg

    def forward(self, probs, candidate_labels):
        """
        probs: predicted probabilities after sigmoid, shape (batch_size, num_labels)
        candidate_labels: candidate label indicator matrix of shape (batch_size, num_labels),
            where 0/1 indicates whether a label is a candidate
        """
        batch_size, num_labels = probs.shape

        # candidate labels
        candidate_mask = (candidate_labels != 0)
        candidate_weights = torch.sigmoid((1.0 - probs) * self.temperature_can).detach()
        high_conf_candidate = probs * candidate_mask * candidate_weights
        sum_high_conf_candidate = high_conf_candidate.sum(dim=1)  # (batch_size,)
        count_high_conf_candidate = (high_conf_candidate > 0).sum(dim=1).clamp(min=1e-6)  # avoid division by zero

        # non-candidate labels
        non_candidate_mask = (candidate_labels == 0)
        low_conf_non_candidate = probs * non_candidate_mask
        sum_low_conf_non_candidate = low_conf_non_candidate.sum(dim=1)
        count_low_conf_non_candidate = (low_conf_non_candidate > 0).sum(dim=1).clamp(min=1e-6)

        # difference between the two means
        mean_high_candidate = sum_high_conf_candidate / count_high_conf_candidate  # (batch_size,)
        mean_low_non_candidate = sum_low_conf_non_candidate / count_low_conf_non_candidate

        # regularization term: enforce mean_high_candidate >= mean_low_non_candidate + margin
        reg_loss = torch.sum(torch.relu(mean_low_non_candidate - mean_high_candidate + self.margin))

        return self.lambda_reg * reg_loss


class DynamicAsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification with a dynamically
    annealed negative focusing parameter.

    When ``dynamic`` is enabled, ``gamma_neg`` decays polynomially from
    ``gamma_neg_init`` to ``gamma_neg_final`` over ``max_epoch`` epochs:

        gamma_neg = init - (init - final) * (epoch / max_epoch) ** decay_power

    Note: this variant expects **probabilities** (post-sigmoid) as input.
    """

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, dynamic=True,
                 disable_torch_grad_focal_loss=True,
                 gamma_neg_init=4, gamma_neg_final=1, max_epoch=30.0, decay_power=1):
        super(DynamicAsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # dynamic annealing of gamma_neg
        self.dynamic = dynamic
        self.gamma_neg_init = gamma_neg_init
        self.gamma_neg_final = gamma_neg_final
        self.max_epoch = max_epoch
        self.decay_power = decay_power

    def forward(self, x_sigmoid, y, weight=None, epoch=0):
        """
        Parameters
        ----------
        x_sigmoid: predicted probabilities (after sigmoid)
        y: targets (multi-label binarized vector; may contain soft entries such
            as 0.5 produced by label disambiguation)
        weight: optional per-element loss weights (e.g. to mask out ambiguous
            labels), same shape as x_sigmoid
        epoch: current epoch, used by the dynamic gamma_neg schedule
        """
        # anneal the negative focusing parameter over the course of training
        if self.dynamic:
            progress = min(epoch / self.max_epoch, 1.0)  # progress in [0, 1]
            self.gamma_neg = self.gamma_neg_init - (self.gamma_neg_init - self.gamma_neg_final) * (
                    progress ** self.decay_power)
        else:
            self.gamma_neg = self.gamma_neg_init

        # Calculating Probabilities
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # Per-element weighting (e.g. zero out ambiguous labels)
        if weight is not None:
            loss = loss * weight

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss