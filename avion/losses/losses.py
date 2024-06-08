import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def gather_sth(
        sth,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_sth = hvd.allgather(sth)
        else:
            with torch.no_grad():
                all_sth = hvd.allgather(sth)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_sth = list(all_sth.chunk(world_size, dim=0))
                gathered_sth[rank] = sth
                all_sth = torch.cat(gathered_sth, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_sth = torch.cat(torch.distributed.nn.all_gather(sth), dim=0)
        else:
            gather_sth = [torch.zeros_like(sth) for _ in range(world_size)]
            dist.all_gather(gather_sth, sth)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gather_sth[rank] = sth
            all_sth = torch.cat(gather_sth, dim=0)
    return all_sth

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / logits_per_image.size(0)
        return {'loss': total_loss, 'clip_acc': acc}


class MaxMarginRankingLoss(nn.Module):

    def __init__(
        self,
        margin=0.2,
        fix_norm=True,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.fix_norm = fix_norm
        self.margin = margin
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def forward(self, image_features, text_features, weight=None):
        # TODO: try gather_from_all in
        # https://github.com/facebookresearch/LaViLa/blob/main/lavila/models/distributed_utils.py
        # all_image_features = gather_from_all(image_features)
        # all_text_features = gather_from_all(text_features)
        all_image_features, all_text_features = gather_features(
            image_features, text_features,
            self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)


        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))
        
        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return {
            'loss': max_margin.mean(),
            'max_margin_loss': max_margin.mean()
        }


class AdaptiveMaxMarginRankingLoss(nn.Module):

    def __init__(
        self,
        margin=0.4,
        temperature = 0.05,
        fix_norm=True,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.fix_norm = fix_norm
        self.margin = margin
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def forward(self, image_features, text_features, weight=None, **kwargs):

        all_image_features, all_text_features = gather_features(
            image_features, text_features,
            self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
        
        weight = gather_sth(weight, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]
        #print(x.size())
        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        #print(w1.size())
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(  w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin =  F.relu( w1_ * self.margin - (x1_ - x2_))

        return {
            'loss': max_margin.mean(),
            'max_margin_loss': max_margin.mean()
        }
    

class SymmetricMultiSimiliarityLoss(nn.Module):

    def __init__(
        self,
        margin=0.6,
        thres = 0.1,
        matrix = None,
        fix_norm=True,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.fix_norm = fix_norm
        self.margin = margin
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.uint_factor = 240
        self.correlation_matrix = torch.tensor(matrix*self.uint_factor, dtype=torch.uint8).to(rank)

        self.eps = 1.1/self.uint_factor
        self.thres = thres

        #self.dis_matrix = SetwiseDistance

    def forward(self, image_features, text_features, weight=None, relevancy_weight=None):

        all_image_features, all_text_features = gather_features(
            image_features, text_features,
            self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
        
        #print(weight, relevancy_weight)
        weight = gather_sth(weight, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]
        #print(x.size())
        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        #print(w1.size())
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        #max_margin = F.relu(  w1 * self.margin - (x1 - x2))

        if relevancy_weight is not None:
            relevancy_weight = gather_sth(relevancy_weight, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            #print(weight.shape, relevancy_weight.shape)
            relevancy_weight = relevancy_weight.view(self.world_size,2,-1)
            relevancy_col, relevancy_row = relevancy_weight.split(split_size=1, dim=1)[0], relevancy_weight.split(split_size=1, dim=1)[1]
            rw = torch.zeros_like(x)
            relevancy_col = relevancy_col.reshape(-1)
            relevancy_row = relevancy_row.reshape(-1)
            rw = self.correlation_matrix[relevancy_row][:, relevancy_col]/self.uint_factor
            #relevancy_matrix = train_dataset.return_relevancy_mat()
            #print(relevancy_col.shape, relevancy_row.shape)

            rw_t = rw.transpose(0, 1).contiguous().view(-1, 1)
            rw_v = rw.view(-1, 1)
            rw = torch.cat((rw_t, rw_v), 0)
        else:
            rw = torch.zeros_like(w1)
        


        #KL_loss = self.KL_loss(all_image_features, all_text_features)

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            rw_ = torch.index_select(rw, dim=0, index=keep_idx)
        
        #print(w1.shape, rw.shape)
        weight_matrix = w1_ - rw_
        max_margin = weight_matrix * self.margin - (x1_ - x2_)
        max_margin =  torch.where(
            weight_matrix > self.eps,
            F.relu(max_margin),
            torch.where(
                torch.abs(weight_matrix) < self.eps, 
                F.relu(torch.abs(max_margin)-self.thres), 
                F.relu(-max_margin))
                )

        return {
            'loss': max_margin.mean(),
            'max_margin_loss': max_margin.mean()
        }