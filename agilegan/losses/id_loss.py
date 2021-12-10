import torch
from mmgen.models.builder import MODULES
from torch import nn

from ..architectures.encoders.model_irse import Backbone


@MODULES.register_module()
class IDLoss(nn.Module):
    def __init__(self, model_path=None):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112,
                                num_layers=50,
                                drop_ratio=0.6,
                                mode='ir_se')
        if model_path is not None:
            self.facenet.load_state_dict(torch.load(model_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count
