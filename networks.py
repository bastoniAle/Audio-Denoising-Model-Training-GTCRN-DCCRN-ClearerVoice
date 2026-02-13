import torch
import torch.nn as nn
import torch.nn.functional as F

class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args
        if args.network == 'FRCRN_SE_16K':
            from models.frcrn.frcrn import FRCRN_SE_16K
            self.se_network = FRCRN_SE_16K(args).model
        elif args.network == 'MossFormer2_SE_48K':
            from models.mossformer2.mossformer2_se_wrapper import MossFormer2_SE_48K
            self.se_network = MossFormer2_SE_48K(args).model
        elif args.network == 'MossFormerGAN_SE_16K':
            from models.mossformer_gan.generator import MossFormerGAN_SE_16K
            self.se_network = MossFormerGAN_SE_16K(args).model
            self.discriminator = MossFormerGAN_SE_16K(args).discriminator
        elif args.network == 'ULUNAS_SE_48K':
            from models.ulunas.ulunas import ULUNAS_SE_48K
            self.se_network = ULUNAS_SE_48K(args)
        elif args.network == 'DCCRN_SE_48K':
            from models.dccrn.dccrn2 import DCCRN_SE_48K
            self.se_network = DCCRN_SE_48K(args)
        elif args.network == 'PHASEN_SE_48K':
            from models.phasen.phasen import PHASEN_SE_48K
            self.se_network = PHASEN_SE_48K(args)
        elif args.network == 'DPCRN_SE_48K':
            from models.dpcrn.dpcrn import DPCRN_SE_48K
            self.se_network = DPCRN_SE_48K(args)
        elif args.network == 'GTCRN_SE_48K':
            from models.gtcrn.gtcrn import GTCRN_SE_48K
            self.se_network = GTCRN_SE_48K(args)
        elif args.network == 'DCUNET_SE_48K':
            from models.dcunet.dcunet import DCUNET_SE_48K
            self.se_network = DCUNET_SE_48K(args)
        elif args.network == 'DTLN_SE_48K':
            from models.dtln.dtln import DTLN_SE_48K
            self.se_network = DTLN_SE_48K(args)
        else:
            print("No network found!")
            return

    def forward(self, mixture, visual=None):
        est_source = self.se_network(mixture)
        return est_source
