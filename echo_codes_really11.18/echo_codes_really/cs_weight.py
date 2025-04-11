import torch
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
from tqdm import tqdm

from dataset_new import *
from models_flow_unet import Unsupervised

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mean, std =0,1
tasks_2 = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
kwargs_2 = {"target_type": tasks_2,
                "mean": mean,
                "std": std
                }
dataset_test = echonet.datasets.Echo(split='test', **kwargs_2)
test2 = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=2, shuffle=False,
                                        pin_memory=(device.type == "cuda"))
#model_weight=torch.load('/media/ubuntu/e/mengjinpeng/echo_codes_really11.18/echo_codes_really/model_weight/training_state.pt')
checkpoint = torch.load('/media/E/mjp/echo_codes_really11.18/echo_codes_really/model_weight/training_state.pt')
model_state_dict = checkpoint['model_state_dict']
mymodel = Unsupervised(conv_predictor="flownet")
mymodel.to(device)
mymodel.load_state_dict(model_state_dict)
def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0

    # pos = 0
    # neg = 0
    # pos_pix = 0
    # neg_pix = 0

    model.eval()

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                y_large = model(large_frame)
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small = model(small_frame)
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                # if train:
                #     optim.zero_grad()
                #     loss.backward()
                #     optim.step()
                #
                # # Accumulate losses and compute baselines
                total += loss.item()
                n += large_trace.size(0)
                # p = pos / (pos + neg)
                # p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)
                #
                # # Show info on process bar
                # pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                # pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (total / n / 112 / 112,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )


val_model=mymodel.predictor_s
loss, large_inter, large_union, small_inter, small_union = run_epoch(val_model, test2, False, None, device)
overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
large_dice = 2 * large_inter / (large_union + large_inter)
small_dice = 2 * small_inter / (small_union + small_inter)
print("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(
                np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)),
                echonet.utils.dice_similarity_coefficient)))
print("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(large_inter,
                                                                                                      large_union,
                                                                        echonet.utils.dice_similarity_coefficient)))
print("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format("test", *echonet.utils.bootstrap(small_inter,
                                                                                                      small_union,
                                                                        echonet.utils.dice_similarity_coefficient)))
