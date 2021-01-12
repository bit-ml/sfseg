import argparse
import sys

import cv2
import torch

from sfseg import SFSegParam

FPS = 25
FOURCC = cv2.VideoWriter_fourcc(*'XVID')


def save_video(save_path, masks_in, masks_out):
    video_h, video_w = masks_in.shape[1:]
    out_video = cv2.VideoWriter(save_path,
                                FOURCC,
                                FPS, (video_w * 2, video_h),
                                isColor=False)

    text = "Input Mask                    SFSeg"
    for i in range(masks_in.shape[0]):
        united = torch.cat([masks_in[i], masks_out[i]], dim=1)
        uint_frame = (united * 255).byte().data.cpu().numpy()
        cv2.putText(uint_frame, text, (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255), 2, cv2.LINE_AA)
        out_video.write(uint_frame)

    out_video.release()


def read_args(args):
    parser = argparse.ArgumentParser(description='SFSeg parameters.')
    parser.add_argument(
        '-inp_path',
        metavar='inp_path',
        type=str,
        default="sample/input_masks.th",
        required=False,
        help='Input segmentation (unary maps), shape N_frames x H x W.')
    parser.add_argument(
        '-feat_path',
        metavar='feat_path',
        type=str,
        default="sample/features.th",
        required=False,
        help='Feature maps (pairwise maps), shape N_frames x H x W.')
    parser.add_argument(
        '-num_iters',
        metavar='num_iters',
        type=int,
        default=10,
        required=False,
        help='Number of SFSeg iterations through the full video.')
    parser.add_argument(
        '-alpha',
        metavar='alpha',
        type=float,
        default=1.,
        required=False,
        help=
        'Hyper-parameter for controling the pairwise term (see Eq. 1 in paper).'
    )
    parser.add_argument(
        '-p',
        metavar='p',
        type=float,
        default=0.5,
        required=False,
        help=
        'Hyper-parameter for controling the unary term (see Eq. 1 in paper).')
    parser.add_argument(
        '-kernel_size',
        metavar='kernel_size',
        nargs='+',
        type=int,
        default=(5, 5, 5),
        required=False,
        help='Hyper-parameter for controlling the 3D pixel neighbourhood size.'
    )
    args = parser.parse_args(args)
    return args


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Init params
    sfseg_params = SFSegParam(args.p, args.alpha, args.kernel_size, device)

    # Input maps
    input_masks = torch.load(args.inp_path, map_location=device)[None]
    features = torch.load(args.feat_path, map_location=device)[None]

    # SFSeg
    output_masks = input_masks.clone()
    for iter_idx in range(args.num_iters):
        one_iter_pi(sfseg_params,
                    output_masks,
                    input_masks,
                    features,
                    binarize=iter_idx > 2)

    # Output video
    save_fname = "sample/out_SFSeg.mp4"
    save_video(save_fname, input_masks[0][2:-2], output_masks[0][2:-2])
    print('Done. Output saved in %s' % save_fname)


if __name__ == "__main__":
    args = read_args(sys.argv[1:])
    main(args)
