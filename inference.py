import os

import cv2
import imagehash
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pygame import freetype
from skimage import io
from skimage.transform import resize

from datagen import To_tensor
from model import Generator
from SRNetDatagen.Synthtext.render_standard_text import make_standard_text


def generate_text_image(image, text, font_path="SRNetDatagen/arial.ttf"):
    freetype.init()

    shape = image.shape[:2]
    i_t = make_standard_text(font_path, text, shape)
    return i_t


def get_images(i_s, text, tmp_folder):
    os.makedirs(tmp_folder, exist_ok=True)
    i_t = generate_text_image(i_s, text, "SRNetDatagen/arial.ttf")

    phash = str(imagehash.phash(Image.fromarray(i_s))) + "_" + text

    i_s_path = os.path.join(tmp_folder, phash + "_i_s.png")
    i_t_path = os.path.join(tmp_folder, phash + "_i_t.png")

    cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    i_s = io.imread(i_s_path)
    i_t = io.imread(i_t_path)
    return i_t, i_s, phash


class SRNetIngerence:
    def __init__(self, checkpoint, tmp_folder, device="cpu"):
        self.device = device
        self.tmp_folder = tmp_folder

        self.G = Generator(in_channels=3).to(device)
        sd = torch.load(checkpoint, map_location="cpu")
        self.G.load_state_dict(sd["generator"])
        self.G = self.G.to(device)
        self.G.eval()

    def preprocess_image(self, i_t, i_s, phash):
        trfms = To_tensor()

        data_shape = [64, None]
        h, w = i_t.shape[:2]
        scale_ratio = data_shape[0] / h
        to_h = data_shape[0]
        to_w = int(round(int(w * scale_ratio) / 8)) * 8
        to_scale = (to_h, to_w)

        i_t = resize(i_t, to_scale, preserve_range=True)
        i_s = resize(i_s, to_scale, preserve_range=True)

        sample = (i_t, i_s, phash)
        sample = trfms(sample)
        return sample

    @torch.no_grad()
    def infer(self, i_s, text):
        i_t, i_s, phash = get_images(image, text, "custom_feed/tmp")
        i_t, i_s, phash = self.preprocess_image(i_t, i_s, phash)

        i_t = i_t[None].to(self.device)
        i_s = i_s[None].to(self.device)

        o_sk, o_t, o_b, o_f = self.G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

        o_f = o_f.squeeze(0).detach().to("cpu")
        return F.to_pil_image((o_f + 1) / 2)


if __name__ == "__main__":
    checkpoint = "trained_final_5M_.model"
    model = SRNetIngerence(checkpoint, "custom_feed/tmp", device="cpu")

    image = cv2.imread("custom_feed/labels/005_i_s.png")
    # original_it = cv2.imread("custom_feed/labels/002_i_t.png")
    text = "ficture"

    res = model.infer(image, text)

    plt.imshow(np.array(image))
    plt.show()
    plt.imshow(np.array(res))
    plt.show()
