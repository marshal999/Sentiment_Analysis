from PIL import Image
import net
import torch
import numpy as np
from face_alignment import align
from face_alignment.align import mtcnn_model
import os

adaface_models = {
    'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location=torch.device('cpu'))['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    output = torch.empty((1, brg_img.shape[2], brg_img.shape[0], brg_img.shape[1]))
    output[0] = torch.tensor(brg_img.transpose(2, 0, 1)).float()
    return output


def get_face_number(img: Image.Image):
    """
    Counts number of faces in an image
    Args:
        img: Input image to count faces in.

    Returns:
        Number faces in the image
    """
    try:
        bboxes, faces = mtcnn_model.align_multi(img)
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        return -1
    return len(faces)


def get_image_similarity(*images):
    """
    Computes similarity between list of images
    Args:
        *images: List of images
    Returns:
        2D tensor of similarities between input images
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    model = load_pretrained_model('ir_50')
    _, _ = model(torch.randn(2, 3, 112, 112))

    features = []
    for img in images:
        aligned_rgb_img = align.get_aligned_face('', img)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    return similarity_scores


def is_two_image_similar(image1: Image.Image, image2: Image.Image, **kwargs):
    """
    Checks if two images similar
    Args:
        image1: The first image
        image2: The second image
        **kwargs: threshold = a threshold to determine if two images are similar

    Returns:
        A bool representing if two images are similar.
    """
    if 'threshold' in kwargs.keys():
        threshold = kwargs['threshold']
    else:
        threshold = 0.4
    return get_image_similarity(image1, image2)[0][1].item() > threshold


def is_images_similar(*images, **kwargs):
    """
    Checks if list of images similar
    Args:
        *images: The list of images
        **kwargs: threshold = a threshold to determine if two images are similar

    Returns:
        2D tensor representing if two images are similar
    """
    if 'threshold' in kwargs.keys():
        threshold = kwargs['threshold']
    else:
        threshold = 0.4
    return get_image_similarity(*images) > threshold


img1 = Image.open('face_alignment/test_images/1.jpeg')
img2 = Image.open('face_alignment/test_images/2.jpeg')
img3 = Image.open('face_alignment/test_images/3.jpeg')
print(get_image_similarity(img1, img2, img3))
print(is_images_similar(img1, img2, img3, threshold=0.8))
print(help(get_face_number))
