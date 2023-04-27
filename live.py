import argparse
from SSD_VGG16D.networks import AuxiliaryNetwork, PredictionNetwork, VGG16DBaseNetwork, DetectionNetwork
from SSD_VGG16D import SSD256
import torch
from torchvision import transforms
from SSD_VGG16D.utils import *
from PIL import Image, ImageDraw, ImageFont
import detect
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def main():
    '''
        Evaluate model by caculate mAP
        model: model SSD
        test_loader: Dataloader for test data

        Out: mAP for test data
    '''
    vid = cv2.VideoCapture(0)

    while 1:
        ret, frame = vid.read()

        annotated_image = detect.detect(detect.model, device, Image.fromarray(frame), min_score=detect.args.min_score,
                                        max_overlap=detect.args.max_overlap, top_k=detect.args.top_k)

        # Display the resulting frame
        cv2.imshow('frame', np.array(annotated_image))

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
