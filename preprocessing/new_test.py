import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt


def noise_Remove(fpath):
    cap = cv2.VideoCapture(fpath)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_frame)

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    output_video = '../results/video/test.m4v'

    try:
        os.remove(output_video)
    except OSError:
        pass
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (544, 960), False)

    try:
        while cap.isOpened():
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            rate = cap.get(cv2.CAP_PROP_FPS)
            # print(rate)
            ret, frame = cap.read()

            if ret:
                img = frame
                height, width, layers = frame.shape
                print(height, width)
                fgmask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = np.ones(fgmask.shape[:2],np.uint8) * 255
                mask[720:-1] = 0
                print(fgmask)
                fgmask = cv2.bitwise_and(fgmask, mask)
                # fgmask = cv2.fastNlMeansDenoising(fgmask)
                # _,fgmask = cv2.threshold(fgmask,100,255,cv2.THRESH_BINARY)
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 600,600)
                cv2.imshow('image', fgmask)
                
                cv2.imwrite('../results/image/test/{}.png'.format(current_frame), fgmask)
                out.write(fgmask)
                print(current_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print('Stopped for ctr-c')
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    video_name = 'test.m4v'
    input_path = '../results/video/'
    fpath = input_path + video_name
    noise_Remove(fpath)

    

if __name__ == "__main__":
    main()
    