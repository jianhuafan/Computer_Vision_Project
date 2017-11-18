import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

def back_Sub(fpath, mode):
    cap = cv2.VideoCapture(fpath)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_frame)
    if mode == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2(30, 15, True) #history, varThreshold, bShadowDetection
    elif mode == 'MOG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(5, 3, 0.7, 0) #history, nmixture, backgroundRatio, noiseSigma
    elif mode == 'GMG':
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(2,0.5)#initializationFrame, decisionThreshold
    elif mode == 'CNT':
        fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(5, True, 15*60, True) #minStability, useHistory, maxStability, isParallel
    else:
        print('We only support methods of MOG2, MOG, GMG, CNT now!')
        exit(1)

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    output_video = '../results/video/{}.m4v'.format(mode)
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
                height, width, layers = frame.shape
                # fgmask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #print(height, width, layers)
                fgmask = fgbg.apply(frame)
                # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(fgmask, connectivity=8)
                # sizes = stats[1:, -1]; nb_components = nb_components - 1
                # min_size = 200
                # img2 = np.zeros((output.shape))
                # for i in range(0, nb_components):
                #     if sizes[i] >= min_size:
                #         img2[output == i + 1] = 255
                #print(type(fgmask))
                # fgmask = frame
                # if current_frame == 0:
                #   background = fgmask
                    # plt.imshow(background, interpolation='bicubic')
                    # cv2.imwrite('../results/image/background.png',background)
                # elif current_frame > 0:
                    # fgmask = np.subtract(fgmask, background)
                if current_frame == 800:
                    cv2.imwrite('../results/image/original_{}.png'.format(current_frame), frame)
                    cv2.imwrite('../results/image/{}_{}.png'.format(mode, current_frame), fgmask)
                out.write(fgmask)
                cv2.imshow('frame', fgmask)
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
    video_name = 'CannyEdge.m4v'
    mode = sys.argv[1]
    input_path = '../results/video/'
    fpath = input_path + video_name
    back_Sub(fpath, mode)

    

if __name__ == "__main__":
    main()
    