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
    output_video = '../results/video/noise_Remove.m4v'

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
                fgmask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(fgmask.dtype)
                background = np.zeros(fgmask.shape, np.uint8)
                # gray = fgmask.copy()
                # mask = np.zeros(gray.shape, np.uint8)
                # contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                # for cnt in contours:
                #     if 200<cv2.contourArea(cnt)<5000:
                #         cv2.drawContours(fgmask,[cnt],0,(0,255,0),2)
                #         cv2.drawContours(mask,[cnt],0,255,-1)


                #cv2.bitwise_not(gray2,gray2,mask)
                #print(height, width, layers)
                #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(fgmask, connectivity=8)
                # sizes = stats[1:, -1]; nb_components = nb_components - 1
                # min_size = 400
                # img2 = np.zeros((output.shape))
                # for i in range(0, nb_components):
                #     if sizes[i] >= min_size:
                #         img2[output == i + 1] = 255
                if current_frame < 100:
                    background += fgmask + background
                else:
                    fgmask -= background

                if current_frame == 800:
                    cv2.imwrite('../results/image/remove_noise_{}.png'.format(current_frame), img2)
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
    input_path = '../results/video/'
    fpath = input_path + video_name
    noise_Remove(fpath)

    

if __name__ == "__main__":
    main()
    