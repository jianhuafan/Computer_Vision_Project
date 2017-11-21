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
    output_video = '../results/video/three_noise_Remove.m4v'

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
                ret, fgmask = cv2.threshold(fgmask, 1, 255, cv2.THRESH_BINARY)
                # fgmask = cv2.fastNlMeansDenoising(fgmask)
                # cv2.imshow('frame', fgmask)
                # cv2.imwrite('../results/image/CNT/{}.png'.format(current_frame), frame)
                # background = np.zeros(fgmask.shape, np.uint8)
                # gray = fgmask.copy()
                # img2 = np.zeros(gray.shape, np.uint8)
                # im2, cnts, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                # for cnt in cnts:
                #     # if 200<cv2.contourArea(cnt):
                #         # cv2.drawContours(fgmask,[cnt],0,(0,255,0),2)
                #     cv2.drawContours(mask,[cnt],0, (0,255,0), 3)
                # cv2.imshow('frame', mask)       
                # _, cnts, _= cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # mask = np.ones(fgmask.shape[:2], dtype="uint8") * 255

                # for c in cnts:
                #     area = cv2.contourArea(c)
                #     if area < 900:
                #         cv2.drawContours(fgmask, [c], -1, 0, -1)
                
                # cv2.imshow('after', image)
                #cv2.bitwise_not(gray2,gray2,mask)
                #print(height, width, layers)
                # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(fgmask, connectivity=4)
                # sizes = stats[1:, -1]; nb_components = nb_components - 1
                # min_size = 100
                # for i in range(0, nb_components):
                #     if sizes[i] >= min_size:
                #         img2[output == i + 1] = 255
                # cv2.imshow('after', img2)
                # if current_frame == 0:
                #     background = fgmask
                # if current_frame  < 40:
                #     background = cv2.bitwise_or(background, fgmask)
                # else:
                #     temp = cv2.bitwise_and(background, fgmask)
                #     fgmask = fgmask - temp
                #     cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                #     cv2.resizeWindow('image', 600,600)
                #     cv2.imshow('image', fgmask)
                minLineLength=10
                new = np.zeros(fgmask.shape[:2], np.uint8)
                lines = cv2.HoughLinesP(image=fgmask,rho=1,theta=np.pi/180, threshold=10,lines=np.array([]), minLineLength=minLineLength,maxLineGap=200)

                a,b,c = lines.shape
                for i in range(a):
                    x = lines[i][0][0] - lines [i][0][2]
                    y = lines[i][0][1] - lines [i][0][3]
                    if x!= 0:
                        if abs(y/x) <1:
                            cv2.line(new, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 1)
                temp = cv2.bitwise_and(new, fgmask)
                fgmask = cv2.subtract(fgmask,temp)
                
                                    
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (1,2))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, se)
                mask = np.ones(fgmask.shape[:2],np.uint8) * 255
                mask[:, :100] = 0
                mask[:, 400:-1]=0
                fgmask = cv2.bitwise_and(fgmask, mask)
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 600,600)
                cv2.imshow('image', fgmask)
                # # if current_frame == 0:
                # #     cv2.imwrite('../results/image/background_{}.png'.format(current_frame), fgmask)
                # if current_frame > 5:
                #     temp = cv2.bitwise_and(background, fgmask)
                #     new = (fgmask - background)
                #     fgmask = new
                #     cv2.namedWindow("output", cv2.WINDOW_NORMAL)
                #     imS = cv2.resize(temp, (0, 0), fx=0.7, fy=0.7) 
                #     cv2.imshow('output', imS)
                #     cv2.imwrite('../results/image/background/{}.png'.format(current_frame), background)
                # if current_frame >= 5:
                #     cv2.imwrite('../results/image/background_{}.png'.format(current_frame), background)
                cv2.imwrite('../results/image/noise/{}.png'.format(current_frame), fgmask)
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
    