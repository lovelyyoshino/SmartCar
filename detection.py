#-*- coding: UTF-8 -*- 
import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib
from data import VOC_CLASSES as labels
import time
import data.config as cfg
import os


from ssd_mobilenetv2 import build_ssd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"#设置型号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#设置使用哪块GPU

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

num_classes = 10#VOC2012的种类，需要修改

image_size = 300#使用的SSD图片大小

def detection_video(path,weight):#识别的video
    global image_size,tracker_rgb,init_num
    flag = 0
    net = build_ssd('test', 300, num_classes)
    net.eval()
    net.load_weights(weight)#导入模型参数

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    init_num=0
    t4=time.time()

    while cap.isOpened():
        ret,image = cap.read()
        if init_num==0:#初始化程序
            flag += 1

            if ret == False:
                print("video is over!")
                break
            if flag % 3 != 0:#每三帧处理一次，为了防止jetson nano速率不够
                continue

            t0 = time.time()
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            resize_image = cv2.resize(image, (300, 300)).astype(np.float32)
            resize_image -= (104, 117, 123)#对SSD实现均值化
            resize_image = resize_image.astype(np.float32)#转为float32
            resize_image = resize_image[:, :, ::-1].copy()

            torch_image = torch.from_numpy(resize_image).permute(2, 0, 1)#重新排列传入torch
            input_image = Variable(torch_image.unsqueeze(0))#扩展第一列
            if torch.cuda.is_available():
                input_image = input_image.cuda()#设置为CUDA形式

            out = net(input_image)#传入到模型当中

            colors = cfg.COLORS

            detections = out.data

            scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)#[ 起始下标 : 终止下标 : 间隔距离 ]


            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)#转化为BGR参数

            idx_obj = -1#初始为-1

            center_point=[0,0]
            gallery_best_draw=[0,0,0,0]

            for i in range(detections.size(1)):#获取所有的参数
                j = 0#都要循环类的次数
                #print(detections.size())
                #print(i)
                if detections[0,i,j,0] >= 0.95:#设定阈值

                    idx_obj += 1#物体数+1

                    score = detections[0,i,j,0]#计算得分
                    label_name = labels[i-1]#得到名称

                    display_txt = '%s %.2f'%(label_name, score)#显示目标物体位置
                    pt = (detections[0,i,j,1:]*scale).cpu().numpy()#获取四个点位置

                    #j += 1

                    # 求得四个边角，并防止溢出
                    pt[0] = max(pt[0],0)
                    pt[1] = max(pt[1],0)
                    pt[2] = min(pt[2],size[1])
                    pt[3] = min(pt[3],size[0])
                    #print(pt[0],pt[3])
                    if  abs(pt[2]-pt[0])*abs(pt[3]-pt[1])>500:
                        print((pt[2]-pt[0])*(pt[3]-pt[1]))
                        if (pt[0]+pt[2])/2>100 and (pt[1]+pt[3])/2>140 and (pt[0]+pt[1]+pt[2]+pt[3])/2>(center_point[0]+center_point[1]):#处理一帧中的最优点
                            center_point=[(pt[0]+pt[2])/2,(pt[1]+pt[3])/2]#更新最优点
                            gallery_best_draw=[pt[0],pt[1],pt[2],pt[3]]
                            #init_num=1
                            #print(pt[0],pt[1],pt[2],pt[3])
                            #print(center_point)
                    else:
                        print("error",(pt[2]-pt[0])*(pt[3]-pt[1]))
                        continue




                    color = colors[idx_obj%len(colors)]#选择颜色

                    textsize = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]#显示文本文字


                    text_x = int(pt[0])#文本位置
                    text_y = int(pt[1])
                    cv2.rectangle(rgb_image,(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])),color,4)#框选位置
                    cv2.putText(rgb_image, display_txt, (text_x + 4, text_y), cv2.FONT_HERSHEY_COMPLEX, 1,(255 - color[0], 255 - color[1], 255 - color[2]), 2)#输出结果

            if gallery_best_draw[0]!=0:
                track_roi=(gallery_best_draw[0],gallery_best_draw[1],abs(gallery_best_draw[2]-gallery_best_draw[0]),abs(gallery_best_draw[3]-gallery_best_draw[1]))
                print("track_roi:",track_roi)
                try:
                    tracker_rgb=cv2.TrackerMOSSE_create()#重置
                    tracker_rgb.init(rgb_image, track_roi)#初始化对应的参数
                except:
                	pass


            #t1 = time.time()

            #cv2.putText(rgb_image, "FPS: %.2f" % (1 / (t1 - t0)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

            #cv2.imshow("result",rgb_image)

        elif init_num==1:
            t0 = time.time()
            images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)#转化为BGR参数
            rgb_image=images.copy()
            (success, box) = tracker_rgb.update(rgb_image)
            if time.time()-t4>10:
                init_num=0
                t4=time.time()
            #print(time.time()-t4)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                csrt_best_draw=[int(x),int(y),int(x+w),int(y+h)]
                cv2.rectangle(rgb_image,tuple(csrt_best_draw),color,4)#框选位置
        
        t1 = time.time()

        cv2.putText(rgb_image, "FPS: %.2f" % (1 / (t1 - t0)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("result",rgb_image)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


def csrt():
    global tracker_rgb
    OPENCV_OBJECT_TRACKERS = {
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    tracker_rgb = OPENCV_OBJECT_TRACKERS["mosse"]()


if __name__ == "__main__":
    weight = 'weights/ssd_mobilenetv2/mobilenetv2_final.pth'
    #path = r"test_images/example.jpg"
    path = r"test_videos/tests.mp4"
    csrt()
    detection_video(path,weight)

