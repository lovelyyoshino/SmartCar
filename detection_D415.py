import os
import torch
from torch.autograd import Variable
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib
from data import VOC_CLASSES as labels
import time
import data.config as cfg
import os
import smbus


from ssd_mobilenetv2 import build_ssd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"#设置型号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#设置使用哪块GPU

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

num_classes = 10#VOC2012的种类，需要修改

image_size = 300#使用的SSD图片大小

def detection_video(pipeline,weight):#识别的video
    global image_size,depth_image,init_num,class_name
    flag = 0
    net = build_ssd('test', 300, num_classes)
    net.eval()
    net.load_weights(weight)#导入模型参数
    init_num=0
    size = (640,480)
    #t4=time.time()#test

    while True:
        frames = pipeline.wait_for_frames()#获取一帧
        depth_frame = frames.get_depth_frame()#深度图
        color_frame = frames.get_color_frame()#颜色图
        if not depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # 在深度图上用颜色渲染
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        if init_num==0:#初始化程序
           
            flag += 1
            if flag % 3 != 0:#每三帧处理一次，为了防止jetson nano速率不够
                continue

            t0 = time.time()
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            resize_image = cv2.resize(color_image, (300, 300)).astype(np.float32)
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
                #center_point=[0,0]
                #print(detections.size())
                if detections[0,i,j,0] >= 0.45:#设定阈值

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
                    
                    if label_name=="cup" or label_name=="battery" or label_name=="bottle" or label_name=="orange" or label_name=="paper":#保证检测到的是垃圾信息
                        if (pt[0]+pt[2])/2>100 and (pt[1]+pt[3])/2>140 and (pt[0]+pt[1]+pt[2]+pt[3])/2>(center_point[0]+center_point[1]) and (pt[2]-pt[0])*(pt[3]-pt[1])>5000:#处理一帧中的最优点
                            center_point=[(pt[0]+pt[2])/2,(pt[1]+pt[3])/2]#更新最优点
                            gallery_best_draw=[pt[0],pt[1],pt[2],pt[3]]
                            init_num=1
                            class_name=label_name#为了判断障碍是属于哪种

                    color = colors[idx_obj%len(colors)]#选择颜色

                    textsize = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]#显示文本文字


                    text_x = int(pt[0])#文本位置
                    text_y = int(pt[1])
                    cv2.rectangle(rgb_image,(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])),color,4)#框选位置
                    cv2.putText(rgb_image, display_txt, (text_x + 4, text_y), cv2.FONT_HERSHEY_COMPLEX, 1,(255 - color[0], 255 - color[1], 255 - color[2]), 2)#输出结果

            if gallery_best_draw[0]!=0:
                #https://blog.csdn.net/weixin_44576543/article/details/96179330
                #https://blog.csdn.net/weixin_44576543/article/details/96175286
                distace=D415_Depth(gallery_best_draw)
                Angle(center_point,distace)
                track_roi=(gallery_best_draw[0],gallery_best_draw[1],abs(gallery_best_draw[2]-gallery_best_draw[0]),abs(gallery_best_draw[3]-gallery_best_draw[1]))
                print("track_roi:",track_roi)
                tracker_rgb=cv2.TrackerMOSSE_create()#重置
                tracker_rgb.init(rgb_image, track_roi)#初始化对应的参数

            '''t1 = time.time()

            cv2.putText(rgb_image, "FPS: %.2f" % (1 / (t1 - t0)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

            cv2.imshow("result",rgb_image)'''

        elif init_num==1:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)#转化为BGR参数
            t0 = time.time()
            rgb_image=color_image.copy()
            (success, box) = tracker_rgb.update(rgb_image)
            # if time.time()-t4>10:#test
            #      init_num=2
            #      t4=time.time()
            #print(time.time()-t4)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                csrt_best_draw=[x,y,x+w,y+h]
                center_point=[(x+w/2),(y+h/2)]#更新最优点
                distace=D415_Depth(csrt_best_draw)
                Angle(center_point,distace)
                cv2.rectangle(rgb_image,tuple(csrt_best_draw),color,4)#框选位置
            cv2.imshow("result",rgb_image)

        elif init_num == 2:  # 初始化程序
            flag += 1
            if flag % 3 != 0:  # 每三帧处理一次，为了防止jetson nano速率不够
                continue

            t0 = time.time()
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            resize_image = cv2.resize(color_image, (300, 300)).astype(np.float32)
            resize_image -= (104, 117, 123)  # 对SSD实现均值化
            resize_image = resize_image.astype(np.float32)  # 转为float32
            resize_image = resize_image[:, :, ::-1].copy()

            torch_image = torch.from_numpy(resize_image).permute(2, 0, 1)  # 重新排列传入torch
            input_image = Variable(torch_image.unsqueeze(0))  # 扩展第一列
            if torch.cuda.is_available():
                input_image = input_image.cuda()  # 设置为CUDA形式

            # if time.time()-t4>10:#test
            #      init_num=0
            #      t4=time.time()

            out = net(input_image)  # 传入到模型当中

            colors = cfg.COLORS

            detections = out.data

            scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)  # [ 起始下标 : 终止下标 : 间隔距离 ]

            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # 转化为BGR参数

            idx_obj = -1  # 初始为-1
            center_point=[0,0]
            gallery_best_draw=[0,0,0,0]
            for i in range(detections.size(1)):  # 获取所有的参数
                j = 0  # 都要循环类的次数
                #center_point = [0, 0]
                # print(detections.size())
                if detections[0, i, j, 0] >= 0.45:  # 设定阈值

                    #idx_obj += 1  # 物体数+1

                    score = detections[0, i, j, 0]  # 计算得分
                    label_name = labels[i - 1]  # 得到名称

                    display_txt = '%s %.2f' % (label_name, score)  # 显示目标物体位置
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()  # 获取四个点位置

                    j += 1

                    # 求得四个边角，并防止溢出
                    pt[0] = max(pt[0], 0)
                    pt[1] = max(pt[1], 0)
                    pt[2] = min(pt[2], size[1])
                    pt[3] = min(pt[3], size[0])

                    if(pt[2]-pt[0])*(pt[3]-pt[1])>5000:
                        if  (class_name=="cup" and label_name=="brown") or (class_name=="battery" and label_name=="red") or (class_name=="bottle" and label_name=="black") or (class_name=="orange" and label_name=="green") or (class_name=="paper" and label_name=="black"):
                            gallery_best_draw = [pt[0], pt[1], pt[2], pt[3]]
                            center_point = [(pt[0] + pt[2]) / 2, (pt[1] + pt[3]) / 2]  # 更新最优点
                            print(center_point)

                    color = colors[idx_obj % len(colors)]  # 选择颜色

                    text_x = int(pt[0])  # 文本位置
                    text_y = int(pt[1])
                    cv2.rectangle(rgb_image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color, 4)  # 框选位置
                    cv2.putText(rgb_image, display_txt, (text_x + 4, text_y), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255 - color[0], 255 - color[1], 255 - color[2]), 2)  # 输出结果

            if gallery_best_draw[0] != 0:
                # https://blog.csdn.net/weixin_44576543/article/details/96179330
                # https://blog.csdn.net/weixin_44576543/article/details/96175286
                distace = D415_Depth(gallery_best_draw)
                Angle(center_point, distace)



        t1 = time.time()

        cv2.putText(rgb_image, "FPS: %.2f" % (1 / (t1 - t0)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("result",rgb_image)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

def csrt():
    global tracker_rgb
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    tracker_rgb = OPENCV_OBJECT_TRACKERS["mosse"]()

def Angle(center_point,distace):
    angle = np.arctan(np.tan(40 / 180 * np.pi) *abs(center_point[0] - 320)/320) * 180 / np.pi
    #需要我们向左转，则取负
    if (center_point[0]-320)<0:
        angle=-angle
    print(angle)
    I2C_Mess(distace, angle)
    #print(distace,angle)
	
def I2C_Mess(distace,angle):
    global bus,addr,recount_th0,init_num
    data=bus.read_byte_data(addr, recount_th0)
    print(data)
    if data==0x02:
        init_num=2#stm32发送是否已经夹取到目标
        bus.write_byte_data(addr, recount_th0, 0x03)
    elif data==0x00:
        init_num=0#stm32发送是否已经放置目标
        bus.write_byte_data(addr, recount_th0, 0x03)

    dis_high=int((distace*1000)//256)#传入高位信息
    dis_low=int(((distace*1000)%256)//1)#低位信息
    bus.write_byte_data(addr,recount_th0+int(1), dis_high)
    bus.write_byte_data(addr, recount_th0+int(2), dis_low)
    ang_i2c=int(angle//1)#舍去小数点
    bus.write_byte_data(addr,recount_th0+ int(3), ang_i2c)



def D415_Depth(best_draw):
    global depth_image
    try:
        cdistance=depth_image[int(best_draw[2]+best_draw[0])//2][int(best_draw[3]+best_draw[1])//2]#获得目标物中心点的深度
        cdistance=cdistance/1000
        print('the center of distace is',cdistance)
        return cdistance

    except:
        return 0


if __name__ == "__main__":
    weight = 'weights/ssd_mobilenetv2/mobilenetv2_final.pth'
    #path = r"test_images/example.jpg"
    #path = r"test_videos/test.avi"
    # 确定传入的信息
    bus = smbus.SMBus(1)
    addr = 0x54
    recount_th0 = 0x30
    bus.write_byte_data(addr, recount_th0, 0x03)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    detection_video(pipeline,weight)
