import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import time

import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords,set_logging

from utils.torch_utils import select_device, time_synchronized

from tracker.matching import linear_assignment
from tracker.bytetrack import ByteTrack
from tracker.deepsort_reid import Extractor
import numpy as np
import cv2 as cv
import json

def reid_distance(bboxs, targets):
    if not bboxs.size:
        return np.empty((0, len(targets)))
    if not targets.size:
        return np.empty((len(bboxs), 0))
    bboxs = bboxs[:, None, ...].repeat(targets.shape[0], axis=1)
    targets = targets[None, ...].repeat(bboxs.shape[0], axis=0)  
    return np.sum(np.abs(targets-bboxs), axis=-1)
    
class Trackconfig:
    img_size = 1080
    conf_thresh = 0.2
    nms_thresh = 0.7
    iou_thresh = 0.3
    track_buffer = 30
    gamma = 0.9
    kalman_format = 'default'
    min_area = 150
    reid_model_path = r'weight/ckpt.t7'
    targets_img_path = r'tracker/test'
    reids_path = r'tracker/reid'
    cuda = True
    reid_min_thres = 11





def detect(s):
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    # generate reid
    '''reid_net = Extractor(Trackconfig.reid_model_path, use_cuda=Trackconfig.cuda)
    targets = os.listdir(Trackconfig.targets_img_path)
    targets = list(filter(lambda x:x.endswith(('jpg', 'png')), targets))
    targets.sort()
    targets_imgs = [cv.imread(os.path.join(Trackconfig.targets_img_path, x)) for x in targets]
    targets_reids = reid_net(targets_imgs) 
    del reid_net
    if Trackconfig.cuda:
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()'''

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # fifo pipe
    fd_cpp2py = os.open("/home/zxj/Desktop/cpp2py", os.O_RDONLY)
    fd_py2cpp = os.open("/home/zxj/Desktop/py2cpp", os.O_WRONLY)

    # track init
    track = ByteTrack(Trackconfig, use_reid=True)
    track2target_pair = {}
    target2track_pair = {}

    # init target reid
    targets_reids = np.empty((0, 512))

    while True:
        #dataset = LoadImages(source, img_size=imgsz, stride=stride)
        data = os.read(fd_cpp2py, 99999999)
        im0s = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR)
        img = letterbox(im0s, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # load target reids
        if len(os.listdir(Trackconfig.reids_path)) != targets_reids.shape[0]:
            reid_names = os.listdir(Trackconfig.reids_path)
            targets_reids = np.array([np.load(os.path.join(Trackconfig.reids_path, x)) for x in reid_names])

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()

        '''try:
            path, img, im0s, vid_cap = next(iter(dataset))
        except: continue'''

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        det = pred[0].cpu()
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        det.numpy()
        tracked, lost, removed = track.update(det, im0s)

        # reid distances
        reid = np.array([x.features[-1] for x in tracked])
        tracked_ids = [x.track_id for x in tracked]
        distances = reid_distance(reid, targets_reids)

        # Prevents linear assignment of already paired targets and tracks
        for i in target2track_pair:
            distances[:, int(i)] = np.inf
        for i, j in enumerate(tracked_ids):
            if str(j) in track2target_pair:
                distances[int(i), :] = np.inf

        # linear assignment
        match_pairs, _, _ = linear_assignment(distances, Trackconfig.reid_min_thres)
        for pair in match_pairs:
            trkid, targetid = tracked[pair[0]].track_id, pair[1]
            track2target_pair[str(trkid)] = targetid
            target2track_pair[str(targetid)] = trkid

        
        s = json.dump({'size':[1920, 1080]})
        if len(tracked) != 0:
            bboxs = []
            for i, trk in enumerate(tracked):
                trk_id, tl, br = trk.track_id, trk.tlbr[:2].astype(np.int16), trk.tlbr[2:4].astype(np.int16)
                if str(trk_id) in track2target_pair:
                    color = (0, 0, 255)
                    colorid = track2target_pair[str(trk_id)]+1
                    text = 'Target'
                else:
                    color = (0, 255, 0)
                    colorid = 0
                    text = f'people-{trk_id}'
                    #text = str(dist[i])[:4]
                x, y, w, h = int((br[0]+tl[0])/2), int((br[1]+tl[1])/2),int(br[0]-tl[0]), int(br[1]-tl[1])
                bboxs.append([x, y, w, h, colorid])
            s += json.dump({'bboxs':bboxs}) 
                        
            '''
            cv.rectangle(im0s, tl, br, color=color, thickness=2)
            cv.putText(im0s, text, tl, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1,
                        color=color, thickness=2)
        
        cv.imshow("fig", im0s)
        cv.waitKey(10)
        '''
        s += '\0'
        os.write(fd_py2cpp, bytes(s, encoding='utf-8'))


    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='v5lite-e.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'/home/zxj/Desktop/input.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        detect()
