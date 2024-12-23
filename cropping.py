import os
import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt

from ultralytics import YOLO
# this version is for system, so we use mode "omr"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RepeatCropping():
    def __init__(self, piece, msmd_test=None, msmd_test_repeat=None, image_path=None, model_path=None, mode="gt", save=None, progress=None):
        # path
        self.msmd_test = msmd_test
        self.gt_path = msmd_test_repeat
        self.image_path = image_path
        self.piece = piece
        self.save = save
        self.mode = mode # gt/omr
        # 
        
        self.progress = progress
        # param
        self.cls_dict = {0: "repeat_start", 1: "repeat_end", 4: "nth_ending", 7: "finalbarline", 8: "none"}
        self.repeat_data = {}
        
        
        # model
        if mode == "omr":
            if os.path.isfile(model_path):
                self.model = YOLO(model_path) 
            else:
                print("Invalid path!")
            self.scores, self.images_path = self.load_images()
        else:
            self.scores = self.load_scores()
        
        self.edge, self.system_id = self.get_edge()
        
        if self.progress != None:
            self.progress(20)
            
        self.repeat_omr()
        
        self.cropping()
        if self.progress != None:
            self.progress(100)
        
        
    def load_scores(self):
        npzfile = np.load(os.path.join(self.msmd_test, self.piece+'.npz'), allow_pickle=True)
        # print(npzfile["sheets"].shape, npzfile["sheets"][0].shape)
        return npzfile["sheets"]
    
    def load_images(self):
        images = []
        images_path = []
        for i in os.listdir(self.image_path):
            path = os.path.join(self.image_path, i)
            image = cv.imread(path)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # # cv.imshow("1", image)
            # # cv.waitKey(0)
            images.append(image)
            images_path.append(path)
        images = np.array(images)
        # print(images.shape)
        return images, images_path
    
    def get_edge(self):
        edge_list = []
        system_id = {}
        c = 0
        for page, p in enumerate(self.scores):
            sysid_tmp = {}
            tmp = []
            # binary
            p = p.astype(np.float32) / 255.0
            p = (p < 0.9).astype(int)
            p = np.sum(p, axis=1)

            ave = p.shape[0] / 2
            p = (p > ave).astype(int)
            
                
            flag = False
            r = []
            for i, pp in enumerate(p):
                if not flag and pp == 1:
                    flag = True
                    r.append(i)
                elif flag and pp == 0:
                    flag = False
                    r.append(i)
                    tmp += [r]
                    r = []
                    
            # print(tmp, len(tmp))
            s = -1
            e = -1
            
            for i, pp in enumerate(tmp):
                # print(i, pp)
                if i%10 == 0:
                    s = pp[0]
                elif i%10 == 9:
                    e = pp[-1]
                # print(s, e)
                if s != -1 and e != -1:
                    # print(s, e)
                    p[s:e] = np.zeros_like(p[s:e])
                    p[e] = 1
                    
                    sysid_tmp[c] = [s, e, ]
                    c += 1
                    s = -1
                    e = -1
            system_id[page] = sysid_tmp
            # print(system_id)
            # print(p, len(p))
            edge = np.where(p == 1)[0].astype(float)/len(p)
            # print(edge, len(p))
            
            # print(p.shape)
            edge_list.append(edge)
            
        # lick wrong edge
        new_edge_list = []
        for edge in edge_list:
            tmp_list = [edge[0] - 0]
            for i in range(len(edge)-1):
                tmp_list.append(edge[i+1] - edge[i])
            tmp_list = np.array(tmp_list)
            ave = np.mean(tmp_list) / 2
           
            new_edge_list.append(np.array(edge[tmp_list > ave]))

        return new_edge_list, system_id
    
    def repeat_omr(self):
        # get bbox position from gt or omr
        for page in range(self.scores.shape[0]):
            if self.progress != None:
                self.progress(80//self.scores.shape[0] * (page + 1))
            # put image to omr model
            # if self.mode == "gt":
            #     # print(img.shape)
            #     # Currently, use GT info cropping
            #     if os.path.isfile(os.path.join(self.gt_path, f"{self.piece}_{page}.txt")):
            #         with open(os.path.join(self.gt_path, f"{self.piece}_{page}.txt"), 'r') as file:
            #             for line in file:
            #                 class_id, x, y, w, h = line.split()  
            #                 # print(line.split())
            #                 if self.cls_dict.get(int(class_id)) == None:
            #                     # print(int(class_id))
            #                     return 1
                            
            #                 if self.repeat_data.get(page) == None:
            #                     self.repeat_data[page] = [[self.cls_dict[int(class_id)], page, float(x), float(y), float(w), float(h)]]
            #                 else:
            #                     self.repeat_data[page].append([self.cls_dict[int(class_id)], page, float(x), float(y), float(w), float(h)])
            #     # print(self.repeat_data)
            # elif self.mode == "omr":
            # print(self.images_path)
            threshold = 0.485
            results = self.model.predict(self.images_path[page], imgsz=640, classes=[0, 1, 4, 7], conf=threshold, verbose=False)
            sh, sw = self.scores[page].shape

            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_cls = results[0].boxes.cls.cpu().numpy()
            # print(page, type(pred_boxes), pred_boxes, pred_boxes.size)
            if pred_boxes.size == 0:
                if self.repeat_data.get(page) == None:
                    self.repeat_data[page] = []
                # print(self.repeat_data[page])
            else:
                for class_id, pred_box in zip(pred_cls, pred_boxes):
                    # print(class_id, pred_box, self.cls_dict[int(class_id)])
                    x1, y1, x2, y2 = pred_box
                    x = (x1 + x2)/2
                    y = (y1 + y2)/2
                    w = abs(x1 - x2)
                    h = abs(y1 - y2)
                    # normalize
                    x /= sw
                    w /= sw
                    y /= sh
                    h /= sh

                    if self.repeat_data.get(page) == None:
                        self.repeat_data[page] = [[self.cls_dict[int(class_id)], page, float(x), float(y), float(w), float(h)]]
                    else:
                        self.repeat_data[page].append([self.cls_dict[int(class_id)], page, float(x), float(y), float(w), float(h)])

    def repeat_sorting(self):
        # get order bbox dict
        for k, v in self.repeat_data.items():
            tmp1 = []
            
            for scanned_h in self.edge[k]:
                tmp2 = []
                for data in v:
                    if data[3] < scanned_h and data not in tmp1 and data not in tmp2:
                        tmp2.append(data)
                if tmp2 != []:
                    
                    tmp2 = sorted(tmp2, key=lambda x: (x[2]))
                    tmp1 += tmp2
            
            # print(tmp1)
            self.repeat_data[k] = tmp1
        # get order bbox list
        
        for k, v in self.repeat_data.items():
            self.repeat_list += v
        self.repeat_list = [['head', 0, 0.0, 0.0, 0.0, 0.0]] + self.repeat_list
        if self.repeat_list[-1][0] == "finalbarline":
            self.repeat_list[-1] = ['end', self.scores.shape[0]-1, 1.0, 1.0, 0.0, 0.0]
        else:
            self.repeat_list += [['end', self.scores.shape[0]-1, 1.0, 1.0, 0.0, 0.0]]
        return -1
       
    def repeat_paring(self):
        semantic = []
        start = self.repeat_list[0]
        tmp = self.repeat_list[0]

        skip = None
        for i, sign_info in enumerate(self.repeat_list):
            sign, p, x, y, w, h = sign_info
            if sign == 'end':
                semantic.append([start, sign_info, skip])
                if skip != None:
                    skip = None
            elif sign == 'repeat_start':
                tmp = sign_info  
            elif sign == 'finalbarline':
                tmp = sign_info  
            elif sign == 'repeat_end':
                semantic.append([start, sign_info, skip])
                
                if skip != None:
                    skip = None
                if self.repeat_list[i-1][0] == "nth_ending":
                    sign_, p_, x_, y_, w_, h_ = self.repeat_list[i-1]
                    skip = [sign_, p_, x_, y, w_, h]
                start = tmp
      
        # mask_list = [] # page, s[x, y, w, h], e[x, y, w, h], skip [x, y, w, h]/None
        for se in semantic:
            
            s, e, skip = se
            if s[1] != e[1]:
                for i in range(s[1], e[1]+1):
                    if skip != None and skip[1] == i:
                        tmp = skip[2:]
                    else:
                        tmp = None
                    if i == s[1]:
                        
                        self.mask_list.append([i, 
                                [s[2], s[3], s[4], s[5]], 
                                [1.0, 1.0, 0.0, 0.0], 
                                tmp])
                    elif i == e[1]:
                        self.mask_list.append([i, 
                                [0.0, 0.0, 0.0, 0.0], 
                                [e[2], e[3], e[4], e[5]], 
                                tmp])
                    else:
                        self.mask_list.append([i, 
                                [0.0, 0.0, 0.0, 0.0], 
                                [1.0, 1.0, 0.0, 0.0],
                                tmp])
                        
            else:
                assert s[1] == e[1]
                if skip != None:
                    tmp = skip[2:]
                else:
                    tmp = None
                self.mask_list.append([s[1], 
                                [s[2], s[3], s[4], s[5]], 
                                [e[2], e[3], e[4], e[5]], 
                                tmp])
    
    def gen_mask(self):
        offset = 25
        scoremask_list = []
        
        
        if self.save != None:
            for i in os.listdir(self.save):
                if str(i).endswith(".jpg"):
                    os.remove(os.path.join(self.save, i))
            
        for p, mask in enumerate(self.mask_list):
            page, start, end, skip = mask
            xs, ys, ws, hs = start
            # print(start, end)
            xe, ye, we, he = end
            
            image_gray = self.scores[page]
        
            scoremask = np.ones_like(image_gray, dtype=np.uint8)
            
            image = cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR)
            
            h, w, n = image.shape
            # print( h, w, n )
            # left top
            image = cv.rectangle(image, (0, 0), (w, int((ys-hs/2) * h)-offset), (255, 255, 255), -1)
            image = cv.rectangle(image, (0, 0), (int((xs-ws/2) * w), int((ys+hs/2) * h)+offset), (255, 255, 255), -1)
            
            image = cv.rectangle(image, (0, int((ye+he/2) * h)+offset), (w, h), (255, 255, 255), -1)
            image = cv.rectangle(image, (int((xe+we/2) * w), int((ye-he/2) * h)-offset), (w,  h), (255, 255, 255), -1)
            
            score_h = int((ys - hs / 2) * h) - offset if int((ys - hs / 2) * h) - offset > 0 else 0
            score_w = w
        
            scoremask[0: score_h, 0:score_w] = 0
     
            score_h = int((ys + hs / 2) * h) + offset if int((ys + hs / 2) * h) + offset > 0 else 0
            score_w = int((xs - ws / 2) * w) if int((xs - ws / 2) * w) > 0 else 0
            scoremask[0: score_h, 0:score_w] = 0
       
            score_h = int((ye + he / 2) * h) + offset if int((ye + he / 2) * h) + offset > 0 else 0
            score_w = w
            scoremask[score_h:h, 0:score_w] = 0
         
            
            score_h = int((ye - he / 2) * h) - offset if int((ye - he / 2) * h) - offset > 0 else 0
            score_w = int((xe + we / 2) * w) if int((xe + we / 2) * w) > 0 else 0
            scoremask[score_h:h , score_w:w] = 0
                   
            # skip
            if skip != None:
                xskip, yskip, wskip, hskip = skip
                image = cv.rectangle(image, (int((xskip-wskip/2) * w), int((yskip-hskip/2) * h) - offset), (int((xskip+wskip/2) * w), int((yskip+hskip/2) * h) + offset), (255, 255, 255), -1)
                
                score_h1 = int((yskip - hskip / 2) * h) - offset if int((yskip - hskip / 2) * h) - offset > 0 else 0
                score_h2 = int((yskip + hskip / 2) * h) + offset if int((yskip + hskip / 2) * h) + offset > 0 else 0
                score_w1 = int((xskip - wskip / 2) * w) if int((xskip - wskip / 2) * w) > 0 else 0
                score_w2 = int((xskip + wskip / 2) * w) if int((xskip + wskip / 2) * w) > 0 else 0
                scoremask[score_h1:score_h2 , score_w1:score_w2] = 0
            
            # plt.imshow(scoremask)
            # plt.show()
            # print(page, scoremask)
            scoremask_list.append([page, scoremask])
            # print(scoremask.shape, type(scoremask), scoremask)
            # plt.imshow(scoremask, cmap="gray")
            # plt.show()
            # input()
            # print(start, end, image.shape)
            # h, w, n = 
            # image = cv.resize(image, (600, 800))
            
            # cv.imshow("masked_img", image)
            if self.save != None:
                
                cv.imwrite(os.path.join(self.save, f'{self.piece}_{p}.jpg'), image)
            # cv.waitKey(0)
        # cv.destroyAllWindows()
        return scoremask_list

    def cropping(self):
        self.mask_list = []
        self.repeat_list = []
        check = self.repeat_sorting()
        
        if check == 0:
            print(piece, "--> Somthing Wrong in Cropping Setting")
        elif check == 1:
            print(piece, "--> Somthing Wrong in gt->can't handel repeat type")
        else:

            self.repeat_paring()
            
            
            self.score_mask = self.gen_mask()
        
        
if "__main__" == __name__:
    piece = []
    for i in os.listdir(r"D:\scorefollowersystem\cyolo_score_following\data\msmd\repeat_subset\test\labels"):
        if i[:-6] not in piece:
            piece.append(i[:-6])
        # print(i[:-6])
    piece.sort()
    print(len(piece))
    total_page = 0
    for id, i in enumerate(piece):
        
        repeatcropping = RepeatCropping(msmd_test = r"D:\scorefollowersystem\cyolo_score_following\data\msmd\msmd_test",
                                        msmd_test_repeat = r"D:\scorefollowersystem\cyolo_score_following\data\msmd\repeat_subset\msmd_test_image",
                                        image_path=r"D:\scorefollowersystem\cyolo_score_following\userinterface\test_score",
                                        model_path=r"D:\scorefollowersystem\cyolo_score_following\userinterface\omr\best.pt",
                                        mode="omr",
                                        piece = i,
                                        save = r"D:\scorefollowersystem\cyolo_score_following\userinterface\cropping_score"
                                        )
        total_page += int(repeatcropping.scores.shape[0])
        input()
        print(id, i, total_page)