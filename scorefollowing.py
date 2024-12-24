import numpy as np
import torch
import torch

from collections import deque
from tqdm import tqdm
from utils.utils import *
from utils.audio_stream import AudioStream
from PyQt5.QtCore import *



class ScoreFollowing(QThread):
    update_data = pyqtSignal(dict)
    
    def __init__(self, model_path, score_dir, crop_dir, audio_dir, piece_name, cropping_info, mode="fullpage", level="note", motif_label=None):
        super().__init__()
        self.param_path = model_path
        self.path = score_dir
        self.crop_path = crop_dir
        self.audio_path = audio_dir
        self.piece_name = piece_name
        self.cropping_info = cropping_info
        # motif
        self.motif_label = motif_label
        
        self.mode = mode
        
        if level == "note":
            # note level
            self.class_idx = 0
        else:
            # Bar
            self.class_idx = 1
        
        self.init_setting()
        
        self.init_model()
    
    def init_setting(self):
        # run status
        
        self.is_piece_end = False
        
        self.hidden = None
        self.signal = deque(np.zeros(FRAME_SIZE), maxlen=FRAME_SIZE)
        
        
        self.actual_page = 0
        self.current_system = 0
        
        # audio file or stream
        self.audio_stream = AudioStream(SAMPLE_RATE, HOP_SIZE*2, self.audio_path)
            
        # autopageturning
        self.th_len = 5
        self.curr_y = deque(np.zeros(self.th_len), maxlen=self.th_len)
        self.curr_x = deque(np.zeros(self.th_len), maxlen=self.th_len)
        self.mean_x = 0
        self.mean_y = 0
        self.bipage_count = 40
        self.cooldown = 0
        
        self.org_scores, self.score, self.crop_org_scores, self.crop_score, self.pad, self.scale_factor = load_piece_for_inference(416, self.mode, self.path, self.piece_name, self.crop_path)
        
    def init_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = load_pretrained_model(self.param_path, self.mode)

        # print(network)
        # print("Putting model to %s ..." % device)
        self.network.to(self.device)
        # print("Number of parameters:", sum(p.numel() for p in self.network.parameters() if p.requires_grad))
        self.network.eval()
        # pdf
        self.score_tensor = torch.from_numpy(self.crop_score).unsqueeze(1).to(self.device)
        
        self.vis_spec = None
        # print(self.crop_org_scores[0].shape) # h, w, n
        if self.mode == "fullpage":
            self.w = self.crop_org_scores[0].shape[1]
        else:
            self.w = self.crop_org_scores[0].shape[1]//2
            self.count = 0
       
    def run(self):
        
        # if self.audio_path:
        #     pbar = tqdm(total=len(self.audio_stream))

        while not self.is_piece_end:
            
            frame = self.audio_stream.get()
            if frame is None:
                break
            self.signal.extend(frame)

            self.autopageturning()
            
            with torch.no_grad():

                sig_excerpt = torch.from_numpy(np.array(self.signal, dtype=np.float64)).float().to(self.device)
                # plt.plot(list(signal))
                spec_frame = self.network.compute_spec([sig_excerpt], tempo_aug=False)[0]

                z, self.hidden = self.network.conditioning_network.get_conditioning(spec_frame, hidden=self.hidden)
    
                t = self.score_tensor[self.actual_page:self.actual_page+1]
          
                # t[:, :, 200:,  :] = 0
                inference_out, pred = self.network.predict(t, z)
                # print(score_tensor[actual_page:actual_page+1].shape, org_scores[actual_page].shape)

            x1, y1, x2, y2 = [], [], [], []
            filtered_inference_out = inference_out[0, inference_out[0, :, -1] == self.class_idx].unsqueeze(0)
            # print(filtered_inference_out.shape)
            _, idx = torch.max(filtered_inference_out[0, :, 4], dim=0)
            box = filtered_inference_out[0, idx, :4].unsqueeze(0)
            x1_, y1_, x2_, y2_ = xywh2xyxy(box).cpu().numpy().T

            if self.mode == "fullpage":
                x1 = x1_ * self.scale_factor - self.pad
                x2 = x2_ * self.scale_factor - self.pad
                y1 = y1_ * self.scale_factor
                y2 = y2_ * self.scale_factor
            else:
                x1 = x1_ * self.scale_factor
                x2 = x2_ * self.scale_factor
                y1 = y1_ * self.scale_factor - self.pad
                y2 = y2_ * self.scale_factor - self.pad

            tmp_x = x1 + (x2 - x1) / 2
            tmp_y = y1 + (y2 - y1) / 2
       
            self.curr_x.append(tmp_x)
            self.curr_y.append(tmp_y)    
            
            # Spectrogram
            if self.vis_spec is not None:
                self.vis_spec = np.roll(self.vis_spec, -1, axis=1)
            else:
                self.vis_spec = np.zeros((spec_frame.shape[-1], 40))

            self.vis_spec[:, -1] = spec_frame[0].cpu().numpy()
            
            # Motif IoU
            self.mean_x = np.mean(self.curr_x)
            self.mean_y = np.mean(self.curr_y)
            
            if type(self.mean_x) == np.float32 and type(self.mean_y) == np.float32:
                mean_pos = [self.mean_x, self.mean_y, 20, 100]
                status, self.class_id = self.motif_iou( self.motif_label, 
                                                        mean_pos, 
                                                        self.crop_org_scores[self.actual_page].shape)
            else:
            # print(mean_pos)
                status, self.class_id = False, "None"
            # 
            # To PyQT UI
            self.update_data.emit({"value" : 0,
                                    "predict" : [self.mean_x, self.mean_y],
                                    "signal": frame,
                                    "spec": self.vis_spec,
                                    "system_id": self.current_system,
                                    "motif_status": status,
                                    "motif_id": self.class_id,
                                    "score_page": self.cropping_info[0][self.actual_page][0],
                                    "masked_score_page": self.actual_page, 
                                    "turning": self.cooldown != 0,
                                    "masked_score": self.crop_org_scores[self.actual_page][::2, ::2, :],
                                    "score": self.org_scores[self.cropping_info[0][self.actual_page][0]]})

    def find_last_system(self):

        mask = self.cropping_info[0][self.actual_page][-1]
        actual_page = self.cropping_info[0][self.actual_page][0] 
        tmp2 = np.sum(mask, axis=1)
        idx = np.nonzero(tmp2)[0][-1]
        system_list = SCORE_HEIGHT * self.cropping_info[2][actual_page]
        # print(mask.shape, idx, self.cropping_info[2][actual_page])
        self.current_system  = find_system_edge(self.mean_y, system_list)
        last_system = find_system_edge(idx, system_list)
        if last_system == 0:
            y1, y2 = 0, int(system_list[last_system])
        else:
            y1, y2 = int(system_list[last_system-1]), int(system_list[last_system])
        # print(y1, y2)
        
        width = np.mean(np.sum(mask[y1:y2, :], axis=1))
        

        if width < SCORE_WIDTH * 0.25 and last_system > 1:
            y1, y2 = int(system_list[last_system-2]), int(system_list[last_system-1])
            width = np.mean(np.sum(mask[y1:y2, :], axis=1))
        # print("!", y1, y2 , int(width),np.mean(self.curr_x), np.mean(self.curr_y), system_list)
        return [y1, y2] , width
            
    def autopageturning(self):
        
        
        in_last_system = False
        in_range = False
        
        system_ys, width = self.find_last_system()
        # if self.class_idx == 0:
        in_range = width > np.mean(self.curr_x) > PAGE_TURNING_THRESHOLD * width
        # else:
        #     in_range = width > np.mean(self.curr_x) > PAGE_TURNING_THRESHOLD2 * width
        in_last_system = system_ys[0] <= np.mean(self.curr_y) <= system_ys[1]

        
        if self.actual_page + 1 < self.score_tensor.shape[0]:
            
            if in_last_system and in_range and self.cooldown == 0 :
                print("Turning...", self.actual_page, self.cropping_info[0][self.actual_page][0])
                print(np.mean(self.curr_x), np.mean(self.curr_y),  PAGE_TURNING_THRESHOLD * width)
                # print(len(self.crop_org_scores))
                # print(self.crop_org_scores[self.actual_page].shape)
                # print(self.motif_label)
                
                self.cooldown = COOLDOWN
                self.hidden = None
                self.curr_y = deque(np.zeros(self.th_len), maxlen=self.th_len)
                self.curr_x = deque(np.zeros(self.th_len), maxlen=self.th_len)
                self.actual_page += 1
        
        if self.cooldown > 0 :
            self.hidden = None
            self.cooldown -= 1
              
    def stop_playing(self):
        self.is_piece_end = True
        self.audio_stream.close()

       
    def motif_iou(self, motifs, position, size):
        # print("!", motifs)
        if motifs != []:
            motif = motifs[self.cropping_info[0][self.actual_page][0]]
            if motif != []:
                # print(motif)
                
                h, w, n = size
                motif = np.array(motif)
                position = np.array(position)
                position = np.tile(position, (motif.shape[0], 1)).astype(float)

                label = motif[:, 0]
                motif = motif[:, 1:].astype(float)
                # print(label, motif, position)
                motif[:, ::2] *= w
                motif[:, 1::2] *= h
                
                metrics = np.diag(box_iou(motif, position)) * 100
                # print(metrics, np.argmax(metrics))
                
                if np.sum(metrics) > 0.0:
                    # print(motif.shape, position.shape, metrics)
                    # print(np.argmax(metrics), label)
                    return True, label[np.argmax(metrics)]
                else: 
                    return False, "None"
        
        return False, "None"
        
   