import sys
import os
import numpy as np
import json
import pickle
import cv2 as cv
from natsort import natsorted

from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from cropping import RepeatCropping
from scorefollowing import ScoreFollowing
from utils.utils import prepare_spec_for_render, plot_box, create_video, CLASS_MAPPING, COLORS, FPS, SAMPLE_RATE, FRAME_SIZE
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System - Main")
        self.width = 1080
        self.height = 960
        
        self.fonttitle = QFont("Arial", 28, QFont.Bold)  # 字體名稱, 字體大小, 粗體
        self.font = QFont("Arial", 14, QFont.Bold)  # 字體名稱, 字體大小, 粗體
        self.font2 = QFont("Arial", 12)  # 字體名稱, 字體大小
        
        self.image_list = [] 
        self.current_image_index = 0 
        self.current_check_index = 0
        self.current_adjust_index = 0
        # motif annotation
        self.annotations = [] 
        
        self.tmp_btn = None
        self.current_bbox = None
        self.labeling = False
        self.rpcropping = None
        self.sf = None
        self.truncate_signal = np.zeros(2 * FRAME_SIZE)
        self.save_video = []

        self.setFixedSize(self.width, self.height)
    
        self.init_button_color()
        self.init_config()
        self.init_pages()

    def init_button_color(self):
        self.light_blue_button = """
            QPushButton {
                background-color: lightblue;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-family: "Arial";
            }
            QPushButton:pressed {
                background-color: deepskyblue;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-family: "Arial";
            }
        """
        self.dark_blue_button = """
            QPushButton {
                background-color: deepskyblue;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-weight: bold;
                font-family: "Arial";
            }
        """
        self.light_gray_button = """
            QPushButton {
                background-color: lightgray;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-family: "Arial";
            }
            QPushButton:pressed {
                background-color: darkgray;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-family: "Arial";
            }
        """
        self.pink_button = """
            QPushButton {
                background-color: pink;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-family: "Arial";
            }
            QPushButton:pressed {
                background-color: hotpink;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-family: "Arial";
            }
        """
        self.hot_pink_button = """
            QPushButton {
                background-color: hotpink;
                border-radius: 20px;
                padding: 20px;
                font-size: 18px;
                font-weight: bold;
                font-family: "Arial";
            }
        """
        self.combo_box_color = """
            QComboBox {
                background-color: lightblue;
                border: 1px solid gray;
                border-radius: 20px;
                padding: 20px;
                font-size: 16px;    
                font-family: "Arial";
            }
           
        """
    
    def init_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "default.json"), "r") as file:
            data = json.load(file)
            # print(data)
            self.motiflbl_path_label = QLabel(os.path.join(current_dir, data["MotifLabel"]))
            self.audio_path_label = QLabel(os.path.join(current_dir, data["Audio"])) 
            self.piece_path_label = QLabel(os.path.join(current_dir, data["Piece"])) 
            self.cropping_piece_path_label = QLabel(os.path.join(current_dir, data["CroppingPiece"]))
            self.save_path_label = QLabel(os.path.join(current_dir, data["Save"]))
            self.omr_path_label = QLabel(os.path.join(current_dir, data["OMRModel"]))
            self.sf_path_label = QLabel(os.path.join(current_dir, data["ScoreFollowingModel"]))
            
            sublist = []
            for i in os.listdir(self.motiflbl_path_label.text()):
                sublist.append(i)
            for i in os.listdir(self.audio_path_label.text()):
                sublist.append(i)
            for i in os.listdir(self.piece_path_label.text()):
                sublist.append(i)
            for i in os.listdir(self.cropping_piece_path_label.text()):
                sublist.append(i)
            for i in os.listdir(self.save_path_label.text()):
                sublist.append(i)
            
            sublist = list(set(sublist))
            for l in sublist:
                if l not in os.listdir(self.motiflbl_path_label.text()):
                    os.makedirs(os.path.join(self.motiflbl_path_label.text(), l) , exist_ok=True)
                if l not in os.listdir(self.audio_path_label.text()):
                    os.makedirs(os.path.join(self.audio_path_label.text(), l) , exist_ok=True)
                if l not in os.listdir(self.piece_path_label.text()):
                    os.makedirs(os.path.join(self.piece_path_label.text(), l) , exist_ok=True)
                if l not in os.listdir(self.cropping_piece_path_label.text()):
                    os.makedirs(os.path.join(self.cropping_piece_path_label.text(), l) , exist_ok=True)
                if l not in os.listdir(self.save_path_label.text()):
                    os.makedirs(os.path.join(self.save_path_label.text(), l) , exist_ok=True)
        
    def init_pages(self):
        self.init_main()
        self.init_motiflabeling()
        self.init_checking()
        self.init_adjusting()
        self.init_scorefollowing()
        
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.motif_labeling_page)
        self.stacked_widget.addWidget(self.masked_score_checking_page)
        self.stacked_widget.addWidget(self.masked_score_adjusting_page)
        self.stacked_widget.addWidget(self.score_following_page)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)    
            
    def init_main(self):
        # Main Page
        self.main_page = QWidget()
        layout = QVBoxLayout()
        
        self.main_page.setObjectName("System - Main")
         
        title = QLabel("Real-Time Score Following Using Score-Audio Synchronization for Music Motif Detection")
        
        title.setFont(self.fonttitle)
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        
        piece = QLabel("Pieces")
        piece.setFont(self.font)
        piece.setAlignment(Qt.AlignCenter)
        piece.setWordWrap(True)

        self.combo_box = QComboBox(self)
        self.combo_box.setFont(self.font2)
        self.piece_list = os.listdir(self.piece_path_label.text())
        self.combo_box.addItems(self.piece_list)
        
        
        piece_hlayout = QVBoxLayout()
        piece_hlayout.addWidget(piece)
        piece_hlayout.addStretch()
        piece_hlayout.addWidget(self.combo_box)
        
        # 3. Dir Path
        directory_label = QLabel("File Directory")
        directory_label.setAlignment(Qt.AlignCenter)
        directory_label.setFont(self.font)
    
        # 
        audio_button = QPushButton("Audio Files", self)
        audio_button.setFont(self.font2)
        audio_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        audio_button.clicked.connect(lambda: self.open_folder(self.audio_path_label.text()))
        # Piece
        piece_button = QPushButton("Pieces", self)
        piece_button.setFont(self.font2)
        piece_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        piece_button.clicked.connect(lambda: self.open_folder(self.piece_path_label.text()))
        # Save
        save_button = QPushButton("Saved Videos", self)
        save_button.setFont(self.font2)
        save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        save_button.clicked.connect(lambda: self.open_folder(self.save_path_label.text()))
        # 
        dirpath_hlayout = QHBoxLayout()
        dirpath_hlayout.addWidget(audio_button)
        dirpath_hlayout.addWidget(piece_button)
        dirpath_hlayout.addWidget(save_button)

        
        btn_right = QPushButton("Score Editing")
        btn_right.setStyleSheet(self.light_blue_button)
        btn_right.clicked.connect(lambda: self.change_page(self.motif_labeling_page))
        
        btn_jump = QPushButton("Score Following")
        btn_jump.setStyleSheet(self.pink_button)
        btn_jump.clicked.connect(lambda: self.change_page(self.score_following_page, addition="jump_check"))
        
        sub_hlayout = QHBoxLayout()
        sub_hlayout.addWidget(btn_right)
        sub_hlayout.addWidget(btn_jump)
        
        layout.addStretch()
        layout.addStretch()
        layout.addWidget(title)
        # layout.addStretch()
        # layout.addWidget(abstract)
        layout.addStretch()
        layout.addLayout(piece_hlayout)
        layout.addStretch()
        layout.addLayout(dirpath_hlayout)
        layout.addStretch()
        layout.addLayout(sub_hlayout)
        layout.addStretch()
        
        self.main_page.setLayout(layout)
        
    def init_motiflabeling(self):
        # Motif Labeling Page
        self.motif_labeling_page = QWidget()
        layout = QVBoxLayout()
        self.motif_labeling_page.setObjectName("System - Motif Labeling")
        
        title = QLabel("Score Motif Labeling")
        
        title.setFont(self.fonttitle)
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        # Left Layout
        left_layout = QVBoxLayout()
        # Label
        btn_label = QPushButton("Label")
        
        btn_label.setStyleSheet(self.light_blue_button)
        btn_label.clicked.connect(lambda: self.start_labeling(btn_label))
        # Remove
        btn_remove = QPushButton("Remove")
        btn_remove.setStyleSheet(self.pink_button)
        btn_remove.clicked.connect(self.remove_label)
        # Score Page
        scorepage_layout = QHBoxLayout()
        prev_button = QPushButton("Previous")
        prev_button.setFont(self.font2)
        prev_button.clicked.connect(self.show_prev_image)
        
        next_button = QPushButton("Next")
        next_button.setFont(self.font2)
        next_button.clicked.connect(self.show_next_image)
        scorepage_layout.addWidget(prev_button)
        scorepage_layout.addWidget(next_button)
        
        textlayout = QHBoxLayout()
        
        texttitle = QLabel("Motif Name")
        texttitle.setFont(self.font2)
        texttitle.setFixedHeight(30)
        texttitle.setAlignment(Qt.AlignCenter)
        # Text
        self.line_edit = QLineEdit(self)
        self.line_edit.setFixedHeight(30)
        self.line_edit.setEnabled(False)
        regex = QRegularExpression("[A-Za-z0-9_]+")
        validator = QRegularExpressionValidator(regex)
        self.line_edit.setValidator(validator)
        self.line_edit.returnPressed.connect(self.on_enter_pressed)
        
        textlayout.addWidget(texttitle)
        textlayout.addWidget(self.line_edit)
        # List
        self.list_widget = QListWidget()
        self.list_widget.setFixedHeight(200)
        self.list_widget.setFont(self.font2)
        self.list_widget.itemClicked.connect(lambda: self.update_image("labeling", self.list_widget.currentRow()))
        self.list_widget.itemClicked.connect(self.motif_naming)
        # Page Button
        btn_right = QPushButton("Generate Masked Score")
        btn_right.setStyleSheet(self.light_blue_button)
        btn_right.clicked.connect(lambda: self.change_page(self.masked_score_checking_page))
        
        self.progress = QProgressBar(self)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100) 
        self.progress.setValue(0) 
        
        left_layout.addWidget(btn_label)
        left_layout.addWidget(btn_remove)
        left_layout.addLayout(scorepage_layout)
        left_layout.addLayout(textlayout)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(btn_right)
        left_layout.addWidget(self.progress)
        
        # Right Layout
        right_layout = QVBoxLayout()
        # Image
        self.image_label = QLabel()
        self.image_label.setFixedSize(555, 785)
        self.image_label.setStyleSheet("background-color: lightgray")  
        self.pixmap = QPixmap(555, 785)
        self.pixmap.fill(Qt.white)
        self.image_label.setPixmap(self.pixmap)
        # Page number
        self.page_label = QLabel()
        self.page_label.setFixedSize(555, 30)
        self.page_label.setAlignment(Qt.AlignCenter)
        
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.page_label)
        # Both
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addLayout(left_layout)
        top_layout.addStretch()
        top_layout.addLayout(right_layout)
        top_layout.addStretch()
        
        layout.addWidget(title)
        layout.addLayout(top_layout)
        self.motif_labeling_page.setLayout(layout)
    
    def init_checking(self):
        # Masked Score Checking Page
        self.masked_score_checking_page = QWidget()
        self.masked_score_checking_page.setObjectName("System - Masked Score Checking")
        layout = QVBoxLayout()
        sub_layout = QHBoxLayout()
        
        
        title = QLabel("Masked Score Checking")
        
        title.setFont(self.fonttitle)
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        
        # Left Image
        left_title = QLabel("Origin")
        left_title.setFont(self.font2)
        left_title.setFixedSize(480, 20)
        left_title.setAlignment(Qt.AlignCenter)
        left_layout = QVBoxLayout()
        
        self.limage_label = QLabel() 
        self.limage_label.setFixedSize(480, 680)  
        self.limage_label.setStyleSheet("background-color: lightgray;")
        self.lcheckimg = QPixmap(480, 680)
        self.lcheckimg.fill(Qt.white)
        self.limage_label.setPixmap(self.lcheckimg)
        
        self.left_page_label = QLabel("x/x", self)
        self.left_page_label.setFont(self.font2)
        self.left_page_label.setFixedSize(480, 30)
        self.left_page_label.setAlignment(Qt.AlignCenter)
        
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.limage_label)
        left_layout.addWidget(self.left_page_label)
        
        
        # Right Image
        right_title = QLabel("Masked")
        right_title.setFont(self.font2)
        right_title.setFixedSize(480, 20)
        right_title.setAlignment(Qt.AlignCenter)
        right_layout = QVBoxLayout()
        
        self.rimage_label = QLabel()  
        self.rimage_label.setFixedSize(480, 680) 
        self.rimage_label.setStyleSheet("background-color: lightgray;") 
        self.rcheckimg = QPixmap(480, 680)
        self.rcheckimg.fill(Qt.white)
        self.rimage_label.setPixmap(self.rcheckimg)
        
        self.right_page_label = QLabel("x/x", self)
        self.right_page_label.setFont(self.font2)
        self.right_page_label.setFixedSize(480, 30)
        self.right_page_label.setAlignment(Qt.AlignCenter)
        
        right_layout.addWidget(right_title)
        right_layout.addWidget(self.rimage_label)
        right_layout.addWidget(self.right_page_label)
        
        sub_layout.addLayout(left_layout)
        sub_layout.addLayout(right_layout)
        # Score Check Page Button
        scorepage_layout = QHBoxLayout()
        prev_button = QPushButton("Previous Page")
        prev_button.setStyleSheet(self.light_gray_button)
        prev_button.clicked.connect(self.check_prev_image)
        
        adjust_button = QPushButton("Adjust")
        adjust_button.setStyleSheet(self.pink_button)
        adjust_button.clicked.connect(lambda: self.change_page(self.masked_score_adjusting_page))
        
        next_button = QPushButton("Next Page")
        next_button.setStyleSheet(self.light_gray_button)
        next_button.clicked.connect(self.check_next_image)
        
        scorepage_layout.addWidget(prev_button)
        scorepage_layout.addWidget(adjust_button)
        scorepage_layout.addWidget(next_button)
        # Status Button
        status_layout = QHBoxLayout()
        btn_left = QPushButton("Done")
        btn_left.setStyleSheet(self.light_blue_button)
        btn_left.clicked.connect(lambda: self.change_page(self.main_page, "save"))
        
        layout.addWidget(title)
        layout.addLayout(sub_layout)
        layout.addLayout(scorepage_layout)
        layout.addLayout(status_layout)
        layout.addWidget(btn_left)
        
        self.masked_score_checking_page.setLayout(layout)
    
    def init_adjusting(self):
        # Masked Score Adjusting Page
        self.masked_score_adjusting_page = QWidget()
        layout = QVBoxLayout()
        self.masked_score_adjusting_page.setObjectName("System - Masked Score Adjusting")
        
   
        title = QLabel("Masked Score Adjusting")
        
        title.setFont(self.fonttitle)
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        # Left
        left_layout = QVBoxLayout()
        # Remove
        btn_remove = QPushButton("Remove")
        btn_remove.setStyleSheet(self.pink_button)
        btn_remove.clicked.connect(self.remove_repeat_label)
        # Score Page
        scorepage_layout = QHBoxLayout()
        prev_button = QPushButton("Prev")
        prev_button.setFont(self.font2)
        prev_button.clicked.connect(self.adjust_prev_image)
        next_button = QPushButton("Next")
        next_button.setFont(self.font2)
        next_button.clicked.connect(self.adjust_next_image)
        scorepage_layout.addWidget(prev_button)
        scorepage_layout.addWidget(next_button)
        # Repeat List
        self.repeat_list = QListWidget()
        self.repeat_list.setFont(self.font2)
        self.repeat_list.itemClicked.connect(lambda: self.update_image("adjusting", self.repeat_list.currentRow()))
        # Repeat Class Button
        rp_start = QPushButton("Repeat Start")
        rp_start.setStyleSheet(self.light_blue_button)
        rp_start.clicked.connect(lambda: self.start_labeling(rp_start, "repeat_start"))
        
        rp_end = QPushButton("Repeat End")
        rp_end.setStyleSheet(self.light_blue_button)
        rp_end.clicked.connect(lambda: self.start_labeling(rp_end, "repeat_end"))
        
        rp_nthending = QPushButton("Nth Ending")
        rp_nthending.setStyleSheet(self.light_blue_button)
        rp_nthending.clicked.connect(lambda: self.start_labeling(rp_nthending, "nth_ending"))
        
        final_barline = QPushButton("Final Barline")
        final_barline.setStyleSheet(self.light_blue_button)
        final_barline.clicked.connect(lambda: self.start_labeling(final_barline, "finalbarline"))
        
        
        # left_layout.addWidget(btn_label)
        left_layout.addWidget(rp_start)
        left_layout.addWidget(rp_end)
        left_layout.addWidget(rp_nthending)
        left_layout.addWidget(final_barline)
        left_layout.addWidget(btn_remove)
        left_layout.addLayout(scorepage_layout)
        left_layout.addWidget(self.repeat_list)

        # Right
        right_layout = QVBoxLayout()
        # Image
        self.adjust_image_label = QLabel()
        self.adjust_image_label.setFixedSize(600, 850)
        self.adjust_image_label.setStyleSheet("background-color: lightgray")  
        self.adjustimg = QPixmap(600, 850)
        self.adjustimg.fill(Qt.white)
        self.adjust_image_label.setPixmap(self.adjustimg)
        # Page number
        self.adjust_page_label = QLabel("x/x", self)
        self.adjust_page_label.setFont(self.font2)
        self.adjust_page_label.setFixedSize(600, 30)
        self.adjust_page_label.setAlignment(Qt.AlignCenter)
        
        right_layout.addWidget(self.adjust_image_label)
        right_layout.addWidget(self.adjust_page_label)
        
        # Page Button
        btn_left = QPushButton("Done")
        btn_left.setStyleSheet(self.light_blue_button)
        btn_left.clicked.connect(lambda: self.change_page(self.masked_score_checking_page, "redo"))

        sub_vlayout = QVBoxLayout()
        sub_vlayout.addLayout(left_layout)
        sub_vlayout.addWidget(btn_left)
        sub_vlayout.addStretch()
        
        sub_layout = QHBoxLayout()
        
        sub_layout.addStretch()
        sub_layout.addLayout(sub_vlayout)
        sub_layout.addLayout(right_layout)
        sub_layout.addStretch()
        
        layout.addWidget(title)
        layout.addLayout(sub_layout)
        
        self.masked_score_adjusting_page.setLayout(layout)
        
    def init_scorefollowing(self):
        # Score Following Page
        self.score_following_page = QWidget()
        self.score_following_page.setObjectName("System - Score Following")
        self.layout = QVBoxLayout()

        self.setting_mode = False
        self.setting_list = []
        self.setting_list2 = []
        
        title = QLabel("Score Following")
        
        title.setFont(self.fonttitle)
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        
        # Left
        signal_layout = QHBoxLayout()
        self.motif_title = QLabel("Motif")
        self.motif_title.setFixedHeight(80)
        self.motif_title.setAlignment(Qt.AlignCenter)
        self.motif_title.setStyleSheet("background-color: darkgray; color: white; padding: 10px; font-size: 20px; font-family: 'Arial';")
        
        self.turning_title = QLabel("Turning")
        self.turning_title.setFixedHeight(80)
        self.turning_title.setAlignment(Qt.AlignCenter)
        self.turning_title.setStyleSheet("background-color: darkgray; color: white; padding: 10px; font-size: 20px; font-family: 'Arial';")
        
        signal_layout.addWidget(self.motif_title)
        signal_layout.addWidget(self.turning_title)
        
        self.setting_list2.append(self.motif_title)
        self.setting_list2.append(self.turning_title)
        
        self.masked_score_label = QLabel()  
        self.masked_score_label.setAlignment(Qt.AlignCenter)
        self.masked_score_label.setFixedSize(300, 400)
        self.masked_score_label.setStyleSheet("background-color: lightgray;")
        self.masked_scoreimg = QPixmap(300, 400)
        self.masked_scoreimg.fill(Qt.white)
        self.masked_score_label.setPixmap(self.masked_scoreimg)
        
        self.info = QTextEdit()
        self.info.append("Masked Score Page:")
        self.info.append("Score Page:")
        self.info.append("Score System:")
        self.info.append("Motif Status:")
        self.info.append("Motif Class:")
        
        self.info.setFont(self.font2)
        self.info.setFixedSize(300, 120)
        self.info.setReadOnly(True)
        
        self.btn_setting = QPushButton("Setting")
        self.btn_setting.setStyleSheet(self.pink_button)
        self.btn_setting.clicked.connect(self.scorefollowing_setting)
        
        ###############################
        # ButtonGroup
        empty1 = QLabel("")
        empty1.setFixedHeight(70)
        empty2 = QLabel("")
        empty2.setFixedHeight(65)
        signal_label = QLabel("Signal Source")
        signal_label.setAlignment(Qt.AlignCenter)
        signal_label.setFont(self.font)
        signal_label.setFixedHeight(80)
        
        self.radio_audio = QRadioButton("Audio", self)
        self.radio_audio.setFont(self.font2)
        self.radio_audio.setFixedHeight(80)
        self.radio_microphone = QRadioButton("Stream", self)
        self.radio_microphone.setFont(self.font2)
        self.radio_microphone.setFixedHeight(80)
        
        self.signal_group = QButtonGroup(self)
        self.signal_group.addButton(self.radio_audio)
        self.signal_group.addButton(self.radio_microphone)
        self.radio_audio.setChecked(True)
        
        signal_hlayout = QHBoxLayout()
        signal_hlayout.addWidget(signal_label)
        signal_hlayout.addStretch()
        signal_hlayout.addWidget(self.radio_audio)
        signal_hlayout.addStretch()
        signal_hlayout.addWidget(self.radio_microphone)

        
        self.setting_list.append(signal_label) 
        self.setting_list.append(self.radio_audio) 
        self.setting_list.append(self.radio_microphone) 
        # ButtonGroup

        level_label = QLabel("Granularty Level")
        level_label.setAlignment(Qt.AlignCenter)
        level_label.setFont(self.font)
        level_label.setFixedHeight(80)
        
        self.note_level = QRadioButton("Note", self)
        self.note_level.setFont(self.font2)
        self.note_level.setFixedHeight(80)
        self.bar_level = QRadioButton("Bar", self)
        self.bar_level.setFont(self.font2)
        self.bar_level.setFixedHeight(80)
        
        self.level_group = QButtonGroup(self)
        self.level_group.addButton(self.note_level)
        self.level_group.addButton(self.bar_level)
        self.note_level.setChecked(True)
        
        level_hlayout = QHBoxLayout()
        level_hlayout.addWidget(level_label)
        level_hlayout.addStretch()
        level_hlayout.addWidget(self.note_level)
        level_hlayout.addStretch()
        level_hlayout.addWidget(self.bar_level)
        level_hlayout.addStretch()
        
        self.score_following_setting = QVBoxLayout()
        self.score_following_setting.addWidget(empty1)
        self.score_following_setting.addLayout(signal_hlayout)
        self.score_following_setting.addLayout(level_hlayout)
        self.score_following_setting.addWidget(empty2)
        
        self.setting_list.append(empty1)
        self.setting_list.append(empty2)
        self.setting_list.append(level_label) 
        self.setting_list.append(self.note_level) 
        self.setting_list.append(self.bar_level) 
        
        for widget in self.setting_list:
            widget.setVisible(self.setting_mode)
            widget.hide()
        # ##########################
        
        self.btn_play_stop = QPushButton("Play")
        self.btn_play_stop.setStyleSheet(self.light_blue_button)
        self.btn_play_stop.clicked.connect(self.play_stop_music)
        
        self.sf_progress = QProgressBar(self)
        self.sf_progress.setTextVisible(False)
        self.sf_progress.setRange(0, 100)
        self.sf_progress.setValue(0)
        
        btn_right = QPushButton("Done")
        btn_right.setStyleSheet(self.light_gray_button)
        btn_right.clicked.connect(lambda: self.play_stop_music("Done"))
        btn_right.clicked.connect(lambda: self.change_page(self.main_page))
        
        
        page_layout = QVBoxLayout()
        page_layout.addStretch()
        page_layout.addLayout(signal_layout)
        page_layout.addWidget(self.info)
        page_layout.addLayout(self.score_following_setting)
        page_layout.addWidget(self.masked_score_label)
        page_layout.addWidget(self.btn_setting)
        page_layout.addWidget(self.btn_play_stop)
        
        page_layout.addWidget(self.sf_progress)
        page_layout.addWidget(btn_right)
        page_layout.addStretch()
        
        self.setting_list2.append(self.info)
        self.setting_list2.append(self.masked_score_label)
        self.setting_list2.append(self.btn_play_stop)
        self.setting_list2.append(self.sf_progress)
        self.setting_list2.append(btn_right)
        # Right
        self.score_label = QLabel()  
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFixedSize(715, 883)
        self.scoreimg = QPixmap(624, 883)
        self.scoreimg.fill(Qt.white)
        self.score_label.setPixmap(self.scoreimg)

        sublayout = QHBoxLayout()

        sublayout.addLayout(page_layout)
        sublayout.addStretch()
        sublayout.addWidget(self.score_label)

        self.setting_list2.append(self.score_label)
        self.layout.addWidget(title)
        self.layout.addLayout(sublayout)
        
        self.score_following_page.setLayout(self.layout)
        
    # -- UI function -- 
    def open_folder(self, folder_path):
        os.startfile(folder_path)  # 僅適用於 Windows
    
    def change_page(self, page, addition=None):
        # print(page.objectName(), addition, type(addition))
        turn = True
        if page.objectName() == "System - Main":
            self.current_image_index = 0 
            self.current_check_index = 0
            self.current_adjust_index = 0
            
            self.score_label.setPixmap(self.scoreimg)
            self.masked_score_label.setPixmap(self.masked_scoreimg)

            if addition == "save":
                self.crop_save()
                
        elif page.objectName() == "System - Demo":
            if addition == "save":
                self.ann_save()
                self.crop_save()

        elif page.objectName() == "System - Motif Labeling":
            self.select_image()
            self.progress.setValue(0)

        elif page.objectName() == "System - Masked Score Checking":
            self.ann_save()
            
            if addition == "redo":
                self.only_cropping()
            else:
                self.rpcropping = None    
                self.repeatcropping()  
                
            origin = self.rpcropping.score_mask[self.current_check_index][0]
            masked = self.current_check_index
            
            self.check_image(origin, masked)          
            
        elif page.objectName() == "System - Masked Score Adjusting":
            
            self.adjust_image(self.current_adjust_index)
            
        elif page.objectName() == "System - Score Following":
            self.sf_progress.setValue(0)
            if addition != None and addition == "jump_check":
                # check motif label
                if os.path.isfile(os.path.join(self.motiflbl_path_label.text(), self.combo_box.currentText(), "annotation.json")):
                    self.ann_load()
                else:
                    print("No Motif Label Annotation Exist!") 
                    turn = False  
                
                # check masked score cropping
                if os.path.isfile(os.path.join(self.cropping_piece_path_label.text(), self.combo_box.currentText(), 'cropping.pkl')):
                    self.crop_load()
                else:
                    print("No Cropping Image Exist!") 
                    turn = False 
                
            elif addition != None and addition != 100:
                turn = False
                    
            if turn:
                print(self.sf)
                self.crop_save()
                if self.sf == None:
                    self.score_following()
                else:
                    self.sf.init_setting()
                
        if turn:
            self.setWindowTitle(page.objectName())
            self.stacked_widget.setCurrentWidget(page)
    
    # motif_labeling
    def select_image(self):
        file_path = os.path.join(self.piece_path_label.text(), self.combo_box.currentText())
        
        if os.path.isdir(file_path):
            file_paths = [os.path.join(file_path, i) for i in os.listdir(file_path)]
            file_paths = natsorted(file_paths)
            
            self.image_list = file_paths
            
            self.ann_load()
            self.current_image_index = 0 
            self.load_image(self.current_image_index)  
            # self.update_image("labeling")
        else:
            print("Error")
       
    def load_image(self, index):
        if 0 <= index < len(self.image_list):
            self.pixmap = QPixmap(self.image_list[index])
            if not self.pixmap.isNull():
                self.update_image("labeling")
                
                self.update_annotation_display()   
                self.page_label.setText(f"{index+1}/{len(self.image_list)}")
    
    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.current_image_index)
           
    def show_next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_image(self.current_image_index)
            
    # labeling & adjusting
    def start_labeling(self, btn, name="none"):       
        self.finished_labeling()
        btn.setStyleSheet(self.dark_blue_button)
        self.tmp_btn = btn
        self.tmp_btnname = name
        self.labeling = True
    
    # motif_labeling
    def finished_labeling(self):
        if self.tmp_btn != None:
            self.tmp_btn.setStyleSheet(self.light_blue_button)
            self.tmp_btn = None
            self.tmp_btnname = None
            self.labeling = False

    def remove_label(self):
        current_row = self.list_widget.currentRow()
        # print(current_row)
        if current_row >= 0: 
            del self.annotations[self.current_image_index][current_row]
            self.list_widget.takeItem(current_row)
            
            self.update_image("labeling")  
        self.line_edit.setText("")
        self.line_edit.setEnabled(False)
        self.list_widget.clearSelection()
        self.list_widget.setCurrentRow(-1)
                  
    def on_enter_pressed(self):
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            
            input_text = self.line_edit.text() 
            self.annotations[self.current_image_index][current_row][0] = input_text
            self.update_annotation_display()
  
        self.line_edit.setEnabled(False)
        self.list_widget.clearSelection()
        self.list_widget.setCurrentRow(-1)
    
    def motif_naming(self):
        
        self.line_edit.setEnabled(True)
        
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            
            input_text = self.annotations[self.current_image_index][current_row][0]
            self.line_edit.setText(input_text) 
            
    def ann_save(self):
        with open(os.path.join(self.motiflbl_path_label.text(), self.combo_box.currentText(), 'annotation.json'), 'w') as file:
            json.dump(self.annotations, file)
    
    def ann_load(self):
        if os.path.isfile(os.path.join(self.motiflbl_path_label.text(), self.combo_box.currentText(), 'annotation.json')):
            with open(os.path.join(self.motiflbl_path_label.text(), self.combo_box.currentText(), 'annotation.json'), 'r') as file:
                self.annotations = json.load(file)
                    
        else:
            print(len(self.image_list))
            self.annotations = []
            for i in range(len(self.image_list)):
                self.annotations.append([])
            
    # labeling & adjusting
    def mousePressEvent(self, event):
        if self.labeling and event.button() == Qt.LeftButton:
            
            if self.stacked_widget.currentWidget() == self.motif_labeling_page and self.image_label.geometry().contains(event.pos()):
                scale = self.image_label.width() / self.pixmap.width()
                self.start_point = event.pos() - self.image_label.pos()
                self.start_point = QPoint(self.start_point.x() / scale, self.start_point.y() / scale)
                self.update_image("labeling")
            elif self.stacked_widget.currentWidget() == self.masked_score_adjusting_page and self.adjust_image_label.geometry().contains(event.pos()):
                self.start_point = event.pos() - self.adjust_image_label.pos()
                
                scale = self.adjust_image_label.width() / self.adjustimg.width()
                self.start_point = event.pos() - self.adjust_image_label.pos()
                self.start_point = QPoint(self.start_point.x() / scale,self.start_point.y() / scale)
                self.update_image("adjusting")
          
    def mouseMoveEvent(self, event):
        if self.labeling and event.buttons() == Qt.LeftButton:
            
            if self.stacked_widget.currentWidget() == self.motif_labeling_page and self.image_label.geometry().contains(event.pos()):
                scale = self.image_label.width() / self.pixmap.width()
                self.end_point = event.pos() - self.image_label.pos()
                # print(self.end_point, event.pos(), self.image_label.pos())
                self.end_point = QPoint(self.end_point.x() / scale, self.end_point.y() / scale)
                
                if self.start_point.x() < self.end_point.x():
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(self.start_point, self.end_point)
                    else:
                        self.current_bbox = QRect(QPoint(self.start_point.x(), self.end_point.y()), 
                                                  QPoint(self.end_point.x(), self.start_point.y()))
                else:
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.start_point.y()), 
                                                  QPoint(self.start_point.x(), self.end_point.y()))
                    else:
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.end_point.y()),
                                                  QPoint(self.start_point.x(), self.start_point.y()))
                   
                self.current_bbox.translate(-10, -10)
                
                self.update_image("labeling")
                
            elif self.stacked_widget.currentWidget() == self.masked_score_adjusting_page and self.adjust_image_label.geometry().contains(event.pos()):
                scale = self.adjust_image_label.width() / self.adjustimg.width()
                self.end_point = event.pos() - self.adjust_image_label.pos()
                self.end_point = QPoint(self.end_point.x() / scale, self.end_point.y() / scale)
                
                if self.start_point.x() < self.end_point.x():
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(self.start_point, self.end_point)
                    else:
                        self.current_bbox = QRect(QPoint(self.start_point.x(), self.end_point.y()), 
                                                  QPoint(self.end_point.x(), self.start_point.y()))
                else:
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.start_point.y()), 
                                                  QPoint(self.start_point.x(), self.end_point.y()))
                    else:
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.end_point.y()),
                                                  QPoint(self.start_point.x(), self.start_point.y()))
                self.current_bbox.translate(-10, -10)
                
                self.update_image("adjusting")
            
    def mouseReleaseEvent(self, event):
        if self.labeling and event.button() == Qt.LeftButton:
           
            if self.stacked_widget.currentWidget() == self.motif_labeling_page and self.image_label.geometry().contains(event.pos()):
                scale = self.image_label.width() / self.pixmap.width()
                self.end_point = event.pos() - self.image_label.pos()
                self.end_point = QPoint(self.end_point.x() / scale, self.end_point.y() / scale)
                if self.start_point.x() < self.end_point.x():
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(self.start_point, self.end_point)
                    else:
                        self.current_bbox = QRect(QPoint(self.start_point.x(), self.end_point.y()), 
                                                  QPoint(self.end_point.x(), self.start_point.y()))
                else:
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.start_point.y()), 
                                                  QPoint(self.start_point.x(), self.end_point.y()))
                    else:
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.end_point.y()),
                                                  QPoint(self.start_point.x(), self.start_point.y()))
                self.current_bbox.translate(-10, -10)
                norm_box = self.normalize("labeling")
                bbox_data = [self.tmp_btnname] + norm_box
                self.current_bbox = None 
                
                self.annotations[self.current_image_index].append(bbox_data)

                self.update_image("labeling")
                self.update_annotation_display()
                
            elif self.stacked_widget.currentWidget() == self.masked_score_adjusting_page and self.adjust_image_label.geometry().contains(event.pos()):
                scale = self.adjust_image_label.width() / self.adjustimg.width()
                self.end_point = event.pos() - self.adjust_image_label.pos()
                self.end_point = QPoint(self.end_point.x() / scale, self.end_point.y() / scale)
                if self.start_point.x() < self.end_point.x():
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(self.start_point, self.end_point)
                    else:
                        self.current_bbox = QRect(QPoint(self.start_point.x(), self.end_point.y()), 
                                                  QPoint(self.end_point.x(), self.start_point.y()))
                else:
                    if self.start_point.y() < self.end_point.y():
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.start_point.y()), 
                                                  QPoint(self.start_point.x(), self.end_point.y()))
                    else:
                        self.current_bbox = QRect(QPoint(self.end_point.x(), self.end_point.y()),
                                                  QPoint(self.start_point.x(), self.start_point.y()))
                self.current_bbox.translate(-10, -10)
                norm_box = self.normalize("adjusting")
                bbox_data = [self.tmp_btnname, self.current_adjust_index] + norm_box
                self.current_bbox = None 
                
                self.rpcropping.repeat_data[self.current_adjust_index].append(bbox_data)
                
                self.update_image("adjusting")
                self.update_repeat_list_display()
        
        self.finished_labeling()
           
    def update_annotation_display(self):
        self.list_widget.clear()
        if self.annotations != []:
            for label_id, bbox in enumerate(self.annotations[self.current_image_index]):
                self.list_widget.addItem(f"{label_id}_{bbox[0]}")
                
    def update_image(self, page, idx=None):
        
        if page == "labeling":

            pixmap_tmp = self.pixmap.copy()
            if self.annotations != []:
                
                self.draw_bbox(pixmap_tmp, self.annotations[self.current_image_index], idx)
       
            if self.current_bbox:    
                self.draw_bbox(pixmap_tmp, [self.current_bbox])
            
            pixmap_tmp = pixmap_tmp.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap_tmp)

        elif page == "adjusting":
            pixmap_tmp = self.adjustimg.copy()

            if self.rpcropping.repeat_data != [] and self.rpcropping.repeat_data[self.current_adjust_index] != []:
                # print(self.rpcropping.repeat_data[self.current_adjust_index], idx)
                self.draw_bbox(pixmap_tmp, self.rpcropping.repeat_data[self.current_adjust_index], idx)
            
            if self.current_bbox:
                self.draw_bbox(pixmap_tmp, [self.current_bbox])

            pixmap_tmp = pixmap_tmp.scaled(self.adjust_image_label.size(), Qt.KeepAspectRatio)
            self.adjust_image_label.setPixmap(pixmap_tmp)
            
    def normalize(self, page):
        if page == "labeling":
            width = self.pixmap.width()
            height = self.pixmap.height()
        elif page == "adjusting":
            width = self.adjustimg.width()
            height = self.adjustimg.height()
        
        
        x = self.current_bbox.x() / width
        y = self.current_bbox.y() / height
        w = self.current_bbox.width() / width
        h = self.current_bbox.height() / height
        return [x + w/2, y + h/2, w, h]

    # cropping 
    def repeatcropping(self):
        self.rpcropping = RepeatCropping(
            image_path=os.path.join(self.piece_path_label.text(), self.combo_box.currentText()),
            model_path=self.omr_path_label.text(),
            mode="omr",
            piece = self.combo_box.currentText(),
            save=os.path.join(self.cropping_piece_path_label.text(), self.combo_box.currentText()),
            progress=self.update_progress
        )
        # print(self.rpcropping.edge)

        self.croppingdata = [self.rpcropping.score_mask, self.rpcropping.system_id, self.rpcropping.edge]
        
    def only_cropping(self):
        self.rpcropping.cropping()
        self.croppingdata = [self.rpcropping.score_mask, self.rpcropping.system_id, self.rpcropping.edge]
        
    def update_progress(self, value):
        self.progress.setValue(value)
    
    def crop_save(self):
        # print("crop_save", os.path.join(self.cropping_piece_path_label.text(), self.combo_box.currentText(), 'cropping.pkl'))
        with open(os.path.join(self.cropping_piece_path_label.text(), self.combo_box.currentText(), 'cropping.pkl'), 'wb') as file:
            pickle.dump(self.croppingdata, file)
            
    def crop_load(self):
        # print("crop_load", os.path.join(self.cropping_piece_path_label.text(), self.combo_box.currentText(), 'cropping.pkl'))
        with open(os.path.join(self.cropping_piece_path_label.text(), self.combo_box.currentText(), 'cropping.pkl'), 'rb') as file:
            self.croppingdata = pickle.load(file)
            # print(self.croppingdata)
    # cropping_check
    def check_image(self, origin_id, masked_id):

        if 0 <= masked_id and masked_id < len(self.rpcropping.mask_list):
            lcheckimg = QPixmap(f"{self.piece_path_label.text()}\\{self.combo_box.currentText()}\\{self.combo_box.currentText()}_{origin_id}.jpg")
            
            
            if not lcheckimg.isNull():
                if self.rpcropping.repeat_data.get(origin_id) != None:
                    repeat_bbox = self.rpcropping.repeat_data[origin_id]
                    self.draw_bbox(lcheckimg, repeat_bbox)
                    
                self.lcheckimg = lcheckimg.scaled(self.limage_label.size(), Qt.KeepAspectRatio)
                self.limage_label.setPixmap(self.lcheckimg)
                self.left_page_label.setText(f"{1}/{len(self.image_list)}")
        
            rcheckimg = QPixmap(f"{self.cropping_piece_path_label.text()}\\{self.combo_box.currentText()}\\{self.combo_box.currentText()}_{masked_id}.jpg")
            if not rcheckimg.isNull():
                self.rcheckimg = rcheckimg.scaled(self.rimage_label.size(), Qt.KeepAspectRatio)
                self.rimage_label.setPixmap(self.rcheckimg)
                self.right_page_label.setText(f"{1}/{len(self.rpcropping.mask_list)}")
    
    def check_prev_image(self):
    
        if self.current_check_index > 0:
            self.current_check_index -= 1
            
            origin = self.rpcropping.score_mask[self.current_check_index][0]
            masked = self.current_check_index
            self.check_image(origin, masked)
            self.left_page_label.setText(f"{origin+1}/{len(self.image_list)}")
            self.right_page_label.setText(f"{masked+1}/{len(self.rpcropping.mask_list)}")
    
    def check_next_image(self):
        if self.current_check_index < len(self.rpcropping.mask_list) - 1:
            self.current_check_index += 1
            
            
            origin = self.rpcropping.score_mask[self.current_check_index][0]
            masked = self.current_check_index
            self.check_image(origin, masked)
            self.left_page_label.setText(f"{origin+1}/{len(self.image_list)}")
            self.right_page_label.setText(f"{masked+1}/{len(self.rpcropping.mask_list)}")
            
    # adjusting
    def adjust_image(self, origin_id, high_light=False):
        
        if 0 <= origin_id < len(self.rpcropping.mask_list):
            
            self.adjustimg = QPixmap(f"{self.piece_path_label.text()}\\{self.combo_box.currentText()}\\{self.combo_box.currentText()}_{origin_id}.jpg")

            if not self.adjustimg.isNull():
                if self.rpcropping.repeat_data.get(origin_id) != None:
                    self.update_image("adjusting")
                    self.update_repeat_list_display()
                    self.adjust_page_label.setText(f"{origin_id+1}/{len(self.image_list)}")
    
    def adjust_prev_image(self):
        if self.current_adjust_index > 0:
            self.current_adjust_index -= 1
            self.adjust_image(self.current_adjust_index)
          
    def adjust_next_image(self):
        if self.current_adjust_index < len(self.rpcropping.repeat_data) - 1:
            self.current_adjust_index += 1
            self.adjust_image(self.current_adjust_index)
 
    def remove_repeat_label(self):
        self.finished_labeling()
        current_row = self.repeat_list.currentRow()
        if current_row >= 0:
            del self.rpcropping.repeat_data[self.current_adjust_index][current_row]
            self.repeat_list.takeItem(current_row)
            self.update_image("adjusting")
            
        self.repeat_list.clearSelection()
        self.repeat_list.setCurrentRow(-1)

    def update_repeat_list_display(self):
        self.repeat_list.clear()
        for label_id, bbox in enumerate(self.rpcropping.repeat_data[self.current_adjust_index]):
            name, cls_id, x, y, w, h = bbox
            self.repeat_list.addItem(f"{label_id}_{name}")
    
    # checking & adjusting   
    def draw_bbox(self, pixmap, bbox_list, idx=None, c=100):
        
        color = QColor(255, 0, 0)
        brush = QColor(255, 0, 0, c)
        
        bbox = QPainter(pixmap)
        bbox.setPen(color) 
        bbox.setBrush(brush) 
    
        for id, box in enumerate(bbox_list):
            if type(box) == QRect:
                rect = box
            else:
                if len(box) == 4:
                    x, y, w, h =  box
                elif len(box) == 5:
                    n, x, y, w, h =  box
                elif len(box) == 6:
                    name, cls_id, x, y, w, h =  box
                    
                rect = QRect(int((x - w/2) * pixmap.width()), 
                            int((y - h/2) * pixmap.height()), 
                            int(w * pixmap.width()), 
                            int(h * pixmap.height()))
                
            bbox.drawRect(rect)

        bbox.end()
        if idx != None:
            color = QColor(0, 255, 0)
            brush = QColor(0, 255, 0, 100)
            hlbbox = QPainter(pixmap)
            hlbbox.setPen(color) 
            hlbbox.setBrush(brush)
        
            box = bbox_list[idx]
            if type(box) == QRect:
                x = box.x()
                y = box.y()
                w = box.width()
                h = box.height()
            else:
                if len(box) == 4:
                    x, y, w, h =  box
                elif len(box) == 5:
                    n, x, y, w, h =  box
                elif len(box) == 6:
                    name, cls_id, x, y, w, h =  box
                    
            if type(box) == QRect:
                rect = QRect(int(x), 
                            int(y), 
                            int(w), 
                            int(h))
            else:
                rect = QRect(int((x - w/2) * pixmap.width()), 
                            int((y - h/2) * pixmap.height()), 
                            int(w * pixmap.width()), 
                            int(h * pixmap.height()))
                
            hlbbox.drawRect(rect)
            
            hlbbox.end()
        
        return pixmap
        
    # Score Following
    def play_stop_music(self, btn=None):
        # print(self.btn_play_stop.text())
        if btn == "Done":
            # if self.save_video != []:
            #         tag = f"_{self.level_group.checkedButton().text()}_{self.level_group.checkedButton().text()}"
            #         create_video(np.array(self.save_video), np.array(self.truncate_signal), self.combo_box.currentText(), FPS, SAMPLE_RATE, tag=tag, path=os.path.join(self.save_path_label.text(), self.combo_box.currentText()), progress=self.sf_progress)
            #         print("!", np.array(self.save_video).shape, self.truncate_signal.shape)
            self.save_video = []
            self.truncate_signal = np.zeros(2 * FRAME_SIZE)
            self.sf.terminate()
            del self.sf
            self.sf = None
            self.btn_play_stop.setText("Play")
        elif self.btn_play_stop.text() == "Play":
            self.sf.init_setting()
            self.sf.start()
            self.btn_play_stop.setText("Stop")
        elif self.btn_play_stop.text() == "Stop":
            if self.sf != None:
                self.sf.stop_playing()
                self.sf.terminate()
 
            self.btn_play_stop.setText("Play")
                    
    def score_following(self):
        
        
        level_button = self.level_group.checkedButton() 
            
        if level_button.text() == "Note":
            mode = "fullpage"
            level = "note"
            model_path = os.path.join(self.sf_path_label.text(), "fullpage", "best_model.pt" ) 
        else:
            mode = "bipage"
            level = "bar"
            model_path = os.path.join(self.sf_path_label.text(), "bipage_sb", "best_model.pt" )        

        selected_button = self.signal_group.checkedButton() 
        if selected_button.text() == "Audio":
            audio_path = os.path.join(self.audio_path_label.text(), self.combo_box.currentText(), f"{self.combo_box.currentText()}.wav")
        else:
            audio_path = None
        
        self.sf = ScoreFollowing(
                # update model path
                model_path=model_path,
                score_dir=self.piece_path_label.text(),
                crop_dir=self.cropping_piece_path_label.text(),
                audio_dir=audio_path,
                piece_name=self.combo_box.currentText(),
                mode=mode,
                level=level,
                motif_label=self.annotations,
                cropping_info=self.croppingdata,
            )
        self.sf.update_data.connect(self.update_info)
        
    def scorefollowing_setting(self):
        self.setting_mode = not self.setting_mode
         
        if self.setting_mode:
            self.masked_score_label.hide()
        else:
            self.masked_score_label.show()
            self.score_following()
            
        for widget in self.setting_list:
            if self.setting_mode:
                widget.setVisible(self.setting_mode)
                widget.show()
            else:
                widget.setVisible(self.setting_mode)
                widget.hide()
                
        for widget in self.setting_list2:
            widget.setEnabled(self.setting_mode != True)
                    
    def update_info(self, data):
        # progress bar
        self.sf_progress.setValue(int(data["value"]))
       
        # visualization   
        cx, cy = data["predict"]
        img_pred = cv.cvtColor(data["score"], cv.COLOR_RGB2BGR)
        maskedimg = cv.cvtColor(data["masked_score"], cv.COLOR_RGB2BGR)
        plot_box([cx - 10, cy - 50, cx + 10, cy + 50], img_pred, label=CLASS_MAPPING[0],
                        color=COLORS[0 % len(COLORS)], line_thickness=2)
        for ann in self.annotations[int(data['score_page'])]:
            if ann != []:
                ih, iw, n = img_pred.shape
                cls, x, y, w, h = ann
                x1 = (x - w/2) * iw
                y1 = (y - h/2) * ih
                x2 = (x + w/2) * iw
                y2 = (y + h/2) * ih
                plot_box([x1, y1, x2, y2], img_pred, label=cls,
                                color=COLORS[1 % len(COLORS)], line_thickness=2)
        
        perf_img = prepare_spec_for_render(data["spec"], img_pred)
        img = np.concatenate((img_pred, perf_img), axis=1)
        img = np.array((img*255), dtype=np.uint8) 
        maskedimg  = np.array((maskedimg*255), dtype=np.uint8) 
        self.plotscore(self.score_label, img)
        # print(maskedimg.shape)
        self.plotscore(self.masked_score_label, maskedimg)
        
        # motif
        if data["motif_status"]:
            self.motif_title.setStyleSheet("background-color: lightpink; color: black; padding: 10px; font-size: 20px; font-weight: bold; font-family: 'Arial';")
        else:
            self.motif_title.setStyleSheet("background-color: darkgray; color: white; padding: 10px; font-size: 20px; font-family: 'Arial';")
        # turning
        if data["turning"]:
            self.turning_title.setStyleSheet("background-color: lightpink; color: black; padding: 10px; font-size: 20px; font-weight: bold; font-family: 'Arial';")
        else:
            self.turning_title.setStyleSheet("background-color: darkgray; color: white; padding: 10px; font-size: 20px; font-family: 'Arial';")
        
        # score info
        self.info.setText(f"")
        self.info.append(f"Masked Score Page: {data['masked_score_page']}")
        self.info.append(f"Score Page: {data['score_page']}")
        self.info.append(f"Score System: {data['system_id']}")
        self.info.append(f"Motif Status: {data['motif_status']}")
        self.info.append(f"Motif Class: {data['motif_id']}")
        
        # video 
        self.save_video.append(img)
        self.truncate_signal = np.concatenate((self.truncate_signal, data['signal']))
        
    def plotscore(self, obj,  image_np):
        if len(image_np.shape) == 2: 
            height, width = image_np.shape
            qimage = QImage(image_np.data, width, height, width, QImage.Format_Grayscale8)
        else: 
            height, width, channel = image_np.shape
            bytes_per_line = 3 * width
            qimage = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(obj.size(), Qt.KeepAspectRatio)
        obj.setPixmap(pixmap)
   
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
