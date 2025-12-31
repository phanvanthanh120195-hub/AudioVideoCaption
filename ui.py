import sys
import json
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QFileDialog, QTextEdit,
    QProgressBar, QGroupBox, QMessageBox, QCheckBox, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QTabWidget, QScrollArea,
    QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor
import io
import sys

# Import functions from app.py
from app import (
    generate_clip_storage, render_video, generate_captions_whisper, 
    burn_captions_to_video, merge_audio_video_files, process_audio_video_batch,
    process_burn_captions_batch,
    get_audio_duration, get_video_duration, format_duration
)

# Import task queue
from task_queue import RenderTask, TaskQueue


class DurationLabel(QWidget):
    """Simple widget to display duration in HH:MM:SS format"""
    
    def __init__(self, duration: float, parent=None):
        super().__init__(parent)
        self.duration = duration
        
        # Set up layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(0)
        
        # Format duration to always show HH:MM:SS
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Duration label
        self.duration_label = QLabel(duration_text)
        self.duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.duration_label.setFont(QFont("Segoe UI", 9))
        self.duration_label.setToolTip(f"{duration:.2f} seconds")
        layout.addWidget(self.duration_label)
        
        self.setLayout(layout)


class OutputRedirector(QObject):
    """Redirect stdout/stderr to QTextEdit widget"""
    output_written = pyqtSignal(str)
    MAX_LINES = 100  # Maximum lines to keep in console
    
    def __init__(self, text_widget, stream_type='stdout'):
        super().__init__()
        self.text_widget = text_widget
        self.stream_type = stream_type
        self.output_written.connect(self.append_text)
        
        # Save original stream
        if stream_type == 'stdout':
            self.original_stream = sys.stdout
        else:
            self.original_stream = sys.stderr
    
    def write(self, text):
        """Write text to both original stream and widget"""
        # Write to original stream (terminal)
        if self.original_stream:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except:
                pass
        
        # Emit signal to update widget
        if text.strip():  # Only emit non-empty text
            self.output_written.emit(text)
    
    def append_text(self, text):
        """Append text to widget (runs in main thread)"""
        if self.text_widget:
            # Ensure each log message ends with newline
            if not text.endswith('\n'):
                text = text + '\n'
            self.text_widget.moveCursor(QTextCursor.MoveOperation.End)
            self.text_widget.insertPlainText(text)
            self.text_widget.moveCursor(QTextCursor.MoveOperation.End)
            
            # Limit number of lines to prevent memory issues
            doc = self.text_widget.document()
            if doc.blockCount() > self.MAX_LINES:
                cursor = QTextCursor(doc)
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                # Remove oldest lines (keep only MAX_LINES)
                lines_to_remove = doc.blockCount() - self.MAX_LINES
                for _ in range(lines_to_remove):
                    cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                    cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
    
    def flush(self):
        """Flush the stream"""
        if self.original_stream:
            try:
                self.original_stream.flush()
            except:
                pass
    
    @property
    def closed(self):
        """Check if stream is closed"""
        return False
    
    def isatty(self):
        """Check if stream is a TTY"""
        return False


class WorkerThread(QThread):
    """Generic worker thread for long-running operations"""
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
            self.finished.emit(True, "Operation completed successfully!")
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class QueueWorkerThread(QThread):
    """Worker thread for processing task queue"""
    task_started = pyqtSignal(str)  # task_id
    task_progress = pyqtSignal(str, int)  # task_id, progress
    task_completed = pyqtSignal(str, bool, str)  # task_id, success, message
    queue_empty = pyqtSignal()
    
    def __init__(self, task_queue: TaskQueue):
        super().__init__()
        self.task_queue = task_queue
        self.running = True
        self.paused = False
    
    def run(self):
        """Process tasks from queue"""
        while self.running:
            # Wait if paused
            if self.paused:
                self.msleep(500)
                continue
            
            # Get next task
            task = self.task_queue.get_next_task()
            if not task:
                self.queue_empty.emit()
                self.msleep(1000)  # Wait 1 second before checking again
                continue
            
            # Process task based on type
            try:
                self.task_started.emit(task.id)
                self.task_queue.update_status(task.id, 'running', 0)
                
                # Check task type
                task_type = getattr(task, 'task_type', 'render')
                
                if task_type == 'clip_generation':
                    # Process clip generation task
                    self._process_clip_generation(task)
                else:
                    # Process render task
                    self._process_render(task)
                
                # Mark as completed
                self.task_queue.update_status(task.id, 'completed', 100)
                self.task_completed.emit(task.id, True, "Task completed successfully")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.task_queue.update_status(task.id, 'failed', 0, error_msg)
                self.task_completed.emit(task.id, False, error_msg)
    
    def _process_clip_generation(self, task):
        """Process a clip generation task"""
        generate_clip_storage(
            count=task.clip_count,
            storage_dir=task.output_path,  # storage_dir stored in output_path
            images_dir=task.audio_path,     # images_dir stored in audio_path
            overlays_dir=task.overlays_dir,
            width=1280,
            height=720,
            fps=24,
            seg_min=90,
            seg_max=150,
            seed=0,
            max_workers=2
        )
        self.task_queue.update_status(task.id, 'running', 100)
    
    def _process_render(self, task):
        """Process a render task"""
        render_video(
            audio_path=task.audio_path,
            images_dir="images",  # TODO: Get from settings
            overlays_dir="overlays",
            out_path=task.output_path,
            seed=0,
            width=1280,
            height=720,
            fps=24,
            seg_min=90,
            seg_max=150,
            crossfade=1.5
        )
        
        self.task_queue.update_status(task.id, 'running', 50)
        
        # Generate captions if requested
        if task.generate_captions:
            srt_path = str(Path(task.output_path).with_suffix('.srt'))
            generate_captions_whisper(
                task.output_path,
                srt_path,
                task.whisper_model,
                language="auto"
            )
            
            self.task_queue.update_status(task.id, 'running', 75)
            
            # Burn captions if requested
            if task.burn_captions:
                output_with_captions = str(
                    Path(task.output_path).parent / 
                    f"{Path(task.output_path).stem}_captioned.mp4"
                )
                burn_captions_to_video(
                    task.output_path,
                    srt_path,
                    output_with_captions
                )
    
    def pause(self):
        """Pause queue processing"""
        self.paused = True
    
    def resume(self):
        """Resume queue processing"""
        self.paused = False
    
    def stop(self):
        """Stop queue processing"""
        self.running = False


class AudiobookVideoCreatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audiobook Video Creator")
        self.setMinimumSize(1000, 1000)
        self.resize(1000, 1000)
        # Settings file
        self.settings_file = "settings.json"
        self.settings = self.load_settings()
        
        # Worker thread
        self.worker = None
        self.queue_worker = None
        
        # Setup UI
        self.init_ui()
        self.apply_styles()
        
        # Center window on screen
        self.center_on_screen()
    
    def center_on_screen(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())
    
    def load_settings(self):
        """Load settings from JSON file"""
        default_settings = {
            "images_dir": "images",
            "overlays_dir": "overlays",
            "storage_dir": "clip_storage",
            "last_audio_path": "",
            "last_output_path": ""
        }
        
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    default_settings.update(loaded)
            except Exception as e:
                print(f"Error loading settings: {e}")
        
        return default_settings
    
    def save_settings(self):
        """Save settings to JSON file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def init_ui(self):
        """Initialize UI components with tab-based layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Main Tab Widget
        self.tabs = QTabWidget()
        
        # Tab 1: Merge MP3 + MP4
        self.merge_tab = QWidget()
        merge_layout = QVBoxLayout()
        # Use existing create_video_rendering_section
        render_group = self.create_video_rendering_section()
        merge_layout.addWidget(render_group)
        # merge_layout.addStretch() # Optional, depends on content
        self.merge_tab.setLayout(merge_layout)
        self.tabs.addTab(self.merge_tab, "Merge Audio/Video")
        
        # Tab 2: Burn Captions
        self.burn_tab = QWidget()
        burn_layout = QVBoxLayout()
        # Create new section
        burn_group = self.create_burn_captions_section()
        burn_layout.addWidget(burn_group)
        # burn_layout.addStretch()
        self.burn_tab.setLayout(burn_layout)
        self.tabs.addTab(self.burn_tab, "Burn Captions in Video")
        
        main_layout.addWidget(self.tabs)
        
        # Console Output Section - Fixed at bottom, always visible
        console_group = self.create_console_section()
        main_layout.addWidget(console_group)
        
        central_widget.setLayout(main_layout)
        
        # Redirect stdout and stderr to console
        self.setup_output_redirection()

    def create_burn_captions_section(self):
        """Create the UI for the Burn Captions tab"""
        group = QGroupBox("🔥 Burn Captions into Video")
        layout = QVBoxLayout()
        
        # Input: Video Folder
        video_layout = QHBoxLayout()
        video_label = QLabel("Video Folder (MP4):")
        video_label.setMinimumWidth(120)
        self.burn_video_path = QLineEdit(self.settings.get("last_burn_video_folder", ""))
        video_btn = QPushButton("Browse")
        video_btn.clicked.connect(self.browse_burn_video_folder)
        video_layout.addWidget(video_label)
        video_layout.addWidget(self.burn_video_path)
        video_layout.addWidget(video_btn)
        layout.addLayout(video_layout)
        
        # Input: SRT Folder
        srt_layout = QHBoxLayout()
        srt_label = QLabel("Subtitle Folder (SRT):")
        srt_label.setMinimumWidth(120)
        self.burn_srt_path = QLineEdit(self.settings.get("last_burn_srt_folder", ""))
        srt_btn = QPushButton("Browse")
        srt_btn.clicked.connect(self.browse_burn_srt_folder)
        srt_layout.addWidget(srt_label)
        srt_layout.addWidget(self.burn_srt_path)
        srt_layout.addWidget(srt_btn)
        layout.addLayout(srt_layout)
        
        # Output: Folder
        out_layout = QHBoxLayout()
        out_label = QLabel("Output Folder:")
        out_label.setMinimumWidth(120)
        self.burn_output_path = QLineEdit(self.settings.get("last_burn_output_folder", ""))
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(self.browse_burn_output_folder)
        out_layout.addWidget(out_label)
        out_layout.addWidget(self.burn_output_path)
        out_layout.addWidget(out_btn)
        layout.addLayout(out_layout)
        
        # Config options (Color, Font Size)
        config_layout = QHBoxLayout()
        
        # Font Size
        config_layout.addWidget(QLabel("Font Size:"))
        self.burn_font_size = QSpinBox()
        self.burn_font_size.setRange(10, 100)
        self.burn_font_size.setValue(24)
        config_layout.addWidget(self.burn_font_size)
        
        # Color
        config_layout.addWidget(QLabel("Color:"))
        self.burn_color_combo = QComboBox()
        self.burn_color_combo.addItems(["white", "yellow", "green", "blue", "red", "black"])
        config_layout.addWidget(self.burn_color_combo)
        
        config_layout.addStretch()
        layout.addLayout(config_layout)
        
        # File List Table using QTableWidget
        list_label = QLabel("Matched Files:")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)
        
        self.burn_file_table = QTableWidget()
        self.burn_file_table.setColumnCount(5)
        self.burn_file_table.setHorizontalHeaderLabels([
            "Video File", "Duration", "Subtitle File", "Status", "Actions"
        ])
        self.burn_file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.burn_file_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.burn_file_table.setAlternatingRowColors(True)
        self.burn_file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.burn_file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.burn_file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.burn_file_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.burn_file_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.burn_file_table.setColumnWidth(1, 90)
        self.burn_file_table.setColumnWidth(3, 100)
        self.burn_file_table.setColumnWidth(4, 80)
        self.burn_file_table.verticalHeader().setDefaultSectionSize(40)
        self.burn_file_table.setMinimumHeight(200)
        layout.addWidget(self.burn_file_table)
        
        # Status Label
        self.burn_status_label = QLabel("Select folders to find matched files")
        self.burn_status_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        self.burn_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.burn_status_label)
        
        # Process Button
        self.burn_process_btn = QPushButton("🔥 Start Burning Captions")
        self.burn_process_btn.setMinimumHeight(40)
        self.burn_process_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #d35400; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.burn_process_btn.clicked.connect(self.process_burn_batch_ui)
        layout.addWidget(self.burn_process_btn)
        
        # Connect text changes to auto-update list
        self.burn_video_path.textChanged.connect(self.update_burn_file_list)
        self.burn_srt_path.textChanged.connect(self.update_burn_file_list)
        
        group.setLayout(layout)
        return group
    
    def create_directory_section(self):
        """Create directory selection section"""
        group = QGroupBox("📁 Directory Settings")
        layout = QVBoxLayout()
        
        # Images Directory
        images_layout = QHBoxLayout()
        images_label = QLabel("Images:")
        images_label.setMinimumWidth(80)
        self.images_path = QLineEdit(self.settings["images_dir"])
        images_btn = QPushButton("Browse")
        images_btn.clicked.connect(lambda: self.browse_directory("images_dir", self.images_path))
        images_layout.addWidget(images_label)
        images_layout.addWidget(self.images_path)
        images_layout.addWidget(images_btn)
        layout.addLayout(images_layout)
        
        # Overlays Directory
        overlays_layout = QHBoxLayout()
        overlays_label = QLabel("Overlays:")
        overlays_label.setMinimumWidth(80)
        self.overlays_path = QLineEdit(self.settings["overlays_dir"])
        overlays_btn = QPushButton("Browse")
        overlays_btn.clicked.connect(lambda: self.browse_directory("overlays_dir", self.overlays_path))
        overlays_layout.addWidget(overlays_label)
        overlays_layout.addWidget(self.overlays_path)
        overlays_layout.addWidget(overlays_btn)
        layout.addLayout(overlays_layout)
        
        # Storage Directory
        storage_layout = QHBoxLayout()
        storage_label = QLabel("Storage:")
        storage_label.setMinimumWidth(80)
        self.storage_path = QLineEdit(self.settings["storage_dir"])
        storage_btn = QPushButton("Browse")
        storage_btn.clicked.connect(lambda: self.browse_directory("storage_dir", self.storage_path))
        storage_layout.addWidget(storage_label)
        storage_layout.addWidget(self.storage_path)
        storage_layout.addWidget(storage_btn)
        layout.addLayout(storage_layout)
        
        group.setLayout(layout)
        return group
    
    def create_clip_generation_section(self):
        """Create clip generation section"""
        group = QGroupBox("🎞️ Generate Clip Storage")
        layout = QVBoxLayout()
        
        # Input row
        input_layout = QHBoxLayout()
        count_label = QLabel("Number of clips:")
        self.clip_count = QSpinBox()
        self.clip_count.setMinimum(1)
        self.clip_count.setMaximum(10000)
        self.clip_count.setValue(100)
        self.clip_count.setSuffix(" clips")
        
        input_layout.addWidget(count_label)
        input_layout.addWidget(self.clip_count)
        input_layout.addStretch()
        layout.addLayout(input_layout)
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("🚀 Generate Clips")
        self.generate_btn.clicked.connect(self.generate_clips)
        buttons_layout.addWidget(self.generate_btn)
        
        self.add_clips_to_queue_btn = QPushButton("➕ Add to Queue")
        self.add_clips_to_queue_btn.clicked.connect(self.add_clip_generation_to_queue)
        buttons_layout.addWidget(self.add_clips_to_queue_btn)
        
        layout.addLayout(buttons_layout)
        
        # Progress bar
        self.clip_progress = QProgressBar()
        self.clip_progress.setVisible(False)
        layout.addWidget(self.clip_progress)
        
        # Status label
        self.clip_status = QLabel("")
        self.clip_status.setWordWrap(True)
        layout.addWidget(self.clip_status)
        
        group.setLayout(layout)
        return group
    
    def create_video_rendering_section(self):
        """Create video rendering section - now for MP3+MP4 merging"""
        group = QGroupBox("🎥 Merge MP3 + MP4 Files")
        layout = QVBoxLayout()
        
        # Audio folder selection
        audio_layout = QHBoxLayout()
        audio_label = QLabel("Audio Folder:")
        audio_label.setMinimumWidth(100)
        self.audio_folder_path = QLineEdit(self.settings.get("last_audio_folder", ""))
        audio_btn = QPushButton("Browse MP3s")
        audio_btn.clicked.connect(self.browse_audio_folder)
        audio_layout.addWidget(audio_label)
        audio_layout.addWidget(self.audio_folder_path)
        audio_layout.addWidget(audio_btn)
        layout.addLayout(audio_layout)
        
        # Video folder selection
        video_layout = QHBoxLayout()
        video_label = QLabel("Video Folder:")
        video_label.setMinimumWidth(100)
        self.video_folder_path = QLineEdit(self.settings.get("last_video_folder", ""))
        video_btn = QPushButton("Browse MP4s")
        video_btn.clicked.connect(self.browse_video_folder)
        video_layout.addWidget(video_label)
        video_layout.addWidget(self.video_folder_path)
        video_layout.addWidget(video_btn)
        layout.addLayout(video_layout)
        
        # Output folder selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        output_label.setMinimumWidth(100)
        self.output_folder_path = QLineEdit(self.settings.get("last_output_folder", ""))
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_folder_path)
        output_layout.addWidget(output_btn)
        layout.addLayout(output_layout)
        
        # File list table
        list_label = QLabel("Matched Files:")
        list_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        layout.addWidget(list_label)
        
        self.file_list_table = QTableWidget()
        self.file_list_table.setColumnCount(6)
        self.file_list_table.setHorizontalHeaderLabels([
            "MP3 File", "Duration", "MP4 File", "Duration", "Status", "Actions"
        ])
        self.file_list_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_list_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.file_list_table.setAlternatingRowColors(True)
        self.file_list_table.horizontalHeader().setStretchLastSection(False)
        self.file_list_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_list_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.file_list_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.file_list_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.file_list_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.file_list_table.setColumnWidth(1, 90)  # MP3 Duration
        self.file_list_table.setColumnWidth(3, 90)  # MP4 Duration
        self.file_list_table.setColumnWidth(4, 100)  # Status
        self.file_list_table.setColumnWidth(5, 80)   # Actions
        self.file_list_table.verticalHeader().setDefaultSectionSize(40)
        self.file_list_table.setMinimumHeight(150)
        self.file_list_table.setMaximumHeight(200)
        layout.addWidget(self.file_list_table)
        
        # Caption options
        caption_options_layout = QHBoxLayout()
        self.caption_checkbox = QCheckBox("Generate captions during render")
        self.caption_checkbox.setChecked(False)
        
        model_label = QLabel("Model:")
        model_label.setMinimumWidth(50)
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["tiny", "base", "small", "medium"])
        self.whisper_model_combo.setCurrentText("tiny")
        self.whisper_model_combo.setEnabled(False)
        self.whisper_model_combo.setMaximumWidth(120)
        
        self.caption_checkbox.stateChanged.connect(
            lambda: self.whisper_model_combo.setEnabled(self.caption_checkbox.isChecked())
        )
        
        caption_options_layout.addWidget(self.caption_checkbox)
        caption_options_layout.addWidget(model_label)
        caption_options_layout.addWidget(self.whisper_model_combo)
        
        # Burn captions checkbox
        self.burn_render_caption_checkbox = QCheckBox("Burn into video")
        self.burn_render_caption_checkbox.setChecked(False)
        self.burn_render_caption_checkbox.setEnabled(False)
        
        # Enable/disable burn checkbox based on caption checkbox
        self.caption_checkbox.stateChanged.connect(
            lambda: self.burn_render_caption_checkbox.setEnabled(self.caption_checkbox.isChecked())
        )
        
        caption_options_layout.addWidget(self.burn_render_caption_checkbox)
        caption_options_layout.addStretch()
        layout.addLayout(caption_options_layout)
        
        # Caption customization row (color and font size)
        caption_custom_layout = QHBoxLayout()
        
        # Caption color
        color_label = QLabel("Caption Color:")
        color_label.setMinimumWidth(100)
        self.caption_color_combo = QComboBox()
        self.caption_color_combo.addItems(["White", "Yellow", "Black", "Red", "Green", "Blue"])
        self.caption_color_combo.setCurrentText("White")
        self.caption_color_combo.setEnabled(False)
        self.caption_color_combo.setMaximumWidth(120)
        
        # Font size
        font_size_label = QLabel("Font Size:")
        font_size_label.setMinimumWidth(80)
        self.caption_font_size = QSpinBox()
        self.caption_font_size.setMinimum(12)
        self.caption_font_size.setMaximum(72)
        self.caption_font_size.setValue(24)
        self.caption_font_size.setSuffix(" px")
        self.caption_font_size.setEnabled(False)
        self.caption_font_size.setMaximumWidth(100)
        
        # Enable/disable color and font size based on caption checkbox
        self.caption_checkbox.stateChanged.connect(
            lambda: self.caption_color_combo.setEnabled(self.caption_checkbox.isChecked())
        )
        self.caption_checkbox.stateChanged.connect(
            lambda: self.caption_font_size.setEnabled(self.caption_checkbox.isChecked())
        )
        
        caption_custom_layout.addWidget(color_label)
        caption_custom_layout.addWidget(self.caption_color_combo)
        caption_custom_layout.addWidget(font_size_label)
        caption_custom_layout.addWidget(self.caption_font_size)
        caption_custom_layout.addStretch()
        layout.addLayout(caption_custom_layout)
        
        # Add spacing before button
        layout.addSpacing(10)
        
        # Process buttons
        buttons_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("🎬 Process All Files")
        self.process_btn.clicked.connect(self.process_audio_video_batch_ui)
        self.process_btn.setMinimumHeight(40)
        buttons_layout.addWidget(self.process_btn)
        
        layout.addLayout(buttons_layout)
        
        # Progress bar
        self.merge_progress = QProgressBar()
        self.merge_progress.setVisible(False)
        layout.addWidget(self.merge_progress)
        
        # Status label
        self.merge_status = QLabel("")
        self.merge_status.setWordWrap(True)
        layout.addWidget(self.merge_status)
        
        group.setLayout(layout)
        return group
    
    def create_caption_generation_section(self):
        """Create standalone caption generation section"""
        group = QGroupBox("💬 Generate Captions (Standalone)")
        layout = QVBoxLayout()
        
        # Video/Audio file selection
        file_layout = QHBoxLayout()
        file_label = QLabel("Video/Audio:")
        file_label.setMinimumWidth(80)
        self.caption_file_path = QLineEdit()
        caption_file_btn = QPushButton("Browse")
        caption_file_btn.clicked.connect(self.browse_caption_file)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.caption_file_path)
        file_layout.addWidget(caption_file_btn)
        layout.addLayout(file_layout)
        
        # Options row
        options_layout = QHBoxLayout()
        
        model_label = QLabel("Model:")
        self.caption_model_combo = QComboBox()
        self.caption_model_combo.addItems(["tiny", "base", "small", "medium"])
        self.caption_model_combo.setCurrentText("base")
        
        self.burn_caption_checkbox = QCheckBox("Burn captions into video")
        self.burn_caption_checkbox.setChecked(True)
        
        options_layout.addWidget(model_label)
        options_layout.addWidget(self.caption_model_combo)
        options_layout.addWidget(self.burn_caption_checkbox)
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Generate button
        self.caption_gen_btn = QPushButton("💬 Generate Captions")
        self.caption_gen_btn.clicked.connect(self.generate_captions_ui)
        layout.addWidget(self.caption_gen_btn)
        
        # Progress bar
        self.caption_progress = QProgressBar()
        self.caption_progress.setVisible(False)
        layout.addWidget(self.caption_progress)
        
        # Status label
        self.caption_status = QLabel("")
        self.caption_status.setWordWrap(True)
        layout.addWidget(self.caption_status)
        
        group.setLayout(layout)
        return group
    
    def create_queue_section(self):
        """Create render queue section"""
        group = QGroupBox("📋 Render Queue")
        layout = QVBoxLayout()
        
        # Queue table
        self.queue_table = QTableWidget()
        self.queue_table.setColumnCount(7)
        self.queue_table.setHorizontalHeaderLabels([
            "Status", "Task Info", "Destination", "Details", "Progress", "Actions", "ID"
        ])
        
        # Hide ID column (used internally)
        self.queue_table.setColumnHidden(6, True)
        
        # Table settings
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.horizontalHeader().setStretchLastSection(False)
        
        # Column resize modes
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.queue_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.queue_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        
        # Column widths
        self.queue_table.setColumnWidth(0, 100)   # Status
        self.queue_table.setColumnWidth(3, 100)   # Details
        self.queue_table.setColumnWidth(4, 100)   # Progress
        self.queue_table.setColumnWidth(5, 200)   # Actions
        # Row height
        self.queue_table.verticalHeader().setDefaultSectionSize(50)
        self.queue_table.verticalHeader().setVisible(False)
        
        self.queue_table.setMinimumHeight(300)
        layout.addWidget(self.queue_table, 1)  # Stretch factor 1 to expand
        
        # Queue controls
        controls_layout = QHBoxLayout()
        
        self.start_queue_btn = QPushButton("▶️ Start Queue")
        self.start_queue_btn.clicked.connect(self.start_queue)
        controls_layout.addWidget(self.start_queue_btn)
        
        self.pause_queue_btn = QPushButton("⏸️ Pause Queue")
        self.pause_queue_btn.clicked.connect(self.pause_queue)
        self.pause_queue_btn.setEnabled(False)
        controls_layout.addWidget(self.pause_queue_btn)
        
        self.clear_completed_btn = QPushButton("🗑️ Clear Completed")
        self.clear_completed_btn.clicked.connect(self.clear_completed_tasks)
        controls_layout.addWidget(self.clear_completed_btn)
        
        controls_layout.addStretch()
        
        # Queue status label
        self.queue_status_label = QLabel("Queue: 0 tasks")
        controls_layout.addWidget(self.queue_status_label)
        
        layout.addLayout(controls_layout)
        
        group.setLayout(layout)
        return group
    
    def create_console_section(self):
        """Create console output section"""
        group = QGroupBox("📋 Console Output")
        layout = QVBoxLayout()
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(120)
        self.console.setMaximumHeight(150)
        self.console.setFont(QFont("Consolas", 9))
        layout.addWidget(self.console)
        
        # Clear button
        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.console.clear)
        layout.addWidget(clear_btn)
        
        group.setLayout(layout)
        return group
    
    def setup_output_redirection(self):
        """Setup stdout/stderr redirection to console widget"""
        # Create redirectors
        self.stdout_redirector = OutputRedirector(self.console, 'stdout')
        self.stderr_redirector = OutputRedirector(self.console, 'stderr')
        
        # Redirect streams
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
        
        print("Console output initialized. All logs will appear here.")
    
    def browse_directory(self, setting_key, line_edit):
        """Browse for directory"""
        current_path = line_edit.text() or "."
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", current_path
        )
        if directory:
            line_edit.setText(directory)
            self.settings[setting_key] = directory
            self.save_settings()
    
    def browse_audio_file(self):
        """Browse for audio file"""
        current_path = self.audio_path.text() or "."
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", current_path,
            "Audio Files (*.mp3 *.wav *.m4a *.mp4 *.m4b);;All Files (*.*)"
        )
        if file_path:
            self.audio_path.setText(file_path)
            
            # Auto-generate output path
            audio_file = Path(file_path)
            output_file = audio_file.parent / f"video_stock_{audio_file.stem}.mp4"
            self.output_path.setText(str(output_file))
            
            self.save_settings()
    
    def browse_output_file(self):
        """Browse for output file"""
        current_path = self.output_path.text() or "."
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video As", current_path,
            "Video Files (*.mp4);;All Files (*.*)"
        )
        if file_path:
            # Ensure .mp4 extension
            if not file_path.endswith('.mp4'):
                file_path += '.mp4'
            self.output_path.setText(file_path)
            self.settings["last_output_path"] = file_path
            self.save_settings()
    
    def browse_caption_file(self):
        """Browse for video/audio file for caption generation"""
        current_path = self.caption_file_path.text() or "."
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video/Audio File", current_path,
            "Media Files (*.mp4 *.mp3 *.wav *.m4a *.avi *.mkv *.mov);;All Files (*.*)"
        )
        if file_path:
            self.caption_file_path.setText(file_path)
    
    def browse_audio_folder(self):
        """Browse for audio folder containing MP3 files"""
        current_path = self.audio_folder_path.text() or "."
        folder = QFileDialog.getExistingDirectory(
            self, "Select Audio Folder (MP3s)", current_path
        )
        if folder:
            self.audio_folder_path.setText(folder)
            self.settings["last_audio_folder"] = folder
            self.save_settings()
            self.update_file_list_table()
    
    def browse_video_folder(self):
        """Browse for video folder containing MP4 files"""
        current_path = self.video_folder_path.text() or "."
        folder = QFileDialog.getExistingDirectory(
            self, "Select Video Folder (MP4s)", current_path
        )
        if folder:
            self.video_folder_path.setText(folder)
            self.settings["last_video_folder"] = folder
            self.save_settings()
            self.update_file_list_table()
    
    def browse_output_folder(self):
        """Browse for output folder"""
        current_path = self.output_folder_path.text() or "."
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", current_path
        )
        if folder:
            self.output_folder_path.setText(folder)
            self.settings["last_output_folder"] = folder
            self.save_settings()
    
    def update_file_list_table(self):
        """Update file list table with matched MP3-MP4 pairs"""
        self.file_list_table.setRowCount(0)
        
        audio_folder = self.audio_folder_path.text()
        video_folder = self.video_folder_path.text()
        
        if not audio_folder or not video_folder:
            return
        
        if not os.path.exists(audio_folder) or not os.path.exists(video_folder):
            return
        
        # Scan for MP3 files
        audio_path = Path(audio_folder)
        mp3_files = {f.stem: f.name for f in audio_path.glob("*.mp3")}
        mp3_files.update({f.stem: f.name for f in audio_path.glob("*.MP3")})
        
        # Scan for MP4 files
        video_path = Path(video_folder)
        mp4_files = {f.stem: f.name for f in video_path.glob("*.mp4")}
        mp4_files.update({f.stem: f.name for f in video_path.glob("*.MP4")})
        
        # Get all unique names
        all_names = set(mp3_files.keys()) | set(mp4_files.keys())
        
        # Extract durations and find max duration for scaling
        file_durations = {}
        max_duration = 0.0
        
        for name in all_names:
            mp3_duration = 0.0
            mp4_duration = 0.0
            
            # Get MP3 duration
            if name in mp3_files:
                mp3_file_path = audio_path / mp3_files[name]
                try:
                    mp3_duration = get_audio_duration(str(mp3_file_path))
                    max_duration = max(max_duration, mp3_duration)
                except Exception as e:
                    print(f"Error getting MP3 duration for {mp3_files[name]}: {e}")
            
            # Get MP4 duration
            if name in mp4_files:
                mp4_file_path = video_path / mp4_files[name]
                try:
                    mp4_duration = get_video_duration(str(mp4_file_path))
                    max_duration = max(max_duration, mp4_duration)
                except Exception as e:
                    print(f"Error getting MP4 duration for {mp4_files[name]}: {e}")
            
            file_durations[name] = (mp3_duration, mp4_duration)
        
        # Add matched pairs to table
        row = 0
        
        for name in sorted(all_names):
            self.file_list_table.insertRow(row)
            
            mp3_name = mp3_files.get(name, "")
            mp4_name = mp4_files.get(name, "")
            mp3_duration, mp4_duration = file_durations[name]
            
            # Column 0: MP3 file name
            mp3_item = QTableWidgetItem(mp3_name if mp3_name else "⚠️ Missing")
            if not mp3_name:
                mp3_item.setForeground(QColor("#FF6B6B"))
            self.file_list_table.setItem(row, 0, mp3_item)
            
            # Column 1: MP3 duration
            if mp3_name and mp3_duration > 0:
                # Format duration to HH:MM:SS
                hours = int(mp3_duration // 3600)
                minutes = int((mp3_duration % 3600) // 60)
                seconds = int(mp3_duration % 60)
                duration_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                mp3_duration_item = QTableWidgetItem(duration_text)
                mp3_duration_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                mp3_duration_item.setToolTip(f"{mp3_duration:.2f} seconds")
                self.file_list_table.setItem(row, 1, mp3_duration_item)
            else:
                empty_item = QTableWidgetItem("")
                self.file_list_table.setItem(row, 1, empty_item)
            
            # Column 2: MP4 file name
            mp4_item = QTableWidgetItem(mp4_name if mp4_name else "⚠️ Missing")
            if not mp4_name:
                mp4_item.setForeground(QColor("#FF6B6B"))
            self.file_list_table.setItem(row, 2, mp4_item)
            
            # Column 3: MP4 duration
            if mp4_name and mp4_duration > 0:
                # Format duration to HH:MM:SS
                hours = int(mp4_duration // 3600)
                minutes = int((mp4_duration % 3600) // 60)
                seconds = int(mp4_duration % 60)
                duration_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                mp4_duration_item = QTableWidgetItem(duration_text)
                mp4_duration_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                mp4_duration_item.setToolTip(f"{mp4_duration:.2f} seconds")
                self.file_list_table.setItem(row, 3, mp4_duration_item)
            else:
                empty_item = QTableWidgetItem("")
                self.file_list_table.setItem(row, 3, empty_item)
            
            # Column 4: Status
            if mp3_name and mp4_name:
                # Check if durations are similar (within 10% difference)
                if mp3_duration > 0 and mp4_duration > 0:
                    duration_diff = abs(mp3_duration - mp4_duration) / max(mp3_duration, mp4_duration)
                    if duration_diff < 0.1:
                        status_item = QTableWidgetItem("✓ Ready")
                        status_item.setForeground(QColor("#51CF66"))
                    else:
                        status_item = QTableWidgetItem("⚠️ Duration Mismatch")
                        status_item.setForeground(QColor("#FFA94D"))
                else:
                    status_item = QTableWidgetItem("✓ Ready")
                    status_item.setForeground(QColor("#51CF66"))
            else:
                status_item = QTableWidgetItem("⚠️ Skip")
                status_item.setForeground(QColor("#FF6B6B"))
            self.file_list_table.setItem(row, 4, status_item)
            
            # Column 5: Actions (Remove button)
            remove_btn = QPushButton("❌")
            remove_btn.setFixedSize(30, 30)
            remove_btn.setToolTip("Remove this file from list")
            remove_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: transparent;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 107, 107, 0.2);
                    border-radius: 4px;
                }
            """)
            # Use closure to capture current row
            remove_btn.clicked.connect(lambda checked, r=row: self.remove_file_row(r))
            
            # Create a container widget for the button
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn_layout.addWidget(remove_btn)
            
            self.file_list_table.setCellWidget(row, 5, btn_widget)
            
            row += 1
        
        # Update status label
        matched_count = sum(1 for name in all_names if name in mp3_files and name in mp4_files)
        self.merge_status.setText(f"Found {matched_count} matched pair(s) ready to process")

    def remove_file_row(self, row):
        """Remove a row from the file list table"""
        # Get the filename to show in confirmation (optional)
        try:
            filename = self.file_list_table.item(row, 0).text()
            if not filename or filename == "⚠️ Missing":
                filename = self.file_list_table.item(row, 2).text()
                
            # Remove the row
            self.file_list_table.removeRow(row)
            
            # Re-bind click events for remaining rows to ensure correct row index
            # This is tricky because lambda captures value at definition time.
            # But when row is removed, indices shift. 
            # Easiest way is to just let it be, but wait - 
            # If I remove row 0, row 1 becomes row 0. But its button still thinks r=1.
            # So clicking it will try to remove row 1 (which might be row 2 now).
            # We need to re-render or update indices. 
            # Re-rendering the whole table is expensive? No, we have the data.
            # But we don't store the data in a persistent list in the class, we parse it from files each time.
            # So just removing visual row is risky if we don't update listeners.
            
            # Better approach: Use finder to find the button's row at runtime.
            # Or simpler: Re-scanning is safest but might reset other things? No.
            # But since we just want to remove from the current list...
            
            # Let's update the buttons.
            for r in range(self.file_list_table.rowCount()):
                widget = self.file_list_table.cellWidget(r, 5)
                if widget:
                    # Find the button inside
                    btn = widget.findChild(QPushButton)
                    if btn:
                        # Disconnect all
                        try: btn.clicked.disconnect() 
                        except: pass
                        # Reconnect
                        btn.clicked.connect(lambda checked, current_r=r: self.remove_file_row(current_r))
                        
            # Update status label
            self.update_ready_count_status()
            
        except Exception as e:
            print(f"Error removing row: {e}")

    def update_ready_count_status(self):
        """Update the status text with current matched pairs count"""
        ready_count = 0
        for row in range(self.file_list_table.rowCount()):
            status_item = self.file_list_table.item(row, 4)
            if status_item and "Ready" in status_item.text():
                ready_count += 1
        self.merge_status.setText(f"Found {ready_count} matched pair(s) ready to process")
    def generate_clips(self):
        """Generate clips in background thread"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An operation is already running!")
            return
        
        # Update settings from UI
        self.settings["images_dir"] = self.images_path.text()
        self.settings["overlays_dir"] = self.overlays_path.text()
        self.settings["storage_dir"] = self.storage_path.text()
        self.save_settings()
        
        # Validate directories
        if not os.path.exists(self.settings["images_dir"]):
            QMessageBox.critical(self, "Error", f"Images directory not found: {self.settings['images_dir']}")
            return
        
        count = self.clip_count.value()
        
        # Show progress
        self.clip_progress.setVisible(True)
        self.clip_progress.setRange(0, 0)  # Indeterminate
        self.clip_status.setText(f"Generating {count} clips...")
        self.generate_btn.setEnabled(False)
        
        # Start worker thread
        self.worker = WorkerThread(
            generate_clip_storage,
            count=count,
            storage_dir=self.settings["storage_dir"],
            images_dir=self.settings["images_dir"],
            overlays_dir=self.settings["overlays_dir"],
            width=1280,
            height=720,
            fps=24,
            seg_min=90,
            seg_max=150,
            seed=0,
            max_workers=2
        )
        self.worker.finished.connect(self.on_clip_generation_finished)
        self.worker.start()
    
    def on_clip_generation_finished(self, success, message):
        """Handle clip generation completion"""
        self.clip_progress.setVisible(False)
        self.generate_btn.setEnabled(True)
        
        if success:
            self.clip_status.setText(f"✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.clip_status.setText(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)
    
    def process_audio_video_batch_ui(self):
        """Process all matched MP3+MP4 pairs in background thread"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An operation is already running!")
            return
        
        # Validate inputs
        audio_folder = self.audio_folder_path.text()
        video_folder = self.video_folder_path.text()
        output_folder = self.output_folder_path.text()
        
        if not audio_folder or not os.path.exists(audio_folder):
            QMessageBox.critical(self, "Error", "Please select a valid audio folder!")
            return
        
        if not video_folder or not os.path.exists(video_folder):
            QMessageBox.critical(self, "Error", "Please select a valid video folder!")
            return
        
        if not output_folder:
            QMessageBox.critical(self, "Error", "Please specify an output folder!")
            return
        
        # Check if there are matched pairs
        if self.file_list_table.rowCount() == 0:
            QMessageBox.critical(self, "Error", "No files to process! Please select audio and video folders.")
            return
        
        # Count ready pairs
        ready_count = 0
        for row in range(self.file_list_table.rowCount()):
            status = self.file_list_table.item(row, 4).text()
            if "Ready" in status:
                ready_count += 1
        
        if ready_count == 0:
            QMessageBox.critical(self, "Error", "No matched file pairs found!")
            return
        
        # Show progress
        self.merge_progress.setVisible(True)
        self.merge_progress.setRange(0, 0)  # Indeterminate
        self.merge_status.setText(f"Processing {ready_count} file pair(s)...")
        self.process_btn.setEnabled(False)
        
        # Get caption settings
        generate_captions = self.caption_checkbox.isChecked()
        whisper_model = self.whisper_model_combo.currentText()
        burn_captions = self.burn_render_caption_checkbox.isChecked()
        caption_color = self.caption_color_combo.currentText().lower()
        caption_font_size = self.caption_font_size.value()
        
        # Start worker thread
        self.worker = WorkerThread(
            process_audio_video_batch,
            audio_folder=audio_folder,
            video_folder=video_folder,
            output_folder=output_folder,
            generate_captions=generate_captions,
            whisper_model=whisper_model,
            burn_captions=burn_captions,
            caption_color=caption_color,
            caption_font_size=caption_font_size
        )
        self.worker.finished.connect(self.on_batch_process_finished)
        self.worker.start()
    
    def on_batch_process_finished(self, success, message):
        """Handle batch processing completion"""
        self.merge_progress.setVisible(False)
        self.process_btn.setEnabled(True)
        
        if success:
            self.merge_status.setText(f"✅ {message}")
            QMessageBox.information(self, "Success", "Batch processing completed!")
        else:
            self.merge_status.setText(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)
    
    def add_batch_to_queue_ui(self):
        """Add batch processing to queue"""
        # Validate inputs
        audio_folder = self.audio_folder_path.text()
        video_folder = self.video_folder_path.text()
        output_folder = self.output_folder_path.text()
        
        if not audio_folder or not os.path.exists(audio_folder):
            QMessageBox.critical(self, "Error", "Please select a valid audio folder!")
            return
        
        if not video_folder or not os.path.exists(video_folder):
            QMessageBox.critical(self, "Error", "Please select a valid video folder!")
            return
        
        if not output_folder:
            QMessageBox.critical(self, "Error", "Please specify an output folder!")
            return
        
        # Create task (simplified - just store folder paths)
        task = RenderTask(
            audio_path=audio_folder,  # Store audio folder
            output_path=output_folder,
            generate_captions=self.caption_checkbox.isChecked(),
            whisper_model=self.whisper_model_combo.currentText(),
            burn_captions=self.burn_render_caption_checkbox.isChecked()
        )
        # Store video folder in a custom attribute
        task.video_folder = video_folder
        task.task_type = 'batch_merge'
        
        self.task_queue.add_task(task)
        self.merge_status.setText(f"✅ Added batch processing to queue")
        QMessageBox.information(self, "Success", "Batch processing task added to queue!")
        
        # Switch to queue tab
        self.tab_widget.setCurrentIndex(1)
    
    def render_video_ui(self):
        """Render video in background thread"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An operation is already running!")
            return
        
        # Update settings from UI
        self.settings["images_dir"] = self.images_path.text()
        self.settings["overlays_dir"] = self.overlays_path.text()
        self.settings["storage_dir"] = self.storage_path.text()
        self.save_settings()
        
        # Validate inputs
        audio_path = self.audio_path.text()
        output_path = self.output_path.text()
        
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.critical(self, "Error", "Please select a valid audio file!")
            return
        
        if not output_path:
            QMessageBox.critical(self, "Error", "Please specify an output path!")
            return
        
        if not os.path.exists(self.settings["images_dir"]):
            QMessageBox.critical(self, "Error", f"Images directory not found: {self.settings['images_dir']}")
            return
        
        # Show progress
        self.render_progress.setVisible(True)
        self.render_progress.setRange(0, 0)  # Indeterminate
        self.render_status.setText("Rendering video...")
        self.render_btn.setEnabled(False)
        
        # Start worker thread
        self.worker = WorkerThread(
            render_video,
            audio_path=audio_path,
            images_dir=self.settings["images_dir"],
            overlays_dir=self.settings["overlays_dir"],
            out_path=output_path,
            seed=0,
            width=1280,
            height=720,
            fps=24,
            seg_min=90,
            seg_max=150,
            crossfade=1.5
        )
        self.worker.finished.connect(self.on_render_finished)
        self.worker.start()
        
        # Store caption settings for post-processing
        self.pending_caption_generation = {
            'enabled': self.caption_checkbox.isChecked(),
            'model': self.whisper_model_combo.currentText(),
            'output_path': output_path,
            'burn': self.burn_render_caption_checkbox.isChecked()
        }
    
    def on_render_finished(self, success, message):
        """Handle video rendering completion"""
        self.render_progress.setVisible(False)
        self.render_btn.setEnabled(True)
        
        if success:
            self.render_status.setText(f"✅ {message}")
            
            # Check if caption generation was requested
            if hasattr(self, 'pending_caption_generation') and self.pending_caption_generation['enabled']:
                self.render_status.setText("✅ Video rendered! Generating captions...")
                self.generate_captions_for_video(
                    self.pending_caption_generation['output_path'],
                    self.pending_caption_generation['model'],
                    self.pending_caption_generation['burn']
                )
            else:
                QMessageBox.information(self, "Success", f"Video rendered successfully!\n\nOutput: {self.output_path.text()}")
        else:
            self.render_status.setText(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)
    
    def generate_captions_ui(self):
        """Generate captions for standalone video/audio file"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An operation is already running!")
            return
        
        file_path = self.caption_file_path.text()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", "Please select a valid video/audio file!")
            return
        
        model = self.caption_model_combo.currentText()
        burn_captions = self.burn_caption_checkbox.isChecked()
        
        # Show progress
        self.caption_progress.setVisible(True)
        self.caption_progress.setRange(0, 0)
        self.caption_status.setText(f"Generating captions with {model} model...")
        self.caption_gen_btn.setEnabled(False)
        
        # Generate SRT path
        file_obj = Path(file_path)
        srt_path = str(file_obj.with_suffix('.srt'))
        
        # Start worker thread
        def caption_worker():
            # Generate SRT
            generate_captions_whisper(file_path, srt_path, model, language="auto")
            
            # Burn captions if requested and input is video
            if burn_captions and file_obj.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                output_with_captions = str(file_obj.parent / f"{file_obj.stem}_captioned.mp4")
                burn_captions_to_video(file_path, srt_path, output_with_captions)
                return output_with_captions
            return srt_path
        
        self.worker = WorkerThread(caption_worker)
        self.worker.finished.connect(self.on_caption_generation_finished)
        self.worker.start()
    
    def generate_captions_for_video(self, video_path, model, burn=False):
        """Generate captions after video rendering"""
        if self.worker and self.worker.isRunning():
            return
        
        # Generate SRT path
        video_obj = Path(video_path)
        srt_path = str(video_obj.with_suffix('.srt'))
        
        self.render_status.setText(f"Generating captions with {model} model...")
        
        # Start worker thread
        def caption_worker():
            generate_captions_whisper(video_path, srt_path, model, language="auto")
            
            # Burn captions if requested
            if burn:
                output_with_captions = str(video_obj.parent / f"{video_obj.stem}_captioned.mp4")
                burn_captions_to_video(video_path, srt_path, output_with_captions)
                return output_with_captions
            return srt_path
        
        self.worker = WorkerThread(caption_worker)
        self.worker.finished.connect(lambda success, msg: self.on_post_render_caption_finished(success, msg, video_path, burn))
        self.worker.start()
    
    def on_caption_generation_finished(self, success, message):
        """Handle standalone caption generation completion"""
        self.caption_progress.setVisible(False)
        self.caption_gen_btn.setEnabled(True)
        
        if success:
            self.caption_status.setText(f"✅ Captions generated successfully!")
            QMessageBox.information(self, "Success", "Captions generated successfully!")
        else:
            self.caption_status.setText(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)
    
    def on_post_render_caption_finished(self, success, message, video_path, burn=False):
        """Handle caption generation after video rendering"""
        if success:
            srt_path = Path(video_path).with_suffix('.srt')
            if burn:
                captioned_video = Path(video_path).parent / f"{Path(video_path).stem}_captioned.mp4"
                self.render_status.setText(f"✅ Video, captions, and burned video generated!")
                QMessageBox.information(
                    self, "Success",
                    f"Video rendered and captions generated!\n\nOriginal: {video_path}\nCaptions: {srt_path}\nBurned: {captioned_video}"
                )
            else:
                self.render_status.setText(f"✅ Video and captions generated!")
                QMessageBox.information(
                    self, "Success",
                    f"Video rendered and captions generated!\n\nVideo: {video_path}\nCaptions: {srt_path}"
                )
        else:
            self.render_status.setText(f"✅ Video rendered (caption generation failed)")
            QMessageBox.warning(
                self, "Partial Success",
                f"Video rendered successfully but caption generation failed:\n{message}"
            )
    
    # ========== Queue Management Methods ==========
    
    def add_to_queue_ui(self):
        """Add current render settings to queue"""
        # Validate inputs
        audio_path = self.audio_path.text()
        output_path = self.output_path.text()
        
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.critical(self, "Error", "Please select a valid audio file!")
            return
        
        if not output_path:
            QMessageBox.critical(self, "Error", "Please specify an output path!")
            return
        
        # Create task
        task = RenderTask.create(
            audio_path=audio_path,
            output_path=output_path,
            generate_captions=self.caption_checkbox.isChecked(),
            whisper_model=self.whisper_model_combo.currentText(),
            burn_captions=self.burn_render_caption_checkbox.isChecked()
        )
        
        # Add to queue
        task_id = self.task_queue.add_task(task)
        
        # Update table
        self.update_queue_table()
        
        # Show confirmation
        QMessageBox.information(
            self, "Added to Queue",
            f"Render task added to queue!\n\nAudio: {Path(audio_path).name}\nPosition: #{len(self.task_queue.get_all_tasks())}"
        )
    
    def add_clip_generation_to_queue(self):
        """Add clip generation task to queue"""
        # Update settings from UI
        self.settings["images_dir"] = self.images_path.text()
        self.settings["overlays_dir"] = self.overlays_path.text()
        self.settings["storage_dir"] = self.storage_path.text()
        self.save_settings()
        
        # Validate directories
        if not os.path.exists(self.settings["images_dir"]):
            QMessageBox.critical(self, "Error", f"Images directory not found: {self.settings['images_dir']}")
            return
        
        count = self.clip_count.value()
        
        # Create clip generation task
        task = RenderTask.create_clip_generation(
            clip_count=count,
            storage_dir=self.settings["storage_dir"],
            images_dir=self.settings["images_dir"],
            overlays_dir=self.settings["overlays_dir"]
        )
        
        # Add to queue
        self.task_queue.add_task(task)
        
        # Update table
        self.update_queue_table()
        
        # Show confirmation
        QMessageBox.information(
            self, "Added to Queue",
            f"Clip generation task added to queue!\n\nClips: {count}\nStorage: {self.settings['storage_dir']}\nPosition: #{len(self.task_queue.get_all_tasks())}"
        )
    
    def update_queue_table(self):
        """Update queue table with current tasks"""
        tasks = self.task_queue.get_all_tasks()
        
        # Update status label
        stats = self.task_queue.get_queue_stats()
        self.queue_status_label.setText(
            f"Queue: {stats['total']} tasks ({stats['pending']} pending, "
            f"{stats['running']} running, {stats['completed']} completed)"
        )
        
        # Update table
        self.queue_table.setRowCount(len(tasks))
        
        for row, task in enumerate(tasks):
            # Get task type
            task_type = getattr(task, 'task_type', 'render')
            
            # Status icon
            status_icons = {
                'pending': '⏳ Pending',
                'running': '▶️ Running',
                'completed': '✅ Done',
                'failed': '❌ Failed'
            }
            status_item = QTableWidgetItem(status_icons.get(task.status, task.status))
            if task.status == 'completed':
                status_item.setForeground(QColor(166, 227, 161))  # Green
            elif task.status == 'failed':
                status_item.setForeground(QColor(243, 139, 168))  # Red
            elif task.status == 'running':
                status_item.setForeground(QColor(137, 180, 250))  # Blue
            self.queue_table.setItem(row, 0, status_item)
            
            # Audio file / Task info
            if task_type == 'clip_generation':
                audio_item = QTableWidgetItem(f"🎞️ Generate {task.clip_count} clips")
                audio_item.setForeground(QColor(250, 204, 21))  # Yellow
            else:
                audio_item = QTableWidgetItem(f"🎥 {Path(task.audio_path).name}")
            self.queue_table.setItem(row, 1, audio_item)
            
            # Output file / Destination
            if task_type == 'clip_generation':
                output_item = QTableWidgetItem(task.output_path)
            else:
                output_item = QTableWidgetItem(Path(task.output_path).name)
            self.queue_table.setItem(row, 2, output_item)
            
            # Captions / Extra info
            if task_type == 'clip_generation':
                caption_item = QTableWidgetItem(f"{task.clip_count} clips")
            else:
                caption_text = "Yes" if task.generate_captions else "No"
                if task.generate_captions and task.burn_captions:
                    caption_text += " (Burn)"
                caption_item = QTableWidgetItem(caption_text)
            self.queue_table.setItem(row, 3, caption_item)
            
            # Progress bar
            progress_widget = QWidget()
            progress_layout = QHBoxLayout(progress_widget)
            progress_layout.setContentsMargins(5, 2, 5, 2)
            progress_bar = QProgressBar()
            progress_bar.setValue(task.progress)
            progress_bar.setMaximumHeight(20)
            progress_layout.addWidget(progress_bar)
            self.queue_table.setCellWidget(row, 4, progress_widget)
            
            # Actions
            actions_widget = QWidget()
            actions_widget.setStyleSheet("background: transparent;")
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 4, 4, 4)
            actions_layout.setSpacing(4)
            
            # Move up button
            up_btn = QPushButton("▲")
            up_btn.setFixedSize(32, 28)
            up_btn.setEnabled(task.status == 'pending' and row > 0)
            up_btn.clicked.connect(lambda checked, tid=task.id: self.move_task_up(tid))
            up_btn.setStyleSheet("""
                QPushButton { 
                    background: rgba(51, 65, 85, 0.8); 
                    border: 1px solid rgba(100, 116, 139, 0.5);
                    border-radius: 4px;
                    color: #e2e8f0;
                    font-size: 12px;
                }
                QPushButton:hover { background: rgba(14, 165, 233, 0.5); }
                QPushButton:disabled { background: rgba(30, 41, 59, 0.5); color: #475569; }
            """)
            actions_layout.addWidget(up_btn)
            
            # Move down button
            down_btn = QPushButton("▼")
            down_btn.setFixedSize(32, 28)
            down_btn.setEnabled(task.status == 'pending' and row < len(tasks) - 1)
            down_btn.clicked.connect(lambda checked, tid=task.id: self.move_task_down(tid))
            down_btn.setStyleSheet("""
                QPushButton { 
                    background: rgba(51, 65, 85, 0.8); 
                    border: 1px solid rgba(100, 116, 139, 0.5);
                    border-radius: 4px;
                    color: #e2e8f0;
                    font-size: 12px;
                }
                QPushButton:hover { background: rgba(14, 165, 233, 0.5); }
                QPushButton:disabled { background: rgba(30, 41, 59, 0.5); color: #475569; }
            """)
            actions_layout.addWidget(down_btn)
            
            # Remove button
            remove_btn = QPushButton("✕")
            remove_btn.setFixedSize(32, 28)
            remove_btn.setEnabled(task.status != 'running')
            remove_btn.clicked.connect(lambda checked, tid=task.id: self.remove_task(tid))
            remove_btn.setStyleSheet("""
                QPushButton { 
                    background: rgba(239, 68, 68, 0.6); 
                    border: 1px solid rgba(239, 68, 68, 0.8);
                    border-radius: 4px;
                    color: #ffffff;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover { background: rgba(239, 68, 68, 0.9); }
                QPushButton:disabled { background: rgba(30, 41, 59, 0.5); color: #475569; border-color: rgba(100, 116, 139, 0.3); }
            """)
            actions_layout.addWidget(remove_btn)
            
            self.queue_table.setCellWidget(row, 5, actions_widget)
            
            # Task ID (hidden)
            id_item = QTableWidgetItem(task.id)
            self.queue_table.setItem(row, 6, id_item)
    
    def start_queue(self):
        """Start processing queue"""
        if self.queue_worker and self.queue_worker.isRunning():
            QMessageBox.information(self, "Info", "Queue is already running!")
            return
        
        # Check if there are pending tasks
        stats = self.task_queue.get_queue_stats()
        if stats['pending'] == 0:
            QMessageBox.information(self, "Info", "No pending tasks in queue!")
            return
        
        # Update settings for queue worker (tasks already have their paths)
        # No need to modify tasks here
        
        # Start queue worker
        self.queue_worker = QueueWorkerThread(self.task_queue)
        self.queue_worker.task_started.connect(self.on_queue_task_started)
        self.queue_worker.task_completed.connect(self.on_queue_task_completed)
        self.queue_worker.queue_empty.connect(self.on_queue_empty)
        self.queue_worker.start()
        
        # Update UI
        self.start_queue_btn.setEnabled(False)
        self.pause_queue_btn.setEnabled(True)
        
        print("Queue processing started")
    
    def pause_queue(self):
        """Pause queue processing"""
        if self.queue_worker:
            self.queue_worker.pause()
            self.start_queue_btn.setEnabled(True)
            self.pause_queue_btn.setEnabled(False)
            print("Queue paused")
    
    def clear_completed_tasks(self):
        """Remove all completed and failed tasks"""
        self.task_queue.clear_completed()
        self.update_queue_table()
        print("Cleared completed tasks")
    
    def move_task_up(self, task_id):
        """Move task up in queue"""
        if self.task_queue.move_task_up(task_id):
            self.update_queue_table()
    
    def move_task_down(self, task_id):
        """Move task down in queue"""
        if self.task_queue.move_task_down(task_id):
            self.update_queue_table()
    
    def remove_task(self, task_id):
        """Remove task from queue"""
        if self.task_queue.remove_task(task_id):
            self.update_queue_table()
    
    def on_queue_task_started(self, task_id):
        """Handle queue task started"""
        self.update_queue_table()
        task = self.task_queue.get_task_by_id(task_id)
        if task:
            print(f"Started processing: {Path(task.audio_path).name}")
    
    def on_queue_task_completed(self, task_id, success, message):
        """Handle queue task completed"""
        self.update_queue_table()
        task = self.task_queue.get_task_by_id(task_id)
        if task:
            status = "✅ Completed" if success else "❌ Failed"
            print(f"{status}: {Path(task.audio_path).name}")
    
    def on_queue_empty(self):
        """Handle queue empty"""
        # Queue is empty, but keep worker running to check for new tasks
        pass
    
    # ========== End Queue Management Methods ==========
    
    # ========== Burn Captions Methods ==========
    
    def browse_burn_video_folder(self):
        """Browse for video folder for burning captions"""
        current_path = self.burn_video_path.text() or "."
        folder = QFileDialog.getExistingDirectory(
            self, "Select Video Folder (MP4s)", current_path
        )
        if folder:
            self.burn_video_path.setText(folder)
            self.settings["last_burn_video_folder"] = folder
            self.save_settings()
            self.update_burn_file_list()

    def browse_burn_srt_folder(self):
        """Browse for subtitle folder for burning captions"""
        current_path = self.burn_srt_path.text() or "."
        folder = QFileDialog.getExistingDirectory(
            self, "Select Subtitle Folder (SRTs)", current_path
        )
        if folder:
            self.burn_srt_path.setText(folder)
            self.settings["last_burn_srt_folder"] = folder
            self.save_settings()
            self.update_burn_file_list()

    def browse_burn_output_folder(self):
        """Browse for output folder for burning captions"""
        current_path = self.burn_output_path.text() or "."
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", current_path
        )
        if folder:
            self.burn_output_path.setText(folder)
            self.settings["last_burn_output_folder"] = folder
            self.save_settings()

    def update_burn_file_list(self):
        """Update burn file list table with matched MP4-SRT pairs"""
        self.burn_file_table.setRowCount(0)
        
        video_folder = self.burn_video_path.text()
        srt_folder = self.burn_srt_path.text()
        
        if not video_folder or not srt_folder:
            return
        
        if not os.path.exists(video_folder) or not os.path.exists(srt_folder):
            return
        
        # Scan for MP4 files
        video_path = Path(video_folder)
        mp4_files = {f.stem: f.name for f in video_path.glob("*.mp4")}
        mp4_files.update({f.stem: f.name for f in video_path.glob("*.MP4")})
        
        # Scan for SRT files
        srt_path = Path(srt_folder)
        srt_files = {f.stem: f.name for f in srt_path.glob("*.srt")}
        srt_files.update({f.stem: f.name for f in srt_path.glob("*.SRT")})
        
        # Get all unique names
        all_names = set(mp4_files.keys()) | set(srt_files.keys())
        
        # Add matched pairs to table
        row = 0
        
        for name in sorted(all_names):
            self.burn_file_table.insertRow(row)
            
            mp4_name = mp4_files.get(name, "")
            srt_name = srt_files.get(name, "")
            
            # Column 0: MP4 file name
            mp4_item = QTableWidgetItem(mp4_name if mp4_name else "⚠️ Missing")
            if not mp4_name:
                mp4_item.setForeground(QColor("#FF6B6B"))
            self.burn_file_table.setItem(row, 0, mp4_item)
            
            # Column 1: MP4 duration (Optional, can include if needed)
            duration_item = QTableWidgetItem("")
            if mp4_name:
                try:
                    duration = get_video_duration(str(video_path / mp4_name))
                    hours = int(duration // 3600)
                    minutes = int((duration % 3600) // 60)
                    seconds = int(duration % 60)
                    duration_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    duration_item.setText(duration_text)
                    duration_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                except:
                    pass
            self.burn_file_table.setItem(row, 1, duration_item)
            
            # Column 2: SRT file name
            srt_item = QTableWidgetItem(srt_name if srt_name else "⚠️ Missing")
            if not srt_name:
                srt_item.setForeground(QColor("#FF6B6B"))
            self.burn_file_table.setItem(row, 2, srt_item)
            
            # Column 3: Status
            if mp4_name and srt_name:
                status_item = QTableWidgetItem("✓ Ready")
                status_item.setForeground(QColor("#51CF66"))
            else:
                status_item = QTableWidgetItem("⚠️ Skip")
                status_item.setForeground(QColor("#FF6B6B"))
            self.burn_file_table.setItem(row, 3, status_item)
            
            # Column 4: Actions (Remove button)
            remove_btn = QPushButton("❌")
            remove_btn.setFixedSize(30, 30)
            remove_btn.setToolTip("Remove this file from list")
            remove_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: transparent;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 107, 107, 0.2);
                    border-radius: 4px;
                }
            """)
            # Use closure to capture current row
            remove_btn.clicked.connect(lambda checked, r=row: self.remove_burn_file_row(r))
            
            # Create a container widget for the button
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn_layout.addWidget(remove_btn)
            
            self.burn_file_table.setCellWidget(row, 4, btn_widget)
            
            row += 1
            
        # Update status
        matched_count = sum(1 for name in all_names if name in mp4_files and name in srt_files)
        self.burn_status_label.setText(f"Found {matched_count} matched pair(s) ready to process")

    def remove_burn_file_row(self, row):
        """Remove a row from the burn file list table"""
        try:
            self.burn_file_table.removeRow(row)
            
            # Update buttons
            for r in range(self.burn_file_table.rowCount()):
                widget = self.burn_file_table.cellWidget(r, 4)
                if widget:
                    btn = widget.findChild(QPushButton)
                    if btn:
                        try: btn.clicked.disconnect() 
                        except: pass
                        btn.clicked.connect(lambda checked, current_r=r: self.remove_burn_file_row(current_r))
            
            # Update status
            ready_count = 0
            for r in range(self.burn_file_table.rowCount()):
                status_item = self.burn_file_table.item(r, 3)
                if status_item and "Ready" in status_item.text():
                    ready_count += 1
            self.burn_status_label.setText(f"Found {ready_count} matched pair(s) ready to process")
            
        except Exception as e:
            print(f"Error removing row: {e}")

    def process_burn_batch_ui(self):
        """Process batch burning of captions"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "An operation is already running!")
            return
        
        video_folder = self.burn_video_path.text()
        srt_folder = self.burn_srt_path.text()
        output_folder = self.burn_output_path.text()
        
        if not video_folder or not os.path.exists(video_folder):
            QMessageBox.critical(self, "Error", "Please select a valid video folder!")
            return
        
        if not srt_folder or not os.path.exists(srt_folder):
            QMessageBox.critical(self, "Error", "Please select a valid subtitle folder!")
            return
            
        if not output_folder:
            QMessageBox.critical(self, "Error", "Please specify an output folder!")
            return
            
        # Check for ready files
        ready_count = 0
        for r in range(self.burn_file_table.rowCount()):
            if "Ready" in self.burn_file_table.item(r, 3).text():
                ready_count += 1
                
        if ready_count == 0:
            QMessageBox.critical(self, "Error", "No matched files to process!")
            return
            
        # Get settings
        font_size = self.burn_font_size.value()
        color = self.burn_color_combo.currentText()
        
        # Disable UI
        self.burn_process_btn.setEnabled(False)
        self.burn_status_label.setText(f"Processing {ready_count} pairs...")
        
        # Start Worker
        self.worker = WorkerThread(
            process_burn_captions_batch,
            video_folder=video_folder,
            srt_folder=srt_folder,
            output_folder=output_folder,
            font_size=font_size,
            font_color=color
        )
        self.worker.finished.connect(self.on_burn_batch_finished)
        self.worker.start()

    def on_burn_batch_finished(self, success, message):
        """Handle completion of burn batch"""
        self.burn_process_btn.setEnabled(True)
        if success:
            self.burn_status_label.setText(f"✅ Batch processing complete!")
            QMessageBox.information(self, "Success", "Batch burning completed successfully!")
        else:
            self.burn_status_label.setText(f"❌ Error: {message}")
            QMessageBox.critical(self, "Error", message)

    def apply_styles(self):
        """Apply premium modern dark theme with glassmorphism"""
        self.setStyleSheet("""
            /* ============================================
               PREMIUM DARK THEME - Audiobook Video Creator
               ============================================ */
            
            /* Main Window - Deep gradient background */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0a0a12, stop:0.5 #12121f, stop:1 #0f0f1a);
            }
            
            /* Base Widget */
            QWidget {
                background: transparent;
                color: #e2e8f0;
                font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
                font-size: 10pt;
            }
            
            /* Central Widget - maintains gradient */
            QMainWindow > QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0a0a12, stop:0.5 #12121f, stop:1 #0f0f1a);
            }
            
            /* Title Styling */
            #title {
                color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #06b6d4, stop:1 #8b5cf6);
                padding: 20px;
                font-size: 26pt;
                font-weight: bold;
            }
            
            /* Subtitle */
            #subtitle {
                color: #64748b;
                padding: 0px 20px 15px 20px;
                font-size: 11pt;
            }
            
            /* Labels */
            QLabel {
                color: #cbd5e1;
                padding: 3px;
                background: transparent;
            }
            
            /* ============================================
               TAB WIDGET - Modern Tab Design
               ============================================ */
            QTabWidget {
                background: transparent;
            }
            
            QTabWidget::pane {
                background-color: rgba(15, 23, 42, 0.5);
                border: 1px solid rgba(100, 116, 139, 0.3);
                border-radius: 12px;
                padding: 10px;
                margin-top: -1px;
            }
            
            QTabBar::tab {
                background: rgba(30, 41, 59, 0.6);
                color: #94a3b8;
                padding: 12px 30px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: bold;
                font-size: 11pt;
                min-width: 120px;
            }
            
            QTabBar::tab:hover {
                background: rgba(51, 65, 85, 0.8);
                color: #e2e8f0;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0ea5e9, stop:1 #0891b2);
                color: #ffffff;
            }
            
            QTabBar::tab:selected:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #38bdf8, stop:1 #0ea5e9);
            }
            
            /* ============================================
               GROUP BOXES - Glassmorphism Cards
               ============================================ */
            QGroupBox {
                background-color: rgba(30, 30, 50, 0.7);
                border: 1px solid rgba(100, 116, 139, 0.3);
                border-radius: 16px;
                margin-top: 20px;
                padding: 25px 20px 20px 20px;
                font-weight: 600;
                font-size: 11pt;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 10px 20px;
                color: #38bdf8;
                font-size: 13pt;
                font-weight: bold;
                background: transparent;
            }
            
            /* ============================================
               INPUT FIELDS - Sleek design
               ============================================ */
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(100, 116, 139, 0.4);
                border-radius: 10px;
                padding: 10px 15px;
                color: #f1f5f9;
                selection-background-color: #0ea5e9;
                font-size: 10pt;
            }
            
            QLineEdit:hover {
                border: 1px solid rgba(56, 189, 248, 0.5);
                background-color: rgba(15, 23, 42, 0.9);
            }
            
            QLineEdit:focus {
                border: 2px solid #0ea5e9;
                background-color: rgba(15, 23, 42, 1);
            }
            
            /* ============================================
               BUTTONS - Gradient & Animations
               ============================================ */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0891b2, stop:1 #0284c7);
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 10pt;
                min-width: 100px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #06b6d4, stop:1 #0ea5e9);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0e7490, stop:1 #0369a1);
            }
            
            QPushButton:disabled {
                background: rgba(51, 65, 85, 0.5);
                color: #64748b;
            }
            
            /* Secondary/Small Buttons */
            QPushButton[flat="true"], 
            QTableWidget QPushButton {
                background: rgba(51, 65, 85, 0.6);
                border: 1px solid rgba(100, 116, 139, 0.3);
                min-width: 30px;
                padding: 6px 12px;
            }
            
            QTableWidget QPushButton:hover {
                background: rgba(71, 85, 105, 0.8);
                border: 1px solid #0ea5e9;
            }
            
            /* ============================================
               SPIN BOXES
               ============================================ */
            QSpinBox {
                background-color: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(100, 116, 139, 0.4);
                border-radius: 10px;
                padding: 10px 15px;
                color: #f1f5f9;
                min-width: 140px;
                font-size: 10pt;
            }
            
            QSpinBox:hover {
                border: 1px solid rgba(56, 189, 248, 0.5);
            }
            
            QSpinBox:focus {
                border: 2px solid #0ea5e9;
            }
            
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: rgba(51, 65, 85, 0.6);
                border-radius: 4px;
                width: 20px;
                margin: 2px;
            }
            
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: rgba(71, 85, 105, 0.8);
            }
            
            QSpinBox::up-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #94a3b8;
                width: 0; height: 0;
            }
            
            QSpinBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #94a3b8;
                width: 0; height: 0;
            }
            
            /* ============================================
               CONSOLE - Terminal Style
               ============================================ */
            QTextEdit {
                background-color: rgba(2, 6, 23, 0.95);
                border: 1px solid rgba(51, 65, 85, 0.5);
                border-radius: 12px;
                padding: 15px;
                color: #4ade80;
                font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
                font-size: 9pt;
                selection-background-color: #0ea5e9;
            }
            
            /* ============================================
               PROGRESS BARS - Animated Gradient
               ============================================ */
            QProgressBar {
                background-color: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(51, 65, 85, 0.5);
                border-radius: 10px;
                text-align: center;
                height: 26px;
                font-weight: bold;
                color: #e2e8f0;
                font-size: 9pt;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0891b2, stop:0.5 #0ea5e9, stop:1 #38bdf8);
                border-radius: 8px;
            }
            
            /* ============================================
               COMBO BOXES
               ============================================ */
            QComboBox {
                background-color: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(100, 116, 139, 0.4);
                border-radius: 10px;
                padding: 10px 15px;
                color: #f1f5f9;
                min-width: 120px;
                font-size: 10pt;
            }
            
            QComboBox:hover {
                border: 1px solid rgba(56, 189, 248, 0.5);
            }
            
            QComboBox:focus, QComboBox:on {
                border: 2px solid #0ea5e9;
            }
            
            QComboBox::drop-down {
                border: none;
                background: transparent;
                width: 30px;
                padding-right: 10px;
            }
            
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #94a3b8;
                width: 0; height: 0;
            }
            
            QComboBox QAbstractItemView {
                background-color: rgba(30, 41, 59, 0.98);
                border: 1px solid #0ea5e9;
                border-radius: 8px;
                selection-background-color: #0ea5e9;
                selection-color: #ffffff;
                color: #e2e8f0;
                padding: 5px;
            }
            
            QComboBox QAbstractItemView::item {
                padding: 8px 12px;
                border-radius: 4px;
            }
            
            QComboBox QAbstractItemView::item:hover {
                background-color: rgba(14, 165, 233, 0.3);
            }
            
            /* ============================================
               CHECK BOXES
               ============================================ */
            QCheckBox {
                color: #cbd5e1;
                spacing: 10px;
                font-size: 10pt;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid rgba(100, 116, 139, 0.5);
                border-radius: 6px;
                background-color: rgba(15, 23, 42, 0.8);
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #0ea5e9;
                background-color: rgba(14, 165, 233, 0.1);
            }
            
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0891b2, stop:1 #0ea5e9);
                border: 2px solid #0ea5e9;
            }
            
            QCheckBox::indicator:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #06b6d4, stop:1 #38bdf8);
            }
            
            /* ============================================
               TABLE WIDGET - Modern Data Grid
               ============================================ */
            QTableWidget {
                background-color: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(51, 65, 85, 0.5);
                border-radius: 12px;
                gridline-color: rgba(51, 65, 85, 0.3);
                color: #e2e8f0;
                font-size: 10pt;
            }
            
            QTableWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid rgba(51, 65, 85, 0.2);
            }
            
            QTableWidget::item:selected {
                background-color: rgba(14, 165, 233, 0.3);
                color: #ffffff;
            }
            
            QTableWidget::item:hover {
                background-color: rgba(51, 65, 85, 0.4);
            }
            
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(51, 65, 85, 0.8), stop:1 rgba(30, 41, 59, 0.8));
                color: #94a3b8;
                padding: 12px 10px;
                border: none;
                border-bottom: 2px solid rgba(14, 165, 233, 0.5);
                font-weight: bold;
                font-size: 9pt;
                text-transform: uppercase;
            }
            
            QHeaderView::section:hover {
                background: rgba(51, 65, 85, 1);
                color: #0ea5e9;
            }
            
            /* Alternating row colors */
            QTableWidget::item:alternate {
                background-color: rgba(30, 41, 59, 0.3);
            }
            
            /* ============================================
               SCROLL BARS - Minimal Design
               ============================================ */
            QScrollBar:vertical {
                background: rgba(15, 23, 42, 0.5);
                width: 10px;
                border-radius: 5px;
                margin: 0;
            }
            
            QScrollBar::handle:vertical {
                background: rgba(100, 116, 139, 0.5);
                border-radius: 5px;
                min-height: 30px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: rgba(14, 165, 233, 0.7);
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            
            QScrollBar:horizontal {
                background: rgba(15, 23, 42, 0.5);
                height: 10px;
                border-radius: 5px;
                margin: 0;
            }
            
            QScrollBar::handle:horizontal {
                background: rgba(100, 116, 139, 0.5);
                border-radius: 5px;
                min-width: 30px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background: rgba(14, 165, 233, 0.7);
            }
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0;
            }
            
            /* ============================================
               MESSAGE BOX
               ============================================ */
            QMessageBox {
                background-color: #1e293b;
            }
            
            QMessageBox QLabel {
                color: #e2e8f0;
                font-size: 10pt;
            }
            
            QMessageBox QPushButton {
                min-width: 80px;
                padding: 8px 20px;
            }
            
            /* ============================================
               TOOLTIPS
               ============================================ */
            QToolTip {
                background-color: rgba(30, 41, 59, 0.95);
                color: #e2e8f0;
                border: 1px solid #0ea5e9;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 9pt;
            }
        """)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Restore original stdout/stderr
        if hasattr(self, 'stdout_redirector') and self.stdout_redirector:
            sys.stdout = self.stdout_redirector.original_stream
        if hasattr(self, 'stderr_redirector') and self.stderr_redirector:
            sys.stderr = self.stderr_redirector.original_stream
        
        # Stop queue worker if running
        if self.queue_worker and self.queue_worker.isRunning():
            self.queue_worker.stop()
            self.queue_worker.wait(2000)  # Wait max 2 seconds
        
        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "An operation is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.worker.terminate()
            self.worker.wait()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = AudiobookVideoCreatorUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
