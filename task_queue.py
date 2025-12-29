"""
Task Queue Management System for Video Rendering

This module provides a thread-safe task queue for managing video rendering jobs.
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import threading


@dataclass
class RenderTask:
    """Represents a single task in the queue (render or clip generation)"""
    id: str
    task_type: str  # 'render' or 'clip_generation'
    audio_path: str  # For render: audio file, For clip: images_dir
    output_path: str  # For render: output video, For clip: storage_dir
    generate_captions: bool
    whisper_model: str
    burn_captions: bool
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: int  # 0-100
    error_message: str
    created_at: str
    completed_at: Optional[str] = None
    # Extra fields for clip generation
    clip_count: int = 0
    overlays_dir: str = ""
    
    @staticmethod
    def create(audio_path: str, output_path: str, generate_captions: bool = False,
               whisper_model: str = "base", burn_captions: bool = False):
        """Create a new render task"""
        return RenderTask(
            id=str(uuid.uuid4()),
            task_type='render',
            audio_path=audio_path,
            output_path=output_path,
            generate_captions=generate_captions,
            whisper_model=whisper_model,
            burn_captions=burn_captions,
            status='pending',
            progress=0,
            error_message='',
            created_at=datetime.now().isoformat()
        )
    
    @staticmethod
    def create_clip_generation(clip_count: int, storage_dir: str, 
                                images_dir: str, overlays_dir: str):
        """Create a new clip generation task"""
        return RenderTask(
            id=str(uuid.uuid4()),
            task_type='clip_generation',
            audio_path=images_dir,  # Store images_dir in audio_path field
            output_path=storage_dir,  # Store storage_dir in output_path field
            generate_captions=False,
            whisper_model="",
            burn_captions=False,
            status='pending',
            progress=0,
            error_message='',
            created_at=datetime.now().isoformat(),
            clip_count=clip_count,
            overlays_dir=overlays_dir
        )
    
    def to_dict(self):
        """Convert task to dictionary"""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: dict):
        """Create task from dictionary"""
        # Handle backward compatibility for old tasks without task_type
        if 'task_type' not in data:
            data['task_type'] = 'render'
        if 'clip_count' not in data:
            data['clip_count'] = 0
        if 'overlays_dir' not in data:
            data['overlays_dir'] = ''
        return RenderTask(**data)


class TaskQueue:
    """Thread-safe task queue manager"""
    
    def __init__(self, queue_file: str = "queue_data.json"):
        self.queue_file = queue_file
        self.tasks: List[RenderTask] = []
        self.lock = threading.Lock()
        self.load_from_file()
    
    def add_task(self, task: RenderTask) -> str:
        """Add a new task to the queue"""
        with self.lock:
            self.tasks.append(task)
            self.save_to_file()
            return task.id
    
    def get_next_task(self) -> Optional[RenderTask]:
        """Get the next pending task"""
        with self.lock:
            for task in self.tasks:
                if task.status == 'pending':
                    return task
            return None
    
    def get_task_by_id(self, task_id: str) -> Optional[RenderTask]:
        """Get task by ID"""
        with self.lock:
            for task in self.tasks:
                if task.id == task_id:
                    return task
            return None
    
    def update_status(self, task_id: str, status: str, progress: int = 0, 
                     error_message: str = ''):
        """Update task status and progress"""
        with self.lock:
            for task in self.tasks:
                if task.id == task_id:
                    task.status = status
                    task.progress = progress
                    task.error_message = error_message
                    if status in ['completed', 'failed']:
                        task.completed_at = datetime.now().isoformat()
                    self.save_to_file()
                    break
    
    def move_task_up(self, task_id: str) -> bool:
        """Move task up in priority (only for pending tasks)"""
        with self.lock:
            task_index = None
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    task_index = i
                    break
            
            if task_index is None or task_index == 0:
                return False
            
            task = self.tasks[task_index]
            if task.status != 'pending':
                return False
            
            # Swap with previous task
            self.tasks[task_index], self.tasks[task_index - 1] = \
                self.tasks[task_index - 1], self.tasks[task_index]
            
            self.save_to_file()
            return True
    
    def move_task_down(self, task_id: str) -> bool:
        """Move task down in priority (only for pending tasks)"""
        with self.lock:
            task_index = None
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    task_index = i
                    break
            
            if task_index is None or task_index >= len(self.tasks) - 1:
                return False
            
            task = self.tasks[task_index]
            if task.status != 'pending':
                return False
            
            # Swap with next task
            self.tasks[task_index], self.tasks[task_index + 1] = \
                self.tasks[task_index + 1], self.tasks[task_index]
            
            self.save_to_file()
            return True
    
    def reorder_task(self, task_id: str, new_position: int) -> bool:
        """Move task to specific position (only for pending tasks)"""
        with self.lock:
            task_index = None
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    task_index = i
                    break
            
            if task_index is None:
                return False
            
            task = self.tasks[task_index]
            if task.status != 'pending':
                return False
            
            if new_position < 0 or new_position >= len(self.tasks):
                return False
            
            # Remove and reinsert at new position
            self.tasks.pop(task_index)
            self.tasks.insert(new_position, task)
            
            self.save_to_file()
            return True
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from queue"""
        with self.lock:
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    # Don't remove running tasks
                    if task.status == 'running':
                        return False
                    self.tasks.pop(i)
                    self.save_to_file()
                    return True
            return False
    
    def clear_completed(self):
        """Remove all completed and failed tasks"""
        with self.lock:
            self.tasks = [t for t in self.tasks 
                         if t.status not in ['completed', 'failed']]
            self.save_to_file()
    
    def get_all_tasks(self) -> List[RenderTask]:
        """Get all tasks"""
        with self.lock:
            return self.tasks.copy()
    
    def get_queue_stats(self) -> dict:
        """Get queue statistics"""
        with self.lock:
            stats = {
                'total': len(self.tasks),
                'pending': sum(1 for t in self.tasks if t.status == 'pending'),
                'running': sum(1 for t in self.tasks if t.status == 'running'),
                'completed': sum(1 for t in self.tasks if t.status == 'completed'),
                'failed': sum(1 for t in self.tasks if t.status == 'failed')
            }
            return stats
    
    def save_to_file(self):
        """Save queue to JSON file"""
        try:
            data = {
                'tasks': [task.to_dict() for task in self.tasks]
            }
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving queue: {e}")
    
    def load_from_file(self):
        """Load queue from JSON file"""
        try:
            if Path(self.queue_file).exists():
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tasks = [RenderTask.from_dict(t) for t in data.get('tasks', [])]
                    # Reset running tasks to pending on load
                    for task in self.tasks:
                        if task.status == 'running':
                            task.status = 'pending'
                            task.progress = 0
        except Exception as e:
            print(f"Error loading queue: {e}")
            self.tasks = []
