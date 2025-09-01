import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import insightface
from insightface.app import FaceAnalysis
import shutil
import os
import tempfile
import base64
import io
from dataclasses import dataclass
import time

# Import functions from photo_cluster_router.py
from photo_cluster_router import (
    FaceRec, Embedder, collect_faces, cluster_faces,
    route_by_clusters, is_image, ensure_dir, load_bgr,
    laplacian_variance, center_crop_square
)

# Page configuration
st.set_page_config(
    page_title="FaceSort - Автоматическая сортировка фото по лицам",
    page_icon="📸",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 1em;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .person-card {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #e8f4f8;
    }
</style>
""", unsafe_allow_html=True)

def cleanup_old_temp_files():
    """Clean up old temporary files (older than 7 days)"""
    try:
        user_home = Path.home()
        temp_base_dir = user_home / "FaceSort_Temp"

        if temp_base_dir.exists():
            import datetime
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=7)

            for temp_dir in temp_base_dir.iterdir():
                if temp_dir.is_dir():
                    try:
                        # Check if directory is old enough to delete
                        stat = temp_dir.stat()
                        dir_date = datetime.datetime.fromtimestamp(stat.st_mtime)

                        if dir_date < cutoff_date:
                            import shutil
                            shutil.rmtree(temp_dir)
                            print(f"Cleaned up old temp directory: {temp_dir}")
                    except Exception as e:
                        print(f"Failed to clean up {temp_dir}: {e}")
    except Exception as e:
        print(f"Error during temp file cleanup: {e}")

def main():
    st.markdown('<h1 class="main-header">📸 FaceSort</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Автоматическая сортировка фотографий по распознанным лицам</p>', unsafe_allow_html=True)

    # Clean up old temporary files on startup
    cleanup_old_temp_files()

    # Initialize default parameters if not set
    if 'params' not in st.session_state:
        st.session_state.params = {
            'eps_sim': 0.55,
            'min_samples': 2,
            'min_face': 110,
            'blur_thr': 45.0,
            'group_thr': 3,
            'det_size': 640,
            'gpu_id': 0
        }

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗂️ Проводник", "⚙️ Параметры", "🚀 Обработка", "📊 Результаты"])

    with tab1:
        show_upload_tab()

    with tab2:
        show_parameters_tab()

    with tab3:
        show_processing_tab()

    with tab4:
        show_results_tab()

def show_upload_tab():
    # Initialize queue if not exists
    if 'processing_queue' not in st.session_state:
        st.session_state.processing_queue = []
    if 'current_processing' not in st.session_state:
        st.session_state.current_processing = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"  # idle, processing, paused

    # Show queue status if there are items
    if st.session_state.processing_queue or st.session_state.current_processing:
        show_processing_queue()

    # Show only the file explorer
    show_file_explorer()

def show_file_uploader():
    """Show traditional file uploader interface"""
    st.subheader("📤 Загрузка файлов")

    uploaded_files = st.file_uploader(
        "Выберите фотографии",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True,
        help="Можно выбрать несколько файлов одновременно или перетащить папку"
    )

    if uploaded_files:
        process_uploaded_files(uploaded_files)

def show_folder_input():
    """Show folder path input interface with folder picker"""
    st.subheader("📁 Выбор папки")

    # Main action button
    if st.button("📂 Выбрать папку через проводник", type="primary", width="stretch"):
        show_folder_picker_instructions()

    # Alternative: Manual path input
    st.markdown("---")
    st.markdown("**Или введите путь вручную:**")

    folder_path = st.text_input(
        "Путь к папке с фотографиями:",
        placeholder="C:/Users/YourName/Pictures/ или /home/user/photos/",
        help="Укажите полный путь к папке с фотографиями"
    )

    if st.button("🔍 Сканировать", width="stretch"):
        if folder_path:
            scan_folder_for_images(folder_path)
        else:
            st.warning("⚠️ Укажите путь к папке")

    # Quick access to common folders
    st.markdown("---")
    st.markdown("**Быстрый доступ:**")

    quick_folders = get_quick_paths()
    if quick_folders:
        # Show only first 4 quick folders
        quick_cols = st.columns(2)
        quick_items = list(quick_folders.items())[:4]

        for i, (name, path) in enumerate(quick_items):
            with quick_cols[i % 2]:
                if st.button(f"📁 {name}", key=f"quick_scan_{i}", width="stretch"):
                    # Update both paths and set as selected
                    st.session_state.explorer_path = path
                    st.session_state.explorer_selected = path
                    st.rerun()

def show_folder_picker_instructions():
    """Show folder selection with direct system dialog"""
    st.markdown("### 📂 Выбор папки через проводник")

    # Main folder picker buttons
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💻 Системный проводник")
        st.info("**Открывает настоящий проводник Windows/Linux/Mac**")

        if st.button("📂 Открыть проводник", type="primary", width="stretch", key="system_picker_main"):
            folder_path = open_system_folder_picker()
            if folder_path:
                st.success(f"✅ Выбрана папка: `{folder_path}`")

                # Count images immediately
                total_images = count_images_in_folder(Path(folder_path))
                st.info(f"📊 В папке найдено {total_images} изображений")

                # Auto-scan the folder
                if st.button("🚀 Начать обработку", type="primary", key="auto_scan_selected"):
                    scan_folder_for_images(folder_path)
            else:
                st.warning("Папка не выбрана или системный диалог недоступен")

    with col2:
        st.markdown("#### 🌐 Через браузер")
        st.info("**Альтернативный способ через браузер**")

        # JavaScript for automatic file dialog
        st.markdown("""
        <script>
        function openBrowserFileDialog() {
            const fileInput = document.querySelector('input[type="file"][key="browser_folder_picker"]');
            if (fileInput) {
                fileInput.click();
            }
        }
        </script>
        """, unsafe_allow_html=True)

        if st.button("📂 Выбрать файлы из папки", key="browser_picker_main"):
            # Trigger JavaScript to open file dialog
            st.markdown("""
            <script>
            openBrowserFileDialog();
            </script>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Hidden file uploader for browser-based selection
    uploaded_files = st.file_uploader(
        "Выберите файлы из папки:",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif', 'tiff'],
        accept_multiple_files=True,
        help="Выберите файлы из папки для автоматического определения пути",
        key="browser_folder_picker",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown("### 📁 Результаты выбора через браузер")

        # Extract folder path from uploaded files
        file_paths = [Path(uploaded_file.name) for uploaded_file in uploaded_files]

        if len(file_paths) > 0:
            common_path = file_paths[0].parent
            for path in file_paths[1:]:
                common_path = find_common_parent(common_path, path.parent)

            folder_path = str(common_path)
            st.success(f"✅ Определена папка: `{folder_path}`")

            # Show preview of selected files
            st.markdown(f"**Выбранные файлы:** {len(uploaded_files)}")
            preview_cols = st.columns(min(4, len(uploaded_files)))
            for i, uploaded_file in enumerate(uploaded_files[:4]):
                with preview_cols[i]:
                    try:
                        image = Image.open(uploaded_file)
                        image.thumbnail((100, 100))
                        st.image(image, caption=uploaded_file.name[:20], width="stretch")
                    except:
                        st.text(f"📄 {uploaded_file.name[:20]}")

            if len(uploaded_files) > 4:
                st.info(f"И ещё {len(uploaded_files) - 4} файлов...")

            # Count total images in the detected folder
            total_images = count_images_in_folder(common_path)
            st.info(f"📊 В папке найдено {total_images} изображений всего")

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Обработать выбранную папку", type="primary", key="process_detected_folder"):
                    scan_folder_for_images(folder_path)
            with col2:
                if st.button("🔄 Выбрать другие файлы", key="reset_selection"):
                    st.rerun()

def open_system_folder_picker():
    """Open system folder picker dialog (works only locally)"""
    try:
        import platform
        system = platform.system()

        if system == "Windows":
            # Windows folder picker using tkinter
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)

                folder_path = filedialog.askdirectory(
                    title="Выберите папку с фотографиями",
                    initialdir=str(Path.home())
                )

                root.destroy()
                return folder_path if folder_path else None

            except ImportError:
                st.warning("⚠️ tkinter не установлен. Установите tkinter для локального выбора папок.")
                return None

        elif system == "Linux":
            # Linux folder picker using zenity
            try:
                import subprocess
                result = subprocess.run(
                    ['zenity', '--file-selection', '--directory',
                     '--title=Выберите папку с фотографиями'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                return None
            except FileNotFoundError:
                st.warning("⚠️ zenity не установлен. Установите zenity или используйте другой метод.")
                return None

        elif system == "Darwin":  # macOS
            # macOS folder picker using AppleScript
            try:
                import subprocess
                script = '''
                tell application "Finder"
                    activate
                    set theFolder to choose folder with prompt "Выберите папку с фотографиями:"
                    return POSIX path of theFolder
                end tell
                '''
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                return None
            except FileNotFoundError:
                st.warning("⚠️ osascript не найден. Используйте другой метод выбора папки.")
                return None

        else:
            st.warning(f"⚠️ Системный выбор папки не поддерживается для {system}")
            return None

    except Exception as e:
        st.error(f"❌ Ошибка при открытии системного диалога: {e}")
        return None

def find_common_parent(path1, path2):
    """Find common parent directory between two paths"""
    try:
        # Convert to Path objects if they're strings
        if isinstance(path1, str):
            path1 = Path(path1)
        if isinstance(path2, str):
            path2 = Path(path2)

        # Find common parts
        parts1 = path1.parts
        parts2 = path2.parts

        common_parts = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common_parts.append(p1)
            else:
                break

        if common_parts:
            return Path(*common_parts)
        else:
            # Return the shorter path as fallback
            return min(path1, path2, key=lambda p: len(str(p)))
    except:
        return path1  # Return first path as fallback

def show_processing_queue():
    """Show processing queue status"""
    st.subheader("📋 Очередь обработки")

    # Current processing status
    if st.session_state.current_processing:
        current_folder = Path(st.session_state.current_processing)
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.session_state.processing_status == "processing":
                st.success(f"🔄 Обрабатывается: {current_folder.name}")
            elif st.session_state.processing_status == "paused":
                st.warning(f"⏸️ Приостановлено: {current_folder.name}")
            else:
                st.info(f"⏳ Ожидает: {current_folder.name}")

        with col2:
            if st.session_state.processing_status == "processing":
                if st.button("⏸️ Пауза", key="pause_processing"):
                    st.session_state.processing_status = "paused"
                    st.rerun()
            elif st.session_state.processing_status == "paused":
                if st.button("▶️ Продолжить", key="resume_processing"):
                    st.session_state.processing_status = "processing"
                    st.rerun()

        with col3:
            if st.button("❌ Остановить", key="stop_processing"):
                st.session_state.processing_status = "idle"
                st.session_state.current_processing = None
                st.rerun()

    # Queue list
    if st.session_state.processing_queue:
        st.markdown("**📝 В очереди:**")

        for i, folder_path in enumerate(st.session_state.processing_queue):
            folder = Path(folder_path)
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                image_count = count_images_in_folder(folder)
                st.info(f"📁 {folder.name} ({image_count} фото)")

            with col2:
                st.write(f"#{i+1}")

            with col3:
                if st.button("❌", key=f"remove_{i}", help=f"Удалить {folder.name} из очереди"):
                    st.session_state.processing_queue.pop(i)
                    st.rerun()

        # Control buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("▶️ Запустить очередь", key="start_queue") and st.session_state.processing_status != "processing":
                start_queue_processing()
        with col2:
            if st.button("🗑️ Очистить очередь", key="clear_queue"):
                st.session_state.processing_queue = []
                st.rerun()
        with col3:
            if st.button("📊 Показать результаты", key="show_results"):
                pass  # Results will be shown in the Results tab

def count_images_in_folder(folder_path):
    """Count image files in folder recursively"""
    try:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        count = 0
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                count += 1
        return count
    except:
        return 0

def add_to_queue(folder_path):
    """Add folder to processing queue"""
    if folder_path not in st.session_state.processing_queue:
        st.session_state.processing_queue.append(folder_path)
        st.success(f"✅ Добавлено в очередь: {Path(folder_path).name}")

        # Add to logs
        if 'processing_logs' not in st.session_state:
            st.session_state.processing_logs = []
        st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ➕ Добавлено в очередь: {Path(folder_path).name}")
    else:
        st.warning("⚠️ Эта папка уже в очереди")

def start_queue_processing():
    """Start processing the queue"""
    if st.session_state.processing_queue:
        st.session_state.current_processing = st.session_state.processing_queue[0]
        st.session_state.processing_status = "processing"
        st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ▶️ Начата обработка очереди ({len(st.session_state.processing_queue)} папок)")
        st.rerun()

def process_next_in_queue():
    """Process next item in queue"""
    if st.session_state.current_processing and st.session_state.current_processing in st.session_state.processing_queue:
        # Remove completed item from queue
        st.session_state.processing_queue.remove(st.session_state.current_processing)
        st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Удалено из очереди: {Path(st.session_state.current_processing).name}")

    # Check if there are more items to process
    if st.session_state.processing_queue and st.session_state.processing_status == "processing":
        next_folder = st.session_state.processing_queue[0]
        st.session_state.current_processing = next_folder
        st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 🔄 Переход к следующей папке: {Path(next_folder).name}")

        # Auto-scan and process next folder
        st.session_state.auto_process_current = True
        scan_folder_for_images(next_folder)
        # Processing will continue automatically
    else:
        # Queue completed
        st.session_state.current_processing = None
        st.session_state.processing_status = "idle"
        # Reset processing state when queue is completed
        st.session_state.processing_state = 'completed'
        if 'processing_logs' in st.session_state:
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 🎉 Обработка всей очереди завершена!")
        st.rerun()

def show_file_explorer():
    """Show simplified and user-friendly file explorer interface"""
    # Initialize session state for file explorer
    if 'explorer_path' not in st.session_state:
        st.session_state.explorer_path = str(Path.home())
    if 'explorer_history' not in st.session_state:
        st.session_state.explorer_history = []
    if 'explorer_selected' not in st.session_state:
        st.session_state.explorer_selected = None

    # Main folder selection - the primary action
    st.markdown("### 📂 Выберите папку с фотографиями")

    # Single prominent button for folder selection
    if st.button("🔍 Выбрать папку через Windows проводник", type="primary", use_container_width=True):
        folder_path = open_system_folder_picker()
        if folder_path:
            st.session_state.explorer_path = folder_path
            st.session_state.explorer_selected = folder_path
            st.success(f"✅ Выбрана папка: {Path(folder_path).name}")
            st.rerun()

    # Show selected folder and its contents
    if st.session_state.explorer_selected and Path(st.session_state.explorer_selected).is_dir():
        current_path = Path(st.session_state.explorer_selected)

        # Folder info
        st.markdown("---")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**📁 {current_path.name}**")
            st.text(str(current_path))

        with col2:
            # Count images in folder
            try:
                image_count = sum(1 for f in current_path.rglob('*')
                                if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'})
                st.metric("Фотографий", image_count)
            except:
                st.metric("Фотографий", "N/A")

        # Simple navigation
        st.markdown("### 📋 Содержимое папки")

        # Quick navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])

        with nav_col1:
            if st.button("⬅️ Назад", disabled=len(st.session_state.explorer_history) == 0):
                if st.session_state.explorer_history:
                    prev_path = st.session_state.explorer_history.pop()
                    st.session_state.explorer_path = prev_path
                    st.session_state.explorer_selected = prev_path
                    st.rerun()

        with nav_col2:
            if st.button("📁 ..", help="Вверх на уровень"):
                parent_path = current_path.parent
                if parent_path != current_path:
                    st.session_state.explorer_history.append(str(current_path))
                    st.session_state.explorer_path = str(parent_path)
                    st.session_state.explorer_selected = str(parent_path)
                    st.rerun()

        # Show folder contents in a simple list
        show_simple_folder_contents(current_path)

        # Action buttons at the bottom
        st.markdown("---")

        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button("➕ Добавить в очередь", use_container_width=True):
                add_to_queue(st.session_state.explorer_selected)
                st.success("✅ Добавлено в очередь обработки!")

        with action_col2:
            if st.button("🚀 Начать обработку", type="primary", use_container_width=True):
                # First scan the folder for images
                scan_folder_for_images(st.session_state.explorer_selected)
                # Then immediately start processing
                if st.session_state.get('uploaded_files'):
                    st.session_state.auto_start_processing = True
                    st.success("✅ Готово! Перейдите на вкладку '🚀 Обработка'")
                    st.info("💡 Обработка начнется автоматически")

    else:
        # Initial state - no folder selected
        st.info("👆 Нажмите кнопку выше, чтобы выбрать папку с фотографиями")

        # Show some quick tips
        with st.expander("💡 Советы по выбору папки"):
            st.markdown("""
            - Выберите папку, содержащую фотографии для обработки
            - Поддерживаемые форматы: JPG, PNG, BMP, WebP, TIF
            - Программа найдет все фотографии автоматически
            - Можно выбрать любую папку на вашем компьютере
            """)

def show_simple_folder_contents(current_path):
    """Show folder contents in a simple, user-friendly way"""
    try:
        items = list(current_path.iterdir())
        folders = [item for item in items if item.is_dir() and not item.name.startswith('.')]
        files = [item for item in items if item.is_file()]

        # Sort folders and files
        folders.sort()
        files.sort()

        # Show folders first
        if folders:
            st.markdown("**Папки:**")
            folder_cols = st.columns(min(3, len(folders)))

            for i, folder in enumerate(folders[:3]):  # Show first 3 folders
                with folder_cols[i]:
                    if st.button(f"📁 {folder.name}", key=f"simple_folder_{folder}",
                               help=f"Открыть папку: {folder.name}"):
                        st.session_state.explorer_history.append(str(current_path))
                        st.session_state.explorer_path = str(folder)
                        st.session_state.explorer_selected = str(folder)
                        st.rerun()

            if len(folders) > 3:
                st.text(f"... и ещё {len(folders) - 3} папок")

        # Show files (images prioritized)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        image_files = [file for file in files if file.suffix.lower() in image_extensions]
        other_files = [file for file in files if file.suffix.lower() not in image_extensions]

        if image_files:
            st.markdown(f"**Фотографии ({len(image_files)}):**")

            # Show first few images as thumbnails
            preview_images = image_files[:6]  # Show first 6 images
            image_cols = st.columns(3)

            for i, img_file in enumerate(preview_images):
                with image_cols[i % 3]:
                    try:
                        image = Image.open(img_file)
                        # Create small thumbnail
                        image.thumbnail((100, 100))
                        st.image(image, caption=img_file.name[:15] + "..." if len(img_file.name) > 15 else img_file.name)
                    except Exception:
                        st.write("🖼️")
                        st.caption(img_file.name[:12] + "..." if len(img_file.name) > 12 else img_file.name)

            if len(image_files) > 6:
                st.text(f"... и ещё {len(image_files) - 6} фотографий")

        if other_files:
            st.markdown(f"**Другие файлы ({len(other_files)}):**")
            other_cols = st.columns(2)

            for i, other_file in enumerate(other_files[:4]):  # Show first 4 other files
                with other_cols[i % 2]:
                    file_icon = get_file_icon(other_file.suffix.lower())
                    st.write(f"{file_icon} {other_file.name}")

            if len(other_files) > 4:
                st.text(f"... и ещё {len(other_files) - 4} файлов")

        if not folders and not files:
            st.info("📂 Папка пуста")

    except PermissionError:
        st.error("❌ Нет доступа к папке")
    except Exception as e:
        st.error(f"❌ Ошибка при чтении папки: {e}")

def show_folder_tree(current_path):
    """Show folder tree navigation"""
    # No header - cleaner interface

    try:
        # Show parent directories
        parent_path = current_path.parent
        if parent_path != current_path:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📁 .. (вверх)", key=f"parent_{parent_path}"):
                    # Update both paths and history
                    st.session_state.explorer_history.append(str(current_path))
                    st.session_state.explorer_path = str(parent_path)
                    st.session_state.explorer_selected = str(parent_path)
                    st.rerun()
            with col2:
                if st.button("🔙", key=f"back_{parent_path}", help="Вернуться назад"):
                    if st.session_state.explorer_history:
                        prev_path = st.session_state.explorer_history.pop()
                        st.session_state.explorer_path = prev_path
                        st.session_state.explorer_selected = prev_path
                        st.rerun()

        # Show subdirectories
        for item in sorted(current_path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                is_selected = str(item) == st.session_state.explorer_selected
                button_text = f"📁 {item.name}"
                if is_selected:
                    button_text = f"▶️ {item.name}"

                if st.button(button_text, key=f"dir_{item}"):
                    st.session_state.explorer_history.append(str(current_path))
                    st.session_state.explorer_path = str(item)
                    st.session_state.explorer_selected = str(item)
                    st.rerun()

    except PermissionError:
        st.error("❌ Нет доступа к папке")
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

def show_folder_contents(current_path):
    """Show contents of selected folder"""
    # No header - cleaner interface

    try:
        items = list(current_path.iterdir())
        folders = [item for item in items if item.is_dir() and not item.name.startswith('.')]
        files = [item for item in items if item.is_file()]

        # Sort folders and files
        folders.sort()
        files.sort()

        # Show folders first
        for folder in folders:
            is_selected = str(folder) == st.session_state.explorer_selected
            col1, col2 = st.columns([3, 1])

            with col1:
                if st.button(f"📁 {folder.name}", key=f"content_dir_{folder}",
                           help=f"Размер: {get_folder_size(folder)} файлов"):
                    st.session_state.explorer_selected = str(folder)
                    st.rerun()

            with col2:
                if is_selected:
                    st.success("✅")

        # Show all files in a compact grid
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        
        # Separate image files and other files
        image_files = [file for file in files if file.suffix.lower() in image_extensions]
        other_files = [file for file in files if file.suffix.lower() not in image_extensions]

        # Combine all files but prioritize images
        all_files = sorted(image_files + other_files, key=lambda x: (x.suffix.lower() not in image_extensions, x.name.lower()))

        if all_files:
            st.markdown("**📁 Содержимое папки:**")

            # Show files in a compact grid (6 columns for better space usage)
            cols = st.columns(6)

            for i, file in enumerate(all_files[:24]):  # Show first 24 items
                with cols[i % 6]:
                    file_path = Path(file)

                    # Create a container for each file item
                    with st.container():
                        # For image files, show thumbnail
                        if file_path.suffix.lower() in image_extensions:
                            try:
                                image = Image.open(file_path)

                                # Create very small thumbnail (60x60)
                                try:
                                    image.thumbnail((60, 60), Image.Resampling.LANCZOS)
                                except AttributeError:
                                    # For older PIL versions
                                    image.thumbnail((60, 60), Image.ANTIALIAS)

                                # Ensure image has consistent format and add small border
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')

                                # Add subtle padding/border effect
                                image = ImageOps.expand(image, border=1, fill='white')

                                st.image(image, width="stretch")

                                # Small caption
                                st.caption(file_path.name[:15] + "..." if len(file_path.name) > 15 else file_path.name)

                            except Exception as e:
                                # If image can't be loaded, show icon
                                st.write("🖼️")
                                st.caption(file_path.name[:12] + "..." if len(file_path.name) > 12 else file_path.name)

                        else:
                            # For non-image files, show file icon
                            file_icon = get_file_icon(file_path.suffix.lower())
                            st.write(file_icon)
                            st.caption(file_path.name[:12] + "..." if len(file_path.name) > 12 else file_path.name)

            if len(all_files) > 24:
                st.info(f"📄 И ещё {len(all_files) - 24} файлов...")
        else:
            st.info("📂 Папка пуста")

    except PermissionError:
        st.error("❌ Нет доступа к папке")
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

def get_quick_paths():
    """Get quick access paths based on OS"""
    import platform
    system = platform.system()

    paths = {}

    if system == "Windows":
        # Windows paths
        user_home = Path.home()
        paths.update({
            "📸 Мои фото": str(user_home / "Pictures"),
            "📁 Документы": str(user_home / "Documents"),
            "📂 Загрузки": str(user_home / "Downloads"),
            "💻 Рабочий стол": str(user_home / "Desktop"),
        })

        # Add drives if accessible
        try:
            import string
            for drive_letter in string.ascii_uppercase:
                drive_path = f"{drive_letter}:\\"
                if Path(drive_path).exists():
                    paths[f"💾 Диск {drive_letter}"] = drive_path
        except:
            pass

    else:
        # Unix/Linux/Mac paths
        user_home = Path.home()
        paths.update({
            "🏠 Домашняя": str(user_home),
            "📸 Изображения": str(user_home / "Pictures"),
            "📂 Документы": str(user_home / "Documents"),
            "📥 Загрузки": str(user_home / "Downloads"),
            "🖥️ Рабочий стол": str(user_home / "Desktop"),
        })

        # Add common system paths
        common_paths = ["/home", "/usr/share", "/opt", "/media", "/mnt"]
        for common_path in common_paths:
            if Path(common_path).exists():
                paths[f"📁 {common_path.split('/')[-1].title()}"] = common_path

    # Filter existing paths
    return {name: path for name, path in paths.items() if Path(path).exists()}

def get_file_icon(extension):
    """Get emoji icon for file extension"""
    icon_map = {
        # Images
        '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️', '.bmp': '🖼️', '.webp': '🖼️',
        '.tif': '🖼️', '.tiff': '🖼️', '.gif': '🖼️', '.svg': '🖼️',
        
        # Documents
        '.pdf': '📄', '.doc': '📄', '.docx': '📄', '.txt': '📄', '.rtf': '📄',
        '.odt': '📄', '.pages': '📄',
        
        # Spreadsheets
        '.xls': '📊', '.xlsx': '📊', '.csv': '📊', '.ods': '📊', '.numbers': '📊',
        
        # Presentations
        '.ppt': '📽️', '.pptx': '📽️', '.key': '📽️', '.odp': '📽️',
        
        # Archives
        '.zip': '📦', '.rar': '📦', '.7z': '📦', '.tar': '📦', '.gz': '📦',
        
        # Audio
        '.mp3': '🎵', '.wav': '🎵', '.flac': '🎵', '.aac': '🎵', '.ogg': '🎵',
        
        # Video
        '.mp4': '🎬', '.avi': '🎬', '.mkv': '🎬', '.mov': '🎬', '.wmv': '🎬',
        
        # Code
        '.py': '🐍', '.js': '📜', '.html': '🌐', '.css': '🎨', '.java': '☕',
        '.cpp': '⚙️', '.c': '⚙️', '.php': '🐘', '.sql': '🗄️',
        
        # Default
        '': '📄'
    }
    
    return icon_map.get(extension.lower(), '📄')

def get_file_info(file_path):
    """Get detailed file information"""
    try:
        stat = file_path.stat()
        size = stat.st_size
        modified = stat.st_mtime

        # Format size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size / (1024 * 1024 * 1024):.1f} GB"

        # Format date
        from datetime import datetime
        date_str = datetime.fromtimestamp(modified).strftime("%d.%m.%Y %H:%M")

        return size_str, date_str
    except:
        return "N/A", "N/A"

def get_folder_size(folder_path):
    """Get approximate folder size (file count)"""
    try:
        return len(list(folder_path.rglob('*')))
    except:
        return 0

def show_processing_logs():
    """Show processing logs"""
    st.subheader("📝 Логи обработки")

    if 'processing_logs' not in st.session_state:
        st.session_state.processing_logs = []

    # Show logs in a scrollable container
    with st.container():
        if st.session_state.processing_logs:
            # Show last 20 logs, most recent first
            recent_logs = st.session_state.processing_logs[-20:][::-1]

            for log_entry in recent_logs:
                # Color code different types of messages
                if "❌" in log_entry:
                    st.error(log_entry)
                elif "✅" in log_entry:
                    st.success(log_entry)
                elif "🚀" in log_entry or "▶️" in log_entry:
                    st.info(log_entry)
                elif "📊" in log_entry or "📁" in log_entry:
                    st.write(f"📊 {log_entry}")
                else:
                    st.write(log_entry)
        else:
            st.info("ℹ️ Логи обработки появятся здесь во время работы")

        # Clear logs button
        if st.button("🗑️ Очистить логи", key="clear_logs"):
            st.session_state.processing_logs = []
            st.rerun()

def scan_folder_for_images(folder_path):
    """Scan folder and subfolders for images"""
    try:
        folder = Path(folder_path)
        if not folder.exists():
            st.error(f"❌ Папка не найдена: {folder_path}")
            return

        if not folder.is_dir():
            st.error(f"❌ Указанный путь не является папкой: {folder_path}")
            return

        # Find all image files recursively
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        image_files = []

        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        if not image_files:
            st.warning(f"⚠️ В папке {folder_path} не найдено фотографий с поддерживаемыми форматами")
            return

        st.success(f"✅ Найдено {len(image_files)} фотографий в папке")

        # Add to processing logs
        if 'processing_logs' not in st.session_state:
            st.session_state.processing_logs = []
        st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 🔍 Найдено {len(image_files)} изображений в папке {folder.name}")

        # Copy files to temp directory for processing
        if 'temp_dir' not in st.session_state:
            # Use user's home directory for temp files instead of system temp
            user_home = Path.home()
            temp_base_dir = user_home / "FaceSort_Temp"
            temp_base_dir.mkdir(exist_ok=True)

            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.temp_dir = str(temp_base_dir / f"temp_{timestamp}")

        temp_dir = Path(st.session_state.temp_dir)
        temp_dir.mkdir(exist_ok=True)

        # Clear previous files
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()

        # Copy found images to temp directory
        copied_files = []
        for src_file in image_files:
            dst_file = temp_dir / f"{src_file.stem}_{hash(str(src_file))}{src_file.suffix}"
            try:
                shutil.copy2(src_file, dst_file)
                copied_files.append(dst_file)
            except Exception as e:
                st.warning(f"Не удалось скопировать {src_file.name}: {e}")
                st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ⚠️ Ошибка копирования: {src_file.name}")

        st.session_state.uploaded_files = copied_files
        st.session_state.input_dir = temp_dir

        # Show preview of found images
        show_image_preview(copied_files, f"Найденные фотографии в папке {folder.name}")

        # Auto-start processing if called from queue
        if st.session_state.get('auto_process_current', False):
            st.session_state.auto_process_current = False
            st.session_state.auto_start_processing = True
            # Processing will start automatically when user switches to Processing tab

    except Exception as e:
        st.error(f"❌ Ошибка при сканировании папки: {e}")

def process_uploaded_files(uploaded_files):
    """Process uploaded files and prepare them for analysis"""
    if 'temp_dir' not in st.session_state:
        # Use user's home directory for temp files instead of system temp
        user_home = Path.home()
        temp_base_dir = user_home / "FaceSort_Temp"
        temp_base_dir.mkdir(exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.temp_dir = str(temp_base_dir / f"upload_{timestamp}")

    temp_dir = Path(st.session_state.temp_dir)
    temp_dir.mkdir(exist_ok=True)

    # Clear previous files
    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            file_path.unlink()

    # Save uploaded files
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)

    st.session_state.uploaded_files = saved_files
    st.session_state.input_dir = temp_dir

    # Display success message and file count
    st.success(f"✅ Загружено {len(saved_files)} фотографий")

    # Show preview
    show_image_preview(saved_files, "Загруженные фотографии")

def show_image_preview(image_files, title):
    """Show preview of images with start processing button"""
    if not image_files:
        return

    st.subheader(f"📸 {title}")

    # Show first 6 images
    preview_files = image_files[:6]
    cols = st.columns(3)

    for i, file_path in enumerate(preview_files):
        with cols[i % 3]:
            try:
                image = Image.open(file_path)
                # Resize for preview
                image.thumbnail((200, 200))
                st.image(image, caption=file_path.name, width="stretch")
            except Exception as e:
                st.error(f"Ошибка загрузки {file_path.name}: {e}")

    if len(image_files) > 6:
        st.info(f"📄 И ещё {len(image_files) - 6} фото...")

    # Start processing button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 СТАРТ ОБРАБОТКИ", type="primary", width="stretch"):
            # Add to logs
            if 'processing_logs' not in st.session_state:
                st.session_state.processing_logs = []
            folder_name = Path(st.session_state.input_dir).name if hasattr(st.session_state, 'input_dir') and st.session_state.input_dir else "фотографии"
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Запуск обработки: {folder_name}")

            # Set flag for auto-starting processing
            st.session_state.auto_start_processing = True
            # Processing will start automatically when user switches to Processing tab

def show_parameters_tab():
    st.header("Настройки обработки")

    if 'uploaded_files' not in st.session_state:
        st.warning("⚠️ Сначала загрузите фотографии")
        return

    st.markdown("""
    Настройте параметры обработки изображений и кластеризации лиц.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Обработка изображений")
        eps_sim = st.slider(
            "Порог схожести лиц",
            min_value=0.1,
            max_value=0.9,
            value=0.55,
            step=0.05,
            help="Чем меньше значение, тем строже кластеризация"
        )

        min_samples = st.slider(
            "Мин. образцов для кластера",
            min_value=1,
            max_value=5,
            value=2,
            help="Минимальное количество лиц для создания кластера"
        )

        group_thr = st.slider(
            "Порог группового фото",
            min_value=1,
            max_value=10,
            value=3,
            help="Фото с большим количеством лиц помещаются в группу"
        )

    with col2:
        st.subheader("Фильтры качества")
        min_face = st.slider(
            "Минимальный размер лица (px)",
            min_value=50,
            max_value=200,
            value=110,
            help="Минимальный размер лица для обработки"
        )

        blur_thr = st.slider(
            "Порог размытия",
            min_value=10.0,
            max_value=100.0,
            value=45.0,
            step=5.0,
            help="Игнорировать слишком размытые изображения"
        )

    # Store parameters in session state
    st.session_state.params = {
        'eps_sim': eps_sim,
        'min_samples': min_samples,
        'min_face': min_face,
        'blur_thr': blur_thr,
        'group_thr': group_thr,
        'det_size': 640,
        'gpu_id': 0
    }

    st.success("✅ Параметры сохранены")

def show_processing_tab():
    st.header("🚀 Обработка фотографий")

    if 'uploaded_files' not in st.session_state:
        st.warning("⚠️ Сначала загрузите фотографии")
        st.info("📤 Перейдите на вкладку 'Загрузка фото' для загрузки изображений")
        return

    # Initialize processing state if not exists
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = 'idle'  # idle, processing, completed, error

    # Auto-start processing if we came from upload tab
    if 'auto_start_processing' in st.session_state and st.session_state.auto_start_processing:
        st.session_state.auto_start_processing = False
        st.session_state.processing_state = 'processing'
        process_images()
        return

    # Create columns for processing controls and logs
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Параметры и управление")

        # Show current processing status
        if st.session_state.processing_state == 'processing':
            st.info("🔄 **Обработка выполняется...**")
        elif st.session_state.processing_state == 'completed':
            st.success("✅ **Обработка завершена!**")
        elif st.session_state.processing_state == 'error':
            st.error("❌ **Ошибка обработки!**")

        # Show parameters summary
        if 'params' in st.session_state:
            params = st.session_state.params
            st.markdown("**📋 Параметры обработки:**")
            st.info(f"**Порог схожести:** {params['eps_sim']}")
            st.info(f"**Мин. образцов:** {params['min_samples']}")
            st.info(f"**Групповой порог:** {params['group_thr']}")
            st.info(f"**Размер лица:** {params['min_face']}px")
            st.info(f"**Порог размытия:** {params['blur_thr']}")

        # Action buttons based on processing state
        st.markdown("---")

        if st.session_state.processing_state == 'idle':
            # Show start button
            if st.button("🚀 НАЧАТЬ ОБРАБОТКУ", type="primary", width="stretch"):
                st.session_state.processing_state = 'processing'
                process_images()
        elif st.session_state.processing_state == 'processing':
            # Show stop button
            if st.button("⏹️ ОСТАНОВИТЬ ОБРАБОТКУ", type="secondary", width="stretch"):
                st.session_state.processing_state = 'idle'
                st.warning("⚠️ Обработка была остановлена пользователем")
                if 'processing_logs' in st.session_state:
                    st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ⏹️ Обработка остановлена пользователем")
        elif st.session_state.processing_state in ['completed', 'error']:
            # Show restart button
            if st.button("🔄 НАЧАТЬ ЗАНОВО", type="primary", width="stretch"):
                # Clear previous results
                if 'results' in st.session_state:
                    del st.session_state.results
                st.session_state.processing_state = 'processing'
                process_images()

    with col2:
        show_processing_logs()

def process_images():
    """Process uploaded images using the face clustering algorithm"""
    # Add log entry
    if 'processing_logs' not in st.session_state:
        st.session_state.processing_logs = []

    log_message = f"🚀 Начата обработка папки: {Path(st.session_state.current_processing or st.session_state.input_dir).name}"
    st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] {log_message}")

    # Set processing state
    st.session_state.processing_state = 'processing'

    with st.spinner("Обработка изображений... Это может занять несколько минут"):

        # Create output directory in user's system
        if 'output_dir' not in st.session_state:
            # Create results folder in user's home directory
            user_home = Path.home()
            results_base_dir = user_home / "FaceSort_Results"

            # Create timestamped subfolder for this processing session
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"facesort_{timestamp}"

            st.session_state.output_dir = results_base_dir / folder_name
            st.session_state.output_dir.mkdir(parents=True, exist_ok=True)

            # Log the output directory
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 📁 Результаты будут сохранены в: {st.session_state.output_dir}")
            st.info(f"📁 Результаты будут сохранены в: {st.session_state.output_dir}")

        output_dir = st.session_state.output_dir
        input_dir = st.session_state.input_dir
        params = st.session_state.params

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Collect faces
            status_text.text("Шаг 1/4: Обнаружение лиц на фотографиях...")
            records = collect_faces(
                input_dir,
                min_face=params['min_face'],
                blur_thr=params['blur_thr'],
                det_size=params['det_size'],
                gpu_id=params['gpu_id']
            )
            progress_bar.progress(0.25)

            if not records:
                st.error("❌ На фотографиях не найдено лиц для обработки")
                st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ❌ Ошибка: лица не найдены в папке {Path(st.session_state.current_processing or st.session_state.input_dir).name}")
                return

            # Step 2: Cluster faces
            status_text.text(f"Шаг 2/4: Кластеризация {len(records)} найденных лиц...")
            labels = cluster_faces(records, params['eps_sim'], params['min_samples'])
            progress_bar.progress(0.5)
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 📊 Найдено {len(set(labels))} кластеров лиц")

            # Step 3: Route by clusters
            status_text.text("Шаг 3/4: Сортировка фотографий по кластерам...")
            cluster_to_name, report_path, eligible_clusters = route_by_clusters(
                records, labels, output_dir, params['group_thr']
            )
            progress_bar.progress(0.75)
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 📁 Создано {len(eligible_clusters)} папок с людьми")

            # Step 4: Finalize
            status_text.text("Шаг 4/4: Завершение обработки...")
            progress_bar.progress(1.0)

            # Store results in session state
            st.session_state.results = {
                'cluster_to_name': cluster_to_name,
                'eligible_clusters': eligible_clusters,
                'output_dir': output_dir,
                'report_path': report_path,
                'total_faces': len(records),
                'total_images': len(st.session_state.uploaded_files)
            }

            status_text.text("✅ Обработка завершена!")
            st.success(f"✅ Обработано {len(records)} лиц на {len(st.session_state.uploaded_files)} фотографиях")
            st.info(f"📁 Результаты сохранены в: {output_dir}")

            # Log completion
            folder_name = Path(st.session_state.current_processing or st.session_state.input_dir).name
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Завершена обработка папки: {folder_name}")
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] 📁 Результаты сохранены в: {output_dir}")

            # Set processing state to completed
            st.session_state.processing_state = 'completed'

            # Process next item in queue
            process_next_in_queue()

        except Exception as e:
            st.error(f"❌ Ошибка обработки: {str(e)}")
            # Set processing state to error
            st.session_state.processing_state = 'error'
            st.session_state.processing_logs.append(f"[{time.strftime('%H:%M:%S')}] ❌ Ошибка обработки: {str(e)}")
            return

def show_results_tab():
    st.header("Результаты обработки")

    if 'results' not in st.session_state:
        st.info("ℹ️ Результаты появятся после обработки фотографий")
        return

    results = st.session_state.results
    output_dir = results['output_dir']

    # Show results location and option to open folder
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**📁 Папка с результатами:** {output_dir}")
        st.markdown("*Результаты сохранены в вашей домашней директории*")

    with col2:
        # Button to open results folder (works when running locally)
        if st.button("📂 Открыть папку", use_container_width=True):
            try:
                import platform
                system = platform.system()
                if system == "Windows":
                    import subprocess
                    subprocess.run(["explorer", str(output_dir)])
                elif system == "Darwin":  # macOS
                    import subprocess
                    subprocess.run(["open", str(output_dir)])
                else:  # Linux
                    import subprocess
                    subprocess.run(["xdg-open", str(output_dir)])
                st.success("✅ Папка открыта!")
            except Exception as e:
                st.warning(f"Не удалось автоматически открыть папку: {e}")
                st.info(f"📁 Откройте папку вручную: {output_dir}")

    st.markdown("---")

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Всего фотографий", results['total_images'])

    with col2:
        st.metric("Найдено лиц", results['total_faces'])

    with col3:
        st.metric("Распознано людей", len(results['eligible_clusters']))

    # Display clusters
    st.subheader("📁 Распознанные люди")

    cluster_to_name = results['cluster_to_name']
    eligible_clusters = sorted(results['eligible_clusters'])

    for cluster_id in eligible_clusters:
        person_name = cluster_to_name[cluster_id]
        person_dir = output_dir / person_name

        if person_dir.exists():
            # Get images for this person
            person_images = list(person_dir.glob("*"))

            if person_images:
                with st.expander(f"👤 {person_name} ({len(person_images)} фото)", expanded=False):
                    # Display images in a grid
                    cols = st.columns(min(4, len(person_images)))

                    for i, img_path in enumerate(person_images[:4]):  # Show first 4 images
                        with cols[i]:
                            image = Image.open(img_path)
                            st.image(image, caption=Path(img_path).stem, width="stretch")

                    if len(person_images) > 4:
                        st.text(f"... и ещё {len(person_images) - 4} фото")

    # Group photos
    group_dir = output_dir / "__group_only__"
    if group_dir.exists() and list(group_dir.glob("*")):
        group_images = list(group_dir.glob("*"))
        st.subheader(f"👥 Групповые фото ({len(group_images)} фото)")

        cols = st.columns(min(3, len(group_images)))
        for i, img_path in enumerate(group_images[:3]):
            with cols[i]:
                image = Image.open(img_path)
                st.image(image, caption=Path(img_path).stem, width="stretch")

    # Unknown photos
    unknown_dir = output_dir / "__unknown__"
    if unknown_dir.exists() and list(unknown_dir.glob("*")):
        unknown_images = list(unknown_dir.glob("*"))
        st.subheader(f"❓ Нераспознанные фото ({len(unknown_images)} фото)")

        cols = st.columns(min(3, len(unknown_images)))
        for i, img_path in enumerate(unknown_images[:3]):
            with cols[i]:
                image = Image.open(img_path)
                st.image(image, caption=Path(img_path).stem, width="stretch")

    # Download results
    st.subheader("📥 Скачать результаты")

    # Create zip archive of results
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zip_file.write(file_path, arcname)

    zip_buffer.seek(0)

    st.download_button(
        label="📦 Скачать все результаты (ZIP)",
        data=zip_buffer,
        file_name="facesort_results.zip",
        mime="application/zip",
        width="stretch"
    )

if __name__ == "__main__":
    main()
