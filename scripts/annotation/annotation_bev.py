#!/usr/bin/env python3
"""
ROS2 Bag → BEVFusion 학습 데이터 완전 자동 생성 (통합 버전)
Annotation 도구 내장 - 지우개 및 이전 프레임 기능 추가
업데이트: 카메라 이미지 추출 및 Annotation 시 시각화 기능 추가
"""

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
import rosbag2_py
import numpy as np
import cv2
import pickle
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


# ============================================================================
# Annotation 도구 (내장) - 개선 버전 (카메라 뷰 추가)
# ============================================================================

class BEVFusionAnnotationTool:
    """BEVFusion Annotation 도구 (Camera View 지원)"""
    
    def __init__(self, bev_image_path, output_dir, camera_image_path=None, target_size=(400, 400)):
        """
        Args:
            bev_image_path: BEV 이미지 경로
            output_dir: 출력 디렉토리
            camera_image_path: 동기화된 카메라 이미지 경로 (옵션)
            target_size: 최종 저장 크기
        """
        # BEV 이미지 로드
        self.bev_image = cv2.imread(bev_image_path, cv2.IMREAD_GRAYSCALE)
        if self.bev_image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {bev_image_path}")
        
        self.h, self.w = self.bev_image.shape
        self.image_path = Path(bev_image_path)
        self.image_id = self.image_path.stem.replace('bev_', '')
        
        # 카메라 이미지 로드 (있으면)
        self.camera_image = None
        if camera_image_path and Path(camera_image_path).exists():
            self.camera_image = cv2.imread(camera_image_path)
            if self.camera_image is not None:
                # BEV 높이에 맞춰 리사이즈 (시각화용)
                cam_h, cam_w = self.camera_image.shape[:2]
                scale = self.h / cam_h
                new_w = int(cam_w * scale)
                self.camera_image = cv2.resize(self.camera_image, (new_w, self.h))
                
                # 카메라 이미지에 텍스트 추가
                cv2.putText(self.camera_image, "Camera View (Reference)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.target_size = target_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 작업용 이미지 (BEV 부분)
        self.bev_vis = cv2.cvtColor(self.bev_image.copy(), cv2.COLOR_GRAY2BGR)
        self.display_image = None # 전체 통합 뷰
        
        # 클래스 정의
        self.classes = {
            0: {'name': 'background', 'color': (0, 0, 0)},
            1: {'name': 'land', 'color': (0, 255, 0)},
            2: {'name': 'debris', 'color': (0, 0, 255)},
        }
        
        self.num_classes = len(self.classes)
        self.current_class = 1  # 기본: land
        
        # Segmentation map
        self.segmentation_map = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # 실행 취소를 위한 히스토리
        self.history = []
        self.max_history = 50
        
        # 기존 annotation 로드
        self.load_existing_annotation()
        
        # 초기 상태 저장
        self.save_to_history()
        
        # 폴리곤
        self.current_polygon = []
        
        # 지우개 모드
        self.eraser_mode = False
        self.eraser_size = 10
        
        # UI
        self.window_name = f'BEVFusion Annotation - {self.image_id}'
    
    def load_existing_annotation(self):
        """기존 annotation 로드"""
        seg_dir = self.output_dir / 'segmentation_maps'
        seg_file = seg_dir / f'{self.image_id}.npy'
        
        if seg_file.exists():
            seg_loaded = np.load(seg_file)
            if seg_loaded.shape != (self.h, self.w):
                self.segmentation_map = cv2.resize(
                    seg_loaded, (self.w, self.h),
                    interpolation=cv2.INTER_NEAREST
                )
                print(f"✓ 기존 annotation 로드 (Resized)")
            else:
                self.segmentation_map = seg_loaded
                print(f"✓ 기존 annotation 로드")
    
    def save_to_history(self):
        self.history.append(self.segmentation_map.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.segmentation_map = self.history[-1].copy()
            print(f"✓ Undo (history: {len(self.history)})")
            self.update_display()
        else:
            print("⚠ 더 이상 되돌릴 수 없습니다")
    
    def clear_all(self):
        self.segmentation_map = np.zeros((self.h, self.w), dtype=np.uint8)
        self.save_to_history()
        print("✓ 전체 annotation 초기화")
        self.update_display()
    
    def update_display(self):
        """디스플레이 업데이트 (BEV + Camera)"""
        # 1. BEV 시각화 생성
        self.bev_vis = cv2.cvtColor(self.bev_image.copy(), cv2.COLOR_GRAY2BGR)
        
        # Segmentation overlay
        for cid in range(0, self.num_classes):
            mask = self.segmentation_map == cid
            if np.any(mask):
                color = self.classes[cid]['color']
                overlay = np.zeros_like(self.bev_vis)
                overlay[mask] = color
                self.bev_vis = cv2.addWeighted(self.bev_vis, 0.7, overlay, 0.3, 0)
        
        # 현재 폴리곤 그리기
        if len(self.current_polygon) > 0:
            color = self.classes[self.current_class]['color']
            for i, pt in enumerate(self.current_polygon):
                cv2.circle(self.bev_vis, pt, 1, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(self.bev_vis, self.current_polygon[i-1], pt, color, 1)
            if len(self.current_polygon) > 2:
                cv2.line(self.bev_vis, self.current_polygon[-1], self.current_polygon[0], color, 1, cv2.LINE_AA)
        
        # UI 텍스트 그리기 (BEV 위에)
        self.draw_ui()
        
        # 2. 전체 화면 구성 (BEV + Camera Side-by-Side)
        if self.camera_image is not None:
            # 구분선 추가
            separator = np.zeros((self.h, 5, 3), dtype=np.uint8)
            separator[:] = (255, 255, 255)
            self.display_image = np.hstack([self.bev_vis, separator, self.camera_image])
        else:
            self.display_image = self.bev_vis
            
    def draw_ui(self):
        """UI 정보 그리기 (BEV 이미지 위에)"""
        # 반투명 배경
        overlay = self.bev_vis.copy()
        cv2.rectangle(overlay, (0, 0), (350, 200), (0, 0, 0), -1)
        self.bev_vis = cv2.addWeighted(self.bev_vis, 0.7, overlay, 0.3, 0)
        
        y = 25
        # 모드 표시
        if self.eraser_mode:
            mode_text = f"Mode: ERASER ({self.eraser_size})"
            mode_color = (0, 255, 255)
        else:
            class_name = self.classes[self.current_class]['name']
            mode_text = f"Mode: {class_name.upper()}"
            mode_color = self.classes[self.current_class]['color']
        
        cv2.putText(self.bev_vis, mode_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        y += 30
        unique_classes = np.unique(self.segmentation_map)
        annotated_classes = [self.classes[c]['name'] for c in unique_classes if c > 0]
        annotated_str = ", ".join(annotated_classes) if annotated_classes else "None"
        cv2.putText(self.bev_vis, f"Classes: {annotated_str}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(self.bev_vis, f"[1]Land [2]Debris [E]raser", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        cv2.putText(self.bev_vis, f"[S]ave [N]ext [P]rev [Q]uit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백"""
        # BEV 이미지 영역(왼쪽)을 벗어나면 무시
        if x >= self.w:
            return
            
        if self.eraser_mode:
            if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
                cv2.circle(self.segmentation_map, (x, y), self.eraser_size, 0, -1)
                self.update_display()
            elif event == cv2.EVENT_LBUTTONUP:
                self.save_to_history()
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_polygon.append((x, y))
                self.update_display()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.current_polygon) >= 3:
                    self.finish_polygon()
    
    def finish_polygon(self):
        if len(self.current_polygon) < 3: return
        
        polygon_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts = np.array(self.current_polygon, dtype=np.int32)
        cv2.fillPoly(polygon_mask, [pts], 255)
        
        white_pixels = self.bev_image > 50
        object_mask = np.logical_and(polygon_mask == 255, white_pixels)
        
        self.segmentation_map[object_mask] = self.current_class
        self.save_to_history()
        
        num_pixels = np.sum(object_mask)
        class_name = self.classes[self.current_class]['name']
        print(f"✓ {class_name}: {num_pixels} pixels added")
        
        self.current_polygon = []
        self.update_display()
    
    def save(self):
        if self.segmentation_map.shape != self.target_size:
            seg_to_save = cv2.resize(self.segmentation_map, self.target_size, interpolation=cv2.INTER_NEAREST)
        else:
            seg_to_save = self.segmentation_map
        
        seg_dir = self.output_dir / 'segmentation_maps'
        seg_dir.mkdir(exist_ok=True)
        np.save(seg_dir / f'{self.image_id}.npy', seg_to_save)
        
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        colored = np.zeros((*self.target_size, 3), dtype=np.uint8)
        for cid in range(self.num_classes):
            mask = seg_to_save == cid
            colored[mask] = self.classes[cid]['color']
        cv2.imwrite(str(vis_dir / f'{self.image_id}_vis.png'), colored)
        
        print(f"✓ 저장 완료: {self.image_id}")
        return True
    
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # 창 크기 조정 (카메라 이미지까지 포함하므로 너비를 늘림)
        width = 1000 + (600 if self.camera_image is not None else 0)
        cv2.resizeWindow(self.window_name, width, 1000)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if ord('1') <= key <= ord('9'):
                new_class = key - ord('0')
                if new_class < self.num_classes:
                    self.current_class = new_class
                    self.eraser_mode = False
                    print(f"Mode changed: {self.classes[self.current_class]['name']}")
                    self.update_display()
            elif key == ord('e'):
                self.eraser_mode = not self.eraser_mode
                mode_text = "ERASER" if self.eraser_mode else "POLYGON"
                print(f"Mode changed: {mode_text}")
                self.current_polygon = []
                self.update_display()
            elif key == ord('+') or key == ord('='):
                self.eraser_size = min(50, self.eraser_size + 2)
                self.update_display()
            elif key == ord('-') or key == ord('_'):
                self.eraser_size = max(2, self.eraser_size - 2)
                self.update_display()
            elif key == 13: # Enter
                if not self.eraser_mode and len(self.current_polygon) >= 3:
                    self.finish_polygon()
            elif key == ord('c'):
                self.current_polygon = []
                self.update_display()
            elif key == ord('C'):
                print("\n전체 초기화 하시겠습니까? (y/n)")
                if (cv2.waitKey(0) & 0xFF) == ord('y'):
                    self.clear_all()
            elif key == ord('z'):
                self.undo()
            elif key == ord('s'):
                self.save()
            elif key == ord('n'):
                self.save()
                cv2.destroyAllWindows()
                return 'next'
            elif key == ord('p'):
                self.save()
                cv2.destroyAllWindows()
                return 'prev'
            elif key == ord('x'):
                print("Skip frame")
                cv2.destroyAllWindows()
                return 'skip'
            elif key == ord('q'):
                self.save()
                cv2.destroyAllWindows()
                return 'quit'


# ============================================================================
# ROS2 Bag → BEVFusion 변환기
# ============================================================================

class ROS2BagToBEVFusion:
    """ROS2 Bag → BEVFusion 완전 자동 변환 (이미지 포함)"""
    
    def __init__(self,
                 x_range=(-50, 50),
                 y_range=(-50, 50),
                 z_range=(-5, 3),
                 resolution=0.5):
        
        self.point_cloud_range = [
            x_range[0], y_range[0], z_range[0],
            x_range[1], y_range[1], z_range[1]
        ]
        
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        
        self.bev_w = int((x_range[1] - x_range[0]) / resolution)
        self.bev_h = int((y_range[1] - y_range[0]) / resolution)
        
        print("=" * 70)
        print("ROS2 Bag → BEVFusion Data Generator")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Range: X={x_range}, Y={y_range}")
        print(f"  BEV Size: {self.bev_w}×{self.bev_h}")
        print()
    
    def points_to_bev(self, points):
        bev_image = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        if len(points) == 0: return bev_image
        
        mask = (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1]) & \
               (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1])
        filtered_points = points[mask]
        if len(filtered_points) == 0: return bev_image
        
        pixel_x = ((filtered_points[:, 0] - self.x_range[0]) / self.resolution).astype(np.int32)
        pixel_y = ((filtered_points[:, 1] - self.y_range[0]) / self.resolution).astype(np.int32)
        pixel_y = self.bev_h - 1 - pixel_y
        
        valid_mask = (pixel_x >= 0) & (pixel_x < self.bev_w) & (pixel_y >= 0) & (pixel_y < self.bev_h)
        bev_image[pixel_y[valid_mask], pixel_x[valid_mask]] = 255
        return bev_image
    
    def decode_image(self, msg):
        try:
            if msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if img.shape[2] == 3: pass 
            return img
        except Exception as e:
            print(f"Image decode error ({msg.encoding}): {e}")
            return None

    def extract_from_bag(self, bag_path, output_dir, pointcloud_topic='/points', image_topic='/image_raw', sample_every=1):
        output_dir = Path(output_dir)
        bev_dir = output_dir / 'bev_images'
        pc_dir = output_dir / 'pointclouds'
        img_dir = output_dir / 'images'
        
        bev_dir.mkdir(parents=True, exist_ok=True)
        pc_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("Step 1: Extracting Data (LiDAR & Camera)")
        print(f"Bag: {bag_path}")
        
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        reader.set_filter(rosbag2_py.StorageFilter(topics=[pointcloud_topic, image_topic]))
        
        frame_idx = 0
        saved_idx = 0
        file_info = []
        latest_image = None
        
        with tqdm(desc="Extracting", unit="msg") as pbar:
            while reader.has_next():
                topic, data, bag_timestamp = reader.read_next()
                
                if topic == image_topic:
                    msg = deserialize_message(data, Image)
                    img = self.decode_image(msg)
                    if img is not None: latest_image = img
                
                elif topic == pointcloud_topic:
                    if frame_idx % sample_every != 0:
                        frame_idx += 1
                        continue
                    
                    msg = deserialize_message(data, PointCloud2)
                    timestamp_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
                    token = datetime.fromtimestamp(timestamp_ns / 1e9).strftime('%Y%m%d_%H%M%S') + f"_{int(timestamp_ns) % 1000000000:09d}"
                    
                    points = []
                    for point in pc2.read_points(msg, skip_nans=True):
                        points.append([point[0], point[1], point[2], point[3] if len(point) > 3 else 0.0])
                    
                    if len(points) == 0:
                        frame_idx += 1
                        continue
                    
                    points_array = np.array(points, dtype=np.float32)
                    bev_image = self.points_to_bev(points_array[:, :3])
                    bev_path = bev_dir / f'bev_{token}.png'
                    cv2.imwrite(str(bev_path), bev_image)
                    
                    pc_path = pc_dir / f'bev_{token}.bin'
                    num_points = self.save_pointcloud_bin(points_array, pc_path)
                    
                    cam_path_str = ""
                    if latest_image is not None:
                        cam_path = img_dir / f'img_{token}.png'
                        cv2.imwrite(str(cam_path), latest_image)
                        cam_path_str = str(cam_path)
                    
                    file_info.append({
                        'idx': saved_idx, 'token': token, 'bev_path': str(bev_path),
                        'pc_path': str(pc_path), 'cam_path': cam_path_str,
                        'num_points': num_points, 'timestamp_ns': int(timestamp_ns)
                    })
                    saved_idx += 1
                    frame_idx += 1
                    pbar.set_postfix({'saved': saved_idx})
                pbar.update(1)
        return file_info
    
    def save_pointcloud_bin(self, points, output_path):
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
               (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
               (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        filtered_points = points[mask]
        filtered_points.astype(np.float32).tofile(output_path)
        return len(filtered_points)

    def run_annotation(self, output_dir, annotation_dir):
        print("\n" + "=" * 70)
        print("Step 2: Annotation")
        print("=" * 70)
        
        bev_dir = Path(output_dir) / 'bev_images'
        image_files = sorted(list(bev_dir.glob('*.png')))
        
        if len(image_files) == 0:
            print("⚠ BEV 이미지가 없습니다.")
            return False
        
        print(f"총 {len(image_files)}개 이미지 Annotation")
        input("Press Enter to start...")
        
        idx = 0
        while idx < len(image_files):
            image_file = image_files[idx]
            token = image_file.stem.replace('bev_', '') # 토큰 추출
            
            # 카메라 이미지 경로 찾기
            cam_path = Path(output_dir) / 'images' / f'img_{token}.png'
            cam_path_str = str(cam_path) if cam_path.exists() else None
            
            print(f"\n[{idx+1}/{len(image_files)}] {image_file.name}")
            if cam_path_str: print(f"  + Camera view: {cam_path.name}")
            
            try:
                tool = BEVFusionAnnotationTool(
                    str(image_file),
                    output_dir=annotation_dir,
                    camera_image_path=cam_path_str, # 카메라 경로 전달
                    target_size=(self.bev_w, self.bev_h)
                )
                
                result = tool.run()
                
                if result == 'quit': break
                elif result == 'next' or result == 'skip': idx += 1
                elif result == 'prev': idx = max(0, idx - 1)
                
            except Exception as e:
                print(f"Error: {e}")
                idx += 1
                continue
        return True
    
    def create_metadata(self, file_info, annotation_dir, output_dir, train_ratio=0.8):
        print("\n" + "=" * 70)
        print("Step 3: Creating Metadata")
        print("=" * 70)
        
        annotation_dir = Path(annotation_dir)
        seg_dir = annotation_dir / 'segmentation_maps'
        seg_files = sorted(list(seg_dir.glob('*.npy')))
        
        if len(seg_files) == 0:
            print(f"⚠ Segmentation 파일이 없습니다.")
            return False
        
        infos = []
        file_info_dict = {item['token']: item for item in file_info}
        
        for idx, seg_file in enumerate(seg_files):
            token = seg_file.stem
            if token in file_info_dict:
                data_info = file_info_dict[token]
                info = {
                    'sample_idx': idx, 'token': token,
                    'pts_path': data_info['pc_path'], 'cam_path': data_info['cam_path'],
                    'seg_path': str(seg_file), 'timestamp': token
                }
                infos.append(info)
        
        n_train = int(len(infos) * train_ratio)
        metadata_dir = annotation_dir / 'metadata'
        metadata_dir.mkdir(exist_ok=True)
        
        with open(metadata_dir / 'train_infos.pkl', 'wb') as f: pickle.dump(infos[:n_train], f)
        with open(metadata_dir / 'val_infos.pkl', 'wb') as f: pickle.dump(infos[n_train:], f)
            
        metadata = {
            'dataset_name': 'Marine_Debris_BEV_Fusion',
            'created': datetime.now().isoformat(),
            'num_classes': 3,
            'classes': {'0': 'background', '1': 'land', '2': 'debris'},
            'num_images': len(infos),
            'bev_config': {
                'x_range': list(self.x_range), 'y_range': list(self.y_range),
                'resolution': self.resolution, 'bev_h': self.bev_h, 'bev_w': self.bev_w
            }
        }
        with open(metadata_dir / 'dataset_info.json', 'w') as f: json.dump(metadata, f, indent=2)
        print(f"✓ Metadata Created: {len(infos)} samples")
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True, help='Input ROS2 bag path')
    parser.add_argument('--output-dir', default='/home/anhong/BEVFusion/data/bev_fusion')
    parser.add_argument('--annotation-dir', default='/home/anhong/BEVFusion/dataset/bev_fusion')
    parser.add_argument('--pointcloud-topic', default='/points')
    parser.add_argument('--image-topic', default='/cam/image_raw')
    parser.add_argument('--x-range', nargs=2, type=float, default=[-20, 20])
    parser.add_argument('--y-range', nargs=2, type=float, default=[-20, 20])
    parser.add_argument('--z-range', nargs=2, type=float, default=[-5, 3])
    parser.add_argument('--resolution', type=float, default=0.1)
    parser.add_argument('--sample-every', type=int, default=5)
    parser.add_argument('--skip-annotation', action='store_true')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    
    args = parser.parse_args()
    
    converter = ROS2BagToBEVFusion(tuple(args.x_range), tuple(args.y_range), tuple(args.z_range), args.resolution)
    file_info = converter.extract_from_bag(args.bag, args.output_dir, args.pointcloud_topic, args.image_topic, args.sample_every)
    if not file_info: return
    
    if not args.skip_annotation:
        converter.run_annotation(args.output_dir, args.annotation_dir)
    
    converter.create_metadata(file_info, args.annotation_dir, args.output_dir, args.train_ratio)

if __name__ == '__main__':
    main()