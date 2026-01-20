import cv2
import numpy as np
import yaml
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_typestore  # [변경] TypeStore 관련 모듈 import
from pathlib import Path
import os

class DepthGTGenerator:
    def __init__(self, bag_path, output_dir):
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 저장 경로 생성
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "depths").mkdir(exist_ok=True)
        (self.output_dir / "viz").mkdir(exist_ok=True)
        
        # [중요] TypeStore 초기화 (ROS2 Humble 기준 표준 메시지 로드)
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        
        # Topic 설정 (Bag 파일 내용을 보고 수정이 필요할 수 있습니다)
        self.img_topic = '/cam/image_raw'      
        self.lidar_topic = '/points' 
        
        # ==========================================
        # 1. 캘리브레이션 데이터
        # ==========================================
        self.W = 640
        self.H = 368
        
        self.K = np.array([
            [278.270488, 0.0,        320.000000],
            [0.0,        278.270488, 184.000000],
            [0.0,        0.0,        1.0]
        ], dtype=np.float32)
        
        self.D = np.array([-0.333915, 0.214450, -0.011128, 0.001014, 0.0], dtype=np.float32)
        
        self.T_lidar2cam = np.array([
            [ 0.0015,  0.9993, -0.0366,  0.0000],
            [ 0.0941, -0.0366, -0.9949,  0.2200],
            [-0.9956, -0.0020, -0.0941,  0.1600],
            [ 0.0000,  0.0000,  0.0000,  1.0000]
        ], dtype=np.float32)
        
        self.rvec, _ = cv2.Rodrigues(self.T_lidar2cam[:3, :3])
        self.tvec = self.T_lidar2cam[:3, 3]

    def process_bag(self):
        print(f"Opening bag: {self.bag_path}")
        
        with Reader2(self.bag_path) as reader:
            # [디버깅] Bag 파일 내 모든 토픽 출력 (설정 확인용)
            print("\n=== Available Topics in Bag ===")
            for connection in reader.connections:
                print(f" - {connection.topic} ({connection.msgtype})")
            print("===============================\n")

            connections = [x for x in reader.connections if x.topic in [self.img_topic, self.lidar_topic]]
            
            if not connections:
                print(f"[오류] 설정한 토픽({self.img_topic}, {self.lidar_topic})을 찾을 수 없습니다.")
                print("위의 Available Topics 목록을 확인하고 self.img_topic, self.lidar_topic 값을 수정하세요.")
                return

            last_img = None
            last_img_ts = 0
            
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # [변경] typestore를 사용하여 deserialize
                msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                if connection.topic == self.img_topic:
                    if hasattr(msg, 'data'):
                        try:
                            # msg.data는 보통 memoryview 또는 array 형태
                            # np.frombuffer로 안전하게 변환
                            data_array = np.frombuffer(msg.data, dtype=np.uint8)
                            img = data_array.reshape(msg.height, msg.width, -1)
                            
                            if msg.encoding == 'rgb8':
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            
                            # 해상도 체크 (캘리브레이션과 다를 경우 경고)
                            if img.shape[:2] != (self.H, self.W):
                                # 필요시 리사이즈 또는 캘리브레이션 수정 필요
                                pass
                                
                            last_img = img
                            last_img_ts = timestamp
                        except Exception as e:
                            print(f"Image decode error: {e}")

                elif connection.topic == self.lidar_topic:
                    if last_img is None:
                        continue
                    
                    if abs(timestamp - last_img_ts) / 1e9 > 0.1:
                        continue
                        
                    points = self.parse_pointcloud(msg)
                    self.process_frame(last_img, points, timestamp)

    def parse_pointcloud(self, msg):
        # msg.data는 uint8 배열 형태
        data = np.frombuffer(msg.data, dtype=np.float32)
        stride = msg.point_step // 4
        points = data.reshape(-1, stride)[:, :3]
        return points

    def project_lidar_to_image(self, points):
        if len(points) == 0:
            return None, None
            
        points_cam = (self.T_lidar2cam[:3, :3] @ points.T).T + self.T_lidar2cam[:3, 3]
        
        # 전방 0.5m 이상만
        mask_z = points_cam[:, 2] > 0.5
        points_filtered = points[mask_z]
        depth_filtered = points_cam[mask_z, 2]
        
        if len(points_filtered) == 0:
            return None, None

        image_points, _ = cv2.projectPoints(
            points_filtered, 
            self.rvec, 
            self.tvec, 
            self.K, 
            self.D
        )
        
        image_points = image_points.squeeze()
        
        if image_points.ndim == 1: # 점이 하나일 경우 차원 유지
            image_points = image_points[np.newaxis, :]

        u = image_points[:, 0]
        v = image_points[:, 1]
        
        mask_uv = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        
        valid_u = u[mask_uv].astype(np.int32)
        valid_v = v[mask_uv].astype(np.int32)
        valid_d = depth_filtered[mask_uv]
        
        return np.stack([valid_u, valid_v], axis=1), valid_d

    def process_frame(self, img, points, timestamp):
        uv, depth = self.project_lidar_to_image(points)
        
        if uv is None:
            return

        vis_img = img.copy()
        depth_map = np.zeros((self.H, self.W), dtype=np.float32)
        
        # 정렬: 먼 점 -> 가까운 점 순서로 그려서 가까운 점이 위에 오도록
        sort_idx = np.argsort(depth)[::-1]
        
        for i in sort_idx:
            u, v = uv[i]
            d = depth[i]
            
            depth_map[v, u] = d
            
            # 시각화 (1m ~ 30m 범위로 색상 표현)
            norm_d = np.clip((d - 1.0) / 29.0, 0, 1)
            # Blue(Far) -> Red(Close)
            # OpenCV는 BGR 순서: (Blue, Green, Red)
            # Far(1) -> Blue(255,0,0), Close(0) -> Red(0,0,255)
            color = (int(255 * norm_d), 0, int(255 * (1 - norm_d)))
            
            cv2.circle(vis_img, (u, v), 1, color, -1)
            
        cv2.putText(vis_img, f"Depth Pts: {len(uv)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis_img, "SPACE: Save, N: Skip, Q: Quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Depth GT Generator", vis_img)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()
        elif key == ord(' '):
            file_name = f"{timestamp}"
            cv2.imwrite(str(self.output_dir / "images" / f"{file_name}.jpg"), img)
            np.save(str(self.output_dir / "depths" / f"{file_name}.npy"), depth_map)
            cv2.imwrite(str(self.output_dir / "viz" / f"{file_name}_vis.jpg"), vis_img)
            print(f"Saved: {file_name}")
        elif key == ord('n'):
            print("Skipped")

if __name__ == "__main__":
    # 파일 경로 확인
    BAG_FILE = "/home/anhong/BEVFusion/data/origin/test1/test1.db3" 
    OUTPUT_DIR = "/home/anhong/BEVFusion/dataset/depth_gt"
    
    gen = DepthGTGenerator(BAG_FILE, OUTPUT_DIR)
    gen.process_bag()