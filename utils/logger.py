import logging
import os
import sys

def setup_logging(save_dir, log_name='train.log'):
    """
    로그 설정을 위한 함수
    Args:
        save_dir: 로그 파일이 저장될 디렉토리
        log_name: 로그 파일 이름
    """
    # Logger 생성
    logger = logging.getLogger('BEVFusion')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 중복 핸들러 방지
    if logger.handlers:
        return logger

    # 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 1. 파일 핸들러 (파일에 기록)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, log_name)
    fh = logging.FileHandler(file_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 2. 콘솔 핸들러 (터미널에 출력)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger