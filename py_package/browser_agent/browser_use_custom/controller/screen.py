import asyncio
import io
import logging
from typing import Optional, Tuple
from PIL import Image, ImageChops
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class VisualChangeDetector:
    """视觉变化检测器（基于屏幕截图比对）"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        pixel_change_threshold: int = 1000,
        resize_width: Optional[int] = 800
    ):
        """
        Args:
            similarity_threshold: 相似度阈值（0-1）
            pixel_change_threshold: 变化像素数量阈值
            resize_width: 缩放宽度（提升性能）
        """
        self.similarity_threshold = similarity_threshold
        self.pixel_threshold = pixel_change_threshold
        self.resize_width = resize_width

    async def capture_screenshot(self, page) -> Image.Image:
        """捕获页面截图（自动优化）"""
        screenshot_bytes = await page.screenshot(timeout=5000,type='jpeg',quality=50,full_page=False)
        img = Image.open(io.BytesIO(screenshot_bytes))
        
        # 统一缩放尺寸（提升比对性能）
        if self.resize_width:
            w, h = img.size
            new_h = int(h * (self.resize_width / w))
            img = img.resize((self.resize_width, new_h))
        
        return img.convert('RGB')  # 确保RGB模式

    def calculate_change(
        self, 
        img1: Image.Image, 
        img2: Image.Image
    ) -> Tuple[float, Image.Image]:
        """
        计算两图差异
        
        Returns:
            tuple: (相似度百分比, 差异图)
        """
        # 确保尺寸一致
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # 转换为numpy数组
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # 计算绝对差异
        diff = np.abs(arr1.astype(int) - arr2.astype(int))
        changed_pixels = np.sum(diff > 0)  # 变化像素总数
        total_pixels = arr1.size // 3      # RGB通道
        
        # 生成差异图（可视化用）
        diff_img = ImageChops.difference(img1, img2)
        similarity = 1 - (changed_pixels / total_pixels)
        
        return similarity, diff_img

    async def detect_change(
        self,
        browser_session,
        reference_img: Optional[Image.Image] = None,
        max_attempts: int = 5,
        attempt_interval: float = 1.5
    ) -> Tuple[bool, Optional[Image.Image], Optional[Image.Image]]:
        """
        检测视觉变化
        
        Args:
            page: 浏览器页面对象
            reference_img: 基准截图（None则自动捕获）
            max_attempts: 最大检测次数
            attempt_interval: 检测间隔（秒）
            
        Returns:
            tuple: (是否变化, 基准截图, 差异图)
        """
        # 首次捕获基准图
        if reference_img is None:
            page = await browser_session.get_current_page()
            reference_img = await self.capture_screenshot(page)
        
        for attempt in range(max_attempts):
            await asyncio.sleep(attempt_interval)
            
            # 捕获当前截图
            page = await browser_session.get_current_page()
            current_img = await self.capture_screenshot(page)
            
            # 计算变化
            similarity, diff_img = self.calculate_change(reference_img, current_img)
            
            # 判断是否显著变化
            if similarity < self.similarity_threshold:
                diff_pixels = np.sum(np.array(diff_img) > 0)
                if diff_pixels > self.pixel_threshold:
                    logger.info(f"视觉变化 detected (相似度: {similarity:.2f}, 变化像素: {diff_pixels})")
                    return True, reference_img, diff_img
            
            reference_img = current_img  # 更新基准图
        
        return False, reference_img, None

class WaitForVisualChangeAction(BaseModel):
    """等待视觉变化的参数模型"""
    timeout: int = 30
    check_interval: float = 2
    similarity_threshold: float = 0.95
    pixel_threshold: int = 5000

async def wait_for_visual_change(
    params: WaitForVisualChangeAction,
    browser_session,
    initial_screenshot: Optional[Image.Image] = None
) -> Tuple[bool, Optional[Image.Image], Optional[Image.Image]]:
    """
    等待页面视觉变化（完整工作流）
    
    Args:
        params: 配置参数
        browser_session: 浏览器会话
        initial_screenshot: 初始截图（可选）
        
    Returns:
        tuple: (是否变化, 初始截图, 差异图)
    """
    detector = VisualChangeDetector(
        similarity_threshold=params.similarity_threshold,
        pixel_change_threshold=params.pixel_threshold
    )
    
    max_attempts = int(params.timeout / params.check_interval)
    
    return await detector.detect_change(
        browser_session=browser_session,
        reference_img=initial_screenshot,
        max_attempts=max_attempts,
        attempt_interval=params.check_interval
    )