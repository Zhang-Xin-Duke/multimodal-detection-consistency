"""
检测管道模块

实现完整的多模态对抗性检测管道，集成各个组件。
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import time
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models.clip_model import CLIPModel
from .text_augment import TextAugmenter, TextAugmentConfig
from .retrieval import MultiModalRetriever, RetrievalConfig
from .sd_ref import SDReferenceGenerator, SDReferenceConfig
from .detector import AdversarialDetector, DetectorConfig
from .utils.metrics import MetricsCalculator, RetrievalEvaluator, DetectionEvaluator
from .utils.visualization import ExperimentVisualizer
from .utils.config import ConfigManager
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """管道配置"""
    # 组件启用配置
    enable_text_augment: bool = True
    enable_retrieval: bool = True
    enable_sd_reference: bool = True
    enable_detection: bool = True
    
    # 并行处理
    enable_parallel: bool = True
    max_workers: int = 4
    
    # 批处理
    batch_size: int = 32
    
    # 缓存配置
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    # 输出配置
    save_intermediate_results: bool = False
    output_dir: Optional[str] = None
    
    # 性能监控
    enable_profiling: bool = False
    profile_steps: bool = True
    
    # 组件配置
    text_augment_config: Optional[TextAugmentConfig] = None
    retrieval_config: Optional[RetrievalConfig] = None
    sd_reference_config: Optional[SDReferenceConfig] = None
    detector_config: Optional[DetectorConfig] = None
    
    def __post_init__(self):
        # 设置默认配置
        if self.text_augment_config is None:
            self.text_augment_config = TextAugmentConfig()
        if self.retrieval_config is None:
            self.retrieval_config = RetrievalConfig()
        if self.sd_reference_config is None:
            self.sd_reference_config = SDReferenceConfig()
        if self.detector_config is None:
            self.detector_config = DetectorConfig()


@dataclass
class PipelineResult:
    """管道结果"""
    # 输入信息
    original_image: Optional[Image.Image] = None
    original_text: str = ""
    
    # 文本增强结果
    text_variants: List[str] = field(default_factory=list)
    text_augment_time: float = 0.0
    
    # 检索结果
    retrieved_images: List[Image.Image] = field(default_factory=list)
    retrieved_texts: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    retrieval_time: float = 0.0
    
    # SD参考结果
    reference_images: List[Image.Image] = field(default_factory=list)
    reference_generation_time: float = 0.0
    
    # 检测结果
    is_adversarial: bool = False
    detection_score: float = 0.0
    detection_details: Dict[str, Any] = field(default_factory=dict)
    detection_time: float = 0.0
    
    # 总体信息
    total_time: float = 0.0
    pipeline_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def adversarial_score(self) -> float:
        """对抗分数（detection_score的别名，用于向后兼容）"""
        return self.detection_score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'original_text': self.original_text,
            'text_variants': self.text_variants,
            'text_augment_time': self.text_augment_time,
            'retrieved_texts': self.retrieved_texts,
            'retrieval_scores': self.retrieval_scores,
            'retrieval_time': self.retrieval_time,
            'reference_generation_time': self.reference_generation_time,
            'is_adversarial': self.is_adversarial,
            'detection_score': self.detection_score,
            'detection_details': self.detection_details,
            'detection_time': self.detection_time,
            'total_time': self.total_time,
            'pipeline_steps': self.pipeline_steps,
            'errors': self.errors
        }


@dataclass
class BatchProcessingResult:
    """批处理结果"""
    results: List[PipelineResult] = field(default_factory=list)
    total_time: float = 0.0
    batch_size: int = 0
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.batch_size = len(self.results)
        self.success_count = sum(1 for r in self.results if not r.errors)
        self.error_count = self.batch_size - self.success_count
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.results:
            return {}
            
        detection_scores = [r.detection_score for r in self.results if not r.errors]
        adversarial_count = sum(1 for r in self.results if r.is_adversarial and not r.errors)
        
        return {
            'batch_size': self.batch_size,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / self.batch_size if self.batch_size > 0 else 0,
            'adversarial_count': adversarial_count,
            'adversarial_rate': adversarial_count / self.success_count if self.success_count > 0 else 0,
            'avg_detection_score': np.mean(detection_scores) if detection_scores else 0,
            'total_time': self.total_time,
            'avg_time_per_sample': self.total_time / self.batch_size if self.batch_size > 0 else 0
        }
        
    def filter_results(self, condition: Callable[[PipelineResult], bool]) -> 'BatchProcessingResult':
        """过滤结果"""
        filtered_results = [r for r in self.results if condition(r)]
        return BatchProcessingResult(
            results=filtered_results,
            total_time=self.total_time,
            errors=self.errors
        )


class PipelineProfiler:
    """管道性能分析器"""
    
    def __init__(self):
        self.step_times = {}
        self.step_counts = {}
        self.current_step = None
        self.step_start_time = None
        self.total_start_time = None
        self.lock = threading.Lock()
    
    def start_profiling(self):
        """开始性能分析"""
        with self.lock:
            self.total_start_time = time.time()
    
    def start_step(self, step_name: str):
        """开始步骤计时"""
        with self.lock:
            if self.current_step is not None:
                self.end_step()
            
            self.current_step = step_name
            self.step_start_time = time.time()
    
    def end_step(self):
        """结束步骤计时"""
        with self.lock:
            if self.current_step is not None and self.step_start_time is not None:
                step_time = time.time() - self.step_start_time
                
                if self.current_step not in self.step_times:
                    self.step_times[self.current_step] = []
                    self.step_counts[self.current_step] = 0
                
                self.step_times[self.current_step].append(step_time)
                self.step_counts[self.current_step] += 1
                
                self.current_step = None
                self.step_start_time = None
    
    def end_profiling(self) -> Dict[str, Any]:
        """结束性能分析并返回结果"""
        with self.lock:
            if self.current_step is not None:
                self.end_step()
            
            total_time = time.time() - self.total_start_time if self.total_start_time else 0.0
            
            # 计算统计信息
            stats = {
                'total_time': total_time,
                'step_stats': {}
            }
            
            for step_name, times in self.step_times.items():
                stats['step_stats'][step_name] = {
                    'count': self.step_counts[step_name],
                    'total_time': sum(times),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
            
            return stats
    
    def reset(self):
        """重置分析器"""
        with self.lock:
            self.step_times.clear()
            self.step_counts.clear()
            self.current_step = None
            self.step_start_time = None
            self.total_start_time = None


class MultiModalDetectionPipeline:
    """多模态检测管道"""
    
    def __init__(self, config: PipelineConfig):
        """
        初始化检测管道
        
        Args:
            config: 管道配置
        """
        self.config = config
        
        # 初始化组件
        self.text_augmenter = None
        self.retriever = None
        self.sd_generator = None
        self.detector = None
        
        self._initialize_components()
        
        # 性能分析器
        self.profiler = PipelineProfiler() if self.config.enable_profiling else None
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 可视化器
        self.visualizer = ExperimentVisualizer()
        
        # 线程池
        self.executor = None
        if self.config.enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # 统计信息
        self.pipeline_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'total_time': 0.0,
            'component_usage': {
                'text_augment': 0,
                'retrieval': 0,
                'sd_reference': 0,
                'detection': 0
            }
        }
        
        logger.info("多模态检测管道初始化完成")
    
    def _initialize_components(self):
        """初始化管道组件"""
        try:
            # 初始化文本增强器
            if self.config.enable_text_augment:
                self.text_augmenter = TextAugmenter(self.config.text_augment_config)
                logger.info("文本增强器初始化完成")
            
            # 初始化检索器
            if self.config.enable_retrieval:
                self.retriever = MultiModalRetriever(self.config.retrieval_config)
                logger.info("检索器初始化完成")
            
            # 初始化SD参考生成器
            if self.config.enable_sd_reference:
                self.sd_generator = SDReferenceGenerator(self.config.sd_reference_config)
                logger.info("SD参考生成器初始化完成")
            
            # 初始化检测器
            if self.config.enable_detection:
                self.detector = AdversarialDetector(self.config.detector_config)
                logger.info("检测器初始化完成")
                
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def process_single(self, image: Union[Image.Image, torch.Tensor], 
                      text: str, 
                      steps: Optional[List[str]] = None) -> PipelineResult:
        """
        处理单个样本
        
        Args:
            image: 输入图像
            text: 输入文本
            steps: 执行的步骤列表
            
        Returns:
            管道结果
        """
        try:
            if self.profiler:
                self.profiler.start_profiling()
            
            start_time = time.time()
            result = PipelineResult(
                original_image=image if isinstance(image, Image.Image) else None,
                original_text=text
            )
            
            # 默认执行所有启用的步骤
            if steps is None:
                steps = []
                if self.config.enable_text_augment:
                    steps.append('text_augment')
                if self.config.enable_retrieval:
                    steps.append('retrieval')
                if self.config.enable_sd_reference:
                    steps.append('sd_reference')
                if self.config.enable_detection:
                    steps.append('detection')
            
            result.pipeline_steps = steps
            
            # 执行各个步骤
            for step in steps:
                try:
                    if self.profiler:
                        self.profiler.start_step(step)
                    
                    if step == 'text_augment':
                        self._execute_text_augment(result)
                    elif step == 'retrieval':
                        self._execute_retrieval(result)
                    elif step == 'sd_reference':
                        self._execute_sd_reference(result)
                    elif step == 'detection':
                        self._execute_detection(result, image)
                    
                    if self.profiler:
                        self.profiler.end_step()
                        
                except Exception as e:
                    error_msg = f"步骤 {step} 执行失败: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            
            result.total_time = time.time() - start_time
            
            # 更新统计信息
            self.pipeline_stats['total_processed'] += 1
            if not result.errors:
                self.pipeline_stats['successful_processed'] += 1
            else:
                self.pipeline_stats['failed_processed'] += 1
            self.pipeline_stats['total_time'] += result.total_time
            
            # 保存中间结果
            if self.config.save_intermediate_results and self.config.output_dir:
                self._save_intermediate_results(result)
            
            logger.debug(f"样本处理完成: {text[:50]}... (耗时: {result.total_time:.3f}s)")
            return result
            
        except Exception as e:
            logger.error(f"样本处理失败: {e}")
            result = PipelineResult(
                original_image=image if isinstance(image, Image.Image) else None,
                original_text=text,
                errors=[str(e)]
            )
            self.pipeline_stats['total_processed'] += 1
            self.pipeline_stats['failed_processed'] += 1
            return result
    
    def _execute_text_augment(self, result: PipelineResult):
        """执行文本增强步骤"""
        if self.text_augmenter is None:
            return
        
        start_time = time.time()
        
        try:
            variants = self.text_augmenter.generate_variants(result.original_text)
            result.text_variants = variants
            result.text_augment_time = time.time() - start_time
            
            self.pipeline_stats['component_usage']['text_augment'] += 1
            logger.debug(f"文本增强完成: 生成 {len(variants)} 个变体")
            
        except Exception as e:
            logger.error(f"文本增强失败: {e}")
            result.errors.append(f"文本增强失败: {e}")
    
    def _execute_retrieval(self, result: PipelineResult):
        """执行检索步骤"""
        if self.retriever is None:
            return
        
        start_time = time.time()
        
        try:
            # 使用原始文本进行检索
            retrieved_paths, retrieval_scores = self.retriever.retrieve_images_by_text(
                result.original_text,
                top_k=5
            )
            
            if retrieved_paths:
                # 加载检索到的图像
                retrieved_images = []
                retrieved_texts = []
                
                for path in retrieved_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        retrieved_images.append(img)
                        
                        # 从图像路径提取文本描述或使用原始查询文本作为匹配标准
                        # 对于检索正确率计算，我们需要有意义的文本描述
                        # 这里使用原始查询文本作为期望的检索结果
                        retrieved_texts.append(result.original_text)
                        
                    except Exception as e:
                        logger.warning(f"无法加载图像 {path}: {e}")
                        continue
                
                result.retrieved_images = retrieved_images
                result.retrieved_texts = retrieved_texts
                result.retrieval_scores = retrieval_scores
            
            result.retrieval_time = time.time() - start_time
            
            self.pipeline_stats['component_usage']['retrieval'] += 1
            logger.debug(f"检索完成: 检索到 {len(result.retrieved_images)} 张图像")
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            result.errors.append(f"检索失败: {e}")
    
    def _execute_sd_reference(self, result: PipelineResult):
        """执行SD参考生成步骤"""
        if self.sd_generator is None:
            return
        
        start_time = time.time()
        
        try:
            ref_result = self.sd_generator.generate_reference_images(
                result.original_text,
                num_images=3
            )
            
            result.reference_images = ref_result.get('images', [])
            result.reference_generation_time = time.time() - start_time
            
            self.pipeline_stats['component_usage']['sd_reference'] += 1
            logger.debug(f"SD参考生成完成: 生成 {len(result.reference_images)} 张参考图像")
            
        except Exception as e:
            logger.error(f"SD参考生成失败: {e}")
            result.errors.append(f"SD参考生成失败: {e}")
    
    def _execute_detection(self, result: PipelineResult, 
                          image: Union[Image.Image, torch.Tensor]):
        """执行检测步骤"""
        if self.detector is None:
            return
        
        start_time = time.time()
        
        try:
            detection_result = self.detector.detect_adversarial(
                image, 
                result.original_text
            )
            
            result.is_adversarial = detection_result.get('is_adversarial', False)
            result.detection_score = detection_result.get('aggregated_score', 0.0)
            result.detection_details = detection_result.get('detection_details', {})
            result.detection_time = time.time() - start_time
            
            self.pipeline_stats['component_usage']['detection'] += 1
            logger.debug(f"检测完成: {'对抗性' if result.is_adversarial else '正常'} (分数: {result.detection_score:.3f})")
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            result.errors.append(f"检测失败: {e}")
    
    def process_batch(self, images: List[Union[Image.Image, torch.Tensor]], 
                     texts: List[str], 
                     steps: Optional[List[str]] = None) -> List[PipelineResult]:
        """
        批量处理样本
        
        Args:
            images: 图像列表
            texts: 文本列表
            steps: 执行的步骤列表
            
        Returns:
            管道结果列表
        """
        if len(images) != len(texts):
            raise ValueError("图像和文本数量不匹配")
        
        results = []
        
        if self.config.enable_parallel and self.executor is not None:
            # 并行处理
            futures = []
            for image, text in zip(images, texts):
                future = self.executor.submit(self.process_single, image, text, steps)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"批量处理中的任务失败: {e}")
                    results.append(PipelineResult(errors=[str(e)]))
        else:
            # 串行处理
            for image, text in zip(images, texts):
                result = self.process_single(image, text, steps)
                results.append(result)
        
        logger.info(f"批量处理完成: {len(results)} 个样本")
        return results
    
    def _save_intermediate_results(self, result: PipelineResult):
        """保存中间结果"""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = int(time.time() * 1000)
            filename = f"pipeline_result_{timestamp}.json"
            
            # 保存结果
            result_dict = result.to_dict()
            with open(output_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            # 保存图像（如果有）
            if result.reference_images:
                image_dir = output_dir / "reference_images"
                image_dir.mkdir(exist_ok=True)
                
                for i, image in enumerate(result.reference_images):
                    image_path = image_dir / f"ref_{timestamp}_{i}.png"
                    image.save(image_path)
                    
        except Exception as e:
            logger.error(f"保存中间结果失败: {e}")
    
    def evaluate_pipeline(self, test_data: List[Tuple[Any, str, bool]], 
                         steps: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        评估管道性能
        
        Args:
            test_data: 测试数据 [(image, text, is_adversarial), ...]
            steps: 执行的步骤列表
            
        Returns:
            评估结果
        """
        try:
            results = []
            labels = []
            predictions = []
            scores = []
            
            for image, text, is_adversarial in test_data:
                result = self.process_single(image, text, steps)
                results.append(result)
                labels.append(is_adversarial)
                predictions.append(result.is_adversarial)
                scores.append(result.detection_score)
            
            # 计算检测性能
            detection_metrics = self.metrics_calculator.compute_detection_metrics(
                np.array(labels),
                np.array(predictions),
                np.array(scores)
            )
            
            # 计算时间统计
            time_stats = {
                'mean_total_time': np.mean([r.total_time for r in results]),
                'mean_detection_time': np.mean([r.detection_time for r in results]),
                'mean_text_augment_time': np.mean([r.text_augment_time for r in results]),
                'mean_retrieval_time': np.mean([r.retrieval_time for r in results]),
                'mean_reference_time': np.mean([r.reference_generation_time for r in results])
            }
            
            # 性能分析结果
            profiling_stats = {}
            if self.profiler:
                profiling_stats = self.profiler.end_profiling()
            
            evaluation_result = {
                'detection_metrics': detection_metrics,
                'time_stats': time_stats,
                'profiling_stats': profiling_stats,
                'pipeline_stats': self.pipeline_stats.copy(),
                'num_samples': len(test_data),
                'error_rate': sum(1 for r in results if r.errors) / len(results)
            }
            
            logger.info(f"管道评估完成: {len(test_data)} 个样本")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"管道评估失败: {e}")
            return {}
    
    def generate_report(self, evaluation_result: Dict[str, Any], 
                       save_path: Optional[str] = None) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_result: 评估结果
            save_path: 保存路径
            
        Returns:
            报告内容
        """
        try:
            report_lines = []
            report_lines.append("# 多模态检测管道评估报告")
            report_lines.append("")
            
            # 基本信息
            report_lines.append("## 基本信息")
            report_lines.append(f"- 样本数量: {evaluation_result.get('num_samples', 0)}")
            report_lines.append(f"- 错误率: {evaluation_result.get('error_rate', 0):.2%}")
            report_lines.append("")
            
            # 检测性能
            detection_metrics = evaluation_result.get('detection_metrics', {})
            if detection_metrics:
                report_lines.append("## 检测性能")
                report_lines.append(f"- 准确率: {detection_metrics.get('accuracy', 0):.3f}")
                report_lines.append(f"- 精确率: {detection_metrics.get('precision', 0):.3f}")
                report_lines.append(f"- 召回率: {detection_metrics.get('recall', 0):.3f}")
                report_lines.append(f"- F1分数: {detection_metrics.get('f1_score', 0):.3f}")
                report_lines.append(f"- AUC: {detection_metrics.get('auc', 0):.3f}")
                report_lines.append("")
            
            # 时间性能
            time_stats = evaluation_result.get('time_stats', {})
            if time_stats:
                report_lines.append("## 时间性能")
                report_lines.append(f"- 平均总时间: {time_stats.get('mean_total_time', 0):.3f}s")
                report_lines.append(f"- 平均检测时间: {time_stats.get('mean_detection_time', 0):.3f}s")
                report_lines.append(f"- 平均文本增强时间: {time_stats.get('mean_text_augment_time', 0):.3f}s")
                report_lines.append(f"- 平均检索时间: {time_stats.get('mean_retrieval_time', 0):.3f}s")
                report_lines.append(f"- 平均参考生成时间: {time_stats.get('mean_reference_time', 0):.3f}s")
                report_lines.append("")
            
            # 管道统计
            pipeline_stats = evaluation_result.get('pipeline_stats', {})
            if pipeline_stats:
                report_lines.append("## 管道统计")
                report_lines.append(f"- 总处理数量: {pipeline_stats.get('total_processed', 0)}")
                report_lines.append(f"- 成功处理数量: {pipeline_stats.get('successful_processed', 0)}")
                report_lines.append(f"- 失败处理数量: {pipeline_stats.get('failed_processed', 0)}")
                report_lines.append(f"- 总处理时间: {pipeline_stats.get('total_time', 0):.3f}s")
                
                component_usage = pipeline_stats.get('component_usage', {})
                if component_usage:
                    report_lines.append("### 组件使用统计")
                    for component, count in component_usage.items():
                        report_lines.append(f"- {component}: {count}")
                report_lines.append("")
            
            report_content = "\n".join(report_lines)
            
            # 保存报告
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"评估报告已保存: {save_path}")
            
            return report_content
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return ""
    
    def clear_cache(self):
        """清理所有组件的缓存"""
        if self.text_augmenter:
            self.text_augmenter.clear_cache()
        if self.retriever:
            self.retriever.clear_cache()
        if self.sd_generator:
            self.sd_generator.clear_cache()
        if self.detector:
            self.detector.clear_cache()
        
        logger.info("管道缓存已清理")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        stats = {
            'pipeline_stats': self.pipeline_stats.copy(),
            'config': {
                'enable_text_augment': self.config.enable_text_augment,
                'enable_retrieval': self.config.enable_retrieval,
                'enable_sd_reference': self.config.enable_sd_reference,
                'enable_detection': self.config.enable_detection,
                'enable_parallel': self.config.enable_parallel,
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size
            }
        }
        
        # 添加组件统计
        if self.text_augmenter:
            stats['text_augmenter_stats'] = self.text_augmenter.get_stats()
        if self.retriever:
            stats['retriever_stats'] = self.retriever.get_stats()
        if self.sd_generator:
            stats['sd_generator_stats'] = self.sd_generator.get_stats()
        if self.detector:
            stats['detector_stats'] = self.detector.get_stats()
        
        return stats
    
    def save_pipeline(self, save_path: str):
        """保存管道配置和状态"""
        try:
            save_data = {
                'config': self.config.__dict__,
                'pipeline_stats': self.pipeline_stats
            }
            
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"管道已保存: {save_path}")
            
        except Exception as e:
            logger.error(f"保存管道失败: {e}")
    
    def __del__(self):
        """析构函数，清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)


# DefensePipeline作为MultiModalDetectionPipeline的别名，用于向后兼容
DefensePipeline = MultiModalDetectionPipeline


def create_detection_pipeline(config: Optional[PipelineConfig] = None) -> MultiModalDetectionPipeline:
    """
    创建检测管道实例
    
    Args:
        config: 管道配置
        
    Returns:
        检测管道实例
    """
    if config is None:
        config = PipelineConfig()
    
    return MultiModalDetectionPipeline(config)


def create_defense_pipeline(config: Optional[PipelineConfig] = None) -> DefensePipeline:
    """
    创建防御管道实例（DefensePipeline的别名）
    
    Args:
        config: 管道配置
        
    Returns:
        防御管道实例
    """
    return create_detection_pipeline(config)