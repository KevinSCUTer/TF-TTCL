"""
Experiment Output Directory Structure:
    exp_res/{data_type}_{open/close}_{yyyymmdd}_{hhmmss}/
        - experiment.log
        - output.jsonl
        - rules.json
        - summary.json
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import statistics
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed, wait as futures_wait

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.utils.llm_client import LLMClient
from source.utils.embedding_client import EmbeddingClient
from source.utils.evaluator import PPLEvaluator
from source.context.prompt_loader import PromptLoader, PromptTemplates
from source.actors import (
    ActorFactory,
    Teacher,
    TeachingAssistant,
    Student,
    GenerationResult,
)
from source.context.static_rule_injector import StaticRuleInjector, StaticRuleConfig
from source.context.context_manager import ContextManager
from source.rag.rag_kernel import (
    RAGKernel as StandardRAGKernel,
    create_rag_kernel as create_standard_rag_kernel,
)
from source.rag.random_kernel import RAGKernel as RandomRAGKernel
from source.compare.selection_kernel import SelectionKernel
from source.summary.summary_module import SummaryModule
from source.utils.batch_adapt import BatchAdapter, BatchSummaryResult
from source.confuse.rule_injector import RuleInjector, InjectedRule
from source.confuse.rule_tracker import RuleTracker, RuleStatus


# 实验结果基础路径
EXP_RES_BASE_PATH = Path(__file__).parent / "exp_res"


def get_utc8_timestamp() -> Tuple[str, str]:
    """Get date and time strings for UTC+8 timezone"""
    utc8 = timezone(timedelta(hours=8))
    now = datetime.now(utc8)
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    return date_str, time_str


def create_experiment_dir(data_type: str, mode: str) -> Path:
    """
    Create experiment directory

    Format: {data_type}_{open/close}_{yyyymmdd}_{hhmmss}

    Args:
        data_type: Data type (e.g., gsm8k, wealth)
        mode: Running mode (open/close)

    Returns:
        Path to the experiment directory
    """
    date_str, time_str = get_utc8_timestamp()
    dir_name = f"{data_type}_{mode}_{date_str}_{time_str}"

    exp_dir = EXP_RES_BASE_PATH / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def extract_data_type(data_path: str) -> str:
    """Extract data type from data path"""
    filename = Path(data_path).stem
    # Extract main part (remove numerical suffix)
    parts = filename.split("_")
    if parts:
        return parts[0]
    return filename


class ExperimentRunner:
    """
    Experiment Runner
    Main controller for the workflow, connects modules, and records logs.
    """

    def __init__(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
        api_base_url: str = "http://localhost:9000/v1",
        api_key: Optional[str] = None,
        model_name: str = "qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 512,
        max_context_tokens: int = 8192,
        limit: Optional[int] = None,
        verbose: bool = True,
        mode: str = "close",
        domain: str = "math",
        embedding_api_url: str = "http://localhost:10000/v1",
        embedding_api_key: Optional[str] = None,
        embedding_model_name: str = "Qwen3-Embedding-0.6B",
        use_rag: bool = True,
        max_positive_rules: int = 10,
        max_negative_rules: int = 10,
        save_variants: bool = False,
        batch_size: int = 50,
        no_batch: bool = False,
        log_prompts: bool = False,
        prompt_paths: Optional[Dict[str, str]] = None,
        student_count: int = 4,
        ablation_config: Optional[Dict[str, bool]] = None,
        pruning_config: Optional[Dict[str, object]] = None,
        static_rules_config: Optional[Dict[str, object]] = None,
        confuse_mode_config: Optional[Dict[str, object]] = None,
        student_timeout: Optional[float] = None,
        prompt_base_dir: Optional[str] = None,
        log_expected_output: bool = True,
        write_labels_to_result: bool = True,
        write_private_labels_file: bool = False,
    ):
        """
        初始化实验运行器

        Args:
            data_path: 数据文件路径
            output_dir: 输出目录 (如果为 None，自动创建)
            api_base_url: API 服务地址
            api_key: API 密钥
            model_name: 模型名称
            max_tokens: 最大生成 token 数
            max_context_tokens: 最大上下文 token 数
            limit: 限制处理的问题数量
            verbose: 是否显示详细输出
            mode: 运行模式 ("close" or "open")
            embedding_api_url: Embedding API 服务地址
            embedding_model_name: Embedding 模型名称
            use_rag: 是否使用 RAG 模式
            max_positive_rules: RAG 最大正向规则数量
            max_negative_rules: RAG 最大负向规则数量
            batch_size: 批次大小 (控制多少个问题后触发规则总结)
            no_batch: 是否禁用批次模式
            log_prompts: 是否在日志中打印完整的提示词
            prompt_paths: 提示词文件路径配置
            student_count: Student 数量
            ablation_config: 消融实验配置
        """
        self.data_path = data_path
        self.limit = limit
        self.verbose = verbose
        self.mode = mode
        self.domain = domain
        self.use_rag = use_rag
        self.save_variants = save_variants
        self.batch_size = batch_size
        self.no_batch = no_batch
        self.log_prompts = log_prompts
        self.prompt_paths = prompt_paths
        self.student_count = student_count
        self.ablation_config = ablation_config or {}
        self.pruning_config = pruning_config or {}
        self.static_rules_config = static_rules_config or {}
        self.confuse_mode_config = confuse_mode_config or {}
        self.student_timeout = student_timeout
        self.prompt_base_dir = prompt_base_dir
        self.log_expected_output = log_expected_output
        self.write_labels_to_result = write_labels_to_result
        self.write_private_labels_file = write_private_labels_file
        
        # Process ablation config: remove_crr (disable RAG retrieval, use LIFO)
        if self.ablation_config.get("remove_crr", False):
            self.use_rag = False
            # Note: We disable use_rag here, so ContextManager automatically switches to LIFO mode.
            # remove_crr explicitly states "no cos calculation", so disabling RAG matches expectations.

        # Extract data type
        self.data_type = extract_data_type(data_path)

        # Create experiment directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = create_experiment_dir(self.data_type, mode)

        # Setup logging (before initializing other components)
        self._setup_logging()

        self.logger.info("Initializing components...")
        self.logger.info(f"Data type: {self.data_type}")
        self.logger.info(f"Running mode: {mode}")
        self.logger.info(f"RAG mode: {'Enabled' if use_rag else 'Disabled'}")
        self.logger.info(f"Save question variants: {'Enabled' if save_variants else 'Disabled'}")
        self.logger.info(f"Print full prompts: {'Enabled' if log_prompts else 'Disabled'}")

        # LLM 客户端
        self.llm_client = LLMClient(
            api_key=api_key,
            base_url=api_base_url,
            model_name=model_name,
            default_max_tokens=max_tokens,
        )

        # PPL 评估器
        self.evaluator = PPLEvaluator()

        # Prompt Loader
        self.prompt_loader = PromptLoader(
            prompt_paths=self.prompt_paths,
            base_dir=self.prompt_base_dir,
        )

        # Embedding client and RAG Kernel
        self.embedding_client = None
        self.rag_kernel = None

        if use_rag or mode == "open":
            self.logger.info("Initializing Embedding client...")
            try:
                self.embedding_client = EmbeddingClient(
                    base_url=embedding_api_url,
                    api_key=embedding_api_key,
                    model_name=embedding_model_name,
                )
                self.logger.info(f"  Embedding API: {embedding_api_url}")
                self.logger.info(f"  Embedding Model: {embedding_model_name}")

                # Parse pruning parameters
                pruning_enabled = self.pruning_config.get("enabled", False)
                pruning_max_repo = (
                    self.pruning_config.get("max_repo_size", 1000)
                    if pruning_enabled
                    else None
                )
                pruning_strategy = self.pruning_config.get("strategy", "fifo")
                pruning_sim_threshold = self.pruning_config.get(
                    "similarity_threshold", 0.95
                )

                # Create RAG Kernel
                if use_rag:
                    if self.ablation_config.get("random_rules", False):
                        self.logger.info(
                            "  [Ablation] random_rules: Using Random Kernel (Random sampling)"
                        )
                        self.rag_kernel = RandomRAGKernel(
                            embedding_client=self.embedding_client,
                            cache_dir=str(self.output_dir / "cache"),
                            max_positive=max_positive_rules,
                            max_negative=max_negative_rules,
                        )
                    else:
                        self.rag_kernel = StandardRAGKernel(
                            embedding_client=self.embedding_client,
                            cache_dir=str(self.output_dir / "cache"),
                            max_positive=max_positive_rules,
                            max_negative=max_negative_rules,
                            max_repo_size=pruning_max_repo,
                            pruning_strategy=pruning_strategy
                            if pruning_enabled
                            else "fifo",
                            similarity_threshold=pruning_sim_threshold,
                        )
                    self.logger.info(
                        f"  RAG Kernel: max_pos={max_positive_rules}, max_neg={max_negative_rules}"
                    )
                    if pruning_enabled:
                        self.logger.info(
                            f"  [Pruning] Pruning enabled: strategy={pruning_strategy}, max_repo_size={pruning_max_repo}"
                        )
                        if pruning_strategy == "similarity":
                            self.logger.info(
                                f"  [Pruning] Similarity merge threshold: {pruning_sim_threshold}"
                            )
            except Exception as e:
                self.logger.warning(
                    f"  Embedding client initialization failed: {e}, will use LIFO mode"
                )
                self.use_rag = False

        # Context manager
        student_prompt = self.prompt_loader.get_student_prompt()

        # [Ablation] remove_crr: Increase max_rules to ensure context can be filled (LIFO)
        # Default 100 rules might not be enough to fill 8k context
        context_max_rules = 100
        if self.ablation_config.get("remove_crr", False):
            context_max_rules = 1000
            self.logger.info(
                f"  [Ablation] remove_crr: max_rules set to {context_max_rules}"
            )

        self.context_manager = ContextManager(
            max_tokens=max_context_tokens,
            system_prompt_tokens=500,
            question_buffer_tokens=300,
            max_rules=context_max_rules,
            rag_kernel=self.rag_kernel,
            use_rag=self.use_rag,
            base_student_prompt=student_prompt,
        )

        # Static rule injection (Optional, experiment specific)
        self._inject_static_rules_if_needed()
        
        # Selection kernel and summary module
        use_max_ppl = self.ablation_config.get(
            "use_max_ppl_for_negative_selection", False
        )
        self.logger.info(
            f"  Ablation Config - use_max_ppl_for_negative_selection: {use_max_ppl}"
        )
        self.selection_kernel = SelectionKernel(
            embedding_client=self.embedding_client,
            use_max_ppl_for_negative_selection=use_max_ppl,
        )
        self.summary_module = SummaryModule(
            self.llm_client, prompt_loader=self.prompt_loader, mode=self.mode
        )

        # Batch Adapter
        self.batch_adapter = None
        if not self.no_batch:
            self.batch_adapter = BatchAdapter(
                batch_size=self.batch_size,
                llm_client=self.llm_client,
                prompt_loader=self.prompt_loader,
                temperature=0.7,
                max_tokens=512,
                mode=self.mode,
            )
            self.logger.info(f"Batch Mode: Enabled (batch_size={self.batch_size})")
        else:
            self.logger.info("Batch Mode: Disabled (Generate exp per question)")

        # ========== Confuse Mode Initialization ==========
        self.rule_injector = None
        self.rule_tracker = None

        if self.confuse_mode_config.get("enabled", False):
            self._init_confuse_mode()

        # Create Teacher, TA, and Students
        teacher_prompt = self.prompt_loader.get_teacher_prompt(
            mode=mode, domain=self.domain
        )
        ta_prompt = self.prompt_loader.get_ta_prompt(mode=mode, domain=self.domain)

        self.teacher, self.ta, self.students = ActorFactory.create_all(
            llm_client=self.llm_client,
            evaluator=self.evaluator,
            teacher_prompt=teacher_prompt,
            ta_prompt=ta_prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            student_count=self.student_count
        )

        # Handle Ablation: remove_qsa -> Teacher config aligns with Students (Temp, Top_p, Top_k)
        if self.ablation_config.get("remove_qsa", False):
            self.logger.info(
                "  [Ablation] remove_qsa: Teacher config will be overridden by Student config"
            )
            if self.students:
                ref_student = self.students[0]
                self.teacher.config.temperature = ref_student.config.temperature
                self.teacher.config.top_p = ref_student.config.top_p
                self.teacher.config.top_k = ref_student.config.top_k

        # Result tracking
        self.results: List[Dict] = []

        # Set output file paths
        self.output_jsonl_path = self.output_dir / "output.jsonl"
        self.result_jsonl_path = self.output_dir / "result.jsonl"  # Standard format output
        self.rules_json_path = self.output_dir / "rules.json"
        self.summary_json_path = self.output_dir / "summary.json"
        self.private_labels_jsonl_path = self.output_dir / "private_labels.jsonl"
        self.variants_jsonl_path = (
            self.output_dir / "question_variants.jsonl"
        )  # Question variants

        # Initialize thread pool
        self.max_workers = max(64, os.cpu_count() * 4 if os.cpu_count() else 64)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Thread-safe lock
        import threading

        self.lock = threading.Lock()

        self.logger.info(f"Initialization complete: Teacher + TA + {len(self.students)} Students")
        self.logger.info(f"Thread Pool: max_workers={self.max_workers}")
        self.logger.info(f"Experiment Directory: {self.output_dir}")
        self.logger.info(f"Write labels to result.jsonl: {self.write_labels_to_result}")
        self.logger.info(f"Log expected_output: {self.log_expected_output}")
        self.logger.info(f"Write private labels file: {self.write_private_labels_file}")
        self._log_actor_configs()

    def _inject_static_rules_if_needed(self):
        """Inject static rules based on configuration"""
        if not self.static_rules_config.get("enabled", False):
            return

        config = StaticRuleConfig(
            enabled=bool(self.static_rules_config.get("enabled", False)),
            path=str(self.static_rules_config.get("path", "")),
            group=str(self.static_rules_config.get("group", "all")),
            max_positive=self.static_rules_config.get("max_positive"),
            max_negative=self.static_rules_config.get("max_negative"),
            strict=bool(self.static_rules_config.get("strict", True)),
        )

        self.logger.info(
            f"Preparing to inject static rules: path={config.path}, group={config.group}, "
            f"max_pos={config.max_positive}, max_neg={config.max_negative}"
        )

        injector = StaticRuleInjector(self.context_manager, log=self.logger)
        stats = injector.inject_from_file(config)
        self.logger.info(
            f"Static rules loaded: total={stats['loaded_total']}, "
            f"pos={stats['loaded_positive']}, neg={stats['loaded_negative']}, skipped={stats['skipped']}"
        )
    
    def _setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / "experiment.log"

        # Configure log format
        log_format = "%(asctime)s | %(levelname)-8s | %(message)s"

        # Create logger
        self.logger = logging.getLogger("ExperimentRunner")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        console_handler.setFormatter(logging.Formatter(log_format))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Let RAG kernel logs write to the same experiment.log (observing similarity_fifo trajectories)
        rag_logger = logging.getLogger("core.rag.rag_kernel")
        rag_logger.setLevel(logging.DEBUG)
        rag_logger.handlers.clear()
        rag_logger.addHandler(file_handler)
        rag_logger.addHandler(console_handler)
        rag_logger.propagate = False

        self.log_file = log_file
        self.logger.info(f"Log file: {log_file}")

    def _log_actor_configs(self):
        """Log Actor configurations"""
        self.logger.info("=" * 60)
        self.logger.info("Actor Configuration:")
        self.logger.info(f"  Teacher: temp={self.teacher.config.temperature}")

        ta_config = self.ta.config
        ta_params = f"temp={ta_config.temperature}"
        if ta_config.top_p:
            ta_params += f", top_p={ta_config.top_p}"
        self.logger.info(f"  TA: {ta_params} (Verbalized Sampling)")

        for student in self.students:
            config = student.config
            params = f"temp={config.temperature}"
            if config.top_p:
                params += f", top_p={config.top_p}"
            if config.top_k:
                params += f", top_k={config.top_k}"
            self.logger.info(f"  {config.role_id}: {params}")
        self.logger.info("=" * 60)

    def _init_confuse_mode(self):
        """Initialize confuse mode"""
        if not self.rag_kernel:
            self.logger.warning("  [Confuse Mode] RAG Kernel not initialized, cannot inject rules")
            return

        self.logger.info("=" * 60)
        self.logger.info("[Confuse Mode] Initializing confuse mode...")

        # Create rule injector
        self.rule_injector = RuleInjector(
            domain=self.domain,
            num_wrong_negative=self.confuse_mode_config.get("num_wrong_negative", 15),
            num_wrong_positive=self.confuse_mode_config.get("num_wrong_positive", 15),
            num_correct_negative=self.confuse_mode_config.get(
                "num_correct_negative", 0
            ),
            num_correct_positive=self.confuse_mode_config.get(
                "num_correct_positive", 0
            ),
            seed=self.confuse_mode_config.get("seed", 42),
        )

        # Generate rules
        injected_rules = self.rule_injector.generate_rules()
        self.logger.info(f"  Generated {len(injected_rules)} injection rules")

        # Inject into RAG Kernel
        inject_stats = self.rule_injector.inject_to_rag_kernel(self.rag_kernel)
        self.logger.info(f"  Injection stats: {inject_stats}")

        # Save injection report
        inject_report_path = self.output_dir / "confuse_injection_report.json"
        self.rule_injector.save_injection_report(str(inject_report_path))
        self.logger.info(f"  Injection report: {inject_report_path}")

        # Create rule tracker
        poison_ids = self.rule_injector.get_poison_rule_ids()
        self.rule_tracker = RuleTracker(
            poison_rule_ids=poison_ids,
            track_top_k=self.confuse_mode_config.get("track_top_k", 10),
        )
        self.logger.info(f"  Tracking {len(poison_ids)} poison rules")

        self.logger.info(f"[Confuse Mode] Initialization complete")
        self.logger.info("=" * 60)

    def load_data(self) -> List[Dict]:
        """Load data"""
        self.logger.info(f"Loading data: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.limit is not None and self.limit > 0:
            data = data[: self.limit]

        self.logger.info(f"Load complete: {len(data)} items")
        return data

    def process_question(self, question_idx: int, question_data: Dict) -> Dict:
        """
        Process a single question (Parallelized version)
        
        Workflow:
        Parallel Layer (~1 LLM call): Teacher generate || (TA generate variants -> Students parallel generate)
        Summary Write-back Layer (~1 LLM call): Selection + Parallel positive/negative rule generation
        """
        instruction = question_data.get("instruction", "").strip('"')
        input_text = question_data.get("input").strip('"')  # Remove surrounding double quotes
        OUTPUT_FORMAT = """
Output your final answer at the end of your response. The final answer must be wrapped in boxed{}.
   - For numerical answers: Provide only the number or fraction (e.g., boxed{42}, boxed{frac{1}{2}}). Do not include units.
   - For text, equations, or mixed answers: Provide the exact final result (e.g., boxed{No}, boxed{y=x^2}, boxed{2:15 PM}).
   - Do NOT include explanatory text like "The answer is" inside the box.
        """
        if self.mode == "close" and self.domain == "math":
            user_instruction = f"{instruction} {input_text} {OUTPUT_FORMAT}".strip()
            # user_instruction = f"{instruction} {OUTPUT_FORMAT} {input_text}".strip()
        else:
            user_instruction = f"{instruction} {input_text}".strip()

        expected_output = question_data.get("output", "")

        self.logger.info("=" * 60)
        self.logger.info(f"Question [{question_idx + 1}]")
        self.logger.debug(f"Question content: {user_instruction[:100]}...")
        if self.log_expected_output:
            self.logger.info(f"Expected output: {expected_output}")

        all_results: List[GenerationResult] = []
        question_variants: List[str] = []
        
        # ========== Stage 1: Teacher & TA Parallel Generation ==========
        self.logger.info("[Stage 1] Teacher & TA Parallel Generation...")
        
        # Build Teacher's enhanced system_prompt (using RAG retrieval)
        with self.lock:
            if self.ablation_config.get("remove_qsa", False):
                teacher_enhanced_prompt, rule_count, rule_tokens = (
                    self.context_manager.build_student_system_prompt(
                        current_question=user_instruction
                    )
                )
            else:
                teacher_enhanced_prompt, rule_count, rule_tokens = (
                    self.context_manager.build_system_prompt_with_rules(
                        base_system_prompt=self.teacher.config.system_prompt,
                        role="teacher",
                        current_question=user_instruction  # Used for RAG retrieval
                    )
                )
        self.logger.info(f"  Teacher Context: {rule_count} rules, ~{rule_tokens} tokens")
        
        # Log full prompt (if enabled)
        if self.log_prompts:
            self.logger.info(f"  [Full Prompt - Teacher]\n{'-' * 80}\n{teacher_enhanced_prompt}\n{'-' * 80}")
        
        # Detail RAG retrieved rules (helpful for analysis)
        if rule_count > 0 and self.use_rag:
            self._log_retrieved_rules(user_instruction)
        
        future_teacher = self.executor.submit(
            self._generate_with_enhanced_system_prompt,
            actor=self.teacher,
            question=user_instruction,
            enhanced_system_prompt=teacher_enhanced_prompt,
        )
        
        # Handle ablation: remove_ta or remove_qsa -> Skip TA
        skip_ta = self.ablation_config.get("remove_ta", False) or self.ablation_config.get("remove_qsa", False)
        
        if not skip_ta:
            future_ta = self.executor.submit(
                self.ta.generate_question_variants,
                original_question=user_instruction,
                max_variants=len(self.students)
            )
        
        # Wait for Teacher result
        teacher_result = None
        try:
            teacher_result = future_teacher.result()
            all_results.append(teacher_result)
            self.logger.info(f"  Teacher complete, PPL: {self.evaluator.format_ppl(teacher_result.ppl_result.ppl)}")
        except Exception as e:
            self.logger.error(f"  Teacher generation failed: {e}")
            if not skip_ta:
                future_ta.cancel()
            return {"error": f"Teacher generation failed: {e}", "question_idx": question_idx}
        
        # Wait for TA result
        if not skip_ta:
            try:
                question_variants = future_ta.result()
                self.logger.info(f"  TA complete, generated {len(question_variants)} question variants")
                
                # Save question variants (if enabled)
                if self.save_variants:
                    with self.lock:
                        self._save_question_variants(
                            question_idx=question_idx,
                            original_question=user_instruction,
                            variants=question_variants,
                        )
            except Exception as e:
                self.logger.error(f"  TA failed to generate question variants: {e}")
                question_variants = [user_instruction] * len(self.students)
        else:
            self.logger.info("  [Ablation] Skipping TA generation (remove_ta/remove_qsa)")
            question_variants = [user_instruction] * len(self.students)
        
        # ========== Stage 2: Students Parallel Generation ==========
        self.logger.info("[Stage 2] Students Parallel Generation...")
        
        # Build Students' enhanced system_prompt (using RAG retrieval)
        with self.lock:
            student_enhanced_prompt, _, _ = \
                self.context_manager.build_student_system_prompt(
                    current_question=user_instruction
                )
        
        # Log full prompt (if enabled)
        if self.log_prompts and student_enhanced_prompt:
            self.logger.info(f"  [Full Prompt - Students]\n{'-' * 80}\n{student_enhanced_prompt}\n{'-' * 80}")
        
        student_futures = {}
        for i, student in enumerate(self.students):
            variant_idx = i % len(question_variants)
            question_variant = question_variants[variant_idx]

            future = self.executor.submit(
                self._generate_with_enhanced_system_prompt,
                actor=student,
                question=question_variant,
                enhanced_system_prompt=student_enhanced_prompt,
            )
            student_futures[future] = (student, variant_idx)
        
        student_results = []
        all_student_futures = list(student_futures.keys())
        
        if self.student_timeout is not None:
            # Use timeout, cancel slow students and proceed with results so far
            done_futures, timed_out_futures = futures_wait(
                all_student_futures, timeout=self.student_timeout
            )
            if timed_out_futures:
                for f in timed_out_futures:
                    f.cancel()
                self.logger.warning(
                    f"  [Straggler] {len(timed_out_futures)}/{len(all_student_futures)} Students "
                    f"timed out (>{self.student_timeout}s), skipped"
                )
        else:
            done_futures = all_student_futures

        for future in done_futures:
            student, variant_idx = student_futures[future]
            try:
                result = future.result()
                student_results.append(result)
                all_results.append(result)
                self.logger.info(f"  {result.role_id} complete, PPL: {self.evaluator.format_ppl(result.ppl_result.ppl)}")
            except Exception as e:
                self.logger.error(f"  {student.config.role_id} generation failed: {repr(e)}")
        
        # ========== Stage 3: Selection & Summary ==========
        self.logger.info("[Stage 3] Selection & Summary...")
        
        if not student_results:
            self.logger.error("All Student generations failed!")
            selection_result = None
        else:
            selection_result = self.selection_kernel.select(
                teacher_result=teacher_result,
                student_results=student_results,
                mode=self.mode,
            )
            self.logger.info(f"  Selection result: {selection_result.description}")
            self.logger.info(
                f"  Best: {selection_result.best_candidate.role_id} (PPL: {selection_result.best_candidate.ppl:.4f})"
            )
            if selection_result.worst_candidate:
                self.logger.info(
                    f"  Worst: {selection_result.worst_candidate.role_id} (PPL: {selection_result.worst_candidate.ppl:.4f})"
                )

        # Determine the final chosen result
        final_best_result = selection_result.best_candidate.result if selection_result else teacher_result
        
        # Summary Module generates rules & updates context
        generated_rules = []

        if selection_result:
            with self.lock:
                if self.ablation_config.get("remove_ced", False):
                    self.logger.info(
                        "  [Ablation] remove_ced: Skipping summary module (no rules generated)"
                    )
                elif self.batch_adapter:
                    should_summarize = self.batch_adapter.add_candidate(
                        question=user_instruction,
                        question_idx=question_idx,
                        selection_result=selection_result,
                    )
                    
                    if should_summarize:
                        batch_result = self.batch_adapter.generate_batch_summary()
                        self._apply_batch_rules(batch_result, generated_rules)
                elif not selection_result.skip_summary:
                    self._process_summary(
                        selection_result=selection_result,
                        question=user_instruction,
                        question_idx=question_idx,
                        generated_rules=generated_rules,
                    )
        
        # Record rule updates
        if generated_rules:
            with self.lock:
                self._append_summary_log(generated_rules)
        
        # Build result
        result = {
            "question_idx": question_idx,
            "instruction": user_instruction,
            "selected_role": final_best_result.role_id,
            "selected_ppl": final_best_result.ppl_result.ppl,
            "selected_answer": final_best_result.content,
            "case_type": selection_result.case_type if selection_result else 0,
            "generated_rules": generated_rules,
            "all_ppls": {r.role_id: r.ppl_result.ppl for r in all_results},
            "all_answers": {r.role_id: r.content for r in all_results}
        }
        if self.log_expected_output:
            result["expected_output"] = expected_output

        # Build general format output
        # Use final chosen result's system_prompt and question_variant
        final_system_prompt = (
            final_best_result.system_prompt
            if final_best_result.system_prompt
            else teacher_enhanced_prompt
        )
        final_user_prompt = (
            final_best_result.question_variant
            if final_best_result.question_variant
            else user_instruction
        )

        general_output = {
            "prompt": f"system\n{final_system_prompt}\nuser\n{final_user_prompt}\nassistant\n",
            "predict": final_best_result.content,
        }
        if self.write_labels_to_result:
            general_output["label"] = expected_output

        with self.lock:
            self._append_result(result, general_output)
            if self.write_private_labels_file:
                with open(self.private_labels_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "question_idx": question_idx,
                                "label": expected_output,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        # [Confuse Mode] 记录 PPL 追踪
        if self.rule_tracker:
            with self.lock:
                self.rule_tracker.record_ppl(final_best_result.ppl_result.ppl)

        return result

    def _update_confuse_tracker(self, question: str, question_idx: int):
        """Update the confuse mode tracker"""
        if not self.rag_kernel or not self.rule_tracker:
            return

        try:
            positive_results, negative_results = self.rag_kernel.retrieve(question)
            snapshot = self.rule_tracker.update_from_retrieval(
                question_idx=question_idx,
                positive_results=positive_results,
                negative_results=negative_results,
            )

            # Periodically output tracking status
            if (question_idx + 1) % 5 == 0 or question_idx == 0:
                stats = self.rule_tracker.get_current_stats()
                self.logger.info(
                    f"  [Confuse Tracker] "
                    f"Poison in Top-K: {snapshot.poison_rules_in_topk}, "
                    f"Remaining: {stats['active_poison_rules']}/{stats['total_poison_rules']}, "
                    f"Elimination: {stats['elimination_rate'] * 100:.1f}%"
                )
        except Exception as e:
            self.logger.error(f"  [Confuse Tracker] Update failed: {e}")

    def _process_summary(
        self,
        selection_result,
        question: str,
        question_idx: int,
        generated_rules: List[Dict],
    ):
        """Handle the summary module and generate rules"""
        case_type = selection_result.case_type

        # Case 1: No consensus -> skip
        if case_type == 1:
            self.logger.info("  Case 1: No consensus, skipping summary module.")
            return
        
        # Case 2/3/4: Generate positive rule
        if selection_result.best_candidate:
            try:
                pos_rule_content = self.summary_module.generate_positive_rule(
                    question=question,
                    best_candidate=selection_result.best_candidate.result
                )
                rule = self.context_manager.add_positive_rule(
                    content=pos_rule_content,
                    question=question,
                    answer=selection_result.best_candidate.result.content,
                    source_role=selection_result.best_candidate.role_id,
                    ppl=selection_result.best_candidate.ppl,
                    question_index=question_idx,
                )
                generated_rules.append(rule.to_dict())
                self.logger.info(f"    [+] Positive rule: {pos_rule_content}")
            except Exception as e:
                self.logger.error(f"    Failed to generate positive rule: {e}")
        
        # Case 3/4: Generate negative rule (if Worst exists)
        if selection_result.worst_candidate and case_type in [3, 4]:
            try:
                neg_rule_content = self.summary_module.generate_negative_rule(
                    question=question,
                    best_candidate=selection_result.best_candidate.result,
                    worst_candidate=selection_result.worst_candidate.result
                )
                rule = self.context_manager.add_negative_rule(
                    content=neg_rule_content,
                    question=question,
                    answer=selection_result.worst_candidate.result.content,
                    source_role=selection_result.worst_candidate.role_id,
                    ppl=selection_result.worst_candidate.ppl,
                    question_index=question_idx,
                )
                generated_rules.append(rule.to_dict())
                self.logger.info(f"    [-] Negative rule: {neg_rule_content}")
            except Exception as e:
                self.logger.error(f"    Failed to generate negative rule: {e}")

    def _apply_batch_rules(
        self, batch_result: BatchSummaryResult, generated_rules: List[Dict]
    ):
        """
        Apply rules from batch summary to the context manager

        Args:
            batch_result: Batch summary result
            generated_rules: List for collecting rules
        """
        # Add positive rule
        if batch_result.positive_exp:
            rule = self.context_manager.add_positive_rule(
                content=batch_result.positive_exp,
                question=f"[Batch {batch_result.batch_id}]",
                answer="",
                source_role="batch_summary",
                ppl=0.0,
                question_index=batch_result.batch_id,
            )
            generated_rules.append(rule.to_dict())
            self.logger.info(
                f"    [+] Batch positive rule: {batch_result.positive_exp[:60]}..."
            )

        # Add negative rule
        if batch_result.negative_exp:
            rule = self.context_manager.add_negative_rule(
                content=batch_result.negative_exp,
                question=f"[Batch {batch_result.batch_id}]",
                answer="",
                source_role="batch_summary",
                ppl=0.0,
                question_index=batch_result.batch_id,
            )
            generated_rules.append(rule.to_dict())
            self.logger.info(
                f"    [-] Batch negative rule: {batch_result.negative_exp[:60]}..."
            )

    def _generate_with_enhanced_system_prompt(
        self, actor, question: str, enhanced_system_prompt: str
    ) -> GenerationResult:
        """Generate response using enhanced system_prompt"""
        messages = actor.llm_client.build_messages(
            system_prompt=enhanced_system_prompt, user_content=question
        )

        response = actor.llm_client.generate(
            messages=messages,
            temperature=actor.config.temperature,
            top_p=actor.config.top_p,
            top_k=actor.config.top_k,
            max_tokens=actor.config.max_tokens,
            logprobs=True,
        )

        ppl_result = actor.evaluator.compute_ppl(response.logprobs)

        return GenerationResult(
            role_id=actor.config.role_id,
            content=response.content,
            ppl_result=ppl_result,
            llm_response=response,
            config=actor.config,
            question_variant=question,
            system_prompt=enhanced_system_prompt,
        )

    def _append_result(self, result: Dict, general_output: Dict):
        """Append results to files in real-time"""
        # Write detailed output (output.jsonl)
        with open(self.output_jsonl_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"result": result, "output": general_output}, ensure_ascii=False
                )
                + "\n"
            )

        # Write standard format result (result.jsonl) - follows evaluation requirements
        with open(self.result_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(general_output, ensure_ascii=False) + "\n")

    def _save_question_variants(
        self, question_idx: int, original_question: str, variants: List[str]
    ):
        """
        Save question variants to question_variants.jsonl

        Format:
        {
            "question_idx": 0,
            "original": "Original Question",
            "variant_count": 4,
            "variants": ["Variant 1", "Variant 2", "Variant 3", "Variant 4"]
        }
        """
        record = {
            "question_idx": question_idx,
            "original": original_question,
            "variant_count": len(variants),
            "variants": variants,
        }

        with open(self.variants_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Display variant details in logs
        self.logger.info(f"  [TA Variants] Saved {len(variants)} variants:")
        for i, v in enumerate(variants, 1):
            preview = v[:80] + "..." if len(v) > 80 else v
            self.logger.info(f"    {i}. {preview}")

    def _append_summary_log(self, rules: List[Dict]):
        """Append rule update records"""
        with open(self.summary_json_path, "a", encoding="utf-8") as f:
            for rule in rules:
                f.write(json.dumps(rule, ensure_ascii=False) + "\n")

    def _log_retrieved_rules(self, question: str):
        """
        Log details of RAG retrieved rules

        Used to analyze RAG effectiveness, showing retrieved rule content and similarity
        """
        if not self.rag_kernel:
            return

        # Get retrieval results
        positive_results, negative_results = self.rag_kernel.retrieve(question)

        if not positive_results and not negative_results:
            self.logger.info("  [RAG] Rule base empty, no retrieval results")
            return

        self.logger.info("  [RAG] Retrieved rules:")

        # Show positive rules
        if positive_results:
            self.logger.info(f"    Positive rules ({len(positive_results)}):")
            for i, r in enumerate(positive_results[:3], 1):  # Display top 3
                content_preview = (
                    r.content[:60] + "..." if len(r.content) > 60 else r.content
                )
                self.logger.info(
                    f"      {i}. [sim={r.similarity:.3f}] {content_preview}"
                )
            if len(positive_results) > 3:
                self.logger.info(f"      ... and {len(positive_results) - 3} more")

        # Show negative rules
        if negative_results:
            self.logger.info(f"    Negative rules ({len(negative_results)}):")
            for i, r in enumerate(negative_results[:3], 1):  # Display top 3
                content_preview = (
                    r.content[:60] + "..." if len(r.content) > 60 else r.content
                )
                self.logger.info(
                    f"      {i}. [sim={r.similarity:.3f}] {content_preview}"
                )
            if len(negative_results) > 3:
                self.logger.info(f"      ... and {len(negative_results) - 3} more")

    def run(self) -> Dict:
        """
        Run experiment (batch-sequential mode)
        
        Parallel within batch (wall-clock = 1 question time), sequential between batches to wait for rule updates.
        Ensures Batch N+1 can see rules generated by Batch N.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Experiment")
        self.logger.info("=" * 60)

        data = self.load_data()
        start_time = datetime.now()
        
        # Determine parallel processing degree
        # If batch_adapter is enabled, use its batch_size as parallelism
        # Otherwise use a reasonable default
        parallel_size = self.batch_size if not self.no_batch else 1
        self.logger.info(f"Parallelism: {parallel_size}")
        
        try:
            # Use a separate thread pool for the outer loop to avoid deadlock with inner executors
            with ThreadPoolExecutor(max_workers=parallel_size) as batch_executor:
                futures = []
                for idx, question_data in enumerate(data):
                    futures.append(batch_executor.submit(self.process_question, idx, question_data))
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        with self.lock:
                            self.results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing question: {e}")
                        with self.lock:
                            self.results.append({"error": str(e)})
            
            # Flush remaining batches
            if self.batch_adapter:
                final_batch = self.batch_adapter.flush()
                if final_batch:
                    self.logger.info(f"Flushing last batch: {final_batch.batch_size} candidates")
                    generated_rules = []
                    self._apply_batch_rules(final_batch, generated_rules)
                    if generated_rules:
                        with self.lock:
                            self._append_summary_log(generated_rules)
                        self.logger.info(f"  [Batch {final_batch.batch_id + 1}] Rules updated: "
                                         f"{self.context_manager.get_stats()['total_rules']} total rules")
        finally:
            self._shutdown_executor()

        # Sort results to ensure consistency with input order
        self.results.sort(key=lambda x: x.get("question_idx", 0))

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = self._generate_summary(duration)
        self._save_results(summary)

        self.logger.info("=" * 60)
        self.logger.info("Experiment Completed!")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)

        return summary

    def _shutdown_executor(self):
        """Shutdown the thread pool"""
        if hasattr(self, "executor") and self.executor:
            self.logger.info("Shutting down thread pool...")
            self.executor.shutdown(wait=True)
            self.logger.info("Thread pool shut down")

    def _generate_summary(self, duration: float) -> Dict:
        """Generate experiment summary"""
        role_counts = self.context_manager.get_role_distribution()
        valid_results = [r for r in self.results if "error" not in r]
        ppls = [r["selected_ppl"] for r in valid_results]
        
        summary = {
            "experiment_info": {
                "data_path": self.data_path,
                "data_type": self.data_type,
                "mode": self.mode,
                "use_rag": self.use_rag,
                "total_questions": len(self.results),
                "successful_questions": len(valid_results),
                "failed_questions": len(self.results) - len(valid_results),
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                "output_dir": str(self.output_dir),
            },
            "role_distribution": role_counts,
            "ppl_stats": {
                "min": min(ppls) if ppls else None,
                "max": max(ppls) if ppls else None,
                "avg": sum(ppls) / len(ppls) if ppls else None,
            },
            "context_stats": self.context_manager.get_stats()
        }

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total questions: {summary['experiment_info']['total_questions']}")
        self.logger.info(
            f"Successfully processed: {summary['experiment_info']['successful_questions']}"
        )
        self.logger.info(f"Role distribution: {role_counts}")
        self.logger.info(
            f"Rule base: {self.context_manager.get_stats()['total_rules']} total rules"
        )
        if ppls:
            self.logger.info(f"PPL statistics: min={min(ppls):.4f}, max={max(ppls):.4f}, avg={sum(ppls)/len(ppls):.4f}")
        
        return summary

    def _save_results(self, summary: Dict):
        """Save summary results"""
        # Save rule base
        self.context_manager.save_to_file(str(self.rules_json_path))

        # Save RAG cache
        if self.rag_kernel:
            self.rag_kernel.save_cache()

        # Append final summary to summary.json
        with open(self.summary_json_path, "a", encoding="utf-8") as f:
            f.write("\n# === EXPERIMENT SUMMARY ===\n")
            f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

        self.logger.info(f"Standard result file: {self.result_jsonl_path}")
        self.logger.info(f"Detailed output file: {self.output_jsonl_path}")
        self.logger.info(f"Rule base file: {self.rules_json_path}")
        self.logger.info(f"Summary file: {self.summary_json_path}")

        # [Confuse Mode] Save tracking report
        if self.rule_tracker:
            self._save_confuse_report()

    def _save_confuse_report(self):
        """Save confuse mode tracking report"""
        if not self.rule_tracker:
            return

        report_path = self.output_dir / "confuse_tracking_report.json"
        self.rule_tracker.save_report(str(report_path))

        # Output summary
        report = self.rule_tracker.generate_report()
        summary = report.get("summary", {})

        self.logger.info("=" * 60)
        self.logger.info("[Confuse Mode] Tracking Report Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"  Poison rules injected: {summary.get('total_poison_injected', 0)}")
        self.logger.info(f"  Poison rules eliminated: {summary.get('poison_eliminated', 0)}")
        self.logger.info(f"  Poison rules remaining: {summary.get('poison_remaining', 0)}")
        self.logger.info(f"  Elimination rate: {summary.get('elimination_rate', 0) * 100:.1f}%")
        self.logger.info(f"  Recovery score: {summary.get('recovery_score', 0):.1f}/100")
        self.logger.info(f"  Tracking report: {report_path}")
        self.logger.info("=" * 60)


import sys
from pathlib import Path
import yaml
import argparse


def apply_overrides(config, overrides):
    """Apply dot-notation overrides to config dict (e.g. 'rules.max_pos_rules=30')."""
    for override in overrides:
        if "=" not in override:
            continue
        key, value_str = override.split("=", 1)
        keys = key.split(".")

        # Navigate to parent of target key
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        # Auto-convert value to int/float/bool/string
        try:
            if value_str.lower() == "true":
                d[keys[-1]] = True
            elif value_str.lower() == "false":
                d[keys[-1]] = False
            elif "." not in value_str and value_str.isdigit():
                d[keys[-1]] = int(value_str)
            elif "." in value_str:
                try:
                    d[keys[-1]] = float(value_str)
                except ValueError:
                    d[keys[-1]] = value_str
            else:
                d[keys[-1]] = value_str
        except ValueError:
            d[keys[-1]] = value_str  # fallback to string


def resolve_path(path_str: Optional[str], base_dir: Path) -> Optional[str]:
    """Resolve a path string against PROJECT_ROOT and config directory."""
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # Prefer PROJECT_ROOT-relative paths first, then config-file-relative fallback.
    candidate_project = (PROJECT_ROOT / p).resolve()
    if candidate_project.exists():
        return str(candidate_project)

    candidate_base = (base_dir / p).resolve()
    return str(candidate_base)


def resolve_config_paths(config: Dict, config_path: Path) -> Dict:
    """Resolve data and prompt paths in-place using PROJECT_ROOT-aware strategy."""
    base_dir = config_path.parent

    experiment = config.get("experiment", {})
    data_path = experiment.get("data_path")
    if data_path:
        experiment["data_path"] = resolve_path(data_path, base_dir)

    prompt_paths = config.get("prompt_file_path", {})
    for key, value in prompt_paths.items():
        prompt_paths[key] = resolve_path(value, base_dir)

    static_rules = config.get("static_rules", {})
    static_rule_path = static_rules.get("path")
    if static_rule_path:
        static_rules["path"] = resolve_path(static_rule_path, base_dir)

    return config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Context Demo - RAG Contextual Contrastive Learning Experiment"
    )
    parser.add_argument(
        "--config", type=str, default="application.yaml", help="Path to config file"
    )

    # Allow command-line overrides for key parameters
    parser.add_argument("--data", type=str, help="Path to data file (overrides config)")
    parser.add_argument(
        "--mode", type=str, choices=["close", "open"], help="Running mode (overrides config)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of questions (overrides config)")

    # General override support
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override any config field using dot notation (e.g. rules.max_pos_rules=30 batch.size=8)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        cwd_candidate = Path.cwd() / config_path
        script_candidate = Path(__file__).parent / config_path
        config_path = cwd_candidate if cwd_candidate.exists() else script_candidate

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = resolve_config_paths(config, config_path)

    # Apply general overrides
    apply_overrides(config, args.override)

    # Print ablation config after overrides (for debugging)
    if args.override:
        print(f"[DEBUG] Applied overrides: {args.override}")
        print(f"[DEBUG] Ablation config after override: {config.get('ablation', {})}")

    # Backward compatibility: legacy command-line overrides
    if args.data:
        config["experiment"]["data_path"] = args.data
    if args.mode:
        config["experiment"]["mode"] = args.mode
    if args.limit:
        config["experiment"]["limit"] = args.limit

    # Data path already normalized by resolve_config_paths
    data_path = Path(config["experiment"]["data_path"])

    # Extract configs
    exp_config = config.get('experiment', {})
    llm_config = config.get('llm', {})
    rag_config = config.get('rag', {})
    rules_config = config.get('rules', {})
    batch_config = config.get('batch', {})
    debug_config = config.get('debug', {})
    prompt_config = config.get('prompt_file_path', {})
    actors_config = config.get('actors', {})
    student_config = config.get('student', {})
    ablation_config = config.get('ablation', {})
    pruning_config = config.get('pruning', {})
    confuse_mode_config = config.get('confuse_mode', {})
    security_config = config.get('security', {})
    
    # Initialize and run experiment
    runner = ExperimentRunner(
        data_path=str(data_path),
        output_dir=exp_config.get("output_dir"),
        api_base_url=llm_config.get("api_url"),
        api_key=llm_config.get("api_key"),
        model_name=llm_config.get("model_name"),
        max_tokens=llm_config.get("max_tokens"),
        max_context_tokens=llm_config.get("max_context_tokens"),
        limit=exp_config.get("limit"),
        verbose=not debug_config.get("quiet", False),
        mode=exp_config.get("mode", "close"),
        domain=exp_config.get("domain", "math"),
        embedding_api_url=rag_config.get("embedding_api_url"),
        embedding_api_key=rag_config.get("embedding_api_key"),
        embedding_model_name=rag_config.get("embedding_model"),
        use_rag=rag_config.get("enabled", True),
        max_positive_rules=rules_config.get("max_pos_rules", 10),
        max_negative_rules=rules_config.get("max_neg_rules", 10),
        save_variants=debug_config.get("save_variants", False),
        batch_size=batch_config.get("size", 50),
        no_batch=not batch_config.get("enabled", True),
        log_prompts=debug_config.get("log_prompts", False),
        prompt_paths=prompt_config,
        student_count=actors_config.get("student_count", student_config.get("number", 4)),
        ablation_config=ablation_config,
        pruning_config=pruning_config,
        confuse_mode_config=confuse_mode_config,
        prompt_base_dir=str(config_path.parent),
        log_expected_output=security_config.get("log_expected_output", True),
        write_labels_to_result=security_config.get("write_labels_to_result", True),
        write_private_labels_file=security_config.get("write_private_labels_file", False),
    )

    summary = runner.run()

    # Print final role distribution
    print("\n" + "=" * 40)
    print("Role Selection Distribution:")
    for role, count in summary.get("role_distribution", {}).items():
        total = summary["experiment_info"]["successful_questions"]
        if total > 0:
            percentage = count / total * 100
            print(f"  {role}: {count} ({percentage:.1f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()
