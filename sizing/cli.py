"""
CLI: Define argumentos de linha de comando.
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIConfig:
    """Configuração derivada dos argumentos CLI."""
    # Seleções
    model_name: str
    server_name: str
    storage_name: str
    
    # NFRs
    concurrency: int
    effective_context: int
    kv_precision: str
    kv_budget_ratio: float
    runtime_overhead_gib: float
    peak_headroom_ratio: float
    
    # Pesos e paralelismo
    weights_precision: Optional[str]
    weights_memory_gib: Optional[float]
    replicas_per_node: int
    tensor_parallel: Optional[int]
    pipeline_parallel: int
    
    # Warmup/Cold Start
    model_artifact_size_gib: Optional[float]
    warmup_concurrency: int
    warmup_read_pattern: str
    warmup_utilization_ratio: float
    
    # Outputs
    executive_report: bool
    verbose: bool


def create_arg_parser() -> argparse.ArgumentParser:
    """Cria parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Sizing de Infraestrutura para Inferência de LLMs em GPUs NVIDIA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Seleções obrigatórias
    parser.add_argument("--model", required=True, help="Nome do modelo (ex: opt-oss-120b)")
    parser.add_argument("--server", required=True, help="Nome do servidor (ex: dgx-b300)")
    parser.add_argument("--storage", required=True, help="Nome do perfil de storage (ex: profile_default)")
    
    # NFRs obrigatórios
    parser.add_argument("--concurrency", type=int, required=True, help="Sessões simultâneas alvo")
    parser.add_argument("--effective-context", type=int, required=True, help="Contexto efetivo em tokens")
    
    # NFRs opcionais
    parser.add_argument("--kv-precision", choices=["fp8", "fp16", "bf16", "int8"], default="fp8",
                        help="Precisão do KV cache (default: fp8)")
    parser.add_argument("--kv-budget-ratio", type=float, default=0.70,
                        help="Fração de HBM disponível para KV (default: 0.70)")
    parser.add_argument("--runtime-overhead-gib", type=float, default=120.0,
                        help="Overhead do runtime em GiB (default: 120)")
    parser.add_argument("--peak-headroom-ratio", type=float, default=0.20,
                        help="Headroom para picos (default: 0.20)")
    
    # Pesos e paralelismo
    parser.add_argument("--weights-precision", choices=["fp16", "bf16", "fp8", "int8", "int4"],
                        help="Precisão dos pesos do modelo (default: do modelo)")
    parser.add_argument("--weights-memory-gib", type=float,
                        help="Memória dos pesos em GiB (override manual)")
    parser.add_argument("--replicas-per-node", type=int, default=1,
                        help="Número de réplicas do modelo por nó (default: 1)")
    parser.add_argument("--tensor-parallel", type=int,
                        help="Grau de paralelismo de tensor (default: GPUs do servidor)")
    parser.add_argument("--pipeline-parallel", type=int, default=1,
                        help="Grau de paralelismo de pipeline (default: 1)")
    
    # Warmup/Cold Start
    parser.add_argument("--model-artifact-size-gib", type=float,
                        help="Tamanho do artefato do modelo em GiB para cálculo de warmup (default: weights_memory)")
    parser.add_argument("--warmup-concurrency", type=int, default=1,
                        help="Pods iniciando em paralelo durante warmup/scale-out (default: 1)")
    parser.add_argument("--warmup-read-pattern", choices=["seq", "rand"], default="seq",
                        help="Padrão de leitura durante warmup: sequencial ou random (default: seq)")
    parser.add_argument("--warmup-utilization-ratio", type=float, default=0.8,
                        help="Fração do storage max utilizável durante warmup (default: 0.8)")
    
    # Saídas
    parser.add_argument("--executive-report", action="store_true",
                        help="Gerar relatório executivo adicional em Markdown")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verboso")
    
    return parser


def parse_cli_args() -> CLIConfig:
    """Parse argumentos CLI e retorna configuração."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    return CLIConfig(
        model_name=args.model,
        server_name=args.server,
        storage_name=args.storage,
        concurrency=args.concurrency,
        effective_context=args.effective_context,
        kv_precision=args.kv_precision,
        kv_budget_ratio=args.kv_budget_ratio,
        runtime_overhead_gib=args.runtime_overhead_gib,
        peak_headroom_ratio=args.peak_headroom_ratio,
        weights_precision=args.weights_precision,
        weights_memory_gib=args.weights_memory_gib,
        replicas_per_node=args.replicas_per_node,
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel,
        model_artifact_size_gib=args.model_artifact_size_gib,
        warmup_concurrency=args.warmup_concurrency,
        warmup_read_pattern=args.warmup_read_pattern,
        warmup_utilization_ratio=args.warmup_utilization_ratio,
        executive_report=args.executive_report,
        verbose=args.verbose
    )
