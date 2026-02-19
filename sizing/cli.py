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
    
    # Política de Capacidade
    capacity_margin: Optional[float]
    target_load_time: Optional[float]
    
    # SLOs de Latência LLM
    ttft: Optional[int]
    ttft_p99: Optional[int]
    tpot: Optional[float]
    
    # Outputs
    executive_report: bool
    verbose: bool
    validate_only: bool


def create_arg_parser() -> argparse.ArgumentParser:
    """Cria parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Sizing de Infraestrutura para Inferência de LLMs em GPUs NVIDIA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Seleções (obrigatórias exceto em --validate-only)
    parser.add_argument("--model", help="Nome do modelo (ex: opt-oss-120b)")
    parser.add_argument("--server", help="Nome do servidor (ex: dgx-b300)")
    parser.add_argument("--storage", help="Nome do perfil de storage (ex: profile_default)")
    
    # NFRs (obrigatórios exceto em --validate-only)
    parser.add_argument("--concurrency", type=int, help="Sessões simultâneas alvo")
    parser.add_argument("--effective-context", type=int, help="Contexto efetivo em tokens")
    
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
    
    # Política de Capacidade
    parser.add_argument("--capacity-margin", type=float,
                        help="Override da margem de capacidade de storage (0.0 a 1.0, ex: 0.30 = 30%%. Default: carregado de parameters.json)")
    parser.add_argument("--target-load-time", type=float,
                        help="Tempo alvo (segundos) para carregar modelo no restart (default: 60s, definido em parameters.json)")
    
    # SLOs de Latência LLM
    parser.add_argument(
        "--ttft", type=int, required=False,
        help="Time to First Token alvo em millisegundos (ms). Define o SLO de latência P50 até primeiro token. "
             "Exemplo: 1000 para 1s. Valores típicos: 500-2000ms. Bom: <500ms, Lento: >2000ms. "
             "Se não especificado, não valida TTFT."
    )
    parser.add_argument(
        "--ttft-p99", type=int, required=False,
        help="TTFT alvo P99 em millisegundos (ms). Default: --ttft * ttft_p99_multiplier (de parameters.json, padrão 2.0)."
    )
    parser.add_argument(
        "--tpot", type=float, required=False,
        help="Time Per Output Token (tokens/segundo) mínimo esperado. Define velocidade de streaming. "
             "Exemplo: 8.0 para 8 tokens/s. Valores típicos: 6-10 tokens/s. Bom: >10 tokens/s, Lento: <6 tokens/s. "
             "Se não especificado, não valida throughput de geração."
    )
    
    # Saídas
    parser.add_argument("--executive-report", action="store_true",
                        help="Gerar relatório executivo adicional em Markdown")
    parser.add_argument("--verbose", action="store_true",
                        help="Modo verboso")
    
    # Validação
    parser.add_argument("--validate-only", action="store_true",
                        help="Apenas validar arquivos JSON (schema e constraints) sem executar sizing")
    
    return parser


def parse_cli_args() -> CLIConfig:
    """Parse argumentos CLI e retorna configuração."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Se --validate-only, não exigir model/server/storage/concurrency/effective-context
    if args.validate_only:
        return CLIConfig(
            model_name=args.model if args.model else "dummy",
            server_name=args.server if args.server else "dummy",
            storage_name=args.storage if args.storage else "dummy",
            concurrency=args.concurrency if args.concurrency else 1,
            effective_context=args.effective_context if args.effective_context else 1,
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
            capacity_margin=args.capacity_margin,
            target_load_time=args.target_load_time,
            ttft=args.ttft,
            ttft_p99=args.ttft_p99,
            tpot=args.tpot,
            executive_report=args.executive_report,
            verbose=args.verbose,
            validate_only=True
        )
    
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
        capacity_margin=args.capacity_margin,
        target_load_time=args.target_load_time,
        ttft=args.ttft,
        ttft_p99=args.ttft_p99,
        tpot=args.tpot,
        executive_report=args.executive_report,
        verbose=args.verbose,
        validate_only=args.validate_only
    )
