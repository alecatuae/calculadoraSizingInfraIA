"""
CLI: Define argumentos de linha de comando.

Dois modos mutuamente exclusivos:
  Modo A — SIZING POR CONCORRÊNCIA: --concurrency (obrigatório)
  Modo B — SIZING POR SLO:          --ttft e --tpot (ambos obrigatórios juntos)
"""

import argparse
import json
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
    concurrency: int           # Valor efetivo para cálculos (CLI em Modo A; default em Modo B)
    concurrency_input: Optional[int]   # Valor digitado pelo usuário (None em Modo B)
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

    # SLOs de Latência (Modo B)
    ttft_input_ms: Optional[int]    # Valor digitado pelo usuário (None em Modo A)
    tpot_input_ms: Optional[float]  # Valor digitado pelo usuário (None em Modo A)
    ttft_p99: Optional[int]         # Derivado automaticamente

    # Modo de operação
    sizing_mode: str  # "concurrency_driven" | "slo_driven"

    # Outputs
    executive_report: bool
    verbose: bool
    validate_only: bool


def _load_default_concurrency_slo_mode() -> int:
    """Carrega default_concurrency_slo_mode de parameters.json com fallback."""
    try:
        with open('parameters.json', 'r', encoding='utf-8') as f:
            params = json.load(f)
            return int(params.get('default_concurrency_slo_mode', 1000))
    except Exception:
        return 1000


def create_arg_parser() -> argparse.ArgumentParser:
    """Cria parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Sizing de Infraestrutura para Inferência de LLMs\n\n"
            "MODO A — SIZING POR CONCORRÊNCIA (--concurrency):\n"
            "  Informa sessões simultâneas → calcula servidores + estima TTFT/TPOT.\n\n"
            "MODO B — SIZING POR SLO (--ttft e --tpot juntos):\n"
            "  Informa metas de latência → calcula concorrência máxima + servidores necessários.\n\n"
            "Os dois modos são MUTUAMENTE EXCLUSIVOS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Seleções (obrigatórias exceto em --validate-only)
    parser.add_argument("--model", help="Nome do modelo (ex: opt-oss-120b)")
    parser.add_argument("--server", help="Nome do servidor (ex: dgx-b300)")
    parser.add_argument("--storage", help="Nome do perfil de storage (ex: profile_default)")

    # ─── MODOS MUTUAMENTE EXCLUSIVOS ──────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--concurrency", type=int,
        help=(
            "[MODO A] Sessões simultâneas alvo. "
            "O sistema calcula servidores necessários e estima TTFT/TPOT resultantes. "
            "NÃO pode ser usado com --ttft ou --tpot."
        )
    )
    mode_group.add_argument(
        "--ttft", type=int,
        help=(
            "[MODO B] Time to First Token alvo em millisegundos. "
            "OBRIGATÓRIO junto com --tpot. "
            "O sistema calcula a concorrência máxima e servidores para atender este SLO. "
            "NÃO pode ser usado com --concurrency."
        )
    )
    # ─────────────────────────────────────────────────────────────────────────

    parser.add_argument(
        "--tpot", type=float,
        help=(
            "[MODO B] Time Per Output Token mínimo em tokens/segundo. "
            "OBRIGATÓRIO junto com --ttft. "
            "Exemplo: 8.0 para 8 tokens/s. Valores típicos: 6-10 tokens/s."
        )
    )

    parser.add_argument("--effective-context", type=int, help="Contexto efetivo em tokens")
    parser.add_argument("--ttft-p99", type=int, required=False,
                        help="TTFT alvo P99 em ms (Modo B). Default: --ttft × ttft_p99_multiplier.")

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
                        help="Tamanho do artefato do modelo em GiB para cálculo de warmup")
    parser.add_argument("--warmup-concurrency", type=int, default=1,
                        help="Pods iniciando em paralelo durante warmup/scale-out (default: 1)")
    parser.add_argument("--warmup-read-pattern", choices=["seq", "rand"], default="seq",
                        help="Padrão de leitura durante warmup: sequencial ou random (default: seq)")
    parser.add_argument("--warmup-utilization-ratio", type=float, default=0.8,
                        help="Fração do storage max utilizável durante warmup (default: 0.8)")

    # Política de Capacidade
    parser.add_argument("--capacity-margin", type=float,
                        help="Override da margem de capacidade de storage (0.0 a 1.0). Default: parameters.json")
    parser.add_argument("--target-load-time", type=float,
                        help="Tempo alvo (segundos) para carregar modelo no restart. Default: parameters.json")

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

    # ── Modo --validate-only ──────────────────────────────────────────────────
    if args.validate_only:
        return CLIConfig(
            model_name=args.model if args.model else "dummy",
            server_name=args.server if args.server else "dummy",
            storage_name=args.storage if args.storage else "dummy",
            concurrency=args.concurrency if args.concurrency else 1,
            concurrency_input=args.concurrency,
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
            ttft_input_ms=None,
            tpot_input_ms=None,
            ttft_p99=None,
            sizing_mode="concurrency_driven",
            executive_report=args.executive_report,
            verbose=args.verbose,
            validate_only=True
        )

    # ── Detectar e validar modo de operação ──────────────────────────────────

    has_concurrency = args.concurrency is not None
    has_ttft = args.ttft is not None
    has_tpot = args.tpot is not None

    # Verificar: --tpot sem --ttft (o grupo mutually exclusive já impede --tpot com --concurrency
    # pois --tpot não está no grupo, então validamos manualmente)
    if has_tpot and has_concurrency:
        parser.error(
            "ERRO: --tpot não pode ser usado com --concurrency.\n"
            "       Use --ttft e --tpot juntos para o MODO B (Sizing por SLO).\n"
            "       Use --concurrency sozinho para o MODO A (Sizing por Concorrência)."
        )

    # MODO B: --ttft e --tpot devem vir JUNTOS
    if has_ttft and not has_tpot:
        parser.error(
            "ERRO: --ttft requer --tpot. TTFT e TPOT são obrigatórios em conjunto no MODO B (Sizing por SLO).\n"
            "       Exemplo: --ttft 2000 --tpot 8.0"
        )

    if has_tpot and not has_ttft:
        parser.error(
            "ERRO: --tpot requer --ttft. TTFT e TPOT são obrigatórios em conjunto no MODO B (Sizing por SLO).\n"
            "       Exemplo: --ttft 2000 --tpot 8.0"
        )

    # Nenhum modo especificado
    if not has_concurrency and not has_ttft:
        parser.error(
            "ERRO: Nenhum modo especificado. Escolha exatamente UM dos modos:\n"
            "       MODO A — Sizing por Concorrência: --concurrency <N>\n"
            "       MODO B — Sizing por SLO:          --ttft <ms> --tpot <tok/s>"
        )

    # ── Modo A: Concorrência-Driven ──────────────────────────────────────────
    if has_concurrency:
        sizing_mode = "concurrency_driven"
        effective_concurrency = args.concurrency
        concurrency_input = args.concurrency
        ttft_input_ms = None
        tpot_input_ms = None
        ttft_p99 = None

    # ── Modo B: SLO-Driven ───────────────────────────────────────────────────
    else:
        sizing_mode = "slo_driven"
        # Usa concorrência padrão de parameters.json para cálculos de VRAM/storage
        effective_concurrency = _load_default_concurrency_slo_mode()
        concurrency_input = None
        ttft_input_ms = args.ttft
        tpot_input_ms = args.tpot
        ttft_p99 = args.ttft_p99

    return CLIConfig(
        model_name=args.model,
        server_name=args.server,
        storage_name=args.storage,
        concurrency=effective_concurrency,
        concurrency_input=concurrency_input,
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
        ttft_input_ms=ttft_input_ms,
        tpot_input_ms=tpot_input_ms,
        ttft_p99=ttft_p99,
        sizing_mode=sizing_mode,
        executive_report=args.executive_report,
        verbose=args.verbose,
        validate_only=args.validate_only if args.validate_only else False
    )
