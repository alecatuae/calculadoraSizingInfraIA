#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sizing.py - Dimensionamento de Inferência de LLMs em GPU NVIDIA (DGX-class)
Autor: Sistema de Sizing de Infraestrutura IA
Data: 2026-02-07

Calcula sizing baseado em memória (KV cache) para inferência de LLMs.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONSTANTES
# ============================================================================
GB_TO_GIB = 1e9 / (1024**3)  # Conversão de GB decimal para GiB

KV_PRECISION_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
}


# ============================================================================
# DATACLASSES
# ============================================================================
@dataclass
class Model:
    """Representa um modelo LLM com seus parâmetros de arquitetura."""
    name: str
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    attention_pattern: str  # "full", "sliding", "hybrid"
    hybrid_full_layers: Optional[int]
    hybrid_sliding_layers: Optional[int]
    sliding_window: Optional[int]
    default_kv_precision: str
    notes: str


@dataclass
class Server:
    """Representa um servidor GPU (nó DGX-class)."""
    name: str
    gpus: int
    hbm_per_gpu_gb: float
    total_hbm_gb: float
    nvlink_bandwidth_tbps: Optional[float]
    system_memory_tb: Optional[float]
    notes: str


@dataclass
class StorageProfile:
    """Representa um perfil de storage com métricas de I/O."""
    name: str
    type: str
    iops_read: int
    iops_write: int
    throughput_read_gbps: float
    throughput_write_gbps: float
    latency_read_ms_p50: float
    latency_read_ms_p99: float
    latency_write_ms_p50: float
    latency_write_ms_p99: float
    notes: str


@dataclass
class SizingResult:
    """Resultado do dimensionamento."""
    model: Model
    server: Server
    storage: StorageProfile
    concurrency: int
    effective_context: int
    kv_precision: str
    kv_budget_ratio: float
    runtime_overhead_gib: float
    peak_headroom_ratio: float
    ha_mode: str
    
    kv_per_session_gib: float
    kv_total_tib: float
    sessions_per_node: int
    nodes_minimum: int
    nodes_with_headroom: int
    nodes_final: int
    
    warnings: List[str]


# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================================================
def load_models(filepath: str = "models.json") -> Dict[str, Model]:
    """Carrega modelos do arquivo JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    models = {}
    for m in data["models"]:
        model = Model(
            name=m["name"],
            num_layers=m["num_layers"],
            num_key_value_heads=m["num_key_value_heads"],
            head_dim=m["head_dim"],
            max_position_embeddings=m["max_position_embeddings"],
            attention_pattern=m["attention_pattern"],
            hybrid_full_layers=m.get("hybrid_full_layers"),
            hybrid_sliding_layers=m.get("hybrid_sliding_layers"),
            sliding_window=m.get("sliding_window"),
            default_kv_precision=m["default_kv_precision"],
            notes=m["notes"]
        )
        models[model.name] = model
    
    return models


def load_servers(filepath: str = "servers.json") -> Dict[str, Server]:
    """Carrega servidores do arquivo JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    servers = {}
    for s in data["servers"]:
        # Calcular total_hbm_gb se não fornecido
        total_hbm = s.get("total_hbm_gb")
        if total_hbm is None:
            total_hbm = s["gpus"] * s["hbm_per_gpu_gb"]
        
        server = Server(
            name=s["name"],
            gpus=s["gpus"],
            hbm_per_gpu_gb=s["hbm_per_gpu_gb"],
            total_hbm_gb=total_hbm,
            nvlink_bandwidth_tbps=s.get("nvlink_bandwidth_tbps"),
            system_memory_tb=s.get("system_memory_tb"),
            notes=s["notes"]
        )
        servers[server.name] = server
    
    return servers


def load_storage_profiles(filepath: str = "storage.json") -> Dict[str, StorageProfile]:
    """Carrega perfis de storage do arquivo JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    profiles = {}
    for p in data["profiles"]:
        profile = StorageProfile(
            name=p["name"],
            type=p["type"],
            iops_read=p["iops_read"],
            iops_write=p["iops_write"],
            throughput_read_gbps=p["throughput_read_gbps"],
            throughput_write_gbps=p["throughput_write_gbps"],
            latency_read_ms_p50=p["latency_read_ms_p50"],
            latency_read_ms_p99=p["latency_read_ms_p99"],
            latency_write_ms_p50=p["latency_write_ms_p50"],
            latency_write_ms_p99=p["latency_write_ms_p99"],
            notes=p["notes"]
        )
        profiles[profile.name] = profile
    
    return profiles


# ============================================================================
# FUNÇÕES PURAS DE CÁLCULO
# ============================================================================
def calc_kv_per_session_gib(
    model: Model,
    effective_context: int,
    kv_precision: str
) -> Tuple[float, List[str]]:
    """
    Calcula o tamanho do KV cache por sessão em GiB.
    
    Fórmula base por camada:
        KV_size = 2 * seq_length * num_key_value_heads * head_dim * bytes_per_element
        (2 = key + value)
    
    Para padrões de atenção:
    - full: todas as camadas usam effective_context
    - sliding: todas as camadas usam sliding_window
    - hybrid: full_layers usam effective_context, sliding_layers usam sliding_window
    
    Returns:
        (kv_size_gib, warnings)
    """
    warnings = []
    bytes_per_elem = KV_PRECISION_BYTES[kv_precision]
    
    # Validar effective_context
    if effective_context > model.max_position_embeddings:
        warnings.append(
            f"AVISO: effective_context={effective_context} excede "
            f"max_position_embeddings={model.max_position_embeddings}. "
            f"Clamping para {model.max_position_embeddings}."
        )
        effective_context = model.max_position_embeddings
    
    total_bytes = 0
    
    if model.attention_pattern == "full":
        # Todas as camadas usam full attention
        seq_len = effective_context
        bytes_per_layer = 2 * seq_len * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_bytes = bytes_per_layer * model.num_layers
    
    elif model.attention_pattern == "sliding":
        # Todas as camadas usam sliding window
        seq_len = model.sliding_window
        bytes_per_layer = 2 * seq_len * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_bytes = bytes_per_layer * model.num_layers
    
    elif model.attention_pattern == "hybrid":
        # Camadas full + camadas sliding
        if model.hybrid_full_layers is None or model.hybrid_sliding_layers is None:
            raise ValueError(
                f"Modelo {model.name} tem attention_pattern='hybrid' mas "
                "hybrid_full_layers ou hybrid_sliding_layers não estão definidos."
            )
        
        # Full layers
        seq_full = effective_context
        bytes_full = 2 * seq_full * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_bytes += bytes_full * model.hybrid_full_layers
        
        # Sliding layers
        seq_sliding = model.sliding_window
        bytes_sliding = 2 * seq_sliding * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_bytes += bytes_sliding * model.hybrid_sliding_layers
    
    else:
        raise ValueError(f"attention_pattern desconhecido: {model.attention_pattern}")
    
    # Converter bytes para GiB
    kv_gib = total_bytes / (1024**3)
    
    # Adicionar avisos sobre precisão
    if kv_precision in ["fp16", "bf16"]:
        warnings.append(
            f"AVISO: kv_precision={kv_precision} usa 2 bytes por elemento. "
            "Considere fp8 (1 byte) para reduzir uso de memória pela metade."
        )
    
    return kv_gib, warnings


def calc_sessions_per_node(
    server: Server,
    kv_per_session_gib: float,
    kv_budget_ratio: float,
    runtime_overhead_gib: float
) -> int:
    """
    Calcula quantas sessões simultâneas cabem em um nó.
    
    Budget disponível para KV:
        total_hbm_gib * kv_budget_ratio - runtime_overhead_gib
    
    Sessões que cabem:
        floor(budget_kv / kv_per_session_gib)
    """
    # Converter total HBM de GB para GiB
    total_hbm_gib = server.total_hbm_gb * GB_TO_GIB
    
    # Budget para KV cache
    budget_kv_gib = total_hbm_gib * kv_budget_ratio - runtime_overhead_gib
    
    if budget_kv_gib <= 0:
        return 0
    
    sessions = int(budget_kv_gib / kv_per_session_gib)
    return max(0, sessions)


def calc_nodes_required(
    concurrency: int,
    sessions_per_node: int,
    peak_headroom_ratio: float,
    ha_mode: str
) -> Tuple[int, int, int]:
    """
    Calcula número de nós necessários.
    
    Returns:
        (nodes_minimum, nodes_with_headroom, nodes_final)
    
    - nodes_minimum: capacidade pura (ceil(concurrency / sessions_per_node))
    - nodes_with_headroom: inclui headroom de pico
    - nodes_final: aplica HA (N+1) se necessário
    """
    if sessions_per_node <= 0:
        # Impossível alocar
        return 0, 0, 0
    
    # Nós mínimos (capacidade pura)
    nodes_minimum = math.ceil(concurrency / sessions_per_node)
    
    # Nós com headroom de pico
    concurrency_with_headroom = concurrency * (1 + peak_headroom_ratio)
    nodes_with_headroom = math.ceil(concurrency_with_headroom / sessions_per_node)
    
    # Nós finais (aplicar HA)
    nodes_final = nodes_with_headroom
    if ha_mode == "n+1":
        nodes_final = nodes_with_headroom + 1
    
    return nodes_minimum, nodes_with_headroom, nodes_final


# ============================================================================
# VALIDAÇÕES E ALERTAS DE STORAGE
# ============================================================================
def generate_storage_warnings(
    storage: StorageProfile,
    model: Model,
    effective_context: int,
    server: Server
) -> List[str]:
    """
    Gera avisos relacionados ao storage.
    
    Embora KV cache fique em HBM, storage é usado para:
    - Cold start (carregar pesos do modelo)
    - Checkpoints
    - Swap/offload (se habilitado)
    - Prefill de contextos longos pode pressionar I/O
    """
    warnings = []
    
    # Estimativa grosseira: prefill de 128k tokens pode gerar leitura de checkpoints
    if effective_context >= 128000:
        warnings.append(
            f"ALERTA STORAGE: Prefill de {effective_context} tokens pode pressionar "
            f"I/O de leitura durante cold-start ou carregamento de modelo. "
            f"Perfil '{storage.name}' fornece {storage.throughput_read_gbps} GB/s leitura, "
            f"latência P99={storage.latency_read_ms_p99} ms."
        )
    
    # Verificar se throughput de leitura é adequado para carregamento de modelo
    # Modelos grandes (>100B parâmetros) podem ter checkpoints de centenas de GB
    if "120b" in model.name.lower() and storage.throughput_read_gbps < 10:
        warnings.append(
            f"ALERTA STORAGE: Modelo grande ({model.name}) com storage de baixo throughput "
            f"({storage.throughput_read_gbps} GB/s). Cold-start pode ser lento."
        )
    
    # Aviso sobre swap/offload
    if storage.type == "cloud_block_storage":
        warnings.append(
            f"AVISO STORAGE: Perfil '{storage.name}' é cloud storage (latências maiores). "
            f"Se usar swap/offload de KV para storage, impacto na latência será significativo "
            f"(P99 leitura: {storage.latency_read_ms_p99} ms)."
        )
    
    return warnings


# ============================================================================
# FUNÇÃO PRINCIPAL DE SIZING
# ============================================================================
def calculate_sizing(
    model: Model,
    server: Server,
    storage: StorageProfile,
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    kv_budget_ratio: float,
    runtime_overhead_gib: float,
    peak_headroom_ratio: float,
    ha_mode: str
) -> SizingResult:
    """
    Calcula o sizing completo.
    """
    warnings = []
    
    # 1) KV cache por sessão
    kv_per_session_gib, kv_warnings = calc_kv_per_session_gib(
        model, effective_context, kv_precision
    )
    warnings.extend(kv_warnings)
    
    # 2) KV total para concorrência
    kv_total_gib = kv_per_session_gib * concurrency
    kv_total_tib = kv_total_gib / 1024
    
    # 3) Sessões por nó
    sessions_per_node = calc_sessions_per_node(
        server, kv_per_session_gib, kv_budget_ratio, runtime_overhead_gib
    )
    
    if sessions_per_node == 0:
        warnings.append(
            "ERRO: Sessões por nó = 0. Budget de HBM insuficiente para uma única sessão. "
            "Ajuste kv_budget_ratio, runtime_overhead_gib, ou use servidor com mais HBM."
        )
    
    # 4) Número de nós
    nodes_minimum, nodes_with_headroom, nodes_final = calc_nodes_required(
        concurrency, sessions_per_node, peak_headroom_ratio, ha_mode
    )
    
    # 5) Avisos de storage
    storage_warnings = generate_storage_warnings(
        storage, model, effective_context, server
    )
    warnings.extend(storage_warnings)
    
    # 6) Criar resultado
    result = SizingResult(
        model=model,
        server=server,
        storage=storage,
        concurrency=concurrency,
        effective_context=effective_context,
        kv_precision=kv_precision,
        kv_budget_ratio=kv_budget_ratio,
        runtime_overhead_gib=runtime_overhead_gib,
        peak_headroom_ratio=peak_headroom_ratio,
        ha_mode=ha_mode,
        kv_per_session_gib=kv_per_session_gib,
        kv_total_tib=kv_total_tib,
        sessions_per_node=sessions_per_node,
        nodes_minimum=nodes_minimum,
        nodes_with_headroom=nodes_with_headroom,
        nodes_final=nodes_final,
        warnings=warnings
    )
    
    return result


# ============================================================================
# FORMATAÇÃO DE SAÍDA
# ============================================================================
def format_report(result: SizingResult) -> str:
    """Formata relatório em texto."""
    lines = []
    lines.append("=" * 80)
    lines.append("RELATÓRIO DE DIMENSIONAMENTO DE INFERÊNCIA LLM")
    lines.append("=" * 80)
    lines.append("")
    
    # Parâmetros do modelo
    lines.append("MODELO:")
    lines.append(f"  Nome: {result.model.name}")
    lines.append(f"  Camadas: {result.model.num_layers}")
    lines.append(f"  KV Heads: {result.model.num_key_value_heads}")
    lines.append(f"  Head Dim: {result.model.head_dim}")
    lines.append(f"  Max Position Embeddings: {result.model.max_position_embeddings:,}")
    lines.append(f"  Padrão de Atenção: {result.model.attention_pattern}")
    
    if result.model.attention_pattern == "hybrid":
        lines.append(f"  - Full Layers: {result.model.hybrid_full_layers}")
        lines.append(f"  - Sliding Layers: {result.model.hybrid_sliding_layers}")
        lines.append(f"  - Sliding Window: {result.model.sliding_window}")
    elif result.model.attention_pattern == "sliding":
        lines.append(f"  - Sliding Window: {result.model.sliding_window}")
    
    lines.append(f"  Precisão KV Padrão: {result.model.default_kv_precision}")
    lines.append(f"  Notas: {result.model.notes}")
    lines.append("")
    
    # Parâmetros do servidor
    lines.append("SERVIDOR:")
    lines.append(f"  Nome: {result.server.name}")
    lines.append(f"  GPUs: {result.server.gpus}")
    lines.append(f"  HBM por GPU: {result.server.hbm_per_gpu_gb:.1f} GB")
    lines.append(f"  HBM Total: {result.server.total_hbm_gb:.1f} GB ({result.server.total_hbm_gb * GB_TO_GIB:.1f} GiB)")
    
    if result.server.nvlink_bandwidth_tbps:
        lines.append(f"  NVLink Bandwidth: {result.server.nvlink_bandwidth_tbps} TB/s")
    if result.server.system_memory_tb:
        lines.append(f"  System Memory: {result.server.system_memory_tb} TB")
    
    lines.append(f"  Notas: {result.server.notes}")
    lines.append("")
    
    # Parâmetros de storage
    lines.append("STORAGE:")
    lines.append(f"  Perfil: {result.storage.name}")
    lines.append(f"  Tipo: {result.storage.type}")
    lines.append(f"  IOPS: {result.storage.iops_read:,} leitura / {result.storage.iops_write:,} escrita")
    lines.append(f"  Throughput: {result.storage.throughput_read_gbps} GB/s leitura / {result.storage.throughput_write_gbps} GB/s escrita")
    lines.append(f"  Latência Leitura: P50={result.storage.latency_read_ms_p50} ms, P99={result.storage.latency_read_ms_p99} ms")
    lines.append(f"  Latência Escrita: P50={result.storage.latency_write_ms_p50} ms, P99={result.storage.latency_write_ms_p99} ms")
    lines.append(f"  Notas: {result.storage.notes}")
    lines.append("")
    
    # Parâmetros de NFR/tráfego
    lines.append("PARÂMETROS NFR/TRÁFEGO:")
    lines.append(f"  Concorrência Alvo: {result.concurrency:,} sessões simultâneas")
    lines.append(f"  Contexto Efetivo: {result.effective_context:,} tokens")
    lines.append(f"  Precisão KV: {result.kv_precision}")
    lines.append(f"  Budget HBM para KV: {result.kv_budget_ratio * 100:.0f}%")
    lines.append(f"  Overhead de Runtime: {result.runtime_overhead_gib:.1f} GiB")
    lines.append(f"  Headroom de Pico: {result.peak_headroom_ratio * 100:.0f}%")
    lines.append(f"  HA: {result.ha_mode}")
    lines.append("")
    
    # Resultados do dimensionamento
    lines.append("RESULTADOS:")
    lines.append(f"  KV Cache por Sessão: {result.kv_per_session_gib:.2f} GiB")
    lines.append(f"  KV Total (concorrência={result.concurrency:,}): {result.kv_total_tib:.2f} TiB ({result.kv_per_session_gib * result.concurrency:.2f} GiB)")
    lines.append(f"  Sessões Simultâneas por Nó: {result.sessions_per_node}")
    lines.append("")
    lines.append(f"  Nós Necessários (capacidade pura): {result.nodes_minimum}")
    lines.append(f"  Nós com Headroom de Pico (+{result.peak_headroom_ratio * 100:.0f}%): {result.nodes_with_headroom}")
    lines.append(f"  Nós Finais (com HA={result.ha_mode}): {result.nodes_final}")
    lines.append("")
    
    # Avisos
    if result.warnings:
        lines.append("AVISOS E ALERTAS:")
        for i, warning in enumerate(result.warnings, 1):
            lines.append(f"  [{i}] {warning}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def result_to_dict(result: SizingResult) -> dict:
    """Converte resultado para dicionário (para JSON)."""
    return {
        "model": {
            "name": result.model.name,
            "num_layers": result.model.num_layers,
            "num_key_value_heads": result.model.num_key_value_heads,
            "head_dim": result.model.head_dim,
            "max_position_embeddings": result.model.max_position_embeddings,
            "attention_pattern": result.model.attention_pattern,
            "hybrid_full_layers": result.model.hybrid_full_layers,
            "hybrid_sliding_layers": result.model.hybrid_sliding_layers,
            "sliding_window": result.model.sliding_window,
            "default_kv_precision": result.model.default_kv_precision,
        },
        "server": {
            "name": result.server.name,
            "gpus": result.server.gpus,
            "hbm_per_gpu_gb": result.server.hbm_per_gpu_gb,
            "total_hbm_gb": result.server.total_hbm_gb,
            "total_hbm_gib": result.server.total_hbm_gb * GB_TO_GIB,
            "nvlink_bandwidth_tbps": result.server.nvlink_bandwidth_tbps,
            "system_memory_tb": result.server.system_memory_tb,
        },
        "storage": {
            "name": result.storage.name,
            "type": result.storage.type,
            "iops_read": result.storage.iops_read,
            "iops_write": result.storage.iops_write,
            "throughput_read_gbps": result.storage.throughput_read_gbps,
            "throughput_write_gbps": result.storage.throughput_write_gbps,
            "latency_read_ms_p50": result.storage.latency_read_ms_p50,
            "latency_read_ms_p99": result.storage.latency_read_ms_p99,
            "latency_write_ms_p50": result.storage.latency_write_ms_p50,
            "latency_write_ms_p99": result.storage.latency_write_ms_p99,
        },
        "parameters": {
            "concurrency": result.concurrency,
            "effective_context": result.effective_context,
            "kv_precision": result.kv_precision,
            "kv_budget_ratio": result.kv_budget_ratio,
            "runtime_overhead_gib": result.runtime_overhead_gib,
            "peak_headroom_ratio": result.peak_headroom_ratio,
            "ha_mode": result.ha_mode,
        },
        "results": {
            "kv_per_session_gib": round(result.kv_per_session_gib, 2),
            "kv_total_tib": round(result.kv_total_tib, 2),
            "sessions_per_node": result.sessions_per_node,
            "nodes_minimum": result.nodes_minimum,
            "nodes_with_headroom": result.nodes_with_headroom,
            "nodes_final": result.nodes_final,
        },
        "warnings": result.warnings,
    }


# ============================================================================
# INTERFACE CLI
# ============================================================================
def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Dimensionamento de Inferência LLM em GPU NVIDIA (DGX-class)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Exemplo 1: opt-oss-120b + dgx300 + 1k concurrency + 128k context + fp8 + N+1
  python sizing.py --model opt-oss-120b --server dgx300 --storage profile_default \\
      --concurrency 1000 --effective-context 131072 --kv-precision fp8 \\
      --kv-budget-ratio 0.70 --runtime-overhead-gib 120 \\
      --peak-headroom-ratio 0.20 --ha n+1

  # Exemplo 2: opt-oss-20b + dgx200 + 1k concurrency + 32k context + fp8 + sem HA
  python sizing.py --model opt-oss-20b --server dgx200 --storage profile_default \\
      --concurrency 1000 --effective-context 32768 --kv-precision fp8 \\
      --kv-budget-ratio 0.70 --runtime-overhead-gib 80 \\
      --peak-headroom-ratio 0.20 --ha none
        """
    )
    
    # Seleção de recursos
    parser.add_argument("--model", required=True, help="Nome do modelo (ex: opt-oss-120b)")
    parser.add_argument("--server", required=True, help="Nome do servidor (ex: dgx300)")
    parser.add_argument("--storage", required=True, help="Perfil de storage (ex: profile_default)")
    
    # Parâmetros NFR/tráfego
    parser.add_argument("--concurrency", type=int, required=True,
                        help="Número de sessões simultâneas alvo")
    parser.add_argument("--effective-context", type=int, required=True,
                        help="Tamanho do contexto efetivo (em tokens)")
    parser.add_argument("--kv-precision", choices=["fp8", "fp16", "bf16", "int8"],
                        default="fp8", help="Precisão do KV cache (padrão: fp8)")
    parser.add_argument("--kv-budget-ratio", type=float, default=0.70,
                        help="Fração da HBM total alocada para KV cache (padrão: 0.70)")
    parser.add_argument("--runtime-overhead-gib", type=float, default=120,
                        help="Overhead de runtime em GiB (modelo, ativações, etc.) (padrão: 120)")
    parser.add_argument("--peak-headroom-ratio", type=float, default=0.20,
                        help="Headroom para picos de tráfego (padrão: 0.20 = 20%%)")
    parser.add_argument("--ha", choices=["none", "n+1"], default="none",
                        help="Modo de alta disponibilidade (padrão: none)")
    
    # Caminhos de arquivos (opcional)
    parser.add_argument("--models-file", default="models.json",
                        help="Caminho para models.json (padrão: models.json)")
    parser.add_argument("--servers-file", default="servers.json",
                        help="Caminho para servers.json (padrão: servers.json)")
    parser.add_argument("--storage-file", default="storage.json",
                        help="Caminho para storage.json (padrão: storage.json)")
    
    # Saída
    parser.add_argument("--json-only", action="store_true",
                        help="Imprimir apenas JSON (sem relatório em texto)")
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Função principal."""
    args = parse_args()
    
    # Carregar dados
    try:
        models = load_models(args.models_file)
        servers = load_servers(args.servers_file)
        storage_profiles = load_storage_profiles(args.storage_file)
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERRO: Erro ao parsear JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validar seleções
    if args.model not in models:
        print(f"ERRO: Modelo '{args.model}' não encontrado.", file=sys.stderr)
        print(f"Modelos disponíveis: {', '.join(models.keys())}", file=sys.stderr)
        sys.exit(1)
    
    if args.server not in servers:
        print(f"ERRO: Servidor '{args.server}' não encontrado.", file=sys.stderr)
        print(f"Servidores disponíveis: {', '.join(servers.keys())}", file=sys.stderr)
        sys.exit(1)
    
    if args.storage not in storage_profiles:
        print(f"ERRO: Perfil de storage '{args.storage}' não encontrado.", file=sys.stderr)
        print(f"Perfis disponíveis: {', '.join(storage_profiles.keys())}", file=sys.stderr)
        sys.exit(1)
    
    # Calcular sizing
    result = calculate_sizing(
        model=models[args.model],
        server=servers[args.server],
        storage=storage_profiles[args.storage],
        concurrency=args.concurrency,
        effective_context=args.effective_context,
        kv_precision=args.kv_precision,
        kv_budget_ratio=args.kv_budget_ratio,
        runtime_overhead_gib=args.runtime_overhead_gib,
        peak_headroom_ratio=args.peak_headroom_ratio,
        ha_mode=args.ha
    )
    
    # Imprimir relatório
    if not args.json_only:
        print(format_report(result))
        print("\nJSON OUTPUT:")
    
    # Imprimir JSON
    print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
