#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sizing.py - Dimensionamento Avançado de Inferência de LLMs em GPU NVIDIA (DGX-class)
Autor: Sistema de Sizing de Infraestrutura IA
Data: 2026-02-08
Versão: 2.0 - Com Racional de Cálculo e 3 Cenários

Calcula sizing baseado em memória (KV cache) com explicações detalhadas.
"""

import argparse
import json
import math
import sys
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any


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
    rack_units_u: int
    power_kw_max: float
    heat_output_btu_hr_max: Optional[float]
    airflow_cfm: Optional[int]
    notes: str
    source: Optional[List[str]] = None


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
class Rationale:
    """Racional de cálculo para um resultado."""
    formula: str
    inputs: Dict[str, Any]
    explanation: str


@dataclass
class ScenarioResult:
    """Resultado de um cenário de dimensionamento."""
    name: str
    peak_headroom_ratio: float
    ha_mode: str
    ha_extra_nodes: int
    kv_budget_ratio: float
    
    kv_per_session_gib: float
    kv_total_gib: float
    kv_total_tib: float
    hbm_total_gib: float
    kv_budget_gib: float
    sessions_per_node: int
    nodes_capacity: int
    nodes_with_headroom: int
    nodes_final: int
    
    # Infraestrutura física
    total_power_kw: float
    total_rack_u: int
    total_heat_btu_hr: Optional[float]
    
    rationale: Dict[str, Rationale]
    warnings: List[str]


# ============================================================================
# FUNÇÕES DE CARREGAMENTO
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
            rack_units_u=s.get("rack_units_u", 10),  # Default 10U se não especificado
            power_kw_max=s.get("power_kw_max", 0.0),
            heat_output_btu_hr_max=s.get("heat_output_btu_hr_max"),
            airflow_cfm=s.get("airflow_cfm"),
            notes=s["notes"],
            source=s.get("source")
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
# DICIONÁRIO DE PARÂMETROS
# ============================================================================
def get_parameter_dictionary() -> Dict[str, Dict[str, str]]:
    """
    Retorna dicionário explicativo de todos os parâmetros usados no sizing.
    
    Para cada parâmetro, fornece:
    - description: O que é
    - source: De onde vem
    - importance: Por que é importante
    - common_errors: Erros comuns
    """
    return {
        "num_layers": {
            "description": "Número total de camadas (layers) do transformer no modelo LLM. Cada camada possui seu próprio conjunto de tensores Key e Value no KV cache.",
            "source": "Parâmetro fixo da arquitetura do modelo, definido em models.json. Não pode ser alterado em runtime.",
            "importance": "Impacta linearmente o tamanho do KV cache. Modelos com mais camadas (ex: 36 vs 24) consomem proporcionalmente mais memória GPU para armazenar o histórico de atenção.",
            "common_errors": "Erro comum: Confundir num_layers com num_hidden_layers ou contar apenas encoder/decoder. Deve ser o total de camadas que mantêm KV cache."
        },
        "num_key_value_heads": {
            "description": "Número de cabeças (heads) de atenção para Key e Value. Em GQA (Grouped Query Attention), este valor pode ser menor que o número de query heads.",
            "source": "Parâmetro fixo da arquitetura do modelo (models.json). Modelos modernos usam GQA para reduzir KV cache.",
            "importance": "Impacta diretamente o tamanho do KV cache. Menos KV heads = menos memória. GQA com 8 KV heads vs 32 representa redução de 4x na memória de KV.",
            "common_errors": "Erro comum: Usar num_attention_heads (query heads) em vez de num_key_value_heads. Em GQA esses valores são diferentes e isso causa superestimação de 4-8x na memória."
        },
        "head_dim": {
            "description": "Dimensionalidade de cada cabeça de atenção (ex: 64, 128). Tamanho do vetor de embedding por head.",
            "source": "Parâmetro fixo da arquitetura do modelo (models.json). Geralmente 64 ou 128.",
            "importance": "Multiplica linearmente o tamanho do KV cache. head_dim=128 vs 64 dobra a memória necessária por head.",
            "common_errors": "Erro comum: Confundir head_dim com hidden_size. hidden_size = num_attention_heads × head_dim. Usar hidden_size diretamente causa erro massivo."
        },
        "max_position_embeddings": {
            "description": "Comprimento máximo de contexto (em tokens) que o modelo foi treinado para suportar. Limite arquitetural do positional embedding.",
            "source": "Parâmetro fixo da arquitetura do modelo (models.json). Definido no training.",
            "importance": "Define o limite superior para effective_context. Tentar usar contextos maiores causa comportamento indefinido (extrapolação de posições).",
            "common_errors": "Erro comum: Ignorar este limite e usar effective_context > max_position_embeddings. Isso leva a resultados incorretos ou crashes em runtime."
        },
        "attention_pattern": {
            "description": "Padrão de atenção usado pelo modelo: 'full' (todas camadas atendem contexto completo), 'sliding' (janela deslizante), ou 'hybrid' (mix de full e sliding).",
            "source": "Parâmetro fixo da arquitetura do modelo (models.json). Define como o modelo processa contexto longo.",
            "importance": "Crítico para cálculo correto de KV cache. Sliding window pode reduzir KV cache drasticamente (ex: 128k context com window=128 usa 1000x menos memória que full attention).",
            "common_errors": "Erro comum: Assumir 'full' para todos os modelos. Modelos modernos usam hybrid/sliding. Usar 'full' quando modelo é 'sliding' superestima memória em ordens de magnitude."
        },
        "sliding_window": {
            "description": "Tamanho da janela de atenção deslizante (em tokens) para camadas com sliding attention. Apenas os últimos N tokens são atendidos.",
            "source": "Parâmetro fixo da arquitetura do modelo (models.json), aplicável apenas se attention_pattern='sliding' ou 'hybrid'.",
            "importance": "Controla o tamanho do KV cache para camadas sliding. Sliding window pequeno (128) vs contexto longo (128k) reduz memória por camada em 1000x.",
            "common_errors": "Erro comum: Não usar sliding_window para camadas sliding, assumindo contexto completo. Isso causa overestimation massiva de memória e sizing incorreto."
        },
        "effective_context": {
            "description": "Tamanho de contexto (em tokens) que sua aplicação efetivamente usará em runtime. Diferente de max_position_embeddings (limite do modelo).",
            "source": "NFR (Non-Functional Requirement) do produto/aplicação. Você define baseado no use case (ex: 4k para chat, 128k para análise de documentos).",
            "importance": "Impacta diretamente o tamanho do KV cache por sessão. Contexto maior = mais memória = menos sessões por nó. Definir incorretamente causa over/under-provisioning.",
            "common_errors": "Erro comum: Usar max_position_embeddings como effective_context. Isso superestima memória se aplicação usa contextos menores, ou causa problemas se excede o limite do modelo."
        },
        "kv_precision": {
            "description": "Precisão numérica usada para armazenar tensores Key e Value: fp8/int8 (1 byte/elemento) ou fp16/bf16 (2 bytes/elemento).",
            "source": "Parâmetro de runtime configurável. fp8 é recomendado para economia de memória com mínima perda de qualidade.",
            "importance": "Impacta diretamente (2x) o tamanho do KV cache. fp16 vs fp8 dobra a memória necessária e reduz pela metade o número de sessões por nó.",
            "common_errors": "Erro comum: Usar fp16 por default sem testar fp8. Muitos casos fp8 tem qualidade equivalente, mas fp16 dobra o custo de infraestrutura desnecessariamente."
        },
        "concurrency": {
            "description": "Número de sessões/requisições simultâneas (concurrent users) que o sistema deve suportar. Métrica de throughput.",
            "source": "NFR do produto, baseado em projeções de tráfego e SLA. Pode vir de análise de uso, teste de carga, ou requisitos de negócio.",
            "importance": "Define quantos nós você precisa. Concurrency mal estimada causa: subdimensionamento (SLA quebrado, throttling) ou superdimensionamento (desperdício de capex).",
            "common_errors": "Erro comum: Confundir concurrency (sessões simultâneas) com RPS (requests per second). Concurrency = sessões ativas ao mesmo tempo. RPS considera latência."
        },
        "kv_budget_ratio": {
            "description": "Fração da HBM total alocada para KV cache (ex: 0.70 = 70%). O restante é para modelo, ativações, overhead de runtime.",
            "source": "Parâmetro de tuning/configuração. Default 0.70 é conservador. Pode ser ajustado baseado em profiling real.",
            "importance": "Define quantas sessões cabem por nó. Budget muito alto (>0.80) causa fragmentação e instabilidade. Budget muito baixo (<0.50) desperdiça HBM.",
            "common_errors": "Erro comum: Alocar 100% da HBM para KV cache, ignorando overhead do modelo, ativações, e buffers do runtime. Isso causa OOM (Out of Memory) em produção."
        },
        "runtime_overhead_gib": {
            "description": "Memória GPU (GiB) reservada para modelo (pesos), ativações de computação, e buffers do runtime de inferência.",
            "source": "Estimativa baseada em tamanho do modelo e framework. Pode ser medido via profiling. Default conservador: 80-150 GiB para modelos grandes.",
            "importance": "Subtrai da HBM disponível antes de calcular budget de KV. Subestimar causa OOM. Superestimar desperdiça capacidade.",
            "common_errors": "Erro comum: Usar overhead muito baixo (<50 GiB) para modelos grandes (>100B parâmetros). Modelo 120B em fp16 sozinho já ocupa ~240 GiB."
        },
        "peak_headroom_ratio": {
            "description": "Fração adicional de capacidade reservada para picos de tráfego (ex: 0.20 = 20% acima da concurrency nominal).",
            "source": "NFR de SRE, baseado em análise de sazonalidade e requisitos de SLO. Típico: 10-30%.",
            "importance": "Garante que sistema aguenta picos sem degradação de SLO. Sem headroom, qualquer pico causa throttling ou violação de SLA.",
            "common_errors": "Erro comum: Não ter headroom (0%) em produção. Tráfego sempre tem variação. Outro erro: headroom excessivo (>50%) que desperdiça capex."
        },
        "ha_mode": {
            "description": "Modo de alta disponibilidade: 'none' (sem redundância), 'n+1' (tolera falha de 1 nó), 'n+2' (tolera 2 nós).",
            "source": "NFR de disponibilidade, baseado em SLA. Produção crítica geralmente requer no mínimo N+1.",
            "importance": "Define quantos nós extras alocar para redundância. N+1 garante que falha de 1 nó não quebra SLA. Sem HA, falha de nó causa degradação imediata.",
            "common_errors": "Erro comum: Não ter HA (none) em produção com SLA > 99%. Falha de hardware é inevitável. Outro erro: N+2 quando N+1 já atende, desperdiçando capex."
        }
    }


# ============================================================================
# FUNÇÕES DE CÁLCULO COM RACIONAL
# ============================================================================
def calc_kv_per_session_with_rationale(
    model: Model,
    effective_context: int,
    kv_precision: str
) -> Tuple[float, Dict[str, Rationale], List[str]]:
    """
    Calcula KV cache por sessão (GiB) com racional detalhado.
    
    Returns:
        (kv_gib, rationale_dict, warnings)
    """
    warnings = []
    rationale = {}
    bytes_per_elem = KV_PRECISION_BYTES[kv_precision]
    
    # Validar e clampar effective_context
    original_context = effective_context
    if effective_context > model.max_position_embeddings:
        warnings.append(
            f"AVISO: effective_context={effective_context} excede "
            f"max_position_embeddings={model.max_position_embeddings}. "
            f"Clampado para {model.max_position_embeddings}."
        )
        effective_context = model.max_position_embeddings
    
    # Calcular baseado no padrão de atenção
    total_bytes = 0
    formula_parts = []
    
    if model.attention_pattern == "full":
        seq_len = effective_context
        bytes_per_layer = 2 * seq_len * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_bytes = bytes_per_layer * model.num_layers
        
        formula_parts.append(f"Full attention: todas {model.num_layers} camadas")
        formula_parts.append(f"bytes_per_layer = 2 × {seq_len} × {model.num_key_value_heads} × {model.head_dim} × {bytes_per_elem}")
        formula_parts.append(f"total = {model.num_layers} × bytes_per_layer")
    
    elif model.attention_pattern == "sliding":
        seq_len = model.sliding_window
        bytes_per_layer = 2 * seq_len * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_bytes = bytes_per_layer * model.num_layers
        
        formula_parts.append(f"Sliding window: todas {model.num_layers} camadas com window={model.sliding_window}")
        formula_parts.append(f"bytes_per_layer = 2 × {seq_len} × {model.num_key_value_heads} × {model.head_dim} × {bytes_per_elem}")
        formula_parts.append(f"total = {model.num_layers} × bytes_per_layer")
    
    elif model.attention_pattern == "hybrid":
        # Full layers
        seq_full = effective_context
        bytes_full = 2 * seq_full * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_full = bytes_full * model.hybrid_full_layers
        
        # Sliding layers
        seq_sliding = model.sliding_window
        bytes_sliding = 2 * seq_sliding * model.num_key_value_heads * model.head_dim * bytes_per_elem
        total_sliding = bytes_sliding * model.hybrid_sliding_layers
        
        total_bytes = total_full + total_sliding
        
        formula_parts.append(f"Hybrid attention: {model.hybrid_full_layers} full + {model.hybrid_sliding_layers} sliding")
        formula_parts.append(f"Full: 2 × {seq_full} × {model.num_key_value_heads} × {model.head_dim} × {bytes_per_elem} × {model.hybrid_full_layers}")
        formula_parts.append(f"Sliding: 2 × {seq_sliding} × {model.num_key_value_heads} × {model.head_dim} × {bytes_per_elem} × {model.hybrid_sliding_layers}")
        formula_parts.append(f"total = full + sliding")
    
    kv_gib = total_bytes / (1024**3)
    
    # Criar racional
    rationale["kv_per_session_gib"] = Rationale(
        formula="\n".join(formula_parts),
        inputs={
            "model": model.name,
            "num_layers": model.num_layers,
            "num_kv_heads": model.num_key_value_heads,
            "head_dim": model.head_dim,
            "attention_pattern": model.attention_pattern,
            "effective_context": effective_context,
            "original_context": original_context if original_context != effective_context else None,
            "sliding_window": model.sliding_window if model.attention_pattern in ["sliding", "hybrid"] else None,
            "kv_precision": kv_precision,
            "bytes_per_element": bytes_per_elem,
            "total_bytes": total_bytes
        },
        explanation=(
            f"KV cache armazena tensores Key e Value de todas as camadas para o contexto da sessão. "
            f"Cada posição no contexto mantém {model.num_key_value_heads} heads × {model.head_dim} dims × {bytes_per_elem} bytes/elem = "
            f"{model.num_key_value_heads * model.head_dim * bytes_per_elem} bytes por posição (K+V separados, daí fator 2). "
            f"Modelo com attention_pattern='{model.attention_pattern}' usa contexto efetivo diferente por camada. "
            f"Total de {kv_gib:.2f} GiB por sessão ativa."
        )
    )
    
    # Avisos adicionais
    if kv_precision in ["fp16", "bf16"]:
        warnings.append(
            f"AVISO: kv_precision={kv_precision} usa 2 bytes/elemento. "
            f"Considere fp8 (1 byte) para reduzir memória pela metade com mínima perda de qualidade."
        )
    
    return kv_gib, rationale, warnings


def calc_scenario_sizing(
    model: Model,
    server: Server,
    storage: StorageProfile,
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    kv_budget_ratio: float,
    runtime_overhead_gib: float,
    peak_headroom_ratio: float,
    ha_extra_nodes: int,
    scenario_name: str
) -> ScenarioResult:
    """
    Calcula sizing completo para um cenário com racional detalhado.
    """
    warnings = []
    all_rationale = {}
    
    # 1) KV por sessão
    kv_per_session_gib, kv_rationale, kv_warnings = calc_kv_per_session_with_rationale(
        model, effective_context, kv_precision
    )
    all_rationale.update(kv_rationale)
    warnings.extend(kv_warnings)
    
    # 2) KV total
    kv_total_gib = kv_per_session_gib * concurrency
    kv_total_tib = kv_total_gib / 1024
    
    all_rationale["kv_total_gib"] = Rationale(
        formula=f"kv_total_gib = kv_per_session_gib × concurrency",
        inputs={
            "kv_per_session_gib": round(kv_per_session_gib, 4),
            "concurrency": concurrency
        },
        explanation=(
            f"Memória total de KV cache necessária para suportar {concurrency:,} sessões simultâneas. "
            f"Cada sessão precisa de {kv_per_session_gib:.2f} GiB, totalizando {kv_total_tib:.2f} TiB "
            f"distribuídos entre os nós do cluster."
        )
    )
    
    # 3) HBM total do servidor
    hbm_total_gib = server.total_hbm_gb * GB_TO_GIB
    
    all_rationale["hbm_total_gib"] = Rationale(
        formula=f"hbm_total_gib = total_hbm_gb × (10^9 / 2^30)",
        inputs={
            "server": server.name,
            "gpus": server.gpus,
            "hbm_per_gpu_gb": server.hbm_per_gpu_gb,
            "total_hbm_gb": server.total_hbm_gb,
            "gb_to_gib_factor": GB_TO_GIB
        },
        explanation=(
            f"Servidor {server.name} tem {server.gpus} GPUs × {server.hbm_per_gpu_gb} GB/GPU = "
            f"{server.total_hbm_gb} GB total. Convertido para GiB (binário): {hbm_total_gib:.1f} GiB. "
            f"Esta é a memória total disponível por nó para modelo, KV cache, ativações e buffers."
        )
    )
    
    # 4) Budget de KV por nó
    kv_budget_gib = max(0, (hbm_total_gib - runtime_overhead_gib) * kv_budget_ratio)
    
    all_rationale["kv_budget_gib"] = Rationale(
        formula=f"kv_budget_gib = max(0, (hbm_total_gib - runtime_overhead_gib) × kv_budget_ratio)",
        inputs={
            "hbm_total_gib": round(hbm_total_gib, 2),
            "runtime_overhead_gib": runtime_overhead_gib,
            "kv_budget_ratio": kv_budget_ratio,
            "available_after_overhead_gib": round(hbm_total_gib - runtime_overhead_gib, 2)
        },
        explanation=(
            f"De {hbm_total_gib:.1f} GiB de HBM, reservamos {runtime_overhead_gib} GiB para modelo+ativações. "
            f"Dos {hbm_total_gib - runtime_overhead_gib:.1f} GiB restantes, alocamos {kv_budget_ratio*100:.0f}% "
            f"({kv_budget_gib:.1f} GiB) para KV cache. O resto ({(1-kv_budget_ratio)*100:.0f}%) fica como buffer "
            f"para fragmentação e overhead de runtime."
        )
    )
    
    # 5) Sessões por nó
    if kv_budget_gib <= 0:
        sessions_per_node = 0
        warnings.append(
            f"ERRO: Budget de KV <= 0 (hbm_total={hbm_total_gib:.1f} GiB, overhead={runtime_overhead_gib} GiB). "
            f"Servidor não tem memória suficiente. Reduza overhead ou use servidor maior."
        )
    else:
        sessions_per_node = int(kv_budget_gib / kv_per_session_gib)
        
        if sessions_per_node == 0:
            warnings.append(
                f"ERRO: Sessões por nó = 0. Uma única sessão precisa de {kv_per_session_gib:.2f} GiB mas budget é {kv_budget_gib:.1f} GiB. "
                f"Soluções: (1) Reduzir effective_context, (2) Usar fp8 em vez de fp16, (3) Reduzir runtime_overhead_gib, "
                f"(4) Aumentar kv_budget_ratio, ou (5) Usar servidor com mais HBM."
            )
    
    all_rationale["sessions_per_node"] = Rationale(
        formula=f"sessions_per_node = floor(kv_budget_gib / kv_per_session_gib)",
        inputs={
            "kv_budget_gib": round(kv_budget_gib, 2),
            "kv_per_session_gib": round(kv_per_session_gib, 4)
        },
        explanation=(
            f"Com {kv_budget_gib:.1f} GiB disponíveis para KV e cada sessão consumindo {kv_per_session_gib:.2f} GiB, "
            f"cada nó pode suportar {sessions_per_node} sessões simultâneas. Este é o limite de capacidade por nó "
            f"baseado exclusivamente em memória de KV cache."
        )
    )
    
    # 6) Nós necessários
    if sessions_per_node > 0:
        nodes_capacity = math.ceil(concurrency / sessions_per_node)
        concurrency_with_headroom = concurrency * (1 + peak_headroom_ratio)
        nodes_with_headroom = math.ceil(concurrency_with_headroom / sessions_per_node)
    else:
        nodes_capacity = 0
        nodes_with_headroom = 0
    
    nodes_final = nodes_with_headroom + ha_extra_nodes
    
    all_rationale["nodes_capacity"] = Rationale(
        formula=f"nodes_capacity = ceil(concurrency / sessions_per_node)",
        inputs={
            "concurrency": concurrency,
            "sessions_per_node": sessions_per_node
        },
        explanation=(
            f"Para atender {concurrency:,} sessões simultâneas com {sessions_per_node} sessões/nó, "
            f"precisamos de no mínimo {nodes_capacity} nós. Este é o dimensionamento de capacidade pura, "
            f"sem considerar headroom para picos ou redundância para HA."
        )
    )
    
    all_rationale["nodes_with_headroom"] = Rationale(
        formula=f"nodes_with_headroom = ceil(concurrency × (1 + peak_headroom_ratio) / sessions_per_node)",
        inputs={
            "concurrency": concurrency,
            "peak_headroom_ratio": peak_headroom_ratio,
            "concurrency_with_headroom": round(concurrency * (1 + peak_headroom_ratio), 1),
            "sessions_per_node": sessions_per_node
        },
        explanation=(
            f"Adicionando {peak_headroom_ratio*100:.0f}% de headroom para picos de tráfego, precisamos suportar "
            f"{concurrency * (1 + peak_headroom_ratio):.0f} sessões simultâneas, resultando em {nodes_with_headroom} nós. "
            f"Headroom garante que o sistema aguenta variações de carga sem degradação de SLO."
        )
    )
    
    all_rationale["nodes_final"] = Rationale(
        formula=f"nodes_final = nodes_with_headroom + ha_extra_nodes",
        inputs={
            "nodes_with_headroom": nodes_with_headroom,
            "ha_extra_nodes": ha_extra_nodes,
            "ha_mode": "n+2" if ha_extra_nodes == 2 else ("n+1" if ha_extra_nodes == 1 else "none")
        },
        explanation=(
            f"Adicionando {ha_extra_nodes} nó(s) para alta disponibilidade, total final é {nodes_final} nós. "
            f"{'Sem HA: qualquer falha de nó causa degradação imediata.' if ha_extra_nodes == 0 else ''}"
            f"{'Com N+1: sistema tolera falha de 1 nó mantendo SLO.' if ha_extra_nodes == 1 else ''}"
            f"{'Com N+2: sistema tolera falha de 2 nós mantendo SLO.' if ha_extra_nodes == 2 else ''}"
        )
    )
    
    # Avisos adicionais
    if effective_context >= 128000:
        warnings.append(
            f"ALERTA: Contexto longo ({effective_context:,} tokens) aumenta TTFT (Time To First Token) e pressiona I/O de storage "
            f"durante prefill. Storage: {storage.name} ({storage.throughput_read_gbps} GB/s read, P99={storage.latency_read_ms_p99} ms)."
        )
    
    if kv_budget_ratio > 0.75:
        warnings.append(
            f"ALERTA: kv_budget_ratio={kv_budget_ratio} é alto (>75%). Risco de fragmentação de memória e instabilidade. "
            f"Considere reduzir para 0.65-0.70 ou usar servidor com mais HBM."
        )
    
    if runtime_overhead_gib < 50:
        warnings.append(
            f"ALERTA: runtime_overhead_gib={runtime_overhead_gib} parece baixo (<50 GiB). "
            f"Modelos grandes (>50B parâmetros) tipicamente precisam de 80-150 GiB. Verifique se não está subestimado."
        )
    
    # 7) Infraestrutura Física
    # Energia total
    total_power_kw = nodes_final * server.power_kw_max
    
    all_rationale["total_power_kw"] = Rationale(
        formula=f"total_power_kw = nodes_final × power_kw_max",
        inputs={
            "nodes_final": nodes_final,
            "power_kw_max": server.power_kw_max,
            "server": server.name
        },
        explanation=(
            f"Consumo total de energia para {nodes_final} nós × {server.power_kw_max} kW/nó = {total_power_kw} kW. "
            f"Este é o dimensionamento de energia máxima do sistema, impactando PDU (Power Distribution Unit), "
            f"capacidade de UPS, e contrato de energia do data center. Considere também eficiência de cooling (PUE ~1.3-1.5x)."
        )
    )
    
    # Espaço em rack
    total_rack_u = nodes_final * server.rack_units_u
    
    all_rationale["total_rack_u"] = Rationale(
        formula=f"total_rack_u = nodes_final × rack_units_u",
        inputs={
            "nodes_final": nodes_final,
            "rack_units_u": server.rack_units_u,
            "server": server.name
        },
        explanation=(
            f"Espaço total de rack necessário: {nodes_final} nós × {server.rack_units_u}U/nó = {total_rack_u}U. "
            f"Considerando racks padrão de 42U, isto equivale a {total_rack_u/42:.1f} racks. "
            f"Impacta densidade de implantação e capacidade física do data center. "
            f"Adicione ~20% para switches, PDUs e espaço de ventilação."
        )
    )
    
    # Heat output (se disponível)
    total_heat_btu_hr = None
    if server.heat_output_btu_hr_max is not None:
        total_heat_btu_hr = nodes_final * server.heat_output_btu_hr_max
        
        all_rationale["total_heat_btu_hr"] = Rationale(
            formula=f"total_heat_btu_hr = nodes_final × heat_output_btu_hr_max",
            inputs={
                "nodes_final": nodes_final,
                "heat_output_btu_hr_max": server.heat_output_btu_hr_max,
                "server": server.name
            },
            explanation=(
                f"Dissipação térmica total: {nodes_final} nós × {server.heat_output_btu_hr_max:,.0f} BTU/hr/nó = "
                f"{total_heat_btu_hr:,.0f} BTU/hr. Isto define a capacidade de refrigeração (cooling capacity) necessária. "
                f"BTU/hr pode ser convertido em toneladas de refrigeração (1 ton = 12,000 BTU/hr): "
                f"{total_heat_btu_hr/12000:.1f} tons. Impacta HVAC e COP (Coefficient of Performance) do data center."
            )
        )
    
    # Criar resultado
    result = ScenarioResult(
        name=scenario_name,
        peak_headroom_ratio=peak_headroom_ratio,
        ha_mode="n+2" if ha_extra_nodes == 2 else ("n+1" if ha_extra_nodes == 1 else "none"),
        ha_extra_nodes=ha_extra_nodes,
        kv_budget_ratio=kv_budget_ratio,
        kv_per_session_gib=kv_per_session_gib,
        kv_total_gib=kv_total_gib,
        kv_total_tib=kv_total_tib,
        hbm_total_gib=hbm_total_gib,
        kv_budget_gib=kv_budget_gib,
        sessions_per_node=sessions_per_node,
        nodes_capacity=nodes_capacity,
        nodes_with_headroom=nodes_with_headroom,
        nodes_final=nodes_final,
        total_power_kw=total_power_kw,
        total_rack_u=total_rack_u,
        total_heat_btu_hr=total_heat_btu_hr,
        rationale=all_rationale,
        warnings=warnings
    )
    
    return result


# ============================================================================
# FUNÇÃO PRINCIPAL DE SIZING (3 CENÁRIOS)
# ============================================================================
def calculate_sizing_all_scenarios(
    model: Model,
    server: Server,
    storage: StorageProfile,
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    kv_budget_ratio: float,
    runtime_overhead_gib: float,
    peak_headroom_ratio: float,
    verbose: bool = False
) -> Dict[str, ScenarioResult]:
    """
    Calcula sizing para os 3 cenários obrigatórios: MÍNIMO, RECOMENDADO, IDEAL.
    """
    scenarios = {}
    
    # CENÁRIO MÍNIMO
    scenarios["minimum"] = calc_scenario_sizing(
        model=model,
        server=server,
        storage=storage,
        concurrency=concurrency,
        effective_context=effective_context,
        kv_precision=kv_precision,
        kv_budget_ratio=kv_budget_ratio,
        runtime_overhead_gib=runtime_overhead_gib,
        peak_headroom_ratio=0.0,  # Sem headroom
        ha_extra_nodes=0,  # Sem HA
        scenario_name="MÍNIMO"
    )
    
    # CENÁRIO RECOMENDADO
    scenarios["recommended"] = calc_scenario_sizing(
        model=model,
        server=server,
        storage=storage,
        concurrency=concurrency,
        effective_context=effective_context,
        kv_precision=kv_precision,
        kv_budget_ratio=kv_budget_ratio,
        runtime_overhead_gib=runtime_overhead_gib,
        peak_headroom_ratio=peak_headroom_ratio,  # Headroom configurado
        ha_extra_nodes=1,  # N+1
        scenario_name="RECOMENDADO"
    )
    
    # CENÁRIO IDEAL
    # IDEAL é mais conservador: headroom mínimo de 30%, N+2, budget ratio mais conservador
    ideal_headroom = max(peak_headroom_ratio, 0.30)
    ideal_budget_ratio = min(kv_budget_ratio, 0.65)
    
    scenarios["ideal"] = calc_scenario_sizing(
        model=model,
        server=server,
        storage=storage,
        concurrency=concurrency,
        effective_context=effective_context,
        kv_precision=kv_precision,
        kv_budget_ratio=ideal_budget_ratio,  # Mais conservador
        runtime_overhead_gib=runtime_overhead_gib,
        peak_headroom_ratio=ideal_headroom,  # Mínimo 30%
        ha_extra_nodes=2,  # N+2
        scenario_name="IDEAL"
    )
    
    return scenarios


# ============================================================================
# FORMATAÇÃO DE SAÍDA
# ============================================================================
def format_report(
    model: Model,
    server: Server,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    verbose: bool = False
) -> str:
    """Formata relatório completo em texto."""
    lines = []
    lines.append("=" * 100)
    lines.append("RELATÓRIO DE DIMENSIONAMENTO AVANÇADO DE INFERÊNCIA LLM")
    lines.append("Sistema de Sizing com Racional de Cálculo e Análise de Cenários")
    lines.append("=" * 100)
    lines.append("")
    
    # ========================================================================
    # SEÇÃO 1: ENTRADAS
    # ========================================================================
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 1: ENTRADAS (Modelo / Servidor / Storage / NFR)".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    lines.append("MODELO:")
    lines.append(f"  Nome: {model.name}")
    lines.append(f"  Camadas: {model.num_layers}")
    lines.append(f"  KV Heads: {model.num_key_value_heads}")
    lines.append(f"  Head Dim: {model.head_dim}")
    lines.append(f"  Max Position Embeddings: {model.max_position_embeddings:,}")
    lines.append(f"  Padrão de Atenção: {model.attention_pattern}")
    if model.attention_pattern == "hybrid":
        lines.append(f"    • Full Layers: {model.hybrid_full_layers}")
        lines.append(f"    • Sliding Layers: {model.hybrid_sliding_layers}")
        lines.append(f"    • Sliding Window: {model.sliding_window}")
    elif model.attention_pattern == "sliding":
        lines.append(f"    • Sliding Window: {model.sliding_window}")
    lines.append(f"  Precisão KV Padrão: {model.default_kv_precision}")
    lines.append("")
    
    lines.append("SERVIDOR:")
    lines.append(f"  Nome: {server.name}")
    lines.append(f"  GPUs: {server.gpus}")
    lines.append(f"  HBM por GPU: {server.hbm_per_gpu_gb} GB")
    lines.append(f"  HBM Total: {server.total_hbm_gb} GB ({server.total_hbm_gb * GB_TO_GIB:.1f} GiB)")
    if server.nvlink_bandwidth_tbps:
        lines.append(f"  NVLink Bandwidth: {server.nvlink_bandwidth_tbps} TB/s")
    lines.append("")
    
    lines.append("STORAGE:")
    lines.append(f"  Perfil: {storage.name}")
    lines.append(f"  Tipo: {storage.type}")
    lines.append(f"  IOPS: {storage.iops_read:,} read / {storage.iops_write:,} write")
    lines.append(f"  Throughput: {storage.throughput_read_gbps} GB/s read / {storage.throughput_write_gbps} GB/s write")
    lines.append(f"  Latência P99: {storage.latency_read_ms_p99} ms read / {storage.latency_write_ms_p99} ms write")
    lines.append("")
    
    lines.append("NFR (Non-Functional Requirements):")
    lines.append(f"  Concorrência Alvo: {concurrency:,} sessões simultâneas")
    lines.append(f"  Contexto Efetivo: {effective_context:,} tokens")
    lines.append(f"  Precisão KV: {kv_precision}")
    lines.append("")
    
    # ========================================================================
    # SEÇÃO 2: DICIONÁRIO DE PARÂMETROS
    # ========================================================================
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 2: DICIONÁRIO DE PARÂMETROS (Explicação e Importância)".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    param_dict = get_parameter_dictionary()
    
    # Mostrar apenas parâmetros mais relevantes no relatório texto (todos vão pro JSON)
    key_params = [
        "num_layers", "num_key_value_heads", "head_dim", "attention_pattern",
        "effective_context", "kv_precision", "concurrency", "kv_budget_ratio",
        "runtime_overhead_gib", "peak_headroom_ratio", "ha_mode"
    ]
    
    for param_name in key_params:
        if param_name in param_dict:
            p = param_dict[param_name]
            lines.append(f"【{param_name}】")
            lines.append(f"  O que é: {p['description']}")
            lines.append(f"  Origem: {p['source']}")
            lines.append(f"  Importância: {p['importance']}")
            lines.append(f"  Erro comum: {p['common_errors']}")
            lines.append("")
    
    lines.append("(Veja JSON para dicionário completo de todos os parâmetros)")
    lines.append("")
    
    # ========================================================================
    # SEÇÃO 3: RESULTADOS POR CENÁRIO
    # ========================================================================
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 3: RESULTADOS POR CENÁRIO (MÍNIMO / RECOMENDADO / IDEAL)".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    for scenario_key in ["minimum", "recommended", "ideal"]:
        scenario = scenarios[scenario_key]
        lines.append("=" * 100)
        lines.append(f"CENÁRIO: {scenario.name}")
        lines.append("=" * 100)
        lines.append(f"  • Peak Headroom: {scenario.peak_headroom_ratio * 100:.0f}%")
        lines.append(f"  • HA Mode: {scenario.ha_mode}")
        lines.append(f"  • KV Budget Ratio: {scenario.kv_budget_ratio * 100:.0f}%")
        lines.append("")
        
        # Resultados com racional
        results_to_show = [
            ("kv_per_session_gib", f"{scenario.kv_per_session_gib:.2f} GiB"),
            ("kv_total_gib", f"{scenario.kv_total_tib:.2f} TiB ({scenario.kv_total_gib:.2f} GiB)"),
            ("hbm_total_gib", f"{scenario.hbm_total_gib:.1f} GiB"),
            ("kv_budget_gib", f"{scenario.kv_budget_gib:.1f} GiB"),
            ("sessions_per_node", f"{scenario.sessions_per_node:,} sessões"),
            ("nodes_capacity", f"{scenario.nodes_capacity} nós"),
            ("nodes_with_headroom", f"{scenario.nodes_with_headroom} nós"),
            ("nodes_final", f"{scenario.nodes_final} nós"),
        ]
        
        for key, value in results_to_show:
            lines.append(f"▸ {key.replace('_', ' ').title()}: {value}")
            
            if key in scenario.rationale:
                rat = scenario.rationale[key]
                lines.append("")
                lines.append("  Racional:")
                lines.append(f"    Fórmula:")
                for formula_line in rat.formula.split('\n'):
                    lines.append(f"      {formula_line}")
                lines.append(f"    Inputs:")
                for input_key, input_val in rat.inputs.items():
                    if input_val is not None:
                        lines.append(f"      • {input_key}: {input_val}")
                lines.append(f"    Interpretação:")
                # Quebrar explanation em linhas de no máximo 90 chars
                explanation_words = rat.explanation.split()
                current_line = "      "
                for word in explanation_words:
                    if len(current_line) + len(word) + 1 <= 96:
                        current_line += word + " "
                    else:
                        lines.append(current_line.rstrip())
                        current_line = "      " + word + " "
                if current_line.strip():
                    lines.append(current_line.rstrip())
                lines.append("")
        
        # Avisos do cenário
        if scenario.warnings:
            lines.append("  ⚠️  AVISOS DESTE CENÁRIO:")
            for i, warning in enumerate(scenario.warnings, 1):
                lines.append(f"    [{i}] {warning}")
            lines.append("")
    
    # ========================================================================
    # SEÇÃO 4: ALERTAS E RISCOS
    # ========================================================================
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 4: ALERTAS E RISCOS OPERACIONAIS".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    # Coletar todos os avisos únicos de todos os cenários
    all_warnings = set()
    for scenario in scenarios.values():
        all_warnings.update(scenario.warnings)
    
    if all_warnings:
        for i, warning in enumerate(sorted(all_warnings), 1):
            lines.append(f"[{i}] {warning}")
    else:
        lines.append("Nenhum alerta crítico detectado.")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("FIM DO RELATÓRIO")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def format_report_markdown(
    model: Model,
    server: Server,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    verbose: bool = False
) -> str:
    """Formata relatório completo em Markdown."""
    lines = []
    
    # Título
    lines.append("# Relatório de Dimensionamento de Inferência LLM")
    lines.append("")
    lines.append("**Sistema de Sizing com Racional de Cálculo e Análise de Cenários**")
    lines.append("")
    lines.append(f"**Data:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Seção 1: Entradas
    lines.append("## 📋 Seção 1: Entradas")
    lines.append("")
    
    lines.append("### Modelo")
    lines.append("")
    lines.append(f"- **Nome:** {model.name}")
    lines.append(f"- **Camadas:** {model.num_layers}")
    lines.append(f"- **KV Heads:** {model.num_key_value_heads}")
    lines.append(f"- **Head Dim:** {model.head_dim}")
    lines.append(f"- **Max Position Embeddings:** {model.max_position_embeddings:,}")
    lines.append(f"- **Padrão de Atenção:** {model.attention_pattern}")
    if model.attention_pattern == "hybrid":
        lines.append(f"  - Full Layers: {model.hybrid_full_layers}")
        lines.append(f"  - Sliding Layers: {model.hybrid_sliding_layers}")
        lines.append(f"  - Sliding Window: {model.sliding_window}")
    elif model.attention_pattern == "sliding":
        lines.append(f"  - Sliding Window: {model.sliding_window}")
    lines.append(f"- **Precisão KV Padrão:** {model.default_kv_precision}")
    lines.append("")
    
    lines.append("### Servidor")
    lines.append("")
    lines.append(f"- **Nome:** {server.name}")
    lines.append(f"- **GPUs:** {server.gpus}")
    lines.append(f"- **HBM por GPU:** {server.hbm_per_gpu_gb} GB")
    lines.append(f"- **HBM Total:** {server.total_hbm_gb} GB ({server.total_hbm_gb * GB_TO_GIB:.1f} GiB)")
    if server.nvlink_bandwidth_tbps:
        lines.append(f"- **NVLink Bandwidth:** {server.nvlink_bandwidth_tbps} TB/s")
    lines.append("")
    
    lines.append("### Storage")
    lines.append("")
    lines.append(f"- **Perfil:** {storage.name}")
    lines.append(f"- **Tipo:** {storage.type}")
    lines.append(f"- **IOPS:** {storage.iops_read:,} read / {storage.iops_write:,} write")
    lines.append(f"- **Throughput:** {storage.throughput_read_gbps} GB/s read / {storage.throughput_write_gbps} GB/s write")
    lines.append(f"- **Latência P99:** {storage.latency_read_ms_p99} ms read / {storage.latency_write_ms_p99} ms write")
    lines.append("")
    
    lines.append("### NFR (Non-Functional Requirements)")
    lines.append("")
    lines.append(f"- **Concorrência Alvo:** {concurrency:,} sessões simultâneas")
    lines.append(f"- **Contexto Efetivo:** {effective_context:,} tokens")
    lines.append(f"- **Precisão KV:** {kv_precision}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Seção 2: Dicionário de Parâmetros (resumido)
    lines.append("## 📚 Seção 2: Dicionário de Parâmetros")
    lines.append("")
    lines.append("Principais parâmetros utilizados no dimensionamento:")
    lines.append("")
    
    param_dict = get_parameter_dictionary()
    key_params = [
        "num_layers", "num_key_value_heads", "effective_context", 
        "kv_precision", "kv_budget_ratio", "ha_mode"
    ]
    
    for param_name in key_params:
        if param_name in param_dict:
            p = param_dict[param_name]
            lines.append(f"### `{param_name}`")
            lines.append("")
            lines.append(f"**O que é:** {p['description']}")
            lines.append("")
            lines.append(f"**Importância:** {p['importance']}")
            lines.append("")
            lines.append(f"**Erro comum:** {p['common_errors']}")
            lines.append("")
    
    lines.append("> ℹ️ Veja JSON para dicionário completo de todos os parâmetros")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Seção 3: Resultados por Cenário
    lines.append("## 🎯 Seção 3: Resultados por Cenário")
    lines.append("")
    
    # Tabela comparativa
    lines.append("### Comparação Rápida")
    lines.append("")
    lines.append("| Métrica | MÍNIMO | RECOMENDADO | IDEAL |")
    lines.append("|---------|--------|-------------|-------|")
    
    min_sc = scenarios["minimum"]
    rec_sc = scenarios["recommended"]
    ideal_sc = scenarios["ideal"]
    
    lines.append(f"| **Headroom** | {min_sc.peak_headroom_ratio*100:.0f}% | {rec_sc.peak_headroom_ratio*100:.0f}% | {ideal_sc.peak_headroom_ratio*100:.0f}% |")
    lines.append(f"| **HA** | {min_sc.ha_mode} | {rec_sc.ha_mode} | {ideal_sc.ha_mode} |")
    lines.append(f"| **Budget KV** | {min_sc.kv_budget_ratio*100:.0f}% | {rec_sc.kv_budget_ratio*100:.0f}% | {ideal_sc.kv_budget_ratio*100:.0f}% |")
    lines.append(f"| **KV/Sessão** | {min_sc.kv_per_session_gib:.2f} GiB | {rec_sc.kv_per_session_gib:.2f} GiB | {ideal_sc.kv_per_session_gib:.2f} GiB |")
    lines.append(f"| **Sessões/Nó** | {min_sc.sessions_per_node} | {rec_sc.sessions_per_node} | {ideal_sc.sessions_per_node} |")
    lines.append(f"| **Nós Finais** | **{min_sc.nodes_final}** | **{rec_sc.nodes_final}** ✅ | **{ideal_sc.nodes_final}** |")
    lines.append("")
    lines.append("> ✅ **RECOMENDADO** é o cenário ideal para produção")
    lines.append("")
    
    # Detalhamento por cenário
    for scenario_key in ["minimum", "recommended", "ideal"]:
        scenario = scenarios[scenario_key]
        
        emoji = "🔴" if scenario_key == "minimum" else ("🟢" if scenario_key == "recommended" else "🔵")
        lines.append(f"### {emoji} Cenário: {scenario.name}")
        lines.append("")
        
        lines.append("**Configuração:**")
        lines.append("")
        lines.append(f"- Peak Headroom: {scenario.peak_headroom_ratio * 100:.0f}%")
        lines.append(f"- HA Mode: {scenario.ha_mode}")
        lines.append(f"- KV Budget Ratio: {scenario.kv_budget_ratio * 100:.0f}%")
        lines.append("")
        
        lines.append("**Resultados:**")
        lines.append("")
        
        results_data = [
            ("KV por Sessão", f"{scenario.kv_per_session_gib:.2f} GiB"),
            ("KV Total", f"{scenario.kv_total_tib:.2f} TiB"),
            ("HBM Total", f"{scenario.hbm_total_gib:.1f} GiB"),
            ("KV Budget", f"{scenario.kv_budget_gib:.1f} GiB"),
            ("Sessões por Nó", f"{scenario.sessions_per_node:,}"),
            ("Nós (Capacidade)", f"{scenario.nodes_capacity}"),
            ("Nós (com Headroom)", f"{scenario.nodes_with_headroom}"),
            ("**Nós Finais**", f"**{scenario.nodes_final}**"),
        ]
        
        for label, value in results_data:
            lines.append(f"- {label}: {value}")
        
        lines.append("")
        
        # Racional resumido para nós finais
        if "nodes_final" in scenario.rationale:
            rat = scenario.rationale["nodes_final"]
            lines.append("<details>")
            lines.append(f"<summary><b>📊 Racional: Nós Finais</b></summary>")
            lines.append("")
            lines.append("**Fórmula:**")
            lines.append("")
            lines.append("```")
            lines.append(rat.formula)
            lines.append("```")
            lines.append("")
            lines.append("**Interpretação:**")
            lines.append("")
            lines.append(rat.explanation)
            lines.append("")
            lines.append("</details>")
            lines.append("")
        
        # Avisos do cenário
        if scenario.warnings:
            lines.append("**⚠️ Avisos:**")
            lines.append("")
            for i, warning in enumerate(scenario.warnings, 1):
                lines.append(f"{i}. {warning}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Seção 4: Alertas e Riscos
    lines.append("## ⚠️ Seção 4: Alertas e Riscos")
    lines.append("")
    
    all_warnings = set()
    for scenario in scenarios.values():
        all_warnings.update(scenario.warnings)
    
    if all_warnings:
        for i, warning in enumerate(sorted(all_warnings), 1):
            lines.append(f"{i}. {warning}")
    else:
        lines.append("✅ Nenhum alerta crítico detectado.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Footer
    lines.append("## 📝 Observações")
    lines.append("")
    lines.append("- Este relatório foi gerado automaticamente pelo sistema de sizing v2.0")
    lines.append("- Para análise completa, consulte também o JSON output")
    lines.append("- Use o **CENÁRIO RECOMENDADO** para produção (N+1, balanceado)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Gerado por: Sistema de Sizing de Infraestrutura IA*")
    
    return "\n".join(lines)


def format_executive_report(
    model: Model,
    server: Server,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    kv_budget_ratio: float,
    runtime_overhead_gib: float,
    verbose: bool = False
) -> str:
    """
    Formata relatório executivo para diretoria e líderes de tecnologia.
    Foco em consumo físico de datacenter, consumo unitário e decisão executiva.
    """
    lines = []
    
    # Header
    lines.append("# RELATÓRIO EXECUTIVO")
    lines.append("## Dimensionamento de Infraestrutura de Inferência LLM")
    lines.append("")
    lines.append(f"**Data:** {__import__('datetime').datetime.now().strftime('%d/%m/%Y')}")
    lines.append(f"**Modelo Analisado:** {model.name}")
    lines.append(f"**Carga Operacional:** {concurrency:,} sessões simultâneas × {effective_context:,} tokens/contexto")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 1. SUMÁRIO EXECUTIVO
    # =======================================================================
    lines.append("## 1. Sumário Executivo")
    lines.append("")
    
    rec = scenarios["recommended"]
    
    lines.append(f"Este relatório dimensiona a infraestrutura necessária para operar o modelo **{model.name}** em produção, "
                 f"sustentando **{concurrency:,} sessões simultâneas** com contexto de **{effective_context:,} tokens**. "
                 f"O principal limitador da operação é a **memória de GPU (HBM)**, especificamente o **KV cache** que mantém o contexto conversacional ativo.")
    lines.append("")
    
    lines.append(f"A análise identifica impacto direto em três dimensões críticas:")
    lines.append(f"- **Servidores**: {rec.nodes_final} nós DGX {server.name} (cenário recomendado)")
    lines.append(f"- **Energia**: {rec.total_power_kw:.1f} kW de consumo contínuo")
    lines.append(f"- **Datacenter**: {rec.total_rack_u}U de espaço em rack ({rec.total_rack_u/42:.1f} racks padrão)")
    lines.append("")
    
    lines.append(f"**Para sustentar a carga avaliada com estabilidade operacional, "
                 f"a plataforma exige múltiplos nós DGX, com impacto direto em energia ({rec.total_power_kw:.1f} kW) "
                 f"e densidade de rack ({rec.total_rack_u}U).**")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 2. CENÁRIOS AVALIADOS (APRESENTAR PRIMEIRO)
    # =======================================================================
    lines.append("## 2. Cenários Avaliados")
    lines.append("")
    
    lines.append("### Tabela – Visão Geral dos Cenários")
    lines.append("")
    lines.append("| Cenário | Objetivo | Tolerância a Falhas | Risco Operacional |")
    lines.append("|---------|----------|---------------------|-------------------|")
    lines.append("| **Mínimo** | Atender no limite | Nenhuma | **Alto** — Falha causa indisponibilidade imediata |")
    lines.append("| **Recomendado** | Produção estável | Falha simples (N+1) | **Médio** — Degradação gerenciável |")
    lines.append("| **Ideal** | Alta resiliência | Falhas múltiplas (N+2) | **Baixo** — Sistema mantém SLA sob adversidades |")
    lines.append("")
    
    lines.append("Os três cenários representam diferentes níveis de **investimento** versus **risco operacional**. "
                 "O cenário **Mínimo** minimiza capex mas expõe a operação a risco de indisponibilidade não gerenciável. "
                 "O cenário **Recomendado** equilibra custo e resiliência, adequado para produção com SLA 99.9%. "
                 "O cenário **Ideal** maximiza disponibilidade, indicado para cargas críticas com requisitos de SLA > 99.95%.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 3. INFORMAÇÕES DO MODELO AVALIADO
    # =======================================================================
    lines.append("## 3. Informações do Modelo Avaliado")
    lines.append("")
    
    lines.append("### Tabela – Perfil do Modelo")
    lines.append("")
    lines.append("| Item | Valor |")
    lines.append("|------|-------|")
    lines.append(f"| **Modelo** | {model.name} |")
    lines.append(f"| **Número de camadas** | {model.num_layers} |")
    lines.append(f"| **KV heads** | {model.num_key_value_heads} |")
    lines.append(f"| **Contexto máximo** | {model.max_position_embeddings:,} tokens |")
    lines.append(f"| **Contexto efetivo usado** | {effective_context:,} tokens |")
    lines.append(f"| **Padrão de atenção** | {model.attention_pattern.capitalize()} |")
    lines.append(f"| **Precisão do KV cache** | {kv_precision.upper()} ({KV_PRECISION_BYTES[kv_precision]} byte/elemento) |")
    lines.append("")
    
    lines.append(f"O modelo {model.name} consome **memória viva** durante a operação para armazenar o **KV cache** — "
                 f"tensores Key e Value que mantêm o contexto conversacional. Este consumo é proporcional ao **contexto efetivo** "
                 f"({effective_context:,} tokens) e à **concorrência** ({concurrency:,} sessões), dominando a capacidade de infraestrutura necessária. "
                 f"Diferente dos pesos do modelo (fixos), o KV cache escala linearmente com o número de sessões ativas.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 4. CONSUMO UNITÁRIO DO MODELO (VISÃO FUNDAMENTAL)
    # =======================================================================
    lines.append("## 4. Consumo Unitário do Modelo")
    lines.append("")
    
    kv_per_session = rec.kv_per_session_gib
    hbm_percent_per_session = (kv_per_session / rec.kv_budget_gib) * 100 if rec.kv_budget_gib > 0 else 0
    power_per_session_w = (server.power_kw_max * 1000) / rec.sessions_per_node if rec.sessions_per_node > 0 else 0
    
    lines.append("### Tabela – Consumo por Sessão")
    lines.append("")
    lines.append("| Recurso | Consumo por Sessão | Significado Operacional |")
    lines.append("|---------|-------------------|------------------------|")
    lines.append(f"| **KV cache** | {kv_per_session:.2f} GiB | Memória GPU ocupada enquanto a sessão está ativa |")
    lines.append(f"| **GPU HBM (%)** | {hbm_percent_per_session:.1f}% de um nó | Fração da capacidade de um servidor consumida |")
    lines.append(f"| **Energia estimada** | {power_per_session_w:.0f} W | Impacto incremental por sessão ativa (aproximado) |")
    lines.append(f"| **Rack** | N/A | Sessão não consome rack diretamente; nó sim ({server.rack_units_u}U/nó) |")
    lines.append("")
    
    lines.append(f"**Cada sessão ativa \"reserva\" {kv_per_session:.2f} GiB de HBM ({hbm_percent_per_session:.1f}% do budget do nó).** "
                 f"A soma dessas reservas define o **limite físico** do servidor: com {rec.kv_budget_gib:.1f} GiB disponíveis para KV, "
                 f"cada nó suporta no máximo **{rec.sessions_per_node} sessões simultâneas**. "
                 f"Exceder este limite causa recusa de novas conexões ou degradação de performance.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 5. CONSUMO AGREGADO DO MODELO (TODAS AS SESSÕES)
    # =======================================================================
    lines.append("## 5. Consumo Agregado da Plataforma")
    lines.append("")
    
    lines.append("### Tabela – Consumo Total (Cenário Recomendado)")
    lines.append("")
    lines.append("| Recurso | Total Consumido |")
    lines.append("|---------|----------------|")
    lines.append(f"| **Sessões simultâneas** | {concurrency:,} |")
    lines.append(f"| **KV total** | {rec.kv_total_tib:.2f} TiB ({rec.kv_total_gib:,.1f} GiB) |")
    lines.append(f"| **Nós DGX** | {rec.nodes_final} |")
    lines.append(f"| **Energia total** | {rec.total_power_kw:.1f} kW ({rec.total_power_kw * 8.76:.0f} MWh/ano) |")
    lines.append(f"| **Espaço em rack** | {rec.total_rack_u}U ({rec.total_rack_u/42:.1f} racks) |")
    if rec.total_heat_btu_hr:
        lines.append(f"| **Dissipação térmica** | {rec.total_heat_btu_hr:,.0f} BTU/hr ({rec.total_heat_btu_hr/12000:.1f} tons) |")
    lines.append("")
    
    lines.append(f"O **consumo agregado** demonstra a diferença entre consumo unitário e impacto total: "
                 f"enquanto uma sessão consome {kv_per_session:.2f} GiB, {concurrency:,} sessões simultâneas consomem "
                 f"{rec.kv_total_tib:.2f} TiB distribuídos entre {rec.nodes_final} nós. "
                 f"**O crescimento de usuários impacta linearmente a infraestrutura**: dobrar concorrência para {concurrency*2:,} sessões "
                 f"dobraria energia para {rec.total_power_kw*2:.1f} kW e rack para {rec.total_rack_u*2}U.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 6. RESULTADOS POR CENÁRIO (COM ENERGIA E RACK)
    # =======================================================================
    lines.append("## 6. Resultados por Cenário")
    lines.append("")
    
    for scenario_key, scenario_label in [("minimum", "MÍNIMO"), ("recommended", "RECOMENDADO"), ("ideal", "IDEAL")]:
        sc = scenarios[scenario_key]
        
        lines.append(f"### Cenário {scenario_label}")
        lines.append("")
        
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| **Nós DGX** | {sc.nodes_final} |")
        lines.append(f"| **Sessões por nó** | {sc.sessions_per_node} |")
        lines.append(f"| **KV por sessão** | {sc.kv_per_session_gib:.2f} GiB |")
        lines.append(f"| **KV total** | {sc.kv_total_tib:.2f} TiB |")
        lines.append(f"| **Energia total** | **{sc.total_power_kw:.1f} kW** ({sc.total_power_kw * 8.76:.0f} MWh/ano) |")
        lines.append(f"| **Espaço em rack** | **{sc.total_rack_u}U** ({sc.total_rack_u/42:.1f} racks) |")
        if sc.total_heat_btu_hr:
            lines.append(f"| **Dissipação térmica** | {sc.total_heat_btu_hr:,.0f} BTU/hr ({sc.total_heat_btu_hr/12000:.1f} tons) |")
        lines.append(f"| **Arquitetura** | {sc.ha_mode.upper() if sc.ha_mode != 'none' else 'Sem redundância'} |")
        lines.append(f"| **Headroom para picos** | {sc.peak_headroom_ratio*100:.0f}% |")
        lines.append("")
        
        # Parágrafo executivo específico por cenário
        if scenario_key == "minimum":
            lines.append(f"**Significado Operacional:** Este cenário dimensiona a infraestrutura no limite absoluto ({sc.nodes_final} nós, {sc.total_power_kw:.1f} kW, {sc.total_rack_u}U). "
                         f"**Sem tolerância a falhas**: qualquer evento de manutenção ou falha de hardware resulta em indisponibilidade imediata. "
                         f"**Sem headroom**: picos de tráfego causam throttling ou recusa de conexões. "
                         f"**Impacto físico mínimo** mas **risco operacional alto**. Adequado apenas para PoC ou ambientes de desenvolvimento.")
        
        elif scenario_key == "recommended":
            lines.append(f"**Significado Operacional:** Dimensionado para produção com resiliência ({sc.nodes_final} nós, {sc.total_power_kw:.1f} kW, {sc.total_rack_u}U). "
                         f"**Tolera falha de 1 nó** sem perda de capacidade crítica. Headroom de {sc.peak_headroom_ratio*100:.0f}% absorve picos de demanda. "
                         f"**Impacto físico:** {sc.total_power_kw:.1f} kW requer PDU com capacidade adequada e UPS dimensionado; "
                         f"{sc.total_rack_u}U equivale a {sc.total_rack_u/42:.1f} racks, gerenciável em datacenter padrão. "
                         f"**Recomendado para produção com SLA 99.9%**.")
        
        else:  # ideal
            lines.append(f"**Significado Operacional:** Máxima resiliência operacional ({sc.nodes_final} nós, {sc.total_power_kw:.1f} kW, {sc.total_rack_u}U). "
                         f"**Tolera falhas simultâneas de até 2 nós**, cenário raro mas possível em eventos de rack ou rede. "
                         f"Headroom de {sc.peak_headroom_ratio*100:.0f}% e budget conservador ({sc.kv_budget_ratio*100:.0f}% HBM) garantem estabilidade máxima. "
                         f"**Impacto físico significativo:** {sc.total_power_kw:.1f} kW pode exigir upgrade de PDU/UPS; "
                         f"{sc.total_rack_u}U requer planejamento de densidade de rack. "
                         f"Indicado para cargas de missão crítica (financeiro, healthcare, SLA > 99.95%).")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # =======================================================================
    # 7. RACIONAL DE CÁLCULO (FORMATO OBRIGATÓRIO)
    # =======================================================================
    lines.append("## 7. Racional de Cálculo")
    lines.append("")
    
    lines.append("### Tabela – Metodologia de Dimensionamento")
    lines.append("")
    lines.append("| Resultado | Fórmula | Parâmetros do Cálculo | Suposição Aplicada | Significado Operacional |")
    lines.append("|-----------|---------|----------------------|-------------------|------------------------|")
    
    # KV por sessão
    if model.attention_pattern == "hybrid":
        formula_kv = "2 × [(full_layers × context) + (sliding_layers × window)] × kv_heads × head_dim × bytes"
    elif model.attention_pattern == "sliding":
        formula_kv = "2 × window × num_layers × kv_heads × head_dim × bytes"
    else:
        formula_kv = "2 × context × num_layers × kv_heads × head_dim × bytes"
    
    lines.append(f"| **KV por sessão** | {formula_kv} | "
                 f"Camadas: {model.num_layers}, Context: {effective_context:,}, KV heads: {model.num_key_value_heads}, Precisão: {kv_precision} | "
                 f"Padrão '{model.attention_pattern}' determina seq_length por camada | "
                 f"Memória reservada por sessão; subdimensionar causa OOM |")
    
    # Sessões por nó
    lines.append(f"| **Sessões por nó** | floor(Budget_KV / KV_per_session) | "
                 f"Budget: {rec.kv_budget_gib:.1f} GiB, KV/sessão: {rec.kv_per_session_gib:.2f} GiB | "
                 f"Budget = (HBM - overhead) × ratio; limitado por memória | "
                 f"Capacidade máxima do servidor; exceder causa recusa de conexões |")
    
    # Nós necessários
    lines.append(f"| **Nós DGX** | ceil(concurrency × (1 + headroom) / sessões_per_nó) + HA | "
                 f"Concorrência: {concurrency:,}, Headroom: {rec.peak_headroom_ratio*100:.0f}%, Sessões/nó: {rec.sessions_per_node}, HA: +{rec.ha_extra_nodes} | "
                 f"Headroom para picos; HA garante continuidade em falhas | "
                 f"Número de servidores a provisionar; define capex e opex |")
    
    # Energia total
    lines.append(f"| **Energia (kW)** | nodes_final × power_kw_max | "
                 f"Nós: {rec.nodes_final}, Power/nó: {server.power_kw_max} kW | "
                 f"Consumo máximo contínuo do sistema | "
                 f"Dimensiona PDU, UPS, contrato de energia; considerar PUE (~1.4x) |")
    
    # Rack total
    lines.append(f"| **Rack (U)** | nodes_final × rack_units_u | "
                 f"Nós: {rec.nodes_final}, U/nó: {server.rack_units_u}U | "
                 f"Cada servidor ocupa {server.rack_units_u}U; racks padrão = 42U | "
                 f"Define densidade e capacidade física; adicionar ~20% para infra |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 8. COMPARAÇÃO EXECUTIVA DOS CENÁRIOS
    # =======================================================================
    lines.append("## 8. Comparação Executiva dos Cenários")
    lines.append("")
    
    min_sc = scenarios["minimum"]
    ideal_sc = scenarios["ideal"]
    
    lines.append("### Tabela – Comparativo")
    lines.append("")
    lines.append("| Critério | Mínimo | Recomendado | Ideal |")
    lines.append("|----------|--------|-------------|-------|")
    lines.append(f"| **Nós DGX** | {min_sc.nodes_final} | {rec.nodes_final} | {ideal_sc.nodes_final} |")
    lines.append(f"| **Energia (kW)** | {min_sc.total_power_kw:.1f} | {rec.total_power_kw:.1f} | {ideal_sc.total_power_kw:.1f} |")
    lines.append(f"| **Rack (U)** | {min_sc.total_rack_u} | {rec.total_rack_u} | {ideal_sc.total_rack_u} |")
    lines.append(f"| **Tolerância a falhas** | Nenhuma | 1 nó (N+1) | 2 nós (N+2) |")
    lines.append(f"| **Headroom** | 0% | {rec.peak_headroom_ratio*100:.0f}% | {ideal_sc.peak_headroom_ratio*100:.0f}% |")
    lines.append(f"| **Risco operacional** | Alto | Médio | Baixo |")
    lines.append(f"| **CapEx relativo** | Baseline | +{int((rec.nodes_final/min_sc.nodes_final - 1) * 100)}% | +{int((ideal_sc.nodes_final/min_sc.nodes_final - 1) * 100)}% |")
    lines.append(f"| **Energia relativa** | Baseline | +{int((rec.total_power_kw/min_sc.total_power_kw - 1) * 100)}% | +{int((ideal_sc.total_power_kw/min_sc.total_power_kw - 1) * 100)}% |")
    lines.append("")
    
    lines.append(f"**O cenário RECOMENDADO oferece o melhor equilíbrio custo × risco.** Com {rec.nodes_final} nós (+{int((rec.nodes_final/min_sc.nodes_final - 1) * 100)}% vs Mínimo), "
                 f"garante operação estável, tolerando falhas e picos. **O impacto físico muda significativamente entre cenários:** "
                 f"Mínimo usa {min_sc.total_power_kw:.1f} kW, Recomendado {rec.total_power_kw:.1f} kW (+{int((rec.total_power_kw/min_sc.total_power_kw - 1) * 100)}%), "
                 f"Ideal {ideal_sc.total_power_kw:.1f} kW (+{int((ideal_sc.total_power_kw/min_sc.total_power_kw - 1) * 100)}%). "
                 f"A escolha deve considerar não apenas servidores, mas capacidade elétrica e densidade de datacenter.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 9. RECOMENDAÇÃO FINAL
    # =======================================================================
    lines.append("## 9. Recomendação Final")
    lines.append("")
    
    lines.append(f"**Recomenda-se o cenário RECOMENDADO**, que equilibra capacidade, consumo energético e tolerância a falhas sem sobrecarregar o datacenter.")
    lines.append("")
    
    lines.append(f"**Justificativa:**")
    lines.append(f"- **Estabilidade:** {rec.nodes_final} nós com N+1 toleram falha de 1 servidor mantendo {rec.sessions_per_node * (rec.nodes_final - 1):,} sessões (suficiente para carga nominal)")
    lines.append(f"- **Energia:** {rec.total_power_kw:.1f} kW requer PDU/UPS padrão de datacenter; PUE 1.4x = {rec.total_power_kw * 1.4:.1f} kW total facility")
    lines.append(f"- **Datacenter:** {rec.total_rack_u}U ({rec.total_rack_u/42:.1f} racks) é gerenciável e não exige reconfiguração física")
    lines.append(f"- **Risco:** Médio, com degradação gerenciável em falhas; adequado para produção com SLA 99.9%")
    lines.append("")
    
    lines.append(f"**Premissas sob governança:**")
    lines.append(f"- Limite de contexto: {effective_context:,} tokens (não liberar contexto máximo sem validação)")
    lines.append(f"- Monitoramento: Alertas quando concorrência ultrapassar {int(rec.sessions_per_node * rec.nodes_final * 0.8):,} sessões (80% capacidade)")
    lines.append(f"- Precisão KV: Manter {kv_precision.upper()} (mudança para FP16 dobraria energia e rack)")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # =======================================================================
    # 10. DICIONÁRIO DE PARÂMETROS (NO FINAL)
    # =======================================================================
    lines.append("## 10. Dicionário de Parâmetros")
    lines.append("")
    
    lines.append("### Tabela – Dicionário")
    lines.append("")
    lines.append("| Parâmetro | Origem | Descrição | Importância |")
    lines.append("|-----------|--------|-----------|------------|")
    
    # Parâmetros do modelo
    lines.append(f"| **num_layers** | Arquitetura do Modelo | Número de camadas do transformer ({model.num_layers}) | Impacta linearmente o KV cache |")
    lines.append(f"| **num_key_value_heads** | Arquitetura do Modelo | Cabeças de atenção para K/V ({model.num_key_value_heads}) | Redução via GQA economiza memória |")
    lines.append(f"| **attention_pattern** | Arquitetura do Modelo | Padrão de atenção: {model.attention_pattern} | Crítico para cálculo correto de KV |")
    
    # Parâmetros do servidor
    lines.append(f"| **total_hbm_gb** | Hardware do Servidor | HBM total do servidor ({server.total_hbm_gb} GB) | Define capacidade bruta de memória |")
    lines.append(f"| **power_kw_max** | Hardware do Servidor | Consumo máximo ({server.power_kw_max} kW) | Define impacto elétrico por nó |")
    lines.append(f"| **rack_units_u** | Hardware do Servidor | Espaço em rack ({server.rack_units_u}U) | Define densidade física |")
    
    # Parâmetros de NFR
    lines.append(f"| **concurrency** | NFR do Produto | Sessões simultâneas ({concurrency:,}) | Define escala e número de nós |")
    lines.append(f"| **effective_context** | NFR do Produto | Contexto efetivo ({effective_context:,} tokens) | Impacta KV por sessão linearmente |")
    lines.append(f"| **kv_precision** | Configuração de Runtime | Precisão do KV ({kv_precision.upper()}) | FP8=1 byte, FP16=2 bytes (dobra memória) |")
    lines.append(f"| **peak_headroom_ratio** | NFR de Resiliência | Folga para picos ({rec.peak_headroom_ratio*100:.0f}%) | Garante absorção de variações de carga |")
    lines.append(f"| **ha_mode** | NFR de Disponibilidade | Alta disponibilidade ({rec.ha_mode.upper()}) | N+1 tolera 1 falha; N+2 tolera 2 falhas |")
    
    lines.append("")
    lines.append("**Nota:** Parâmetros de modelo e servidor são fixos. Parâmetros de NFR e runtime são ajustáveis conforme requisitos de negócio.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Footer
    lines.append("## Informações do Relatório")
    lines.append("")
    lines.append(f"- **Sistema:** Sizing de Infraestrutura IA v2.0")
    lines.append(f"- **Data de Geração:** {__import__('datetime').datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    lines.append(f"- **Metodologia:** Dimensionamento baseado em memória GPU (KV cache) com impacto físico de datacenter")
    lines.append(f"- **Servidor de Referência:** {server.name} ({server.gpus} GPUs, {server.total_hbm_gb} GB HBM, {server.power_kw_max} kW, {server.rack_units_u}U)")
    lines.append("")
    lines.append("*Este relatório foi gerado automaticamente. Decisões de investimento devem ser revisadas por arquitetos de infraestrutura e finance.*")
    
    return "\n".join(lines)


def scenarios_to_dict(
    model: Model,
    server: Server,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    kv_budget_ratio: float,
    runtime_overhead_gib: float,
    peak_headroom_ratio: float
) -> dict:
    """Converte resultados para dicionário JSON."""
    
    def rationale_to_dict(r: Rationale) -> dict:
        return {
            "formula": r.formula,
            "inputs": r.inputs,
            "explanation": r.explanation
        }
    
    def scenario_to_dict(s: ScenarioResult) -> dict:
        return {
            "name": s.name,
            "configuration": {
                "peak_headroom_ratio": s.peak_headroom_ratio,
                "ha_mode": s.ha_mode,
                "ha_extra_nodes": s.ha_extra_nodes,
                "kv_budget_ratio": s.kv_budget_ratio
            },
            "results": {
                "kv_per_session_gib": round(s.kv_per_session_gib, 4),
                "kv_total_gib": round(s.kv_total_gib, 2),
                "kv_total_tib": round(s.kv_total_tib, 4),
                "hbm_total_gib": round(s.hbm_total_gib, 2),
                "kv_budget_gib": round(s.kv_budget_gib, 2),
                "sessions_per_node": s.sessions_per_node,
                "nodes_capacity": s.nodes_capacity,
                "nodes_with_headroom": s.nodes_with_headroom,
                "nodes_final": s.nodes_final
            },
            "rationale": {k: rationale_to_dict(v) for k, v in s.rationale.items()},
            "warnings": s.warnings
        }
    
    return {
        "inputs": {
            "model": {
                "name": model.name,
                "num_layers": model.num_layers,
                "num_key_value_heads": model.num_key_value_heads,
                "head_dim": model.head_dim,
                "max_position_embeddings": model.max_position_embeddings,
                "attention_pattern": model.attention_pattern,
                "hybrid_full_layers": model.hybrid_full_layers,
                "hybrid_sliding_layers": model.hybrid_sliding_layers,
                "sliding_window": model.sliding_window,
                "default_kv_precision": model.default_kv_precision
            },
            "server": {
                "name": server.name,
                "gpus": server.gpus,
                "hbm_per_gpu_gb": server.hbm_per_gpu_gb,
                "total_hbm_gb": server.total_hbm_gb,
                "total_hbm_gib": round(server.total_hbm_gb * GB_TO_GIB, 2),
                "nvlink_bandwidth_tbps": server.nvlink_bandwidth_tbps,
                "system_memory_tb": server.system_memory_tb
            },
            "storage": {
                "name": storage.name,
                "type": storage.type,
                "iops_read": storage.iops_read,
                "iops_write": storage.iops_write,
                "throughput_read_gbps": storage.throughput_read_gbps,
                "throughput_write_gbps": storage.throughput_write_gbps,
                "latency_read_ms_p99": storage.latency_read_ms_p99,
                "latency_write_ms_p99": storage.latency_write_ms_p99
            },
            "nfr": {
                "concurrency": concurrency,
                "effective_context": effective_context,
                "kv_precision": kv_precision,
                "kv_budget_ratio": kv_budget_ratio,
                "runtime_overhead_gib": runtime_overhead_gib,
                "peak_headroom_ratio": peak_headroom_ratio
            }
        },
        "parameter_dictionary": get_parameter_dictionary(),
        "scenarios": {
            "minimum": scenario_to_dict(scenarios["minimum"]),
            "recommended": scenario_to_dict(scenarios["recommended"]),
            "ideal": scenario_to_dict(scenarios["ideal"])
        },
        "alerts": list(set([w for s in scenarios.values() for w in s.warnings]))
    }


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    """Parse argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Dimensionamento Avançado de Inferência LLM com Racional de Cálculo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Seleção
    parser.add_argument("--model", required=True, help="Nome do modelo")
    parser.add_argument("--server", required=True, help="Nome do servidor")
    parser.add_argument("--storage", required=True, help="Perfil de storage")
    
    # NFRs
    parser.add_argument("--concurrency", type=int, required=True, help="Sessões simultâneas")
    parser.add_argument("--effective-context", type=int, required=True, help="Contexto efetivo (tokens)")
    parser.add_argument("--kv-precision", choices=["fp8", "fp16", "bf16", "int8"], default="fp8")
    parser.add_argument("--kv-budget-ratio", type=float, default=0.70)
    parser.add_argument("--runtime-overhead-gib", type=float, default=120)
    parser.add_argument("--peak-headroom-ratio", type=float, default=0.20)
    
    # Arquivos
    parser.add_argument("--models-file", default="models.json")
    parser.add_argument("--servers-file", default="servers.json")
    parser.add_argument("--storage-file", default="storage.json")
    
    # Output
    parser.add_argument("--output-json-file", help="Salvar JSON em arquivo")
    parser.add_argument("--output-markdown-file", help="Salvar relatório em Markdown (.md)")
    parser.add_argument("--executive-report", action="store_true", help="Gerar relatório executivo (para diretoria)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--markdown-only", action="store_true", help="Gerar apenas relatório Markdown (sem JSON no stdout)")
    
    return parser.parse_args()


def print_executive_summary(
    model: Model,
    server: Server,
    scenarios: Dict[str, Any],
    concurrency: int,
    effective_context: int,
    kv_precision: str
) -> None:
    """Imprime resumo executivo no terminal (saída interativa)."""
    
    print("=" * 80)
    print("RESUMO EXECUTIVO - SIZING DE INFERÊNCIA LLM")
    print("=" * 80)
    print()
    
    # Cabeçalho
    print(f"Modelo:              {model.name}")
    print(f"Servidor:            {server.name}")
    print(f"Contexto Efetivo:    {effective_context:,} tokens")
    print(f"Concorrência Alvo:   {concurrency:,} sessões simultâneas")
    print(f"Precisão KV Cache:   {kv_precision.upper()}")
    print()
    
    # Tabela resumida
    print("-" * 100)
    print(f"{'Cenário':<15} {'Nós DGX':>10} {'Energia (kW)':>14} {'Rack (U)':>12} {'Sessões/Nó':>12} {'KV/Sessão (GiB)':>16}")
    print("-" * 100)
    
    for scenario_key, scenario_label in [("minimum", "MÍNIMO"), ("recommended", "RECOMENDADO"), ("ideal", "IDEAL")]:
        sc = scenarios[scenario_key]
        
        print(f"{scenario_label:<15} {sc.nodes_final:>10} {sc.total_power_kw:>14.1f} {sc.total_rack_u:>12} {sc.sessions_per_node:>12} {sc.kv_per_session_gib:>16.2f}")
    
    print("-" * 100)
    print()
    
    # Status final
    rec = scenarios["recommended"]
    if rec.sessions_per_node == 0:
        print("⚠️  ERRO: Não cabe nem 1 sessão por nó. Ajuste contexto, precisão ou servidor.")
    elif rec.nodes_final <= 3:
        print(f"✓ Cenário RECOMENDADO ({rec.nodes_final} nós, {rec.total_power_kw:.1f} kW, {rec.total_rack_u}U) atende os requisitos com tolerância a falhas (N+1).")
    elif rec.nodes_final <= 10:
        print(f"✓ Cenário RECOMENDADO ({rec.nodes_final} nós, {rec.total_power_kw:.1f} kW, {rec.total_rack_u}U) atende os requisitos. Considere otimizações para grandes cargas.")
    else:
        print(f"⚠️  Cenário RECOMENDADO requer {rec.nodes_final} nós ({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U). Revise NFRs ou considere modelo menor.")
    
    print()
    
    # Alertas críticos (apenas os mais importantes)
    critical_alerts = []
    for sc in scenarios.values():
        for warning in sc.warnings:
            if "excede" in warning or "ERRO" in warning or "dobra" in warning:
                if warning not in critical_alerts:
                    critical_alerts.append(warning)
    
    if critical_alerts:
        print("ALERTAS CRÍTICOS:")
        for alert in critical_alerts[:3]:  # Máximo 3 alertas
            print(f"  • {alert}")
        print()
    
    print("=" * 80)


def save_reports(
    model: Model,
    server: Server,
    storage: StorageProfile,
    scenarios: Dict[str, Any],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    kv_budget_ratio: float,
    runtime_overhead_gib: float,
    peak_headroom_ratio: float,
    verbose: bool = False
) -> Tuple[str, str]:
    """
    Salva relatórios completos em arquivos.
    
    Returns:
        (caminho_txt, caminho_json)
    """
    # Criar diretório relatorios/ se não existir
    os.makedirs("relatorios", exist_ok=True)
    
    # Gerar timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Nomes dos arquivos
    filename_base = f"sizing_{model.name}_{server.name}_{timestamp}"
    txt_path = os.path.join("relatorios", f"{filename_base}.txt")
    json_path = os.path.join("relatorios", f"{filename_base}.json")
    
    # Gerar relatório completo em texto
    report_text = format_report(
        model=model,
        server=server,
        storage=storage,
        scenarios=scenarios,
        concurrency=concurrency,
        effective_context=effective_context,
        kv_precision=kv_precision,
        verbose=verbose
    )
    
    # Gerar JSON completo
    json_output = scenarios_to_dict(
        model=model,
        server=server,
        storage=storage,
        scenarios=scenarios,
        concurrency=concurrency,
        effective_context=effective_context,
        kv_precision=kv_precision,
        kv_budget_ratio=kv_budget_ratio,
        runtime_overhead_gib=runtime_overhead_gib,
        peak_headroom_ratio=peak_headroom_ratio
    )
    
    # Salvar arquivos
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    return txt_path, json_path


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
        print(f"ERRO: JSON inválido: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validar seleções
    if args.model not in models:
        print(f"ERRO: Modelo '{args.model}' não encontrado.", file=sys.stderr)
        print(f"Disponíveis: {', '.join(models.keys())}", file=sys.stderr)
        sys.exit(1)
    
    if args.server not in servers:
        print(f"ERRO: Servidor '{args.server}' não encontrado.", file=sys.stderr)
        print(f"Disponíveis: {', '.join(servers.keys())}", file=sys.stderr)
        sys.exit(1)
    
    if args.storage not in storage_profiles:
        print(f"ERRO: Storage '{args.storage}' não encontrado.", file=sys.stderr)
        print(f"Disponíveis: {', '.join(storage_profiles.keys())}", file=sys.stderr)
        sys.exit(1)
    
    # Calcular sizing (3 cenários)
    scenarios = calculate_sizing_all_scenarios(
        model=models[args.model],
        server=servers[args.server],
        storage=storage_profiles[args.storage],
        concurrency=args.concurrency,
        effective_context=args.effective_context,
        kv_precision=args.kv_precision,
        kv_budget_ratio=args.kv_budget_ratio,
        runtime_overhead_gib=args.runtime_overhead_gib,
        peak_headroom_ratio=args.peak_headroom_ratio,
        verbose=args.verbose
    )
    
    # Salvar relatórios completos automaticamente
    txt_path, json_path = save_reports(
        model=models[args.model],
        server=servers[args.server],
        storage=storage_profiles[args.storage],
        scenarios=scenarios,
        concurrency=args.concurrency,
        effective_context=args.effective_context,
        kv_precision=args.kv_precision,
        kv_budget_ratio=args.kv_budget_ratio,
        runtime_overhead_gib=args.runtime_overhead_gib,
        peak_headroom_ratio=args.peak_headroom_ratio,
        verbose=args.verbose
    )
    
    # Imprimir resumo executivo no terminal
    print_executive_summary(
        model=models[args.model],
        server=servers[args.server],
        scenarios=scenarios,
        concurrency=args.concurrency,
        effective_context=args.effective_context,
        kv_precision=args.kv_precision
    )
    
    # Mensagem final indicando onde estão os relatórios
    print(f"📄 Relatórios completos salvos em:")
    print(f"   • Texto:  {txt_path}")
    print(f"   • JSON:   {json_path}")
    print()
    
    # Manter compatibilidade com flags legadas (se usuário explicitamente pedir arquivos específicos)
    if args.output_markdown_file:
        markdown_report = format_report_markdown(
            model=models[args.model],
            server=servers[args.server],
            storage=storage_profiles[args.storage],
            scenarios=scenarios,
            concurrency=args.concurrency,
            effective_context=args.effective_context,
            kv_precision=args.kv_precision,
            verbose=args.verbose
        )
        with open(args.output_markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        print(f"   • Markdown: {args.output_markdown_file}")
    
    if args.executive_report:
        executive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        executive_filename = f"relatorios/executive_{models[args.model].name}_{servers[args.server].name}_{executive_timestamp}.md"
        
        executive_report = format_executive_report(
            model=models[args.model],
            server=servers[args.server],
            storage=storage_profiles[args.storage],
            scenarios=scenarios,
            concurrency=args.concurrency,
            effective_context=args.effective_context,
            kv_precision=args.kv_precision,
            kv_budget_ratio=args.kv_budget_ratio,
            runtime_overhead_gib=args.runtime_overhead_gib,
            verbose=args.verbose
        )
        
        with open(executive_filename, "w", encoding="utf-8") as f:
            f.write(executive_report)
        
        print(f"   • Executivo: {executive_filename}")
        print()


if __name__ == "__main__":
    main()
