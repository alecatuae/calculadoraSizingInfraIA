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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    
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
    
    # Gerar outputs
    if not args.json_only:
        report = format_report(
            model=models[args.model],
            server=servers[args.server],
            storage=storage_profiles[args.storage],
            scenarios=scenarios,
            concurrency=args.concurrency,
            effective_context=args.effective_context,
            kv_precision=args.kv_precision,
            verbose=args.verbose
        )
        print(report)
        print("\n")
        print("=" * 100)
        print("JSON OUTPUT (3 CENÁRIOS):")
        print("=" * 100)
    
    # JSON
    json_output = scenarios_to_dict(
        model=models[args.model],
        server=servers[args.server],
        storage=storage_profiles[args.storage],
        scenarios=scenarios,
        concurrency=args.concurrency,
        effective_context=args.effective_context,
        kv_precision=args.kv_precision,
        kv_budget_ratio=args.kv_budget_ratio,
        runtime_overhead_gib=args.runtime_overhead_gib,
        peak_headroom_ratio=args.peak_headroom_ratio
    )
    
    json_str = json.dumps(json_output, indent=2, ensure_ascii=False)
    print(json_str)
    
    # Salvar em arquivo se solicitado
    if args.output_json_file:
        with open(args.output_json_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\nJSON salvo em: {args.output_json_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
