"""
Cálculos de dimensionamento de storage para operação de inferência de LLM.

Storage é um recurso crítico para:
- Operação contínua (pesos do modelo, cache, logs)
- Estabilidade (recuperação de falhas, scale-out)
- Governança (auditoria, métricas, traces)
"""

from dataclasses import dataclass
from typing import Dict, Any
from .models import ModelSpec
from .servers import ServerSpec
from .storage import StorageProfile


@dataclass
class StorageRequirements:
    """Requisitos de storage calculados para um cenário."""
    
    # Volumetria BASE (TB) - valores técnicos calculados
    storage_model_base_tb: float
    storage_cache_base_tb: float
    storage_logs_base_tb: float
    storage_operational_base_tb: float
    platform_volume_total_tb: float  # Volume da plataforma (SO, AI Enterprise, runtime)
    storage_total_base_tb: float
    
    # Volumetria RECOMENDADA (TB) - valores estratégicos com margem
    storage_model_recommended_tb: float
    storage_cache_recommended_tb: float
    storage_logs_recommended_tb: float
    storage_operational_recommended_tb: float
    platform_volume_recommended_tb: float  # Volume da plataforma com margem
    storage_total_recommended_tb: float
    
    # Detalhamento da plataforma (valores por servidor, sem margem)
    platform_per_server_gb: float
    platform_per_server_tb: float
    
    # Margem aplicada
    margin_applied: bool
    margin_percent: float
    
    # IOPS
    iops_read_peak: int
    iops_write_peak: int
    iops_read_steady: int
    iops_write_steady: int
    
    # Throughput (GB/s)
    throughput_read_peak_gbps: float
    throughput_write_peak_gbps: float
    throughput_read_steady_gbps: float
    throughput_write_steady_gbps: float
    
    # Metadata
    rationale: Dict[str, Any]


def calc_storage_model_tb(
    model: ModelSpec,
    weights_precision: str,
    num_nodes: int,
    replicas_per_node: int = 1
) -> tuple[float, Dict[str, Any]]:
    """
    Calcula volumetria de storage para pesos do modelo.
    
    Considera:
    - Pesos do modelo em disco (checkpoints/shards)
    - Replicação para tolerância a falhas
    - Margem para versionamento (rollback)
    
    Retorna: (storage_tb, rationale)
    """
    # Obter tamanho dos pesos baseado na precisão
    weights_gib = 0.0
    if weights_precision == "fp16" or weights_precision == "bf16":
        weights_gib = model.weights_memory_gib_fp16 or 0.0
    elif weights_precision == "fp8":
        weights_gib = model.weights_memory_gib_fp8 or 0.0
    elif weights_precision == "int8":
        weights_gib = model.weights_memory_gib_int8 or 0.0
    elif weights_precision == "int4":
        weights_gib = model.weights_memory_gib_int4 or 0.0
    
    # Se não disponível, estimar
    if weights_gib == 0.0 and model.total_params_b:
        bytes_per_param = {
            "fp16": 2.0, "bf16": 2.0,
            "fp8": 1.0, "int8": 1.0,
            "int4": 0.5
        }.get(weights_precision, 2.0)
        weights_gib = (model.total_params_b * 1e9 * bytes_per_param) / (2**30)
    
    # Converter GiB para TB
    weights_tb = weights_gib / 1024.0
    
    # Total de réplicas na plataforma
    total_replicas = num_nodes * replicas_per_node
    
    # Storage base: 1 cópia por réplica + 1 cópia de backup + 1 versão anterior
    # Fator de replicação conservador: 2.5x
    storage_factor = 2.5
    storage_model_tb = weights_tb * total_replicas * storage_factor
    
    rationale = {
        "formula": "storage_model_tb = weights_tb * total_replicas * storage_factor",
        "inputs": {
            "weights_gib": round(weights_gib, 2),
            "weights_tb": round(weights_tb, 4),
            "num_nodes": num_nodes,
            "replicas_per_node": replicas_per_node,
            "total_replicas": total_replicas,
            "storage_factor": storage_factor
        },
        "assumption": "Fator 2.5x inclui: 1 cópia ativa + 1 backup + 0.5 para versionamento/rollback",
        "operational_meaning": f"Armazena {total_replicas} réplicas do modelo com backup e capacidade de rollback. Crítico para restart rápido e scale-out."
    }
    
    return storage_model_tb, rationale


def calc_storage_cache_tb(
    num_nodes: int,
    sessions_per_node: int,
    scenario: str = "recomendado"
) -> tuple[float, Dict[str, Any]]:
    """
    Calcula volumetria de cache local/runtime por nó.
    
    Considera:
    - Cache de engine compilado (TensorRT-LLM, NIM)
    - Cache de artefatos (tokenizers, configs)
    - Cache temporário de execução
    
    Retorna: (storage_cache_tb, rationale)
    """
    # Base: 50 GiB por nó (engine compilado + artefatos)
    base_cache_gib = 50.0
    
    # Adicional por sessão: 1 GiB (buffers temporários)
    per_session_cache_gib = 1.0
    
    # Fator de cenário
    scenario_factor = {
        "minimo": 1.0,      # Apenas cache essencial
        "recomendado": 1.5, # Margem operacional
        "ideal": 2.0        # Margem ampla para picos
    }.get(scenario.lower(), 1.5)
    
    cache_per_node_gib = (base_cache_gib + per_session_cache_gib * sessions_per_node) * scenario_factor
    cache_total_gib = cache_per_node_gib * num_nodes
    cache_total_tb = cache_total_gib / 1024.0
    
    rationale = {
        "formula": "storage_cache_tb = ((base_cache + per_session_cache * sessions) * scenario_factor * num_nodes) / 1024",
        "inputs": {
            "base_cache_gib": base_cache_gib,
            "per_session_cache_gib": per_session_cache_gib,
            "sessions_per_node": sessions_per_node,
            "num_nodes": num_nodes,
            "scenario_factor": scenario_factor,
            "cache_per_node_gib": round(cache_per_node_gib, 2)
        },
        "assumption": f"Cenário {scenario}: fator {scenario_factor}x para cache local e temporário",
        "operational_meaning": "Cache de engine compilado, artefatos e buffers temporários. Essencial para latência e throughput."
    }
    
    return cache_total_tb, rationale


def calc_storage_logs_tb(
    concurrency: int,
    num_nodes: int,
    retention_days: int,
    scenario: str = "recomendado"
) -> tuple[float, Dict[str, Any]]:
    """
    Calcula volumetria de logs, métricas e auditoria.
    
    Considera:
    - Logs de requisições
    - Métricas de inferência
    - Traces (se habilitado)
    - Retenção por cenário
    
    Retorna: (storage_logs_tb, rationale)
    """
    # Retenção por cenário
    retention_policy = {
        "minimo": 7,        # 7 dias
        "recomendado": 30,  # 30 dias
        "ideal": 90         # 90 dias
    }
    retention_days = retention_policy.get(scenario.lower(), retention_days)
    
    # Estimativa: 10 KB por requisição (log + métricas)
    bytes_per_request = 10 * 1024
    
    # Requisições por dia (assumindo operação 24/7 com concorrência média)
    # Tempo médio por requisição: 2 segundos (estimativa conservadora)
    avg_request_duration_sec = 2.0
    requests_per_second = concurrency / avg_request_duration_sec
    requests_per_day = requests_per_second * 86400
    
    # Storage de logs por dia
    logs_per_day_bytes = requests_per_day * bytes_per_request
    logs_per_day_gib = logs_per_day_bytes / (2**30)
    
    # Storage total
    logs_total_gib = logs_per_day_gib * retention_days
    logs_total_tb = logs_total_gib / 1024.0
    
    rationale = {
        "formula": "storage_logs_tb = (concurrency / avg_duration * 86400 * bytes_per_req * retention_days) / (1024^4)",
        "inputs": {
            "concurrency": concurrency,
            "num_nodes": num_nodes,
            "avg_request_duration_sec": avg_request_duration_sec,
            "requests_per_second": round(requests_per_second, 2),
            "requests_per_day": round(requests_per_day, 0),
            "bytes_per_request": bytes_per_request,
            "retention_days": retention_days,
            "logs_per_day_gib": round(logs_per_day_gib, 2)
        },
        "assumption": f"Cenário {scenario}: retenção de {retention_days} dias. 10KB/req (logs+métricas), duração média 2s/req.",
        "operational_meaning": "Logs e métricas são críticos para debugging, auditoria e conformidade. Retenção inadequada compromete troubleshooting."
    }
    
    return logs_total_tb, rationale


def calc_storage_operational_tb(
    num_nodes: int,
    scenario: str = "recomendado"
) -> tuple[float, Dict[str, Any]]:
    """
    Calcula volumetria de dados operacionais (configs, metadados, artefatos auxiliares).
    
    Retorna: (storage_operational_tb, rationale)
    """
    # Base por nó: 10 GiB (configs, metadados, artefatos)
    operational_per_node_gib = 10.0
    
    # Fator de cenário
    scenario_factor = {
        "minimo": 1.0,
        "recomendado": 1.5,
        "ideal": 2.0
    }.get(scenario.lower(), 1.5)
    
    operational_total_gib = operational_per_node_gib * num_nodes * scenario_factor
    operational_total_tb = operational_total_gib / 1024.0
    
    rationale = {
        "formula": "storage_operational_tb = (operational_per_node * num_nodes * scenario_factor) / 1024",
        "inputs": {
            "operational_per_node_gib": operational_per_node_gib,
            "num_nodes": num_nodes,
            "scenario_factor": scenario_factor
        },
        "assumption": f"Cenário {scenario}: fator {scenario_factor}x para dados operacionais",
        "operational_meaning": "Configurações, metadados e artefatos auxiliares. Essencial para orquestração e recuperação."
    }
    
    return operational_total_tb, rationale


def calc_storage_iops(
    concurrency: int,
    num_nodes: int,
    storage_model_tb: float,
    scenario: str = "recomendado"
) -> tuple[Dict[str, int], Dict[str, Any]]:
    """
    Calcula IOPS de leitura e escrita para operação de inferência.
    
    Considera:
    - Leitura: startup, restart, scale-out (pesos do modelo)
    - Escrita: logs, métricas, checkpoints
    - Steady-state vs. peak
    
    Retorna: (iops_dict, rationale)
    """
    # IOPS de leitura PEAK (startup/restart de múltiplos nós simultâneos)
    # Assumindo restart de 25% dos nós simultaneamente no pior caso
    nodes_restarting = max(1, int(num_nodes * 0.25))
    
    # IOPS por nó durante startup: 50k IOPS (leitura de pesos)
    iops_read_per_node_startup = 50000
    iops_read_peak = nodes_restarting * iops_read_per_node_startup
    
    # IOPS de leitura STEADY (operação normal, cache hits)
    # Muito baixo, pesos já em memória: 1k IOPS por nó
    iops_read_per_node_steady = 1000
    iops_read_steady = num_nodes * iops_read_per_node_steady
    
    # IOPS de escrita PEAK (logs + métricas em burst)
    # Estimativa: 1 write op por requisição completada
    # Burst: 2x concurrency
    writes_per_second_peak = concurrency * 2
    iops_write_peak = int(writes_per_second_peak)
    
    # IOPS de escrita STEADY (operação normal)
    writes_per_second_steady = concurrency
    iops_write_steady = int(writes_per_second_steady)
    
    # Ajuste por cenário
    scenario_factor = {
        "minimo": 1.0,
        "recomendado": 1.5,
        "ideal": 2.0
    }.get(scenario.lower(), 1.5)
    
    iops_read_peak = int(iops_read_peak * scenario_factor)
    iops_write_peak = int(iops_write_peak * scenario_factor)
    
    iops_dict = {
        "iops_read_peak": iops_read_peak,
        "iops_write_peak": iops_write_peak,
        "iops_read_steady": iops_read_steady,
        "iops_write_steady": iops_write_steady
    }
    
    rationale = {
        "formula": "IOPS = f(nodes_restarting, concurrency, scenario_factor)",
        "inputs": {
            "concurrency": concurrency,
            "num_nodes": num_nodes,
            "nodes_restarting": nodes_restarting,
            "iops_read_per_node_startup": iops_read_per_node_startup,
            "iops_read_per_node_steady": iops_read_per_node_steady,
            "scenario_factor": scenario_factor
        },
        "assumption": f"Cenário {scenario}: {nodes_restarting} nós reiniciando simultaneamente (25%). Fator {scenario_factor}x para margem.",
        "operational_meaning": f"IOPS pico ({iops_read_peak:,} R / {iops_write_peak:,} W) suporta restart de {nodes_restarting} nós + burst de logs. IOPS steady ({iops_read_steady:,} R / {iops_write_steady:,} W) para operação normal."
    }
    
    return iops_dict, rationale


def calc_storage_throughput(
    concurrency: int,
    num_nodes: int,
    storage_model_tb: float,
    scenario: str = "recomendado"
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calcula throughput de leitura e escrita (GB/s) para operação de inferência.
    
    Considera:
    - Leitura: carga de pesos durante startup/restart
    - Escrita: flush de logs, checkpoints
    - Steady-state vs. peak
    
    Retorna: (throughput_dict, rationale)
    """
    # Throughput de leitura PEAK (startup/restart)
    # Meta: carregar modelo completo em < 60 segundos
    # Assumindo restart de 25% dos nós simultaneamente
    nodes_restarting = max(1, int(num_nodes * 0.25))
    model_per_node_gib = (storage_model_tb * 1024) / num_nodes
    target_load_time_sec = 60.0
    
    throughput_read_per_node = model_per_node_gib / target_load_time_sec
    throughput_read_peak_gbps = throughput_read_per_node * nodes_restarting
    
    # Throughput de leitura STEADY (cache hits, minimal disk read)
    throughput_read_steady_gbps = 0.5 * num_nodes  # 0.5 GB/s por nó
    
    # Throughput de escrita PEAK (flush de logs em batch)
    # Estimativa: 10 KB por requisição, flush a cada 10 segundos
    bytes_per_request = 10 * 1024
    requests_per_flush = concurrency * 10  # 10 segundos de buffer
    bytes_per_flush = bytes_per_request * requests_per_flush
    throughput_write_peak_gbps = (bytes_per_flush / 10) / (2**30)  # GB/s
    
    # Throughput de escrita STEADY (flush contínuo)
    throughput_write_steady_gbps = throughput_write_peak_gbps * 0.5
    
    # Ajuste por cenário
    scenario_factor = {
        "minimo": 1.0,
        "recomendado": 1.5,
        "ideal": 2.0
    }.get(scenario.lower(), 1.5)
    
    throughput_read_peak_gbps *= scenario_factor
    throughput_write_peak_gbps *= scenario_factor
    
    throughput_dict = {
        "throughput_read_peak_gbps": round(throughput_read_peak_gbps, 2),
        "throughput_write_peak_gbps": round(throughput_write_peak_gbps, 2),
        "throughput_read_steady_gbps": round(throughput_read_steady_gbps, 2),
        "throughput_write_steady_gbps": round(throughput_write_steady_gbps, 2)
    }
    
    rationale = {
        "formula": "Throughput = f(model_size, nodes_restarting, target_load_time, log_flush_rate)",
        "inputs": {
            "storage_model_tb": round(storage_model_tb, 2),
            "model_per_node_gib": round(model_per_node_gib, 2),
            "num_nodes": num_nodes,
            "nodes_restarting": nodes_restarting,
            "target_load_time_sec": target_load_time_sec,
            "scenario_factor": scenario_factor
        },
        "assumption": f"Cenário {scenario}: carregar modelo em <{target_load_time_sec}s. {nodes_restarting} nós reiniciando (25%). Fator {scenario_factor}x.",
        "operational_meaning": f"Throughput pico ({throughput_dict['throughput_read_peak_gbps']:.2f} R / {throughput_dict['throughput_write_peak_gbps']:.2f} W GB/s) garante restart rápido. Throughput steady ({throughput_dict['throughput_read_steady_gbps']:.2f} R / {throughput_dict['throughput_write_steady_gbps']:.2f} W GB/s) para operação contínua."
    }
    
    return throughput_dict, rationale


def calc_storage_requirements(
    model: ModelSpec,
    server: ServerSpec,
    storage: StorageProfile,
    concurrency: int,
    num_nodes: int,
    sessions_per_node: int,
    weights_precision: str,
    replicas_per_node: int,
    capacity_policy,  # CapacityPolicy instance
    platform_storage_profile,  # PlatformStorageProfile instance
    scenario: str = "recomendado",
    retention_days: int = 30
) -> StorageRequirements:
    """
    Calcula requisitos completos de storage para um cenário de inferência.
    
    Args:
        model: Especificação do modelo
        server: Especificação do servidor
        storage: Perfil de storage
        concurrency: Concorrência alvo
        num_nodes: Número de nós
        sessions_per_node: Sessões simultâneas por nó
        weights_precision: Precisão dos pesos
        replicas_per_node: Réplicas por nó
        capacity_policy: Política de margem de capacidade
        platform_storage_profile: Profile de storage da plataforma (SO, AI Enterprise, runtime)
        scenario: "minimo", "recomendado" ou "ideal"
        retention_days: Dias de retenção de logs (sobrescrito por cenário)
    
    Returns:
        StorageRequirements com todos os cálculos e rationale
    """
    # Calcular volumetria BASE (valores técnicos)
    storage_model_base_tb, rationale_model = calc_storage_model_tb(
        model, weights_precision, num_nodes, replicas_per_node
    )
    
    storage_cache_base_tb, rationale_cache = calc_storage_cache_tb(
        num_nodes, sessions_per_node, scenario
    )
    
    storage_logs_base_tb, rationale_logs = calc_storage_logs_tb(
        concurrency, num_nodes, retention_days, scenario
    )
    
    storage_operational_base_tb, rationale_operational = calc_storage_operational_tb(
        num_nodes, scenario
    )
    
    # Calcular volume da plataforma (SO, AI Enterprise, runtime, etc.)
    platform_volume_total_tb = platform_storage_profile.calc_total_platform_volume_tb(num_nodes)
    platform_rationale = platform_storage_profile.get_rationale(num_nodes)
    
    # Storage Total BASE inclui plataforma + modelo + cache + logs + operational
    storage_total_base_tb = (
        storage_model_base_tb + 
        storage_cache_base_tb + 
        storage_logs_base_tb + 
        storage_operational_base_tb +
        platform_volume_total_tb
    )
    
    # Aplicar margem de capacidade (valores estratégicos)
    storage_model_recommended_tb = capacity_policy.apply_margin(
        storage_model_base_tb, "storage_model"
    )
    storage_cache_recommended_tb = capacity_policy.apply_margin(
        storage_cache_base_tb, "storage_cache"
    )
    storage_logs_recommended_tb = capacity_policy.apply_margin(
        storage_logs_base_tb, "storage_logs"
    )
    storage_operational_recommended_tb = capacity_policy.apply_margin(
        storage_operational_base_tb, "storage_operational"
    )
    # Volume da plataforma também recebe margem
    platform_volume_recommended_tb = capacity_policy.apply_margin(
        platform_volume_total_tb, "storage_total"  # Usa o mesmo target para aplicar margem
    )
    storage_total_recommended_tb = capacity_policy.apply_margin(
        storage_total_base_tb, "storage_total"
    )
    
    # Calcular IOPS (baseado em valores recomendados para garantir margem)
    iops_dict, rationale_iops = calc_storage_iops(
        concurrency, num_nodes, storage_total_recommended_tb, scenario
    )
    
    # Calcular Throughput (baseado em valores recomendados)
    throughput_dict, rationale_throughput = calc_storage_throughput(
        concurrency, num_nodes, storage_total_recommended_tb, scenario
    )
    
    # Consolidar rationale
    rationale = {
        "storage_model": rationale_model,
        "storage_cache": rationale_cache,
        "storage_logs": rationale_logs,
        "storage_operational": rationale_operational,
        "platform_storage": platform_rationale,
        "storage_total": {
            "formula": "storage_total_tb = storage_model + storage_cache + storage_logs + storage_operational + platform_volume",
            "inputs": {
                "storage_model_base_tb": round(storage_model_base_tb, 3),
                "storage_cache_base_tb": round(storage_cache_base_tb, 3),
                "storage_logs_base_tb": round(storage_logs_base_tb, 3),
                "storage_operational_base_tb": round(storage_operational_base_tb, 3),
                "platform_volume_total_tb": round(platform_volume_total_tb, 3),
                "storage_total_base_tb": round(storage_total_base_tb, 3)
            },
            "assumption": f"Cenário {scenario}: soma de todos os componentes de storage incluindo volume estrutural da plataforma. "
                         f"Margem de {capacity_policy.margin_percent*100:.0f}% aplicada conforme política de capacidade.",
            "operational_meaning": f"Total BASE de {storage_total_base_tb:.2f} TB (inclui {platform_volume_total_tb:.2f} TB de plataforma). "
                                  f"Total RECOMENDADO de {storage_total_recommended_tb:.2f} TB com margem de {capacity_policy.margin_percent*100:.0f}% para crescimento, retenção adicional e resiliência. "
                                  f"Subdimensionamento compromete tempo de recuperação."
        },
        "capacity_policy": {
            "margin_percent": capacity_policy.margin_percent,
            "source": capacity_policy.source,
            "notes": capacity_policy.notes
        },
        "iops": rationale_iops,
        "throughput": rationale_throughput
    }
    
    return StorageRequirements(
        # Valores BASE
        storage_model_base_tb=storage_model_base_tb,
        storage_cache_base_tb=storage_cache_base_tb,
        storage_logs_base_tb=storage_logs_base_tb,
        storage_operational_base_tb=storage_operational_base_tb,
        platform_volume_total_tb=platform_volume_total_tb,
        storage_total_base_tb=storage_total_base_tb,
        # Valores RECOMENDADOS (com margem)
        storage_model_recommended_tb=storage_model_recommended_tb,
        storage_cache_recommended_tb=storage_cache_recommended_tb,
        storage_logs_recommended_tb=storage_logs_recommended_tb,
        storage_operational_recommended_tb=storage_operational_recommended_tb,
        platform_volume_recommended_tb=platform_volume_recommended_tb,
        storage_total_recommended_tb=storage_total_recommended_tb,
        # Detalhamento da plataforma
        platform_per_server_gb=platform_storage_profile.total_per_server_gb,
        platform_per_server_tb=platform_storage_profile.total_per_server_tb,
        # Margem aplicada
        margin_applied=True,
        margin_percent=capacity_policy.margin_percent,
        # IOPS e Throughput
        iops_read_peak=iops_dict["iops_read_peak"],
        iops_write_peak=iops_dict["iops_write_peak"],
        iops_read_steady=iops_dict["iops_read_steady"],
        iops_write_steady=iops_dict["iops_write_steady"],
        throughput_read_peak_gbps=throughput_dict["throughput_read_peak_gbps"],
        throughput_write_peak_gbps=throughput_dict["throughput_write_peak_gbps"],
        throughput_read_steady_gbps=throughput_dict["throughput_read_steady_gbps"],
        throughput_write_steady_gbps=throughput_dict["throughput_write_steady_gbps"],
        rationale=rationale
    )
